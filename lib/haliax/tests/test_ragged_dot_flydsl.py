# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

"""``ragged_dot(implementation="flydsl")``: jax-flydsl's grouped GEMMs on AMD Instinct GPUs. The kernel tests skip
without such a GPU and jax-flydsl; the selection and error-path tests run anywhere."""

import importlib
import sys
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from haliax.nn import ragged_dot

ragged_dot_module = importlib.import_module("haliax.nn.ragged_dot")


@pytest.fixture(autouse=True)
def _no_ragged_dot_impl_override(monkeypatch):
    """RAGGED_DOT_IMPL beats an explicit implementation=, so a launch env that sets it would test the wrong kernel."""
    monkeypatch.delenv("RAGGED_DOT_IMPL", raising=False)


def _to_cpu_f32(*arrays):
    cpu = jax.devices("cpu")[0]
    return tuple(jax.device_put(x.astype(jnp.float32), cpu) for x in arrays)


def _require_flydsl_gpu():
    if jax.default_backend() != "gpu" or "Instinct" not in jax.devices()[0].device_kind:
        pytest.skip("requires an AMD Instinct GPU")
    pytest.importorskip("jax_flydsl", reason="requires the optional jax-flydsl package")


def _skewed_group_sizes(num_groups: int, active_rows: int, seed: int) -> list[int]:
    """Lognormal group sizes summing to ``active_rows``, with every fourth group empty."""
    rng = np.random.default_rng(seed)
    weights = rng.lognormal(0.0, 1.0, num_groups)
    weights[::4] = 0.0
    sizes = np.floor(weights / weights.sum() * active_rows).astype(np.int64)
    sizes[np.argmax(sizes)] += active_rows - sizes.sum()
    return sizes.tolist()


# (rows, K, N, group sizes). Each sums to at most rows, and the rows past the sum hold NaN.
_FLYDSL_CASES = {
    # 1000 rows exercise ragged_dot's padding to a multiple of 512; one group is empty.
    "small_unaligned": (1000, 256, 384, [300, 0, 517, 100]),
    "small_skewed": (1024, 128, 192, [1, 0, 1000, 3, 0, 20]),
    "small_all_empty": (512, 128, 128, [0, 0, 0, 0]),
    # June's 32 local experts and the up (2560 x 2560) and down (1280 x 2560) projections, 97% of the rows in groups.
    "june_up": (8192, 2560, 2560, _skewed_group_sizes(32, 7946, seed=0)),
    "june_down": (32768, 1280, 2560, _skewed_group_sizes(32, 31784, seed=1)),
}
# bf16 rounds to 8 significant bits: rounding the f32 accumulator to bf16 is off by up to 2**-8 (3.9e-3) of each
# element and about 1.4e-3 on average. On gfx950 the max error is 2-3.4e-3 of the largest magnitude.
_FLYDSL_MAX_ERROR = 5e-3
_FLYDSL_MEAN_ERROR = 2e-3


@pytest.mark.parametrize("rows, k, n, group_sizes", list(_FLYDSL_CASES.values()), ids=list(_FLYDSL_CASES))
def test_flydsl_ragged_dot_value_and_gradients_match_xla_f32(rows, k, n, group_sizes):
    """bf16 FlyDSL output and VJP match ``implementation="xla"`` in float32 on the same bf16 values."""
    _require_flydsl_gpu()
    active = sum(group_sizes)
    key_lhs, key_rhs, key_dout = jax.random.split(jax.random.key(2), 3)
    lhs = jax.random.normal(key_lhs, (rows, k), jnp.float32).astype(jnp.bfloat16)
    rhs = (jax.random.normal(key_rhs, (len(group_sizes), k, n), jnp.float32) / np.sqrt(k)).astype(jnp.bfloat16)
    dout = jax.random.normal(key_dout, (rows, n), jnp.float32).astype(jnp.bfloat16)
    sizes = jnp.array(group_sizes, dtype=jnp.int32)
    in_groups = (jnp.arange(rows) < active)[:, None]
    # No group may read the rows past the last group.
    poisoned_lhs = jnp.where(in_groups, lhs, jnp.nan)
    poisoned_dout = jnp.where(in_groups, dout, jnp.nan)

    @jax.jit
    def flydsl_vjp(lhs, rhs, sizes, dout):
        out, vjp = jax.vjp(lambda a, b: ragged_dot(a, b, sizes, implementation="flydsl"), lhs, rhs)
        return (out, *vjp(dout))

    @jax.jit
    def xla_f32_vjp(lhs, rhs, sizes, dout):
        with jax.default_matmul_precision("highest"):
            out, vjp = jax.vjp(lambda a, b: ragged_dot(a, b, sizes, implementation="xla"), lhs, rhs)
            return (out, *vjp(dout))

    actual = flydsl_vjp(poisoned_lhs, rhs, sizes, poisoned_dout)
    expected = _to_cpu_f32(
        *xla_f32_vjp(
            jnp.where(in_groups, lhs, 0).astype(jnp.float32),
            rhs.astype(jnp.float32),
            sizes,
            jnp.where(in_groups, dout, 0).astype(jnp.float32),
        )
    )

    for name, value, reference in zip(("out", "dlhs", "drhs"), actual, expected, strict=True):
        assert value.dtype == jnp.bfloat16, name
        (value,) = _to_cpu_f32(value)
        scale = max(float(jnp.max(jnp.abs(reference), initial=0.0)), 1.0)
        assert float(jnp.max(jnp.abs(value - reference), initial=0.0)) <= _FLYDSL_MAX_ERROR * scale, name
        mean_error = float(jnp.mean(jnp.abs(value - reference)))
        assert mean_error <= _FLYDSL_MEAN_ERROR * max(float(jnp.mean(jnp.abs(reference))), 1e-6), name
    out, dlhs, drhs = _to_cpu_f32(*actual)
    assert bool(jnp.all(out[active:] == 0))
    assert bool(jnp.all(dlhs[active:] == 0))
    assert bool(jnp.all(drhs[np.array(group_sizes) == 0] == 0))


@pytest.mark.parametrize("active", [0, 64, 917, 1023, 1024])
def test_flydsl_tail_kernel_zeroes_only_the_rows_past_the_groups(active):
    """The in-place tail kernel zeroes exactly the rows from ``sum(group_sizes)`` on and keeps every other row."""
    if jax.default_backend() != "gpu" or not ragged_dot_module._has_pallas_triton:
        pytest.skip("requires a GPU with Pallas Triton")
    rows, n = 1024, 192
    stale = jax.random.normal(jax.random.key(3), (rows, n), jnp.float32).astype(jnp.bfloat16)
    stale = stale.at[rows // 2].set(jnp.nan)
    sizes = jnp.array([active // 2, 0, active - active // 2], jnp.int32)

    out = jax.jit(ragged_dot_module._zero_rows_past_groups_in_place)(stale, sizes)

    out, stale = _to_cpu_f32(out, stale)
    np.testing.assert_array_equal(np.asarray(out[:active]), np.asarray(stale[:active]))
    assert bool(jnp.all(out[active:] == 0))


def test_flydsl_ragged_dot_without_jax_flydsl_raises_import_error(monkeypatch):
    """An explicit flydsl request fails instead of falling back when the optional package is missing."""
    monkeypatch.setattr(ragged_dot_module, "_has_pallas_triton", True)
    for module in ("jax_flydsl", "jax_flydsl.ops", "jax_flydsl.ops.grouped_gemm_bf16"):
        monkeypatch.setitem(sys.modules, module, None)
    lhs = jnp.zeros((512, 128), jnp.bfloat16)
    rhs = jnp.zeros((2, 128, 128), jnp.bfloat16)

    with pytest.raises(ImportError, match="optional jax-flydsl package"):
        ragged_dot(lhs, rhs, jnp.array([256, 256], jnp.int32), implementation="flydsl")


def test_ragged_dot_env_override_selects_flydsl_and_reraises_its_error(monkeypatch):
    """RAGGED_DOT_IMPL=flydsl is the default implementation; it leaves nothing to fall back to, so its own error
    surfaces even with "auto"."""
    assert not ragged_dot_module.flydsl_selected()
    monkeypatch.setenv("RAGGED_DOT_IMPL", "flydsl")
    assert ragged_dot_module.flydsl_selected()

    def fail(*args):
        raise RuntimeError("flydsl compile failed")

    monkeypatch.setattr(ragged_dot_module, "_ragged_dot_flydsl_impl", fail)
    monkeypatch.setattr(ragged_dot_module, "_flydsl_grouped_gemm_ops", lambda: None)
    lhs, rhs, sizes = jnp.zeros((512, 128), jnp.bfloat16), jnp.zeros((2, 128, 128), jnp.bfloat16), jnp.array([1, 2])

    with pytest.raises(RuntimeError, match="flydsl compile failed"):
        ragged_dot(lhs, rhs, sizes.astype(jnp.int32))


def test_flydsl_arch_mismatch_raises_instead_of_compiling(monkeypatch):
    """FlyDSL's silent gfx942 fallback on another GPU raises a clear error rather than aborting inside LLVM."""

    class _Device:
        compute_capability = "gfx950"

    monkeypatch.setattr(ragged_dot_module.jax, "devices", lambda *args: [_Device()])
    monkeypatch.setitem(sys.modules, "jax_flydsl", types.SimpleNamespace(get_rocm_arch_str=lambda: "gfx942"))
    ragged_dot_module._check_flydsl_arch.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="FLYDSL_GPU_ARCH=gfx950"):
            ragged_dot_module._check_flydsl_arch()
    finally:
        ragged_dot_module._check_flydsl_arch.cache_clear()
