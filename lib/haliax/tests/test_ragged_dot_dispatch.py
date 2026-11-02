# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
import types
from typing import Literal

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


def _inputs():
    contraction_dim = 32
    output_dim = 17
    lhs = jnp.arange(3 * contraction_dim, dtype=jnp.float32).reshape(3, contraction_dim) / 100
    rhs = jnp.arange(2 * contraction_dim * output_dim, dtype=jnp.float32).reshape(2, contraction_dim, output_dim) / 100
    group_sizes = jnp.array([2, 1], dtype=jnp.int32)
    return lhs, rhs, group_sizes


def _accelerator_implementation() -> Literal["megablox", "triton"]:
    backend = jax.default_backend()
    if backend == "gpu" and ragged_dot_module._has_pallas_triton:
        return "triton"
    if backend == "tpu" and ragged_dot_module._gmm_megablox is not None:
        return "megablox"
    pytest.skip("requires a supported accelerator ragged-dot implementation")


def test_accelerator_implementation_value_and_gradients_match_xla():
    lhs, rhs, group_sizes = _inputs()
    implementation = _accelerator_implementation()

    def loss(lhs, rhs, implementation):
        return jnp.sum(ragged_dot(lhs, rhs, group_sizes, implementation=implementation) ** 2)

    # At default precision, ROCm may run f32 dots in reduced-precision xf32 on MI300X, and its choice can
    # differ between the two paths and between processes, giving ~7e-4 relative error on either side.
    with jax.default_matmul_precision("highest"):
        actual_value, actual_gradients = jax.value_and_grad(loss, argnums=(0, 1))(lhs, rhs, implementation)
        expected_value, expected_gradients = jax.value_and_grad(loss, argnums=(0, 1))(lhs, rhs, "xla")

    assert jnp.allclose(actual_value, expected_value, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(actual_gradients[0], expected_gradients[0], rtol=1e-5, atol=1e-5)
    assert jnp.allclose(actual_gradients[1], expected_gradients[1], rtol=1e-5, atol=1e-5)


def test_triton_kernel_traces_with_jax_0_9_pallas_memory_api_on_cpu_interpreter():
    if not ragged_dot_module._has_pallas_triton:
        pytest.skip("Pallas Triton backend is not available")

    lhs, grouped_rhs, _ = _inputs()
    lhs = lhs[:2]
    rhs = grouped_rhs[0]
    lo = jnp.array(0, dtype=jnp.int32)
    hi = jnp.array(lhs.shape[0], dtype=jnp.int32)
    pallas_call = ragged_dot_module.pl.pallas_call(
        lambda a, b, lo, hi, out: ragged_dot_module._triton_ragged_dot_kernel(
            a, b, lo, hi, out, block_m=lhs.shape[0], block_k=lhs.shape[1], n=rhs.shape[1]
        ),
        out_shape=jax.ShapeDtypeStruct((lhs.shape[0], rhs.shape[1]), lhs.dtype),
        in_specs=[ragged_dot_module.pl.no_block_spec] * 4,
        out_specs=ragged_dot_module.pl.no_block_spec,
        grid=(1, 1),
        interpret=True,
    )

    assert jnp.allclose(pallas_call(lhs, rhs, lo, hi), lhs @ rhs, rtol=1e-5, atol=1e-5)


def test_gpu_auto_selects_triton_instead_of_xla(monkeypatch):
    lhs, rhs, group_sizes = _inputs()
    expected = jnp.full((lhs.shape[0], rhs.shape[2]), 17.0, dtype=lhs.dtype)
    monkeypatch.setattr(ragged_dot_module.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(ragged_dot_module, "_has_pallas_triton", True)

    def triton_result(lhs, rhs, group_sizes):
        return jnp.full((lhs.shape[0], rhs.shape[2]), expected[0, 0], dtype=lhs.dtype)

    def unexpected_xla(*args):
        raise AssertionError("GPU auto dispatch selected XLA instead of Triton")

    monkeypatch.setattr(ragged_dot_module, "_ragged_dot_triton_impl", triton_result)
    monkeypatch.setattr(ragged_dot_module, "_ragged_dot_xla_impl", unexpected_xla)

    auto_out = ragged_dot(lhs, rhs, group_sizes, implementation="auto")

    assert jnp.array_equal(auto_out, expected)


def test_triton_default_block_sizes_use_blackwell_n_tile(monkeypatch):
    monkeypatch.setattr(ragged_dot_module, "_is_blackwell_gpu_backend", lambda: True)

    assert ragged_dot_module._triton_default_block_sizes(32768, 5120, 5120) == (128, 256, 32)


def test_triton_default_block_sizes_keep_non_blackwell_n_tile(monkeypatch):
    monkeypatch.setattr(ragged_dot_module, "_is_blackwell_gpu_backend", lambda: False)

    assert ragged_dot_module._triton_default_block_sizes(32768, 5120, 5120) == (128, 128, 32)


def test_triton_custom_vjp_routes_backward_through_triton_layouts(monkeypatch):
    lhs, rhs, group_sizes = _inputs()
    calls = []

    def fake_triton_pallas_call(
        lhs,
        rhs,
        group_sizes,
        ragged_dot_dimension_numbers=ragged_dot_module._DEFAULT_DIM_NUMS,
    ):
        calls.append(ragged_dot_dimension_numbers)
        return jax.lax.ragged_dot_general(
            lhs=lhs,
            rhs=rhs,
            group_sizes=group_sizes,
            ragged_dot_dimension_numbers=ragged_dot_dimension_numbers,
        )

    monkeypatch.setattr(ragged_dot_module, "_has_pallas_triton", True)
    monkeypatch.setattr(ragged_dot_module, "_triton_pallas_call", fake_triton_pallas_call)

    def triton_loss(lhs, rhs):
        return jnp.sum(ragged_dot_module._ragged_dot_triton_impl(lhs, rhs, group_sizes))

    def xla_loss(lhs, rhs):
        return jnp.sum(ragged_dot(lhs, rhs, group_sizes, implementation="xla"))

    # At default precision, ROCm may run f32 dots in reduced-precision xf32 on MI300X, and its choice can
    # differ between the two paths and between processes, giving ~7e-4 relative error on either side.
    with jax.default_matmul_precision("highest"):
        triton_value, triton_grads = jax.value_and_grad(triton_loss, argnums=(0, 1))(lhs, rhs)
        xla_value, xla_grads = jax.value_and_grad(xla_loss, argnums=(0, 1))(lhs, rhs)

    assert jnp.allclose(triton_value, xla_value, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(triton_grads[0], xla_grads[0], rtol=1e-5, atol=1e-5)
    assert jnp.allclose(triton_grads[1], xla_grads[1], rtol=1e-5, atol=1e-5)
    assert calls == [
        ragged_dot_module._DEFAULT_DIM_NUMS,
        ragged_dot_module._DLHS_DIM_NUMS,
        ragged_dot_module._DRHS_DIM_NUMS,
    ]


# Group sizes over 300 rows and 4 groups, chosen so that row tiles straddle group boundaries.
_GROUP_SIZE_CASES = {
    "unaligned": [37, 101, 0, 162],
    "empty_first_and_last": [0, 150, 150, 0],
    "single_group_all_rows": [0, 0, 300, 0],
    "rows_past_last_group": [64, 0, 100, 20],
    "all_empty": [0, 0, 0, 0],
}
# Odd sizes: rows, contraction and columns are not multiples of the tile-map blocks below.
_ROWS, _K, _N, _GROUPS = 300, 80, 200, 4
_LAYOUTS = [layout.value for layout in ragged_dot_module.RaggedLayout]


def _require_triton_gpu():
    if jax.default_backend() != "gpu" or not ragged_dot_module._has_pallas_triton:
        pytest.skip("requires the Pallas Triton GPU backend")


def _require_kernel_family(family: str):
    """Both families run on a GPU; the tile-map kernels also run on CPU through the Pallas interpreter."""
    if family == "tile_map" and jax.default_backend() == "cpu" and ragged_dot_module._has_pallas_triton:
        return
    _require_triton_gpu()


def _layout_operands(layout: str, dtype, group_sizes: list[int]):
    key_lhs, key_rhs = jax.random.split(jax.random.key(0))
    if layout == "fwd":
        lhs_shape, rhs_shape = (_ROWS, _K), (_GROUPS, _K, _N)
    elif layout == "dlhs":
        lhs_shape, rhs_shape = (_ROWS, _N), (_GROUPS, _K, _N)
    else:
        lhs_shape, rhs_shape = (_ROWS, _K), (_ROWS, _N)
    lhs = jax.random.normal(key_lhs, lhs_shape, jnp.float32).astype(dtype)
    rhs = jax.random.normal(key_rhs, rhs_shape, jnp.float32).astype(dtype)
    return lhs, rhs, jnp.array(group_sizes, dtype=jnp.int32)


def _to_cpu_f32(*arrays):
    cpu = jax.devices("cpu")[0]
    return tuple(jax.device_put(x.astype(jnp.float32) if x.dtype != jnp.int32 else x, cpu) for x in arrays)


def _cpu_f32_reference(lhs, rhs, group_sizes, dim_nums):
    """``jax.lax.ragged_dot_general`` in float32 on the CPU backend, as an independent oracle."""
    lhs, rhs, group_sizes = _to_cpu_f32(lhs, rhs, group_sizes)
    return jax.lax.ragged_dot_general(
        lhs, rhs, group_sizes, ragged_dot_dimension_numbers=dim_nums, precision=jax.lax.Precision.HIGHEST
    )


def _assert_close_to_reference(actual, expected, rtol):
    """Max abs error, relative to the reference's largest magnitude, stays within ``rtol``."""
    (actual,) = _to_cpu_f32(actual)
    error = jnp.abs(actual - expected)
    scale = max(float(jnp.max(jnp.abs(expected), initial=0.0)), 1.0)
    assert bool(jnp.all(jnp.isfinite(actual)))
    max_error = float(jnp.max(error, initial=0.0))
    assert max_error <= rtol * scale, f"max abs error {max_error:.3e}, scale {scale:.3e}"


def _triton_f32_dots_use_tf32() -> bool:
    """Triton's dot defaults f32 inputs to TF32 on NVIDIA GPUs and on gfx942 (MI300X, MI325X), and to IEEE on gfx950."""
    device = jax.devices()[0]
    # ROCm reports the gfx architecture name as the compute capability.
    return device.platform == "gpu" and ("NVIDIA" in device.device_kind or device.compute_capability == "gfx942")


def _run_kernel_family(family: str, lhs, rhs, sizes, layout: str):
    layout = ragged_dot_module.RaggedLayout(layout)
    if family == "group_grid":
        return jax.jit(lambda a, b, g: ragged_dot_module._group_grid_pallas_call(a, b, g, layout))(lhs, rhs, sizes)
    # Small blocks put several row and column tiles in each group and leave partial edge tiles. With
    # num_xcds=8 the launch grid is padded and renumbered, so padding programs must not write, and
    # group_m=3 leaves a partial band of tile rows.
    config = ragged_dot_module.TritonBlockConfig(
        block_m=64, block_n=64, block_k=32, num_warps=4, num_stages=2, num_xcds=8, group_m=3
    )
    return jax.jit(lambda a, b, g: ragged_dot_module._tile_map_pallas_call(a, b, g, layout, config))(lhs, rhs, sizes)


@pytest.mark.parametrize("dtype, rtol", [(jnp.float32, 1e-4), (jnp.bfloat16, 1e-2)], ids=["f32", "bf16"])
@pytest.mark.parametrize("group_sizes", list(_GROUP_SIZE_CASES.values()), ids=list(_GROUP_SIZE_CASES))
@pytest.mark.parametrize("layout", _LAYOUTS)
@pytest.mark.parametrize("family", ["group_grid", "tile_map"])
def test_triton_kernel_families_match_ragged_dot_general(family, layout, group_sizes, dtype, rtol):
    _require_kernel_family(family)
    if dtype == jnp.float32 and _triton_f32_dots_use_tf32():
        # About 3e-4 off here on NVIDIA and 7e-4 on MI300X; XLA's own ragged dot on NVIDIA is off by as much.
        pytest.skip("f32 Triton dots use TF32 on NVIDIA and gfx942")
    lhs, rhs, sizes = _layout_operands(layout, dtype, group_sizes)

    actual = _run_kernel_family(family, lhs, rhs, sizes, layout)
    expected = _cpu_f32_reference(
        lhs, rhs, sizes, ragged_dot_module._LAYOUT_DIM_NUMS[ragged_dot_module.RaggedLayout(layout)]
    )

    if family == "group_grid" and layout != "drhs":
        # The group-grid kernels leave rows past sum(group_sizes) unwritten; the tile-map kernels zero them.
        actual, expected = actual[: sum(group_sizes)], expected[: sum(group_sizes)]
    _assert_close_to_reference(actual, expected, rtol)


# H100 picks its drhs blocks by rows per group, below and from 2048.
@pytest.mark.parametrize(
    "group_sizes",
    [[512, 512, 512, 512], [5, 1000, 0, 1043], [2048, 2048, 2048, 2048]],
    ids=["balanced", "skewed", "many_rows_per_group"],
)
def test_triton_ragged_dot_value_and_gradients_match_reference_with_device_blocks(group_sizes):
    _require_triton_gpu()
    rows, k, n = sum(group_sizes), 256, 384
    key_lhs, key_rhs, key_dout = jax.random.split(jax.random.key(1), 3)
    lhs = jax.random.normal(key_lhs, (rows, k), jnp.float32).astype(jnp.bfloat16)
    rhs = jax.random.normal(key_rhs, (len(group_sizes), k, n), jnp.float32).astype(jnp.bfloat16)
    dout = jax.random.normal(key_dout, (rows, n), jnp.float32).astype(jnp.bfloat16)
    sizes = jnp.array(group_sizes, dtype=jnp.int32)

    out, vjp = jax.vjp(lambda a, b: ragged_dot(a, b, sizes, implementation="triton"), lhs, rhs)
    dlhs, drhs = vjp(dout)

    def reference(a, b):
        return _cpu_f32_reference(a, b, sizes, ragged_dot_module._DEFAULT_DIM_NUMS)

    expected_out, reference_vjp = jax.vjp(reference, *_to_cpu_f32(lhs, rhs))
    expected_dlhs, expected_drhs = reference_vjp(*_to_cpu_f32(dout))
    _assert_close_to_reference(out, expected_out, 1e-2)
    _assert_close_to_reference(dlhs, expected_dlhs, 1e-2)
    _assert_close_to_reference(drhs, expected_drhs, 1e-2)


@pytest.mark.parametrize("implementation", ["xla", "megablox"])
def test_ragged_dot_zeroes_rows_past_the_last_group(implementation):
    """Rows outside every group come back as zeros, so callers may leave padding rows ungrouped."""
    if implementation == "megablox" and ragged_dot_module._gmm_megablox is None:
        pytest.skip("megablox is not available")
    lhs, rhs, sizes = _layout_operands("fwd", jnp.float32, _GROUP_SIZE_CASES["rows_past_last_group"])
    active = sum(_GROUP_SIZE_CASES["rows_past_last_group"])

    out = jax.jit(lambda a, b, g: ragged_dot(a, b, g, implementation=implementation))(lhs, rhs, sizes)
    expected = _cpu_f32_reference(lhs, rhs, sizes, ragged_dot_module._DEFAULT_DIM_NUMS)

    (out,) = _to_cpu_f32(out)
    assert bool(jnp.all(out[active:] == 0))
    _assert_close_to_reference(out[:active], expected[:active], 1e-4)


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
    # June's 32 local experts and expert dims for the up (2560 x 2560) and down (1280 x 2560) projections, with
    # 97% of the rows in groups. Fewer rows than June's 131072 keep the f32 XLA reference's compile under a minute.
    "june_up": (8192, 2560, 2560, _skewed_group_sizes(32, 7946, seed=0)),
    "june_down": (32768, 1280, 2560, _skewed_group_sizes(32, 31784, seed=1)),
}
# bf16 rounds to 8 significant bits, so rounding the f32 accumulator to the bf16 result is off by up to 2**-8
# (3.9e-3) of each element and by about 1.4e-3 on average; the f32 accumulation order adds far less. On gfx950
# the max error is 2-3.4e-3 of the largest magnitude and the mean 1.41e-3 of the mean magnitude.
_FLYDSL_MAX_ERROR = 5e-3
_FLYDSL_MEAN_ERROR = 2e-3


@pytest.mark.timeout(300)
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
    # No group may read the rows past the last group. This does not show that those output rows get zeroed: the
    # kernels never write them, and a fresh output buffer often reads 0 anyway (see the tail-kernel test below).
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
    # The reference sees zeros past the last group, which no group may read.
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
        _assert_close_to_reference(value, reference, _FLYDSL_MAX_ERROR)
        (value,) = _to_cpu_f32(value)
        mean_error = float(jnp.mean(jnp.abs(value - reference)))
        assert mean_error <= _FLYDSL_MEAN_ERROR * max(float(jnp.mean(jnp.abs(reference))), 1e-6), name
    out, dlhs, drhs = _to_cpu_f32(*actual)
    assert bool(jnp.all(out[active:] == 0))
    assert bool(jnp.all(dlhs[active:] == 0))
    assert bool(jnp.all(drhs[np.array(group_sizes) == 0] == 0))


@pytest.mark.parametrize("active", [0, 64, 917, 1023, 1024])
def test_flydsl_tail_kernel_zeroes_only_the_rows_past_the_groups(active):
    """The in-place tail kernel zeroes exactly the rows from ``sum(group_sizes)`` on and keeps every other row.

    The FlyDSL kernels never write those rows, so this kernel alone makes them zero; reused buffers inside a
    scanned or rematerialized train step hold stale values there.
    """
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


def test_ragged_dot_env_override_reraises_the_implementation_error(monkeypatch):
    """A RAGGED_DOT_IMPL override leaves nothing to fall back to, so its own error surfaces even with "auto"."""
    monkeypatch.setenv("RAGGED_DOT_IMPL", "flydsl")

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
