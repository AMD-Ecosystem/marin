# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ZeRO-1 over the replicas (``GrugTrainerConfig.zero1``) in the June train step, on eight CPU devices.

With ``zero1`` the expert weights' float32 parameters, optimizer state and EMA are sharded over ``replica_dcn`` as well
as ``expert``; the step reduces each expert gradient into that shard and all-gathers the updated compute-dtype
parameters back to the model's sharding after the update. On the meshes of the gradient accumulation tests
(replica_dcn=4 x expert=2, replica_dcn=2 x data=2 x expert=2), with AdamW and with the production MuonH, the losses and
parameters match the replicated update; the state is 1/replica_dcn per device; the expert weights cross the replicas
once per step each way, after the microbatch loop; and a checkpoint saved with ZeRO-1 restores with it off and on.

On CPU XLA lowers the reduce-scatter to an all-reduce and a slice, so the HLO test counts the elements each
reduction covers rather than requiring a reduce-scatter.

Each case runs in a subprocess so XLA can configure the CPU device count before JAX initializes.
"""

import textwrap

import pytest

from tests.june_cpu_devices import run_on_cpu_devices
from tests.test_june_grad_accum import _PRELUDE as _GRAD_ACCUM_PRELUDE

_PRELUDE = (
    _GRAD_ACCUM_PRELUDE
    + r"""
import tempfile

from jax.sharding import PartitionSpec as P
from levanter.checkpoint import load_checkpoint, save_checkpoint
from levanter.optim.config import AdamConfig

from experiments.june_tpu_67b_a2b.moe.optimizer import GrugMoeMuonHConfig

# Eight experts, so that they split over expert=2 x replica_dcn=4.
ZERO1_CFG = dataclasses.replace(BASE, num_experts=8, remat_mode="save_moe_attn")
POLICY = jmp.get_policy("p=f32,c=bfloat16")
OPTIMIZERS = {
    "adamw": lambda: AdamConfig(learning_rate=1e-2, warmup=0.0, weight_decay=0.1, max_grad_norm=1.0).build(10),
    "muonh": lambda: GrugMoeMuonHConfig(learning_rate=2e-2, adam_lr=1e-2, warmup=0.0).build(10),
}
EMA_BETA = 0.9
GATHERS = ("all-gather", "all-gather-start")


def zero1_state(mesh, optimizer, *, zero1):
    with jax.set_mesh(mesh):
        return jax.jit(
            lambda key: june_train.initial_state(
                ZERO1_CFG, optimizer=optimizer, mp=POLICY, key=key, ema_beta=EMA_BETA, carry_compute_params=True,
                zero1=zero1,
            )
        )(jax.random.key(0))


def zero1_step(optimizer, *, zero1, microbatches=4):
    return june_train._make_train_step(
        optimizer, POLICY, z_loss_weight=0.0, ema_beta=EMA_BETA, microbatches=microbatches, zero1=zero1
    )


def is_expert(path):
    return "expert_mlp" in jax.tree_util.keystr(path)


def specs(tree):
    return {jax.tree_util.keystr(path): x.sharding.spec for path, x in jax.tree_util.tree_leaves_with_path(tree)}


def zero1_spec(spec):
    # The expert dimension (1) of a stacked expert weight also split over the replicas.
    entries = list(spec)
    entries[1] = (entries[1], "replica_dcn")
    return P(*entries)


def relative_errors(tree, reference):
    return {
        jax.tree_util.keystr(path): np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)
        for (path, a), b in zip(
            jax.tree_util.tree_leaves_with_path(jax.tree.map(lambda x: np.asarray(x, np.float32), tree)),
            leaves(reference),
            strict=True,
        )
    }
"""
)


def _run(body: str) -> None:
    run_on_cpu_devices(_PRELUDE + textwrap.dedent(body), device_count=8)


@pytest.mark.timeout(900)
@pytest.mark.parametrize("optimizer", ["adamw", "muonh"])
def test_zero1_matches_the_replicated_update(optimizer):
    # After one step only the order of the optimizer's norm sums differs (the global clip norm, MuonH's per-layer
    # hyperball norm), a few float32 ulps: below 1e-6 relative, 4.6e-7 observed. Later steps start from compute copies
    # that can round differently to bfloat16 and train on, so after three steps every leaf need only be within one
    # bfloat16 rounding (2**-8) of the replicated run's; MuonH, which normalizes each update, shows 4.2e-4. The losses,
    # means over the batch, move far less: they matched exactly.
    _run(
        f"""
        for name, mesh in MESHES.items():
            optimizer = OPTIMIZERS[{optimizer!r}]()
            runs = {{}}
            for zero1 in (False, True):
                state = zero1_state(mesh, optimizer, zero1=zero1)
                train_step = zero1_step(optimizer, zero1=zero1)
                losses, after_one_step = [], None
                with jax.set_mesh(mesh):
                    for _ in range(3):
                        state, metrics, _ = train_step(state, BATCH)
                        losses.append(float(metrics["train/loss"]))
                        if after_one_step is None:
                            after_one_step = jax.tree.map(np.asarray, (state.params, state.ema_params))
                runs[zero1] = (losses, after_one_step, jax.tree.map(np.asarray, (state.params, state.ema_params)))
            (losses, one, three), (zero1_losses, zero1_one, zero1_three) = runs[False], runs[True]
            np.testing.assert_allclose(zero1_losses, losses, rtol=1e-5, err_msg=name)
            errors = relative_errors(zero1_one, one)
            assert max(errors.values()) <= 1e-6, (name, errors)
            errors = relative_errors(zero1_three, three)
            assert max(errors.values()) <= 2**-8, (name, errors)
        """
    )


@pytest.mark.timeout(900)
def test_zero1_shards_the_expert_state_over_the_replicas():
    # The expert weights' parameters, EMA and every optimizer state leaf of their shape are sharded over the replicas
    # as well, so each device keeps 1/replica_dcn of them; everything else keeps the replicated run's sharding, and
    # the compute copy keeps the model's sharding after the update.
    _run(
        """
        for optimizer_name, make_optimizer in OPTIMIZERS.items():
            optimizer = make_optimizer()
            for name, mesh in MESHES.items():
                msg = f"{name} {optimizer_name}"
                replicas = mesh.shape["replica_dcn"]
                with jax.set_mesh(mesh):
                    states = {}
                    for zero1 in (False, True):
                        initial = jax.eval_shape(
                            lambda key: june_train.initial_state(
                                ZERO1_CFG, optimizer=optimizer, mp=POLICY, key=key, ema_beta=EMA_BETA,
                                carry_compute_params=True, zero1=zero1,
                            ),
                            jax.random.key(0),
                        )
                        states[zero1] = jax.eval_shape(zero1_step(optimizer, zero1=zero1), initial, BATCH)[0]
                replicated, sharded = states[False], states[True]
                expert_shapes = {
                    x.shape for path, x in jax.tree_util.tree_leaves_with_path(replicated.params) if is_expert(path)
                }
                assert len(expert_shapes) == 2, expert_shapes
                for tree in ("params", "ema_params"):
                    for path, before, after in zip(
                        specs(getattr(replicated, tree)),
                        specs(getattr(replicated, tree)).values(),
                        specs(getattr(sharded, tree)).values(),
                        strict=True,
                    ):
                        expected = zero1_spec(before) if "expert_mlp" in path else before
                        assert after == expected, (msg, tree, path, before, after)
                assert specs(sharded.compute_params) == specs(replicated.compute_params), msg

                def state_bytes(tree, expert):
                    return sum(
                        int(np.prod(x.sharding.shard_shape(x.shape))) * x.dtype.itemsize
                        for x in jax.tree.leaves(tree)
                        if hasattr(x, "sharding") and (x.shape in expert_shapes) == expert
                    )

                for expert in (False, True):
                    before = state_bytes((replicated.params, replicated.opt_state, replicated.ema_params), expert)
                    after = state_bytes((sharded.params, sharded.opt_state, sharded.ema_params), expert)
                    assert after * (replicas if expert else 1) == before, (msg, expert, before, after)
        """
    )


@pytest.mark.timeout(900)
def test_zero1_crosses_the_replicas_once_per_step_after_the_microbatch_loop():
    # On top of the replicated run's collectives across the replicas, each expert weight is all-gathered once per step
    # after the microbatch loop, its compute copy's shard in full; no gather runs in a loop that the replicated run does
    # not have. The gradient reductions across the replicas still cover every weight's gradient once, after the loop.
    # The step is compiled for the state one step has written, so a compute copy left sharded over the replicas would
    # show up as gathers in the layer loop.
    _run(
        """
        for name, mesh in MESHES.items():
            optimizer = optax.adam(1e-2)
            with jax.set_mesh(mesh):
                marked = jax.eval_shape(
                    june_train._mark_batch_reduced, zero1_state(mesh, optimizer, zero1=False).compute_params
                )
            sizes = [
                int(np.prod(x.sharding.shard_shape(x.shape)))
                for path, x in jax.tree_util.tree_leaves_with_path(marked)
                if getattr(path[-1], "name", None) != "router_bias"
            ]
            for k in (2, 4):
                msg = f"{name} K={k}"
                gathers, reductions = {}, {}
                for zero1 in (False, True):
                    train_step = zero1_step(optimizer, zero1=zero1, microbatches=k)
                    with jax.set_mesh(mesh):
                        state, _, _ = train_step(zero1_state(mesh, optimizer, zero1=zero1), BATCH)
                        hlo = train_step.lower(state, BATCH).compile().as_text()
                    gathers[zero1] = replica_reductions(hlo, mesh, GATHERS)
                    reductions[zero1] = replica_reductions(hlo, mesh)
                in_loop = [sorted((r.elements, r.op_name) for r in gathers[z] if r.in_loop) for z in (False, True)]
                assert in_loop[1] == in_loop[0], (msg, in_loop)
                added = collections.Counter(r.elements for r in gathers[True] if not r.in_loop)
                added.subtract(r.elements for r in gathers[False] if not r.in_loop)
                expert_shards = collections.Counter(
                    int(np.prod(x.sharding.shard_shape(x.shape)))
                    for path, x in jax.tree_util.tree_leaves_with_path(state.compute_params)
                    if is_expert(path)
                )
                assert +added == expert_shards and -added == collections.Counter(), (msg, added, expert_shards)
                in_loop = [r for r in reductions[True] if r.in_loop]
                assert not any("transpose(" in r.op_name and not r.scalar for r in in_loop), (msg, in_loop)
                after_loop = sum(r.elements for r in reductions[True] if not r.in_loop)
                assert sum(sizes) <= after_loop < sum(sizes) + min(sizes), (msg, after_loop, sum(sizes))
        """
    )


@pytest.mark.timeout(900)
def test_zero1_checkpoint_restores_with_zero1_off_and_on():
    # The checkpoint keeps the tree and the global shapes, so the exemplar's sharding decides what each device reads.
    # The compute copy rebuilt from the restored parameters has the model's sharding either way.
    _run(
        """
        mesh = MESHES["replica4_expert2"]
        optimizer = OPTIMIZERS["adamw"]()
        with jax.set_mesh(mesh):
            state, _, _ = zero1_step(optimizer, zero1=True)(zero1_state(mesh, optimizer, zero1=True), BATCH)
        saved = jax.tree.map(np.asarray, state)
        with tempfile.TemporaryDirectory() as tmpdir:
            with jax.set_mesh(mesh):
                save_checkpoint(state, step=1, checkpoint_path=tmpdir)
            for zero1 in (False, True):
                exemplar = zero1_state(mesh, optimizer, zero1=zero1)
                with jax.set_mesh(mesh):
                    restored = load_checkpoint(exemplar, tmpdir, mesh=mesh)
                    rebuilt = june_train.rebuilt_compute_params(restored, POLICY)
                assert specs(restored) == specs(exemplar), zero1
                assert specs(rebuilt.compute_params) == specs(exemplar.compute_params), zero1
                for a, b in zip(jax.tree.leaves(restored), jax.tree.leaves(saved), strict=True):
                    np.testing.assert_array_equal(np.asarray(a), b, err_msg=str(zero1))
                for a, b in zip(
                    jax.tree.leaves(rebuilt.compute_params), jax.tree.leaves(POLICY.cast_to_compute(saved.params))
                ):
                    np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=str(zero1))
        """
    )
