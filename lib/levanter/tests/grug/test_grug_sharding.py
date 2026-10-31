# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from levanter.testing.cpu_devices import run_on_cpu_devices


def test_keep_reduced_leaves_out_axes_the_spec_shards_over():
    # A weight reduced over replica_dcn and data enters a shard_map that shards it over data and all-gathers it there,
    # as the chunked sonic_cute path does. Its gradient stays unreduced over both axes and sums to the plain gradient.
    run_on_cpu_devices(
        """
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax import shard_map
        from jax.sharding import AxisType, NamedSharding, PartitionSpec as P, reshard
        from levanter.grug.sharding import keep_reduced

        mesh = jax.make_mesh((2, 2, 2), ("replica_dcn", "data", "expert"), axis_types=(AxisType.Explicit,) * 3)
        batch = ("replica_dcn", "data", "expert")
        reduced = frozenset({"replica_dcn", "data"})

        def loss(w, x):
            spec = keep_reduced(P("expert", "data", None), w)
            assert spec.reduced == (jax.typeof(w).sharding.spec.reduced - {"data"}), spec

            def local(w, x):
                return jnp.sum((x @ jax.lax.all_gather(w, "data", axis=1, tiled=True)[0]) ** 2)[None]

            sharded = shard_map(local, mesh=mesh, in_specs=(spec, P(batch, None)), out_specs=P(batch), check_vma=False)
            return jnp.sum(sharded(reshard(w, spec), x))

        def reduced_grad(w, x):
            grad = jax.grad(loss)(reshard(w, P("expert", None, None, reduced=reduced)), x)
            assert jax.typeof(grad).sharding.spec.unreduced == reduced, jax.typeof(grad)
            return reshard(grad, P("expert", None, None))

        with jax.set_mesh(mesh):
            w = jax.device_put(np.arange(24, dtype=np.float32).reshape(2, 4, 3) / 10, NamedSharding(mesh, P("expert")))
            x = jax.device_put(np.arange(32, dtype=np.float32).reshape(8, 4) / 7, NamedSharding(mesh, P(batch)))
            np.testing.assert_allclose(jax.jit(reduced_grad)(w, x), jax.jit(jax.grad(loss))(w, x), rtol=1e-6)
        """,
        device_count=8,
    )
