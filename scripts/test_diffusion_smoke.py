#!/usr/bin/env python3
"""
Smoke test for the spectral diffusion-style generator.
- Builds DiffusionGenerator
- Forwards random inputs and prints diagnostics
- Runs a short param update loop (single-step and small loop)
"""
import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model.ML.architectures.diffusion import DiffusionGenerator
from model.ML.utils.utils import module_to_single

def arr_stats(name, a):
    a = jnp.asarray(a)
    a_cpu = np.array(jax.device_get(a))
    return f"{name}: shape={a_cpu.shape}, dtype={a_cpu.dtype}, min={a_cpu.min():.6g}, max={a_cpu.max():.6g}, mean={a_cpu.mean():.6g}, std={a_cpu.std():.6g}, has_nan={np.isnan(a_cpu).any()}"

def print_param_leaves(params, n=8):
    leaves, _ = jax.tree_util.tree_flatten(params)
    print(f"PARAM LEAVES: count={len(leaves)}")
    for i, leaf in enumerate(leaves[:n]):
        leaf_cpu = np.array(jax.device_get(jnp.asarray(leaf)))
        print(f"  leaf[{i}]: shape={leaf_cpu.shape}, mean={leaf_cpu.mean():.6g}, std={leaf_cpu.std():.6g}, has_nan={np.isnan(leaf_cpu).any()}")

def main():
    key = jax.random.PRNGKey(0)
    key, k1 = jax.random.split(key)
    nz = 1
    nx = 64
    ny = nx

    # build diffusion generator
    gen = DiffusionGenerator(channels=nz, cutoff=8.0, n_steps=3, alpha=0.2, hidden=64, key=k1)
    gen = module_to_single(gen)  # match repo expectation

    # random input
    q = jax.random.normal(key, (nz, ny, nx), dtype=jnp.float32) * 0.1
    print(arr_stats("input q", q))

    # forward
    out = gen(q, verbose=True)
    print(arr_stats("generator output", out))

    # inspect params
    params, static = eqx.partition(gen, eqx.is_array)
    print_param_leaves(params, n=12)

    # do one optimizer update to 'fit' zeros (toy)
    target = jnp.zeros_like(out)
    optim = optax.adam(1e-3)
    opt_state = optim.init(params)

    @eqx.filter_jit
    def loss_and_grads(p, q, tgt):
        model = eqx.combine(p, static)
        pred = model(q, verbose=False)
        loss = jnp.mean((pred - tgt) ** 2)
        grads = jax.grad(lambda pr: jnp.mean((eqx.combine(pr, static)(q) - tgt) ** 2))(p)
        return loss, grads

    loss, grads = loss_and_grads(params, q, target)
    print(f"initial loss: {float(loss):.6e}")
    flat_grads, _ = jax.tree_util.tree_flatten(grads)
    print("any NaNs in grads:", any([jnp.isnan(g).any().item() for g in flat_grads]))

    updates, opt_state = optim.update(grads, opt_state, params)
    new_params = eqx.apply_updates(params, updates)
    print("After one update:")
    print_param_leaves(new_params, n=8)

    # short loop
    p = new_params
    s = static
    losses = []
    for i in range(10):
        key = jax.random.fold_in(key, i)
        q_batch = jax.random.normal(key, (nz, ny, nx), dtype=jnp.float32) * 0.1
        loss, grads = loss_and_grads(p, q_batch, jnp.zeros_like(q_batch))
        updates, opt_state = optim.update(grads, opt_state, p)
        p = eqx.apply_updates(p, updates)
        losses.append(float(loss))
        if (i + 1) % 5 == 0:
            print(f"step {i+1} loss={losses[-1]:.6e}")

    print("final loss", losses[-1])
    # final forward-check
    final_gen = eqx.combine(p, s)
    out_final = final_gen(q, verbose=False)
    print(arr_stats("final output", out_final))
    print("Done.")

if __name__ == '__main__':
    main()
