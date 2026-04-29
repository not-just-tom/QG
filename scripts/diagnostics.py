#!/usr/bin/env python3
"""
Smoke test for FNO in this repo:
- builds closure via build_closure(cfg)
- forwards random input and prints diagnostics
- does one optimizer step and a short training loop
- prints parameter stats before/after and checks for NaNs
"""
import sys
from pathlib import Path
import yaml
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
from types import SimpleNamespace

# repo-root aware import
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model.ML.architectures.build_model import build_closure

def load_cfg(path):
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    # Small wrapper so code can use cfg.get(...) or cfg.ml.attr
    class Cfg:
        def __init__(self, dd):
            self.__dict__.update({k: v for k, v in dd.items()})
            # make ml attribute an object for attribute access
            if "ml" in dd and isinstance(dd["ml"], dict):
                self.ml = SimpleNamespace(**dd["ml"])
        def get(self, k, default=None):
            return self.__dict__.get(k, default)
    return Cfg(d)

def arr_stats(name, a):
    a = jnp.asarray(a)
    a_cpu = np.array(jax.device_get(a))
    return f"{name}: shape={a_cpu.shape}, dtype={a_cpu.dtype}, min={a_cpu.min():.6g}, max={a_cpu.max():.6g}, mean={a_cpu.mean():.6g}, std={a_cpu.std():.6g}, has_nan={np.isnan(a_cpu).any()}"

def print_param_leaves(params, n=8):
    leaves, _ = jax.tree_util.tree_flatten(params)
    print(f"PARAM LEAVES: count={len(leaves)}")
    for i, leaf in enumerate(leaves[:n]):
        leaf = jnp.asarray(leaf)
        leaf_cpu = np.array(jax.device_get(leaf))
        print(f"  leaf[{i}]: shape={leaf_cpu.shape}, dtype={leaf_cpu.dtype}, mean={leaf_cpu.mean():.6g}, std={leaf_cpu.std():.6g}, min={leaf_cpu.min():.6g}, max={leaf_cpu.max():.6g}, has_nan={np.isnan(leaf_cpu).any()}")

def spectral_highk_fraction(x, cutoff):
    # x: (C,H,W) or (B,C,H,W)
    if x.ndim == 3:
        x = x[None, ...]
    ft = jnp.fft.fft2(x, axes=(-2, -1))
    ps = jnp.abs(ft) ** 2
    B, C, H, W = ps.shape
    # define high-k as modes where |kx|>cutoff or |ky|>cutoff (use index threshold)
    kx = jnp.fft.fftfreq(W) * W
    ky = jnp.fft.fftfreq(H) * H
    kx_mask = jnp.abs(kx) > cutoff
    ky_mask = jnp.abs(ky) > cutoff
    # broadcast to 2D mask
    mask2d = (ky_mask[:, None] | kx_mask[None, :])
    mask2d = mask2d[None, None, :, :]  # B,C,H,W broadcast dims
    high = jnp.sum(ps * mask2d)
    total = jnp.sum(ps)
    return jnp.where(total == 0, 0.0, (high / total).item())

def main():
    cfg_path = REPO_ROOT / "config" / "default.yaml"
    cfg = load_cfg(cfg_path)

    # force FNO usage
    cfg.ml.model_type = "fno"

    print("Loading FNO closure via build_closure(...)")
    closure = build_closure(cfg)

    # grid shapes from config
    nz = int(cfg.params.get("nz", 1))
    nx = int(cfg.params.get("nx", 64))
    ny = nx

    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (nz, ny, nx), dtype=jnp.float32) * 0.1

    print(arr_stats("input q", q))

    # forward
    try:
        out = closure(q.astype(jnp.float32))
    except Exception as e:
        print("Forward failed:", e)
        raise

    print(arr_stats("closure output", out))
    print("Spectral high-k fraction (cutoff=modes2 from config):",
          spectral_highk_fraction(out, cutoff=int(cfg.get("architectures", {}).get("fno", {}).get("modes2", 8))))

    # inspect params
    params, static = eqx.partition(closure, eqx.is_array)
    print_param_leaves(params, n=12)

    # Simple single-step training: target zeros (you can change)
    target = jnp.zeros_like(out)

    optim = optax.adam(float(cfg.get("architectures", {}).get("fno", {}).get("learning_rate", 1e-4)))
    opt_state = optim.init(params)

    @eqx.filter_jit
    def loss_and_grads(params, q, target):
        closure_comb = eqx.combine(params, static)
        pred = closure_comb(q.astype(jnp.float32))
        loss = jnp.mean((pred - target) ** 2)
        return loss, jax.grad(lambda p: jnp.mean((eqx.combine(p, static)(q.astype(jnp.float32)) - target) ** 2))(params)

    loss, grads = loss_and_grads(params, q, target)
    print(f"Initial loss: {float(loss):.6e}")
    # check grads for NaNs
    flat_grads, _ = jax.tree_util.tree_flatten(grads)
    has_nan_grads = any([jnp.isnan(g).any().item() for g in flat_grads])
    print("Any NaNs in grads before update:", has_nan_grads)

    updates, opt_state = optim.update(grads, opt_state, params)
    new_params = eqx.apply_updates(params, updates)

    print("After one update, param leaf stats (first few):")
    print_param_leaves(new_params, n=8)

    # quick small training loop with random inputs
    n_steps = 20
    losses = []
    p = params
    s = static
    opt_s = opt_state
    for i in range(n_steps):
        key = jax.random.fold_in(key, i)
        q_batch = jax.random.normal(key, (nz, ny, nx), dtype=jnp.float32) * 0.1
        loss, grads = loss_and_grads(p, q_batch, jnp.zeros_like(q_batch))
        updates, opt_s = optim.update(grads, opt_s, p)
        p = eqx.apply_updates(p, updates)
        losses.append(float(loss))
        if (i + 1) % 5 == 0:
            print(f"step {i+1}/{n_steps} loss={losses[-1]:.6e}")

    print("Final sample loss:", losses[-1])
    print("Final param leaf snapshot:")
    print_param_leaves(p, n=8)

    # final forward-check
    closure_final = eqx.combine(p, s)
    out_final = closure_final(q)
    print(arr_stats("final closure output", out_final))
    print("Final spectral high-k fraction:",
          spectral_highk_fraction(out_final, cutoff=int(cfg.get("architectures", {}).get("fno", {}).get("modes2", 8))))
    print("Done.")

if __name__ == "__main__":
    main()