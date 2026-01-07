"""Small demo training loop for spectral correction model.

This provides a convenience function `train_spectral_demo` that synthesizes a
"truth" spectral correction and fits the spectral parameters to it using SGD.
It is intentionally simple and meant as a starting point for more advanced
training (e.g., differentiable time integration, multi-step losses, etc.).
"""
from __future__ import annotations

import time
import jax
import jax.numpy as jnp
from .spectral import init_spectral_params, predict, sgd_step, make_gaussian_target


def train_spectral_demo(grid, rng_key=None, steps=200, lr=1e-2, report_every=50):
    rng_key = jax.random.PRNGKey(0) if rng_key is None else rng_key
    rng_key, k0 = jax.random.split(rng_key)

    target = make_gaussian_target(grid, k0=8.0, sigma=2.0)
    params = init_spectral_params(grid, k0, scale=1e-2)

    losses = []
    t0 = time.time()
    for i in range(steps):
        params, loss = sgd_step(params, target, lr)
        losses.append(float(loss))
        if (i + 1) % report_every == 0:
            print(f"step {i+1}/{steps} loss={loss:.6g}")

    print("training finished in %.2f s" % (time.time() - t0))
    return params, jnp.array(losses)
