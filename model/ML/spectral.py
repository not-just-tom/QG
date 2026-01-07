"""Simple spectral correction model and training utilities.

This file provides a minimal trainable spectral correction representation
(useful for prototyping). The model parameters are an array of the same
rFFT shape as qh and training uses plain SGD on mean-squared error to a
synthetic target.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Tuple


def init_spectral_params(grid, rng_key: jax.random.PRNGKey, scale: float = 1e-3) -> jnp.ndarray:
    """Initialize a small random spectral correction array matching qh shape."""
    shape = (grid.ny, grid.nx // 2 + 1)
    key1, key2 = jax.random.split(rng_key)
    return scale * jax.random.normal(key1, shape)


@jax.jit
def predict(params: jnp.ndarray, qh: jnp.ndarray, key, grid, params_cfg=None) -> jnp.ndarray:
    """Return the spectral correction (here a simple identity mapping of params).

    This function can be replaced by a more complex mapping (e.g., a small NN
    applied in physical space), but this constant spectral field is useful to
    prototype training and integration with Solver.rhs.
    """
    return params


@jax.jit
def loss_fn(params: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((params - target) ** 2)


@jax.jit
def sgd_step(params: jnp.ndarray, target: jnp.ndarray, lr: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Perform one SGD step and return (new_params, loss)."""
    loss, grad = jax.value_and_grad(loss_fn)(params, target)
    new_params = params - lr * grad
    return new_params, loss


def make_gaussian_target(grid, k0: float = 8.0, sigma: float = 2.0) -> jnp.ndarray:
    """Create a radial Gaussian target in spectral amplitude (rfft shape).

    The target is real-valued spectral amplitudes (can be complex if desired),
    but we return a real array to be added into spectral forcing representation.
    """
    K = grid.Kmag
    targ = jnp.exp(- (K - k0) ** 2 / (2 * sigma ** 2))
    targ = jnp.where(grid.K2 == 0, 0.0, targ)
    return targ
