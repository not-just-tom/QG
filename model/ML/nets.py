"""ML closure helpers: build an ml_forcing callable to plug into Solver.rhs.

The returned callable must have signature:
    ml_forcing(qh, key, grid, params) -> spectral_forcing (shape like qh)

"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx

def mse_loss(model, q, target):
    pred = jax.vmap(model)(q)
    return jnp.mean((pred - target) ** 2)


@eqx.filter_value_and_grad
def loss_and_grad(model, q, target):
    return mse_loss(model, q, target)


def zero_ml_forcing(qh, key, grid, params):
    # return zeros in spectral shape
    return jnp.zeros_like(qh)

class QGClosureCNN(eqx.Module):
    layers: list

    def __init__(self, key, in_channels=1, out_channels=1, width=64):
        keys = jax.random.split(key, 4)
        self.layers = [
            eqx.nn.Conv2d(in_channels, width, kernel_size=3, padding=1, key=keys[0]),
            jax.nn.relu,
            eqx.nn.Conv2d(width, width, kernel_size=3, padding=1, key=keys[1]),
            jax.nn.relu,
            eqx.nn.Conv2d(width, out_channels, kernel_size=3, padding=1, key=keys[2]),
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
