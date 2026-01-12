import jax
import jax.numpy as jnp
import equinox as eqx


class CNN(eqx.Module):
    """Hardcoded CNN model for now"""
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

    def __call__(self, qh):
        x = qh[None, :, :] # add batch dim
        for layer in self.layers:
            x = layer(x)
        return x[0, :, :] # remove batch dim