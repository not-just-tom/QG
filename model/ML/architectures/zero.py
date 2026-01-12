import jax.numpy as jnp
import equinox as eqx


class ZeroModel(eqx.Module):
    """Just a zero test model"""
    def __init__(self, img_size=None, n_layers_in=None, n_layers_out=None, key=None, **kwargs):
        return None

    def __call__(self, x):
        return jnp.zeros_like(x)