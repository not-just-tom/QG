import jax.numpy as jnp
import equinox as eqx


class ZeroModel(eqx.Module):
    """Just a zero test model"""
    def __init__(self, **kwargs):
        # No parameters to initialise for the zero model.
        # Keep the constructor compatible with other architectures.
        pass

    def __call__(self, x):
        return jnp.zeros_like(x)