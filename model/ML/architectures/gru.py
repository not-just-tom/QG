import jax
import jax.numpy as jnp
import equinox as eqx


class GRU(eqx.Module):
    """
    GRU closure based on Kurz and Beck - due to the short term memory, 
    we have to pass some stuff through, making this a little more complex.
    """
    ml = True
    memory = True
    layers: list

    def __init__(
        self,
        key=jax.random.PRNGKey(0),
    ):
        pass

    def __call__(self, qh):
        pass