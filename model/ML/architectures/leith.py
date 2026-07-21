"""
Leith model closure stand-in to provide the terms and a baseline for the ML performance. 

It stands in as a closure with trainable params within the equinox environment, but the overhead of training the
analytical closures is not a primary concern
"""
import equinox as eqx
import jax.numpy as jnp

class LeithClosure(eqx.Module):
    ml = False
    leith_coeff: float
    delta: float

    def __init__(self, leith_coeff, cfg, **kwargs):
        self.leith_coeff = leith_coeff
        self.delta = cfg.params.Lx/cfg.params.nx


    def __call__(self , state, **kwargs):
        """Compute the Leith closure term for the given state."""
        
        return self.leith_coeff * self.delta**3 * abs(state.dqhdt) # <-- this last bit doesnt exist