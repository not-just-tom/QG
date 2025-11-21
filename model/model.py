"""
Model basis and defintion stuff - make this sound smarter eventually eh

To do:
Check that this is GPU compatible 
On that note, JAX optimise
Figure out the grid formulation properly 
Check Arakawa Jacobian discretisation
Add initialise function which can clear field variables <---
^^^ these are kept IN the function, so thats why we need a thing for them....


"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.numpy.fft import rfftn, irfftn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import cached_property


import importlib
import model.utils
importlib.reload(model.utils)
from .utils import Solver


class QGM(Solver):
    """Finite difference solver for the 2D barotropic vorticity equation on a
    beta-plane"""

    def __init__(self, parameters, *, idtype=None):
        super().__init__(
            parameters, field_states={"F_1"})

    def initialize(self, zeta=None):
        super().initialize(zeta=zeta)
        if zeta is None:
            self._update_fields(self._initial)
        else:
            self._update_fields(zeta)
        self.fields.zero("F_1")

    def _update_fields(self, zeta):
        """
        Update model fields using a spectral Poisson solve.
        zeta : physical-space vorticity (2D array)
        """
        # Compute streamfunction in Fourier space
        zetah = rfftn(zeta, axes=(-2, -1))
        # Avoid division by zero at k=0
        psih = -zetah * self.grid.invK2
        psi = irfftn(psih, axes=(-2, -1)).real

        self.fields["psi"] = psi
        self.fields["zeta"] = zeta


    def step(self):
        qh = rfftn(self.fields["zeta"], axes=(-2, -1))
        # Use solver's stored key and advance it each timestep
        key = getattr(self, '_key', None)
        if key is None:
            key = self.parameters.get('key', jax.random.PRNGKey(self.n))
        state = (qh, key)

        state = self.RK4(state, self.grid, self.parameters, self.forcing_spectrum)
        qh_new, key = state
        # store advanced key for next step?
        self._key = key

        zeta_new = irfftn(qh_new, axes=(-2, -1)).real
        self._update_fields(zeta_new)
        super().step()

    def compute_rhines_length(model):
        """
        Compute the instantaneous Rhines scale L_β:

            L_β = sqrt(U_rms / β)

        where U_rms = sqrt( <u^2 + v^2> ).
        """
        grid = model.grid
        beta = model.parameters["beta"]

        # Get streamfunction
        psi = model.fields["psi"]

        # Compute velocity from ψ
        u =  jnp.gradient(psi, grid.dy, axis=-2)     #  dont like this gradient call rlly 
        v = -jnp.gradient(psi, grid.dx, axis=-1)   
        # RMS velocity
        U_rms = jnp.sqrt(0.5 * jnp.mean(u**2 + v**2))

        # Rhines scale
        L_beta = jnp.sqrt(U_rms / beta)

        return float(L_beta)

        









        