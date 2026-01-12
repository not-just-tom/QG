
import jax 
import jax.numpy as jnp
import numpy as np
import logging
from abc import ABC, abstractmethod
from functools import cached_property
import keras
from jax.numpy.fft import rfftn, irfftn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm

from .states import States, IOInterface
from .grid import Grid
from ..utils.pytree import PytreeNode

optional = object()
required = object()

class Solver(PytreeNode, ABC):
    """
    My model to try and get this thing started
    """

    _registry = {}

    _defaults = {"Lx": required,  # x \in [-Lx, Lx]
                 "nx": required,
                 "beta": required,
                 "dt": required}
    
    def __init__(self, cfg, initial: jnp.ndarray | None = None, ml_closure:jnp.ndarray| None = None, field_states=None, prescribed_field_states=None):
        # ==== Parameter Initialisation ===  
        self._cfg = cfg
        self.ml_closure = ml_closure

        # PRNG key for the solver; stored and advanced each step
        self._key = jax.random.PRNGKey(cfg.params.seed)

        # ==== Grid and Params Initialisation ===
        self.grid = grid = Grid(cfg)
        self.kmin = cfg.params.kmin
        self.kmax = cfg.params.kmax

        initial = self._make_initial(cfg)
        self._initial = jax.device_put(initial)

        if field_states is None:
            field_states = set()
        else:
            field_states = set(field_states)
        field_states.update({"psi", "zeta", "Q"})
        if prescribed_field_states is None:
            prescribed_field_states = {"Q"}
        
        self._fields = States(grid, field_states)
        self._prescribed_field_states = tuple(sorted(prescribed_field_states))

        self.zero_prescribed() # zeros the things we're interested in
        self.initialize() # removes irrelevant states from another run

    def steps(self, n, *, unroll=8): 
        model = jax.lax.fori_loop(0, n, self._step, self, unroll=unroll)
        self.update(model)

    def _make_initial(self, cfg):
        """
        Create a band-passed random initial condition in physical space, depending on config.
        """

        forcing = getattr(cfg, "forcing")
        grid = self.grid

        if custom_init := getattr(cfg, "initial_condition", None):
            #logger.info("Using custom initial condition from input")
            # === i dont use this yet but i need a check to make sure it fits the grid ===
            raise NotImplementedError("Custom not implemented yet, needs grid shape check")
        elif forcing is None:
            #logger.info("Using random initial, as forcing absent")
            key = jax.random.PRNGKey(cfg.params.seed) 
            key, k1, k2 = jax.random.split(key)
            noise_real = jax.random.normal(k1, (grid.ny, grid.nx // 2 + 1)) 
            noise_imag = jax.random.normal(k2, (grid.ny, grid.nx // 2 + 1)) 
            qh = noise_real + 1j * noise_imag 
            qh = qh.at[:, 0].set(jnp.real(qh[:, 0]))  
            initial = Solver.dealias(qh, grid, s=8) # dealiasing - should this 8 be hardcoded?

            # --- band-pass filter ---
            band_mask = (grid.Kmag >= self.kmin) & (grid.Kmag <= self.kmax)
            qh = qh * band_mask
            qh = qh.at[:, 0].set(0.0)

            #  physical-space initial field
            initial = irfftn(qh, axes=(-2, -1), norm='ortho').real
            assert initial.shape == (grid.ny, grid.nx)
            return initial
        else:
            #logger.info("Using pseudo-random initial generated from forcing wavenumber")
            k_f = getattr(forcing, 'k_f', 8.0)
            key = jax.random.PRNGKey(cfg.params.seed) 
            key, k1 = jax.random.split(key)


            # pseudo-random initial spectrum peaked at k_f
            qh = Solver.pseudo_randomiser(grid, k_f, k1)
            qh = Solver.dealias(qh, grid, s=8)

            # --- band-pass filter ---
            band_mask = (grid.Kmag >= self.kmin) & (grid.Kmag <= self.kmax)
            qh = qh * band_mask
            qh = qh.at[:, 0].set(0.0)

            #  physical-space initial field
            initial = irfftn(qh, axes=(-2, -1), norm='ortho').real
            assert initial.shape == (grid.ny, grid.nx)
            return initial

    
    @staticmethod
    @jax.jit
    def dealias(y, grid, s=8):
        """Apply a precomputed dealias mask from the grid if available.

        Falls back to a component-wise 2/3 rule mask if `grid.dealias_mask` is
        not present.
        """
        # come back to this i can make it better i recon
        mask = getattr(grid, "dealias_mask", None)
        if mask is None:
            # Fallback 2/3 rule mask- this needs to be exponential soon
            kxmax = jnp.max(jnp.abs(grid.kx))
            kymax = jnp.max(jnp.abs(grid.ky))
            kxcut = 2/3 * kxmax
            kycut = 2/3 * kymax
            mask_x = jnp.where(jnp.abs(grid.kx) <= kxcut, 1.0, 0.0)
            mask_y = jnp.where(jnp.abs(grid.ky) <= kycut, 1.0, 0.0)
            mask = mask_y[:, None] * mask_x[None, :]
        return y * mask
    
    @staticmethod
    def RK4(state, grid, cfg, ml_closure=None):
        """RK4 stepping for a single timestep. 
        - need to build this to choose a custom stepper eventually
        """
        qh, key = state
        # Draw one noise key per timestep and reuse it for all RK stages?
        key, k_noise = jax.random.split(key, 2)
        dt = cfg.params.dt

        rhs1, key = Solver.rhs(qh, k_noise, grid, cfg, ml_closure)
        rhs2, key = Solver.rhs(qh + 0.5 * dt * rhs1, k_noise, grid, cfg, ml_closure)
        rhs3, key = Solver.rhs(qh + 0.5 * dt * rhs2, k_noise, grid, cfg, ml_closure)
        rhs4, key = Solver.rhs(qh + dt * rhs3, k_noise, grid, cfg, ml_closure)
        qh_new = qh + (dt/6.0)*(rhs1 + 2*rhs2 + 2*rhs3 + rhs4)
        return (qh_new, key)
    
    @staticmethod
    def rhs(qh, key, grid, cfg, ml_closure=None): 
        """
        Compute the RHS of the QG equation in spectral space, with optional ML works. 
        Should return to make sure it is jitted properly.
        """
        key, key_noise = jax.random.split(key, 2)

        # PV to velocity in Fourier space
        psih = -qh * grid.invK2
        uh = -1j * grid.KY * psih
        vh =  1j * grid.KX * psih

        # Back to physical space, avoiding imaginary components(?)
        u, v, q = irfftn(jnp.stack([uh, vh, qh], axis=0), axes=(-2, -1), norm='ortho').real

        # Compute derivatives in physical space via spectral method
        dqdx = irfftn(1j * grid.KX * rfftn(q, axes=(-2,-1), norm='ortho'), axes=(-2,-1), norm='ortho').real
        dqdy = irfftn(1j * grid.KY * rfftn(q, axes=(-2,-1), norm='ortho'), axes=(-2,-1), norm='ortho').real

        # Skew-symmetric form
        nonlinear_phys = 0.5 * (u * dqdx + v * dqdy + dqdx * u + dqdy * v)  #0.5*(u·∇q + ∇·(uq))

        # Transform back to Fourier space and apply dealias filter
        nonlinear = rfftn(nonlinear_phys, axes=(-2,-1), norm='ortho')
        nonlinear = Solver.dealias(nonlinear, grid, s=8)

        # Beta term
        beta_term = - cfg.params.beta * 1j * grid.KX * psih

        # --- forcing (stochastic) ---
        if hasattr(cfg, 'forcing'): 
            noise_phys = jax.random.normal(key_noise, (grid.ny, grid.nx))
            forcing_h = rfftn(noise_phys, axes=(-2, -1), norm='ortho')

            k_f = cfg.forcing.k_f
            k_width = cfg.forcing.k_width
            L_rh = cfg.params.L_rh
            epsilon = 1e-6 *cfg.params.beta**3*L_rh**5 # target energy injection rate using C_eps = 1e-3

            # ---  (setting energy input for forcing) ---
            forcing_spectrum = jnp.exp(- (grid.Kmag - k_f)**2 / (2 * k_width**2)) # standard Gaussian annulus - im happy w this
            forcing_spectrum = jnp.where(grid.K2 == 0, 0.0, forcing_spectrum)

            eps0 = jnp.sum(forcing_spectrum * grid.invK2 / 2)
            forcing_spectrum = forcing_spectrum * (epsilon / eps0) # check this
            forcing = forcing_h * jnp.sqrt(forcing_spectrum/cfg.params.dt)

            forcing = Solver.dealias(forcing, grid, s=8)
        else:
            forcing = 0.0

        # === Model closure addition === #
        if ml_closure is not None:
            sgs_addition = ml_closure(qh)
        else:
            sgs_addition = 0.0 


        rhs =  nonlinear + beta_term + forcing + sgs_addition

        # --- Shape check ---
        assert u.shape == (grid.ny, grid.nx)
        assert qh.shape == (grid.ny, grid.nx//2 + 1), f"qh shape {qh.shape} unexpected"
        assert qh.shape == (grid.ny, grid.nx//2 + 1), f"qh shape {qh.shape} unexpected"

        return rhs, key
    
    def pseudo_randomiser(grid, k_peak, key):
        '''This is a method to make noise align better with the forcing wavenumber
        It returns qh right now, IS dealiased!!! No normalisation!!!!
        '''
        k0 = k_peak * 2 * jnp.pi / grid.Lx
        key_r, key_i = jax.random.split(key)

        modpsi = (grid.Kmag**2 * (1 + (grid.Kmag / k0)**4))**(-0.5)
        modpsi = modpsi.at[0,0].set(0.0)
        modpsi = jnp.clip(modpsi, a_min=1e-8, a_max=1e8) # is this clipping necessary?

        # this sets the psuedo-random using modpsi scaling
        phase = jax.random.normal(key_r, modpsi.shape) + 1j * jax.random.normal(key_i, modpsi.shape)
        psih = phase * modpsi
        psih = Solver.dealias(psih, grid, s=8)

        qh = -grid.K2 * psih
        qh = qh.at[:, 0].set(jnp.real(qh[:, 0]))

        return qh

    def write(self, h, path="solver"):
        h = IOInterface(h)
        g = h.create_group(path)
        del h

        g.attrs["type"] = type(self).__name__
        g.attrs["n"] = int(self.n)
        self.cfg.write(g.h, "parameters")
        self.fields.write(g.h, "fields")

        return g.h

    def new(self, *, copy_prescribed=False):
        model = type(self)(self.cfg)
        if copy_prescribed:
            for state in self.prescribed_field_states:
                model.fields[state] = self.fields[state]
        return model

    def update(self, model):
        self.fields.update(model.fields)
        self.n = model.n

    def flatten(self):
        """Return a JAX flattened representation.

        Returns
        -------

        Sequence[Sequence[object, ...], Sequence[object, ...]]
        """

        return ((dict(self.fields), self.n),
                (self.cfg, self.ml_closure))

    @classmethod
    def unflatten(cls, aux_data, children):
        """Unpack a JAX flattened representation.
        """

        cfg, ml_closure= aux_data
        fields, n = children

        model = cls(cfg, ml_closure=ml_closure)
        model.fields.update({key: value for key, value in fields.items() if type(value) is not object})
        if type(n) is not object:
            model.n = n

        return model
    
    @property
    def cfg(self):
        return self._cfg

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, g):
        self._grid = g

    @property
    def fields(self):
        return self._fields
    
    @property
    def n(self):
        return self._n
    
    @property
    def initial(self):
        """Return the initial condition in physical space."""
        return self._initial

    @n.setter
    def n(self, n):
        self._n = n

    @cached_property
    def beta(self):
        return jnp.array(self.cfg.params.beta)
    
    @cached_property
    def dt(self):
        return jnp.array(self.cfg.params.dt)

    @property
    def prescribed_field_states(self):
        return self._prescribed_field_states

    def zero_prescribed(self):
        """Zero prescribed fields.
        """

        self.fields.zero(*self.prescribed_field_states)

    @abstractmethod
    def initialize(self, zeta=None):
        """Initialize the model"""

        self.fields.clear(keep_states=self.prescribed_field_states)
        self._n = 0

    @abstractmethod 
    def step(self):
        """Take a timestep.
        """

        self._n += 1

    @staticmethod
    @jax.jit 
    def _step(_, model):
        model.step()
        return model

    def plot_field_at_step(self, step=0, frame_interval=1, vmin=None, vmax=None):
        if step > 0:
            self.steps(step * frame_interval)

        zeta = np.array(self.fields["zeta"])
        grid = self.grid
        nan_count = jnp.isnan(zeta).sum()
        print(f"Number of NaNs (black squares) at step {step}: {nan_count}")

        if vmin is None:
            vmin = jnp.nanmin(zeta)
        if vmax is None:
            vmax = jnp.nanmax(zeta)

        cmap = cm.get_cmap('RdBu_r').copy()
        cmap.set_bad(color='black')
        zeta_masked = np.ma.masked_invalid(zeta)

        plt.figure(figsize=(6,5))
        im = plt.imshow(
            zeta_masked,
            origin='lower',
            cmap=cmap,
            extent=(-grid.Lx/2, grid.Lx/2, -grid.Ly/2, grid.Ly/2),
            vmin=vmin,
            vmax=vmax
        )
        plt.colorbar(im, label=r'$\zeta$')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Vorticity Field at Step {step}')
        plt.show()



