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
import importlib
import model.core.TwoLayer
importlib.reload(model.core.TwoLayer)
import model.core.grid
importlib.reload(model.core.grid)
from model.core.grid import Grid
import model.utils.pytree as Pytree
import model.core.kernels
importlib.reload(model.core.kernels)
from model.core.kernels import SingleLayerKernel, TwoLayerKernel
import model.core.states as states


def create_model(params, n_layers=1):
    """Factory function to create QG models based on configuration.
    
    Parameters
    ----------
    params : dict
        Configuration dictionary with model parameters
    n_layers : int, optional
        Number of layers: 1 for single-layer (default), 2 for two-layer model
        
    Returns
    -------
    Model or TwoLayerModel
        Appropriate model instance
    """
    if n_layers == 1:
        return SingleLayerModel(params)
    elif n_layers == 2:
        # Import here to avoid circular imports
        from model.core.TwoLayer import TwoLayerModel
        return TwoLayerModel.from_params(params)
    else:
        raise ValueError(f"Unsupported n_layers={n_layers}. Use 1 or 2.")


def grid(nx, ny, Lx, Ly):
    dx = Lx / nx
    dy = Ly / ny
    x = jnp.linspace(-Lx/2 + dx/2, Lx/2 - dx/2, nx)
    y = jnp.linspace(-Ly/2 + dy/2, Ly/2 - dy/2, ny)
    return jnp.meshgrid(x, y)


@Pytree.register_pytree_class_attrs(
    children=[],
    static_attrs=["nx", "ny", "Lx", "Ly", "beta", "kmin", "kmax"],
)
class SingleLayerModel(SingleLayerKernel):
    """Base single-layer QG model with spectral solver.
    
    Requires params dict with keys:
    - nx, ny: grid resolution
    - Lx, Ly: domain size
    - beta: beta-plane parameter
    - kmin, kmax: wavenumber band for initialization
    """
    
    def __init__(self, params):
        """Initialize model from params dict.
        """
        super().__init__(params=params)

    def get_full_state(self, state: states.State) -> states.TempStates:
        """Compute all the temp values for a step in kernel.
        """
        full_state = super()._get_state(state)
        full_state = self._do_external_forcing(full_state)
        return full_state

    def get_grid(self) -> Grid:
        """Retrieve the grid for this model."""
        return Grid(
            nx=self.nx,
            ny=self.ny,
            Lx=self.Lx,
            Ly=self.Ly,
        )

    def _do_external_forcing(self, state: states.TempStates) -> states.TempStates:
        # put machine leanring here?
        return state
    
    def initialise(self, seed, *, tune=False, target_Lr=None, cfl=0.1, nspin=50, dt_spin=1e-4, tol=0.05, max_iter=10):
        """Initialize state. Optionally tune initial PV amplitude so Rhines length matches target.

        Parameters
        ----------
        seed : int
            RNG seed for initial condition
        tune : bool
            If True, perform bisection on amplitude with short spin-ups to match `target_Lr`.
        target_Lr : float or None
            Desired Rhines length. If None and `tune` is True, uses Lx/8.
        cfl : float
            CFL fraction used for suggested dt estimate (returned via `estimate_cfl_dt`).
        nspin : int
            Number of short spin-up steps per trial amplitude.
        dt_spin : float
            Time step used during short Euler spin-ups.
        tol : float
            Relative tolerance for matching target Rhines length.
        max_iter : int
            Maximum bisection iterations.

        Returns
        -------
        State
            Tuned initial `State` (spectral `qh`)
        """
        base_state = super().initialise(seed)
        if not tune:
            return base_state

        # determine default target Rhines length if not provided
        grid = self.get_grid()
        if target_Lr is None:
            target_Lr = float(grid.Lx) / 8.0

        # bisection bracket for amplitude
        a_lo, a_hi = 1e-6, 1e3

        def spin_up(state, dt, nsteps):
            s = state
            for _ in range(int(nsteps)):
                updates = self.get_updates(s)
                # simple explicit Euler step for spin-up
                s = s.update(qh=(s.qh + dt * updates.qh))
            return s

        best_state = base_state
        for _ in range(max_iter):
            a_mid = float((a_lo * a_hi) ** 0.5)
            trial = base_state.update(qh=base_state.qh * a_mid)
            spun = spin_up(trial, dt_spin, nspin)
            Lr, U = self.rhines_length(spun)
            if abs(Lr - target_Lr) / target_Lr < tol:
                best_state = spun
                break
            # if Lr < target -> increase amplitude (U too small)
            if Lr < target_Lr:
                a_lo = a_mid
            else:
                a_hi = a_mid
            best_state = spun

        # Optionally compute suggested dt from CFL using the spun-up state
        dt_suggest, umax, vmax = self.estimate_cfl_dt(best_state, cfl=cfl)
        print(dt_suggest)

        return best_state
        

    
    def _get_dealias_filter(self, alpha=36, p=8):
        """Apply a precomputed dealias mask from the grid if available.
        """
        return jnp.exp(-alpha * (self.Kmag / jnp.max(self.Kmag)) ** p)
    
    @property
    def _dealias(self):
        """Dealias filter as a property."""
        return self._get_dealias_filter()

    @property
    def dx(self):
        return self.get_grid().dx

    @property
    def dy(self):
        return self.get_grid().dy
    
    @property
    def kx(self):
        return jnp.fft.rfftfreq(self.nx, d=self.dx / (2 * jnp.pi))

    @property
    def ky(self):
        return jnp.fft.fftfreq(self.ny, d=self.dy / (2 * jnp.pi))

    @property
    def Kmag(self):
        KX, KY = jnp.meshgrid(self.kx, self.ky)
        return jnp.sqrt(KX**2 + KY**2)

    @property
    def x(self):
        return grid(
            nx=self.nx,
            ny=self.ny,
            Lx=self.Lx,
            Ly=self.Ly,
        )[0]

    @property
    def y(self):
        return grid(
            nx=self.nx,
            ny=self.ny,
            Lx=self.Lx,
            Ly=self.Ly,
        )[1]

    def rhines_length(self, state: states.State):
        """Estimate Rhines length from a `State` by computing U_rms and Lr = sqrt(U/beta).

        Returns (Lr, U_rms) as floats.
        """
        full = self.get_full_state(state)
        u = full.u
        v = full.v
        U_rms = float(jnp.sqrt(jnp.mean(u ** 2 + v ** 2)))
        beta = float(self.beta)
        if beta == 0:
            return float('inf'), U_rms
        Lr = float(jnp.sqrt(U_rms / beta))
        return Lr, U_rms

    def estimate_cfl_dt(self, state: states.State, cfl=0.1):
        """Estimate a stable `dt` based on CFL: dt = cfl * min(dx/umax, dy/vmax).

        Returns (dt, umax, vmax).
        """
        full = self.get_full_state(state)
        u = full.u
        v = full.v
        umax = float(jnp.max(jnp.abs(u)))
        vmax = float(jnp.max(jnp.abs(v)))
        dx = float(self.dx)
        dy = float(self.dy)
        eps = 1e-12
        dt = float(cfl * min(dx / (umax + eps), dy / (vmax + eps))) #is min best here?
        return dt, umax, vmax
