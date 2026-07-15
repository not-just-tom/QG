"""Two-layer quasi-geostrophic model implementation."""

import inspect
import logging
import jax
import jax.numpy as jnp
from model.core.kernel import Kernel
import model.core.states as states
import model.utils.pytree as Pytree
from model.core.grid import Grid
from model.utils.forcing_diagnostics import tune_forcing_parameters

logger = logging.getLogger(__name__)

def grid(nx, ny, Lx, Ly):
    x, y = jnp.meshgrid(
        (jnp.arange(0.5, nx, 1.0) / nx) * Lx,
        (jnp.arange(0.5, ny, 1.0) / ny) * Ly,
    )
    return x, y

@Pytree.register_pytree_class_attrs(
    children=["beta", "rd", "delta", "U1", "U2", "H1"],
    static_attrs=["params"],
)
class QGM(Kernel):
    """multi-layer quasi-geostrophic model.
    """
    def __init__(self, params):
        self.params = params
        # Use safe dict lookups so missing keys (e.g. 'ny') don't raise KeyError
        nx = params.get('nx')
        ny = params.get('ny', nx)
        nz = params.get('nz', 1)
        rek = params.get('rek')
        kmin = params.get('kmin')
        kmax = params.get('kmax')
        forcing_center = params.get('forcing_center', 6.5)
        forcing_width = params.get('forcing_width', 2.0)
        forcing_amplitude = params.get('forcing_amplitude', 0.0)
        seed = params.get('seed', 0)
        self.beta = params.get('beta', 10.0)
        self.Lx = params.get('Lx', 6.28)
        self.Ly = params.get('Ly', self.Lx)
        self._Lz = params.get('Lz', 500)
        self.filterfac = params.get('filterfac', 23.6)
        self.g = params.get('g', 9.81)
        self.f = params.get('f', None)
        self.rd = params.get('rd', 15.0)
        self.delta = params.get('delta', 0.25)
        self.U1 = params.get('U1', 0.0)
        self.U2 = params.get('U2', 0.0)

        super().__init__(
            nx=nx,
            ny=ny,
            nz=nz,
            rek=rek,
            kmin=kmin,
            kmax=kmax,
            forcing_center=forcing_center,
            forcing_width=forcing_width,
            forcing_amplitude=forcing_amplitude,
            seed=seed,
        )

        # Precompute spectral grids and dealias filter to avoid recomputation
        # during tight stepping loops.
        grid = self.get_grid()
        # spectral frequencies (note normalization matches previous properties)
        self._kx = jnp.fft.rfftfreq(self.nx, d=(grid.dx / (2 * jnp.pi)))
        self._ky = jnp.fft.fftfreq(self.ny, d=(grid.dy / (2 * jnp.pi)))
        self._KX, self._KY = jnp.meshgrid(self._kx, self._ky)
        self._Kmag = jnp.sqrt(self._KX ** 2 + self._KY ** 2)
        self._K2 = self._Kmag ** 2        
        # Compute forcing mask based on forcing type
        self._update_forcing_mask()
        # Precompute two-layer elliptic inversion matrix A such that ph = A qh.
        # The (k,l)=(0,0) mode is singular and is explicitly set to zero.
        det = self._K2 * (self._K2 + self.F1 + self.F2)
        det_inv = jnp.where(det != 0, 1.0 / det, 0.0)
        det_inv = det_inv.at[0, 0].set(0.0)
        A = jnp.zeros((2, 2, self.ny, self.nx // 2 + 1), dtype=det_inv.dtype)
        A = A.at[0, 0].set(-(self._K2 + self.F2))
        A = A.at[0, 1].set(-self.F1)
        A = A.at[1, 0].set(-self.F2)
        A = A.at[1, 1].set(-(self._K2 + self.F1))
        self._A = A * det_inv
        # Use the same default dealiasing form as before (alpha=36, p=8)
        self._dealias_mask = jnp.exp(-36 * (self._Kmag / jnp.max(self._Kmag)) ** 8)
        # Optional exact post-step spectral damping (qg_closure-style)
        cphi = 0.65 * jnp.pi
        wvx = jnp.sqrt((self._KX * grid.dx) ** 2 + (self._KY * grid.dx) ** 2)
        exact_filter = jnp.exp(-self.filterfac * (wvx - cphi) ** 4)
        self._exact_step_filter = jnp.where(wvx <= cphi, 1.0, exact_filter)

    def _update_forcing_mask(self):
        """Recompute the spectral forcing mask from the current forcing parameters."""
        k_diff = jnp.abs(self.Kmag - self.forcing_center)
        self.forcing_mask = jnp.exp(-(k_diff / self.forcing_width) ** 2)

    def _tune_forcing_parameters(self, state, n_jets, verbose=False):
        """Choose forcing parameters that target the requested jet count while staying stable."""
        return tune_forcing_parameters(self, state, n_jets, verbose=verbose)

    def initialise(
        self,
        key,
        n_jets=None,
        tune=False,
        pseudo=False,
        verbose=False,
    ):
        """This still needs a lot of work - i need an auto replacing dt with the suggested dt from cfl, 
        and probably change to a energy level ? figure whether i should step the model/filter out some noise later
        """
        if pseudo and n_jets is None:
            raise ValueError("n_jets must be specified for pseudo random initialisation.")
        if tune and n_jets is None:
            raise ValueError("n_jets must be specified for tuning.")
        
        base_state = super().initialise(key, n_jets)
        if not tune:
            return base_state

        self._tune_forcing_parameters(base_state, n_jets, verbose=verbose)

        U_target = self.beta * (self.Ly / (jnp.pi * n_jets))**2
        U_rms = self.rhines_length(base_state)[1] # i actually think im not using rhines here despite the name - just U_rms


        scaler = U_target / (U_rms + 1e-12)
        qh = base_state.qh * scaler
        if verbose:
            logger.info(f"Initialised state with U_rms={U_rms:.3f}, scaled to U_target={U_target:.3f} with scale factor {scaler:.3f}")

        # Compute suggested dt only for debugging/logging to avoid extra work in vmapped init.
        scaled_state = base_state.update(qh=qh)
        if verbose:
            suggest_dt = self.estimate_cfl_dt(scaled_state)
            logger.info(f"Suggested initial dt for stability: {float(suggest_dt):.3f}")

        return scaled_state
    
    def set_initial(self, qh, _q_shape=None):
        """Set the initial state from a given spectral PV array `qh`."""
        return states.State(qh=qh, _q_shape=_q_shape)
    
    def get_full_state(self, state, forcing_key=None):
        return super().get_full_state(state, forcing_key=forcing_key)   

    def get_grid(self) -> Grid:
        """Retrieve the grid for this model."""
        return Grid(
            nz=getattr(self, "nz", 1),
            nx=self.nx,
            ny=self.ny,
            Lx=self.Lx,
            Ly=self.Ly,
        )
    
    def _get_dealias_filter(self, alpha=36, p=8):
        """Apply a precomputed dealias mask from the grid if available.
        """
        # fall back to precomputed mask when using default params
        if alpha == 36 and p == 8:
            return self._dealias_mask
        return jnp.exp(-alpha * (self.Kmag / jnp.max(self.Kmag)) ** p)
    
    @property
    def _dealias(self):
        """Dealias filter as a property."""
        return self._get_dealias_filter()

    def apply_exact_step_filter(self, state: states.State) -> states.State:
        """Apply exact spectral damping after each explicit step."""
        return state.update(qh=state.qh * jnp.expand_dims(self._exact_step_filter, 0))
    
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
    
    @property
    def dx(self):
        return self.get_grid().dx

    @property
    def dy(self):
        return self.get_grid().dy
    
    @property
    def ky(self):
        return self._ky

    @property
    def kx(self):
        return self._kx

    @property
    def KX(self):
        return self._KX

    @property
    def KY(self):
        return self._KY

    @property
    def ik(self):
        return 1j * self.KX

    @property
    def il(self):
        return 1j * self.KY

    @property
    def Kmag(self):
        """Total wavenumber magnitude."""
        return self._Kmag

    @property
    def K2(self):
        return self._K2
    
    @property
    def U(self):
        return self.U1 - self.U2

    @property
    def Lz(self):
        """Layer thicknesses: [H1, H2]"""
        return jnp.array(
            [self._Lz, self._Lz / self.delta]
        )

    @property
    def Ubg(self):
        return jnp.array([self.U1, self.U2])

    @property
    def F1(self):
        return self.rd**-2 / (1 + self.delta)

    @property
    def F2(self):
        return self.delta * self.F1

    @property
    def Qy1(self):
        return self.beta + self.F1 * (self.U1 - self.U2)

    @property
    def Qy2(self):
        return self.beta - self.F2 * (self.U1 - self.U2)

    @property
    def Qy(self):
        return jnp.array([self.Qy1, self.Qy2])

    @property
    def ikQy1(self):
        return self.Qy1 * 1j * self.k

    @property
    def ikQy2(self):
        return self.Qy2 * 1j * self.k

    @property
    def ikQy(self):
        return jnp.stack([self.ikQy1, self.ikQy2], axis=-3)

    @property
    def ilQx(self):
        return 0

    @property
    def del1(self):
        return self.delta / (self.delta + 1)

    @property
    def del2(self):
        return (self.delta + 1) ** -1

    def _apply_a_ph(self, state):
        qh = state.qh

        # find layer axis (size == self.nz)
        qh_shape = qh.shape
        try:
            layer_axis = next(i for i, s in enumerate(qh_shape) if s == self.nz)
        except StopIteration:
            raise ValueError("Could not find layer axis in state.qh")

        # move layer axis to the last position: (..., nl, nk, nz)
        qh_last = jnp.moveaxis(qh, layer_axis, -1)

        # qg_closure-style stable inversion: ph = A qh with A[...,0,0] mode zeroed.
        A = self._A.astype(qh.dtype)
        ph_last = jnp.einsum("ijlk,...lkj->...lki", A, qh_last)
        ph = jnp.moveaxis(ph_last, -1, layer_axis)
        return ph

    def rhines_length(self, state: states.State):
        """Estimate Rhines length from a `State` by computing U_rms and Lr = sqrt(U/beta).

        Returns (Lr, U_rms) as JAX scalars (trace-safe under jit/vmap).
        """
        full = self.get_full_state(state)
        u = full.u
        v = full.v
        U_rms = jnp.sqrt(jnp.mean(u ** 2 + v ** 2))
        beta = jnp.asarray(self.beta, dtype=U_rms.dtype)
        safe_beta = jnp.where(beta == 0, jnp.inf, beta)
        Lr = jnp.sqrt(U_rms / safe_beta)
        return Lr, U_rms

    def estimate_cfl_dt(self, state: states.State, cfl=0.1):
        """Estimate a stable `dt` based on CFL: dt = courant_no. * x_lengthscale/abs(U)
        """
        full = self.get_full_state(state)
        U_rms = jnp.sqrt(jnp.mean(full.u ** 2 + full.v ** 2))
        # Return a JAX scalar so this function remains safe under jit/vmap.
        dt = jnp.asarray(cfl, dtype=U_rms.dtype) * jnp.asarray(self.dx, dtype=U_rms.dtype) / (jnp.abs(U_rms) + 1e-12)
        return dt

    def compute_kinetic_energy(self, state: states.State) -> jnp.ndarray:
        """Compute total domain-averaged kinetic energy.
        
        Returns a JAX scalar representing the mean KE = 0.5 * (u^2 + v^2)
        """
        full = self.get_full_state(state)
        u = full.u
        v = full.v
        ke = jnp.mean(0.5 * (u ** 2 + v ** 2))
        return ke

    def compute_energy_spectrum(self, state: states.State) -> jnp.ndarray:
        """Compute kinetic energy spectrum E(k) in spectral space.
        
        Returns shape (nz, ny, nx//2+1) with energy per wavenumber mode.
        E(k) = 0.5 * (|uh|^2 + |vh|^2)
        """
        full = self.get_full_state(state)
        uh = full.uh
        vh = full.vh
        energy_spectrum = 0.5 * (jnp.abs(uh) ** 2 + jnp.abs(vh) ** 2)
        return energy_spectrum

    def get_forcing_scale_factor(self, state: states.State, target_energy_fraction: float = 0.1) -> jnp.ndarray:
        """Compute a normalization factor for forcing to maintain energy balance.
        
        This returns a scale factor such that the forcing amplitude is proportional to
        the system's initial kinetic energy. This prevents forcing from overwhelming
        or being negligible for different resolutions and initial conditions.
        
        Parameters
        ----------
        state : states.State
            Current state of the system
        target_energy_fraction : float
            Target fraction of initial energy to be added per step by forcing.
            Smaller values = weaker forcing. Default 0.1 (10% per step is aggressive).
            
        Returns
        -------
        scale_factor : jax.Array
            Scalar in [0, inf) used to multiply forcing_amplitude
        """
        ke = self.compute_kinetic_energy(state)
        # Avoid division by zero: if KE is very small, use a minimum scale
        safe_ke = jnp.where(ke > 1e-12, ke, 1e-12)
        # Scale factor: larger KE → larger scale factor (forcing scales with energy)
        scale_factor = jnp.sqrt(safe_ke) / jnp.sqrt(target_energy_fraction + 1e-12)
        return scale_factor

    @classmethod
    def from_params(cls, params):
        """Factory method to create `QGM` from a params dict.
        """
        # Filter params to only include those accepted by __init__
        sig = inspect.signature(cls.__init__)
        valid_params = {k: v for k, v in params.items() if k in sig.parameters}
        return cls(**valid_params)



