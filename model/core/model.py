"""Two-layer quasi-geostrophic model implementation."""

import inspect
import logging
import jax.numpy as jnp
from model.core.kernel import Kernel
import model.core.states as states
import model.utils.pytree as Pytree
from model.core.grid import Grid

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
        # Use the same default dealiasing form as before (alpha=36, p=8)
        self._dealias_mask = jnp.exp(-36 * (self._Kmag / jnp.max(self._Kmag)) ** 8)

    def initialise(
        self,
        key,
        n_jets=None,
        tune=False,
        pseudo=False,
        verbose=False
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

        U_target = self.beta * (self.Ly / (jnp.pi * n_jets))**2
        U_rms = self.rhines_length(base_state)[1] # i actually think im not using rhines here despite the name - just U_rms


        scaler = U_target / (U_rms + 1e-12)
        qh = base_state.qh * scaler
        if verbose:
            logger.info(f"Initialised state with U_rms={U_rms:.3f}, scaled to U_target={U_target:.3f} with scale factor {scaler:.3f}")

        # Compute suggested dt on the scaled state 
        scaled_state = base_state.update(qh=qh)
        suggest_dt = self.estimate_cfl_dt(scaled_state)
        if verbose:
            logger.info(f"Suggested initial dt for stability: {suggest_dt:.3f}")

        return scaled_state
    
    def set_initial(self, qh, _q_shape=None):
        """Set the initial state from a given spectral PV array `qh`."""
        return states.State(qh=qh, _q_shape=_q_shape)
    
    def get_full_state(self, state: states.State) -> states.FullState:
        """Expand a partial state into a full state with all computed values.
        """
        full_state = super().get_full_state(state)
        return full_state

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

        # Choose working dtypes to avoid unnecessary upcasts. Use complex64/float32
        # when input is complex64, otherwise preserve complex128/float64.
        if qh.dtype == jnp.complex128:
            c_dtype = jnp.complex128
            f_dtype = jnp.float64
        else:
            c_dtype = jnp.complex64
            f_dtype = jnp.float32

        K2 = self._K2.astype(f_dtype)
        F1 = jnp.array(self.F1, dtype=f_dtype)
        F2 = jnp.array(self.F2, dtype=f_dtype)

        a00 = -(K2 + F1)
        a01 = jnp.full_like(K2, F1)
        a10 = jnp.full_like(K2, F2)
        a11 = -(K2 + F2)

        # inv_mat shape (..., nl, nk, 2, 2) as complex for solve
        inv_mat = jnp.stack(
            [jnp.stack([a00, a01], axis=-1), jnp.stack([a10, a11], axis=-1)],
            axis=-2,
        ).astype(c_dtype)

        rhs = jnp.expand_dims(qh_last.astype(c_dtype), axis=-1)

        sol = jnp.linalg.solve(inv_mat, rhs).astype(state.qh.dtype)
        sol = jnp.squeeze(sol, axis=-1)
        ph = jnp.moveaxis(sol, -1, layer_axis)
        return ph

    def rhines_length(self, state: states.State):
        """Estimate Rhines length from a `State` by computing U_rms and Lr = sqrt(U/beta).

        Returns (Lr, U_rms) as floats.
        """
        full = self.get_full_state(state)
        u = full.u
        v = full.v
        U_rms = jnp.sqrt(jnp.mean(u ** 2 + v ** 2))
        beta = self.beta
        if beta == 0:
            return float('inf'), float(U_rms)
        Lr = jnp.sqrt(U_rms / beta)
        return float(Lr), float(U_rms)

    def estimate_cfl_dt(self, state: states.State, cfl=0.1):
        """Estimate a stable `dt` based on CFL: dt = courant_no. * x_lengthscale/abs(U)
        """
        full = self.get_full_state(state)
        U_rms = jnp.sqrt(jnp.mean(full.u ** 2 + full.v ** 2))
        # convert to python float for use in dt selection
        Ur = float(U_rms)
        dt = float(cfl * self.dx / (abs(Ur) + 1e-12))
        return dt

    @classmethod
    def from_params(cls, params):
        """Factory method to create `QGM` from a params dict.
        """
        # Filter params to only include those accepted by __init__
        sig = inspect.signature(cls.__init__)
        valid_params = {k: v for k, v in params.items() if k in sig.parameters}
        return cls(**valid_params)



