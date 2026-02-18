"""Two-layer quasi-geostrophic model implementation."""

import inspect
import jax
import jax.numpy as jnp
from model.core.kernel import Kernel
import model.core.states as states
import model.utils.pytree as Pytree
from model.core.grid import Grid

def grid(nx, ny, Lx, Ly):
    x, y = jnp.meshgrid(
        (jnp.arange(0.5, nx, 1.0) / nx) * Lx,
        (jnp.arange(0.5, ny, 1.0) / ny) * Ly,
    )
    return x, y

@Pytree.register_pytree_class_attrs(
    children=["beta", "rd", "delta", "U1", "U2", "H1"],
    static_attrs=[],
)
class QGM(Kernel):
    """multi-layer quasi-geostrophic model.
    """
    def __init__(self, params):
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

    def initialise(
        self,
        seed,
        n_jets=None,
        tune=False,
        scale=1
    ):
        """This still needs a lot of work - i need an auto replacing dt with the suggested dt from cfl, 
        and probably change to a energy level ? figure whether i should step the model/filter out some noise later
        """
        
        base_state = super().initialise(seed, n_jets)
        if not tune:
            return base_state
        
        if n_jets is None:
            raise ValueError("n_jets must be specified when tune=True")

        U_target = self.beta * (self.Ly / (jnp.pi * n_jets))**2
        U_rms = self.rhines_length(base_state)[1] # i actually think im not using rhines here despite the name - just U_rms


        scaler = U_target / (U_rms + 1e-12)
        qh = base_state.qh * scaler
        print(f"Initialised state with U_rms={U_rms:.3f}, scaled to U_target={U_target:.3f} with scale factor {scaler:.3f}")

        # Compute suggested dt on the scaled state 
        scaled_state = base_state.update(qh=qh)
        suggest_dt = self.estimate_cfl_dt(scaled_state)
        print(f"Suggested initial dt for stability: {suggest_dt:.3f}")

        return scaled_state
    
    def set_initial(self, qh):
        """Set the initial state from a given spectral PV array `qh`."""
        return states.State(qh=qh)
    
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
        return jnp.fft.fftfreq(self.ny, d=(self.dy / (2 * jnp.pi)))

    @property
    def kx(self):
        return jnp.fft.rfftfreq(self.nx, d=(self.dx / (2 * jnp.pi)))

    @property
    def KX(self):
        return jnp.meshgrid(self.kx, self.ky)[0]

    @property
    def KY(self):
        return jnp.meshgrid(self.kx, self.ky)[1]

    @property
    def ik(self):
        return 1j * self.KX

    @property
    def il(self):
        return 1j * self.KY

    @property
    def Kmag(self):
        """Total wavenumber magnitude."""
        return jnp.sqrt(self.KX**2 + self.KY**2)

    @property
    def K2(self):
        return self.Kmag**2
    
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
        # Double precision inversion for better stability in the inversion step
        qh = state.qh  # shape (..., nz, nl, nk) or (nz, nl, nk)

        # find which axis corresponds to the layer dimension (nz)
        qh_shape = qh.shape
        try:
            layer_axis = next(i for i, s in enumerate(qh_shape) if s == self.nz)
        except StopIteration:
            raise ValueError("Could not find layer axis in state.qh")

        # move layer axis to the last position: (..., nl, nk, nz)
        qh_last = jnp.moveaxis(qh, layer_axis, -1)

        # build the 2x2 coefficient matrix per (nl, nk)
        #print('model, before: ', qh.shape)
        qh = qh.reshape((-1, 2))
        #print('model, after: ', qh.shape)

        K2 = self.K2.astype(jnp.float64)
        F1 = jnp.array(self.F1, dtype=jnp.float64)
        F2 = jnp.array(self.F2, dtype=jnp.float64)

        a00 = -(K2 + F1)
        a01 = jnp.full_like(K2, F1)
        a10 = jnp.full_like(K2, F2)
        a11 = -(K2 + F2)

        # inv_mat shape (nl, nk, 2, 2)
        inv_mat = jnp.stack(
            [jnp.stack([a00, a01], axis=-1), jnp.stack([a10, a11], axis=-1)],
            axis=-2,
        )

        # RHS for solve: shape (..., nl, nk, 2, 1)
        rhs = jnp.expand_dims(qh_last.astype(jnp.complex128), axis=-1)

        # Solve per-wavenumber: inv_mat broadcasts over any leading dims
        sol = jnp.linalg.solve(inv_mat, rhs).astype(state.qh.dtype)

        # sol shape (..., nl, nk, 2, 1) -> squeeze last dim -> (..., nl, nk, 2)
        sol = jnp.squeeze(sol, axis=-1)

        # move layer dimension (size 2) back to its original position
        ph = jnp.moveaxis(sol, -1, layer_axis)

        return ph

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
        return dt

    @classmethod
    def from_params(cls, params):
        """Factory method to create `QGM` from a params dict.
        """
        # Filter params to only include those accepted by __init__
        sig = inspect.signature(cls.__init__)
        valid_params = {k: v for k, v in params.items() if k in sig.parameters}
        return cls(**valid_params)



