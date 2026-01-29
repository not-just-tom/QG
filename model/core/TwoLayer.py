"""Two-layer quasi-geostrophic model implementation."""

import inspect
import jax
import jax.numpy as jnp
from abc import abstractmethod
from model.core.kernels import TwoLayerKernel
import model.core.states as states
import model.utils.pytree as Pytree
from model.core.grid import Grid


@Pytree.register_pytree_class_attrs(
    children=["L", "W", "filterfac", "g", "f"],
    static_attrs=[],
)
class TwoLayerCalling(TwoLayerKernel):
    def __init__(
        self,
        *,
        # grid size parameters
        nz=1,
        ny=None,
        nx=64,
        Lx=6.28,
        Ly=None,
        # friction parameters
        rek=5.787e-7,
        filterfac=23.6,
        # constants
        f=None,
        g=9.81,
    ):
        super().__init__(
            nz=nz,
            ny=ny if ny is not None else nx,
            nx=nx,
            rek=rek,
        )
        self.Lx = Lx
        self.Ly = Ly if Ly is not None else Lx
        self.filterfac = filterfac
        self.g = g
        self.f = f

    def get_full_state(
        self, state: states.State
    ) -> states.TempStates:
        """Expand a partial state into a full state with all computed values.

        Parameters
        ----------
        state : PseudoSpectralState
            The partial state to be expanded.

        Returns
        -------
        FullPseudoSpectralState
            New state object with all computed fields derived from `state`.
        """
        full_state = super().get_full_state(state)
        full_state = self._do_external_forcing(full_state)
        return full_state

    def get_grid(self) -> Grid:
        """Retrieve information on the model grid.

        .. versionadded:: 0.8.0

        Returns
        -------
        Grid
            A grid instance with attributes giving information on the
            spatial and spectral model grids.
        """
        return Grid(
            nz=self.nz,
            ny=self.ny,
            nx=self.nx,
            Lx=self.Lx,
            Ly=self.Ly,
        )

    def _do_external_forcing(
        self, state: states.TempStates
    ) -> states.TempStates:
        return state

    def dealias(self, state: states.State) -> states.State:
        """Apply spectral filter to state.
        
        Parameters
        ----------
        state : State
            The state to filter
            
        Returns
        -------
        State
            The filtered state
        """
        return state.update(qh=jnp.expand_dims(self.filtr, 0) * state.qh)

    @property
    def f2(self):
        if self.f is not None:
            return self.f**2
        else:
            return None

    @property
    def dk(self):
        return self.get_grid().dk

    @property
    def dl(self):
        return self.get_grid().dl

    @property
    def dx(self):
        return self.get_grid().dx

    @property
    def dy(self):
        return self.get_grid().dy

    @property
    def M(self):
        return self.nx * self.ny

    @property
    @abstractmethod
    def Hi(self) -> jax.Array:
        pass

    @property
    def x(self):
        return _grid_xy(
            nx=self.nx,
            ny=self.ny,
            Lx=self.Lx,
            Ly=self.Ly,
        )[0]

    @property
    def y(self):
        return _grid_xy(
            nx=self.nx,
            ny=self.ny,
            Lx=self.Lx,
            Ly=self.Ly,
        )[1]

    @property
    def ll(self):
        return jnp.fft.fftfreq(
            self.ny,
            d=(self.Ly / (2 * jnp.pi * self.ny)),
        )

    @property
    def kk(self):
        return jnp.fft.rfftfreq(
            self.nx,
            d=(self.Lx / (2 * jnp.pi * self.nx)),
        )

    @property
    def k(self):
        return _grid_kl(kk=self.kk, ll=self.ll)[0]

    @property
    def l(self):
        return _grid_kl(kk=self.kk, ll=self.ll)[1]

    @property
    def ik(self):
        return 1j * self.k

    @property
    def il(self):
        return 1j * self.l

    @property
    def wv(self):
        """Total wavenumber magnitude."""
        return jnp.sqrt(self.k**2 + self.l**2)

    @property
    def wv2(self):
        return self.wv**2

    @property
    def wv2i(self):
        return jnp.where((self.wv2 != 0), jnp.power(self.wv2, -1), self.wv2)

    @property
    def filtr(self):
        cphi = 0.65 * jnp.pi
        wvx = jnp.sqrt((self.k * self.dx) ** 2 + (self.l * self.dy) ** 2)
        filtr = jnp.exp(-self.filterfac * (wvx - cphi) ** 4)
        return jnp.where(wvx <= cphi, 1, filtr)



@Pytree.register_pytree_class_attrs(
    children=["beta", "rd", "delta", "U1", "U2", "H1"],
    static_attrs=[],
)
class TwoLayerModel(TwoLayerCalling):
    """Two-layer quasi-geostrophic model.
    
    Parameters
    ----------
    nx : int, optional
        Number of grid points in x direction (default: 64)
    ny : int, optional
        Number of grid points in y direction (default: nx)
    L : float, optional
        Domain length in x direction (default: 1e6)
    W : float, optional
        Domain length in y direction (default: L)
    rek : float, optional
        Linear drag in lower layer (default: 5.787e-7)
    filterfac : float, optional
        Amplitude of spectral filter (default: 23.6)
    f : float, optional
        Coriolis parameter (default: None)
    g : float, optional
        Gravitational acceleration (default: 9.81)
    beta : float, optional
        Gradient of Coriolis parameter (default: 1.5e-11)
    rd : float, optional
        Deformation radius (default: 15000.0)
    delta : float, optional
        Layer thickness ratio H1/H2 (default: 0.25)
    H1 : float, optional
        Upper layer thickness (default: 500)
    U1 : float, optional
        Upper layer flow (default: 0.025)
    U2 : float, optional
        Lower layer flow (default: 0.0)
    """
    
    def __init__(
        self,
        *,
        # grid size parameters
        nx=64,
        ny=None,
        L=1e6,
        W=None,
        # friction parameters
        rek=5.787e-7,
        filterfac=23.6,
        # constants
        f=None,
        g=9.81,
        # Additional model parameters
        beta=1.5e-11,
        rd=15000.0,
        delta=0.25,
        H1=500,
        U1=0.025,
        U2=0.0,
    ):
        super().__init__(
            nz=2,
            ny=ny if ny is not None else nx,
            nx=nx,
            rek=rek,
        )
        self.L = L
        self.W = W if W is not None else L
        self.filterfac = filterfac
        self.g = g
        self.f = f
        self.beta = beta
        self.rd = rd
        self.delta = delta
        self.U1 = U1
        self.U2 = U2
        self.H1 = H1

    @classmethod
    def from_params(cls, params):
        """Factory method to create TwoLayerModel from params dict.
        
        Parameters
        ----------
        params : dict
            Configuration dictionary with all required parameters
            
        Returns
        -------
        TwoLayerModel
            Initialized two-layer model instance
        """
        # Filter params to only include those accepted by __init__
        sig = inspect.signature(cls.__init__)
        valid_params = {k: v for k, v in params.items() if k in sig.parameters}
        return cls(**valid_params)

    def initialise(self, seed):
        """Create a new initial state with random initialization.

        Parameters
        ----------
        key : jax.random.key
            The PRNG state used as the random key for initialization.

        Returns
        -------
        PseudoSpectralState
            The new state with random initialization.
        """
        state = super().create_initial_state()
        # initial conditions (pv anomalies)
        key = jax.random.PRNGKey(seed)
        rng_a, rng_b = jax.random.split(key, num=2)
        q1 = 1e-7 * jax.random.uniform(
            rng_a, shape=(self.ny, self.nx)) + 1e-6 * (
            jnp.ones((self.ny, 1))
            * jax.random.uniform(rng_b, shape=(1, self.nx))
        )
        q2 = jnp.zeros_like(self.x)
        state = state.update(q=jnp.stack([q1, q2], axis=-3))
        return state

    @property
    def U(self):
        return self.U1 - self.U2

    @property
    def Hi(self):
        return jnp.array(
            [self.H1, self.H1 / self.delta]
        )

    @property
    def H(self):
        return self.get_grid().H

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
        # Double precision inversion, mirroring pyqg_jax
        qh = jnp.moveaxis(state.qh, 0, -1)
        qh_orig_shape = qh.shape
        qh = qh.reshape((-1, 2))

        wv2 = self.wv2.astype(jnp.float64)
        F1 = jnp.array(self.F1, dtype=jnp.float64)
        F2 = jnp.array(self.F2, dtype=jnp.float64)

        inv_mat2 = jnp.moveaxis(
            jnp.array(
                [
                    [
                        # 0, 0
                        -(wv2 + F1),
                        # 0, 1
                        jnp.full_like(wv2, F1),
                    ],
                    [
                        # 1, 0
                        jnp.full_like(wv2, F2),
                        # 1, 1
                        -(wv2 + F2),
                    ],
                ],
                dtype=jnp.float64,
            ),
            (0, 1),
            (-2, -1),
        ).reshape((-1, 2, 2))[1:]

        ph_tail = jnp.squeeze(
            jnp.linalg.solve(
                inv_mat2,
                jnp.expand_dims(qh[1:].astype(jnp.complex128), -1),
            ).astype(state.qh.dtype),
            -1,
        )
        ph_head = jnp.expand_dims(jnp.zeros_like(qh[0]), 0)
        ph = jnp.concatenate([ph_head, ph_tail], axis=0).reshape(qh_orig_shape)
        return jnp.moveaxis(ph, -1, 0)
    
def _grid_xy(nx, ny, Lx, Ly):
    x, y = jnp.meshgrid(
        (jnp.arange(0.5, nx, 1.0) / nx) * Lx,
        (jnp.arange(0.5, ny, 1.0) / ny) * Ly,
    )
    return x, y


def _grid_kl(kk, ll):
    k, l = jnp.meshgrid(kk, ll)
    return k, l




