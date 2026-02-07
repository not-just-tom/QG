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
        # initialization parameters
        kmin=0.0,
        kmax=1.0e9,
        # constants
        f=None,
        g=9.81,
    ):
        super().__init__(
            nz=nz,
            ny=ny if ny is not None else nx,
            nx=nx,
            rek=rek,
            kmin=kmin,
            kmax=kmax,
        )
        self.Lx = Lx
        self.Ly = Ly if Ly is not None else Lx
        self.filterfac = filterfac
        self.g = g
        self.f = f

    def get_full_state(self, state: states.State) -> states.TempStates:
        """Expand a partial state into a full state with all computed values.
        """
        full_state = super().get_full_state(state)
        full_state = self._do_external_forcing(full_state)
        return full_state

    def get_grid(self) -> Grid:
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
        """
        return state.update(qh=jnp.expand_dims(self._dealias, 0) * state.qh)

    @property
    def f2(self):
        if self.f is not None:
            return self.f**2
        else:
            return None

    @property
    def dx(self):
        return self.get_grid().dx

    @property
    def dy(self):
        return self.get_grid().dy

    @property
    @abstractmethod
    def Lz(self) -> jax.Array:
        pass

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
    def wv2i(self):
        return jnp.where((self.wv2 != 0), jnp.power(self.wv2, -1), self.wv2)

    @property
    def _dealias(self):
        """Dealias for 2/3 strict rn
        """
        Kmag = jnp.sqrt(self.KX ** 2 + self.KY ** 2)
        kcut = (2.0 / 3.0) * jnp.max(Kmag)
        return jnp.where(Kmag <= kcut, 1.0, 0.0)



@Pytree.register_pytree_class_attrs(
    children=["beta", "rd", "delta", "U1", "U2", "H1"],
    static_attrs=[],
)
class TwoLayerModel(TwoLayerCalling):
    """Two-layer quasi-geostrophic model.
    """
    def __init__(
        self,
        *,
        # grid size parameters
        nx=64,
        ny=None,
        nz=1,
        Lx=6.28,
        Ly=None,
        Lz=500, #not sure if this is really a grid size param but hey

        # friction parameters
        rek=5.787e-7,
        filterfac=23.6,
        # constants
        f=None,
        g=9.81,
        # Additional model parameters
        beta=10,
        rd=15.0,
        delta=0.25,
        U1=0.0,
        U2=0.0,
        # initialization parameters
        kmin=3.0,
        kmax=10,
    ):
        super().__init__(
            nx=nx,
            ny=ny if ny is not None else nx,
            nz=nz, # atm this is where im changing the layers from. works for single but not for more. Something to do with the NNParams. 
            rek=rek,
            kmin=kmin,
            kmax=kmax,
        )
        self.Lx = Lx
        self.Ly = Ly if Ly is not None else Lx
        self._Lz = Lz
        self.filterfac = filterfac
        self.g = g
        self.f = f
        self.beta = beta
        self.rd = rd
        self.delta = delta
        self.U1 = U1
        self.U2 = U2

    

    @classmethod
    def from_params(cls, params):
        """Factory method to create TwoLayerModel from params dict.
        """
        # Filter params to only include those accepted by __init__
        sig = inspect.signature(cls.__init__)
        valid_params = {k: v for k, v in params.items() if k in sig.parameters}
        return cls(**valid_params)

    def initialise(self, seed):
        key = jax.random.PRNGKey(seed)
        qh = self._pseudo_random(key)
        return states.State(qh=qh, _q_shape=(self.ny, self.nx))

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
        qh = jnp.moveaxis(state.qh, 0, -1)
        qh_orig_shape = qh.shape
        qh = qh.reshape((-1, 2))

        K2 = self.K2.astype(jnp.float64)
        F1 = jnp.array(self.F1, dtype=jnp.float64)
        F2 = jnp.array(self.F2, dtype=jnp.float64)

        inv_mat2 = jnp.moveaxis(
            jnp.array(
                [
                    [
                        # 0, 0
                        -(K2 + F1),
                        # 0, 1
                        jnp.full_like(K2, F1),
                    ],
                    [
                        # 1, 0
                        jnp.full_like(K2, F2),
                        # 1, 1
                        -(K2 + F2),
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
    
def grid(nx, ny, Lx, Ly):
    x, y = jnp.meshgrid(
        (jnp.arange(0.5, nx, 1.0) / nx) * Lx,
        (jnp.arange(0.5, ny, 1.0) / ny) * Ly,
    )
    return x, y




