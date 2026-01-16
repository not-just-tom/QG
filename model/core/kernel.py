from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
import model.core.states as states
import model.utils.pytree as Pytree
from model.core.grid import Grid
from jax.numpy.fft import rfftn, irfftn

@Pytree.register_pytree_class_attrs(
    children=[],
    static_attrs=["params"],
)
class Kernel(ABC):
    def __init__(
        self,
        params
    ):
        # Store small, fundamental properties (others will be computed on demand)
        self.Lx = params['Lx']
        self.Ly = params['Ly'] if hasattr(params, 'Ly') else params['Lx'] 
        self.nx = params['nx']
        self.ny = params['ny'] if hasattr(params, 'ny') else params['nx'] 
        self.beta = params['beta']
        self.grid = Grid(self.Lx, self.nx)
        self.kmin = params['kmin']
        self.kmax = params['kmax']
        self.filtr = self._dealias
        
    # ==== Stuff to pass upwards. ==== # 
    def get_updates(self, state: states.State) -> states.State:
        full_state = self._get_state(state)
        return states.State(
            qh=full_state.dqhdt,
            _q_shape=self.get_grid().real_state_shape[-2:],
        )

    def dealias(self, state: states.State) -> states.State:
        # describe this 
        return state.update(qh=self.filtr*state.qh)

    def initialise(self, seed) -> states.State:
        # pseudo-random initial condition
        key = jax.random.PRNGKey(seed)
        qh = self._pseudo_random(key)
        return states.State(qh=qh, _q_shape=self.get_grid().real_state_shape[-2:],)
        
    @abstractmethod
    def get_grid(self) -> Grid:
        pass

    @property
    @abstractmethod
    def kk(self) -> jax.Array:
        pass

    @property
    def _ik(self):
        return 1j * self.kk

    @property
    @abstractmethod
    def ll(self) -> jax.Array:
        pass

    @property
    def _il(self):
        return 1j * self.ll

    # ==== Internals ==== #
    def _get_state(self, state: states.State) -> states.TempStates:
        def _empty_real():
            return jnp.zeros(
                self.get_grid().real_state_shape)

        def _empty_com():
            return jnp.zeros(
                self.get_grid().spectral_state_shape)

        #self._state_shape_check(state)
        full_state = states.TempStates(
            state=state,
            ph=_empty_com(),
            u=_empty_real(),
            v=_empty_real(),
            dqhdt=_empty_com(),
        )
        full_state = self._invert(full_state)
        full_state = self._rhs_term(full_state)
        return full_state

    def _invert(self, state: states.TempStates) -> states.TempStates:
        # invert PV to streamfunction in spectral space
        # qh lives on the state
        ph = -state.state.qh * self.get_grid().invK2

        # spectral velocities
        uh = -1j * self.get_grid().KY * ph
        vh =  1j *self.get_grid().KX * ph

        # transform to physical
        u = irfftn(uh, axes=(-2,-1), norm='ortho').real
        v = irfftn(vh, axes=(-2,-1), norm='ortho').real

        return state.update(ph=ph, u=u, v=v) 

    @property
    @abstractmethod
    def _dealias(self, qh):
        pass

    def _pseudo_random(self, key):
        # pseudo-random PV in spectral space
        key_r, key_i = jax.random.split(key)
        noise_real = jax.random.normal(key_r, (self.ny, self.nx//2+1))
        noise_imag = jax.random.normal(key_i, (self.ny, self.nx//2+1))
        qh = noise_real + 1j*noise_imag
        qh = qh.at[:, 0].set(jnp.real(qh[:, 0]))  
        qh = self.filtr*qh

        # then bandmask it
        band_mask = (self.get_grid().Kmag >= self.kmin) & (self.get_grid().Kmag <= self.kmax)
        qh = qh * band_mask
        qh = qh.at[:, 0].set(0.0)
        qh = self.filtr*qh
        return qh

    def _rhs_term(self, state: states.TempStates) -> states.TempStates:
        # multiply to get advective flux in space
        uq = state.u * state.q
        vq = state.v * state.q
        uqh = states._generic_rfftn(uq)
        vqh = states._generic_rfftn(vq)
        # spectral divergence
        dqhdt = jnp.negative( # this is wrong and needs changing. 
            1j *self.grid.KX * uqh
            + 1j * self.grid.KY * vqh
            + self.beta * 1j * self.grid.KX * state.ph
        )
        return state.update(dqhdt=dqhdt)

