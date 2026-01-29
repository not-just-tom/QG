from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
import model.core.states as states
import model.utils.pytree as Pytree
from model.core.grid import Grid
from jax.numpy.fft import rfftn, irfftn

@Pytree.register_pytree_class_attrs(
    children=[],
    static_attrs=["nx", "ny", "Lx", "Ly", "beta", "kmin", "kmax"],
)
class SingleLayerKernel(ABC):
    def __init__(
        self,
        params
    ):
        # Store small, fundamental properties (others will be computed on demand)
        # Extract primitive values from params and register them as static_attrs
        self.nx = params['nx']
        self.ny = params.get('ny', params['nx'])
        self.Lx = params['Lx']
        self.Ly = params.get('Ly', params['Lx'])
        self.beta = params['beta']
        self.kmin = params['kmin']
        self.kmax = params['kmax']
        
    # ==== Stuff to pass upwards. ==== # 
    def get_updates(self, state: states.State) -> states.State:
        full_state = self._get_state(state)
        # Ensure _q_shape is Python ints, not JAX arrays
        ny, nx = int(self.ny), int(self.nx)
        return states.State(
            qh=full_state.dqhdt,
            _q_shape=self.get_grid().real_state_shape[-2:],
        )

    def dealias(self, state: states.State) -> states.State:
        # describe this 
        return state.update(qh=self._dealias*state.qh)

    def initialise(self, seed) -> states.State:
        # pseudo-random initial condition
        key = jax.random.PRNGKey(seed)
        qh = self._pseudo_random(key)
        # Ensure _q_shape is Python ints, not JAX arrays
        ny, nx = int(self.ny), int(self.nx)
        return states.State(qh=qh, _q_shape=(ny, nx))
        
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
    
    @property
    @abstractmethod
    def _dealias(self) -> jax.Array:
        """Dealias filter to be applied to spectral quantities."""
        pass
    
    @property
    def grid(self) -> Grid:
        """Lazily compute the grid on demand."""
        return Grid(self.Lx, self.nx, self.Ly, self.ny)
    
    @property
    def params(self) -> dict:
        """Reconstruct params dict from static attributes."""
        return {
            'nx': self.nx,
            'ny': self.ny,
            'Lx': self.Lx,
            'Ly': self.Ly,
            'beta': self.beta,
            'kmin': self.kmin,
            'kmax': self.kmax,
        }

        

    # ==== Internals ==== #
    def _get_state(self, state: states.State) -> states.TempStates:
        def _empty_real():
            return jnp.zeros(self.get_grid().real_state_shape)

        def _empty_com():
            return jnp.zeros(self.get_grid().spectral_state_shape)

        #self._state_shape_check(state)
        full_state = states.TempStates(
            state=state,
            ph=_empty_com(),
            u=_empty_real(),
            v=_empty_real(),
            dqhdt=_empty_com(),
        )
        # Debug: trace shapes/types flowing through the kernel
        full_state = self._invert(full_state)
        full_state = self._rhs_term(full_state)
        return full_state

    def _invert(self, state: states.TempStates) -> states.TempStates:
        # invert PV to streamfunction in spectral space
        # qh lives on the state
        ph = -state.state.qh * self.get_grid().invK2

        # spectral velocities
        uh = -1j * self.get_grid().KY * ph
        vh =  1j * self.get_grid().KX * ph

        # transform to physical
        u = irfftn(uh, axes=(-2,-1), norm='ortho').real
        v = irfftn(vh, axes=(-2,-1), norm='ortho').real

        return state.update(ph=ph, u=u, v=v) 

    def _pseudo_random(self, key):
        # pseudo-random PV in spectral space
        key_r, key_i = jax.random.split(key)
        noise_real = jax.random.normal(key_r, (self.ny, self.nx//2+1))
        noise_imag = jax.random.normal(key_i, (self.ny, self.nx//2+1))
        qh = noise_real + 1j*noise_imag
        qh = qh.at[:, 0].set(jnp.real(qh[:, 0]))  
        qh = self._dealias*qh

        # then bandmask it
        band_mask = (self.get_grid().Kmag >= self.kmin) & (self.get_grid().Kmag <= self.kmax)
        qh = qh * band_mask
        qh = qh.at[:, 0].set(0.0)
        qh = self._dealias*qh
        return qh

    def _rhs_term(self, state: states.TempStates) -> states.TempStates:
        # multiply to get advective flux in space
        q_real = state.q

        uq = state.u * q_real
        vq = state.v * q_real
        uqh = states._generic_rfftn(uq)
        vqh = states._generic_rfftn(vq)
        
        # spectral divergence
        beta_term = self.beta * 1j * self.grid.KX * state.ph
        dqhdt = jnp.negative(
            1j *self.grid.KX * uqh
            + 1j * self.grid.KY * vqh
            + beta_term
        )
        return state.update(dqhdt=dqhdt)

@Pytree.register_pytree_class_attrs(
    children=["rek"],
    static_attrs=["nz", "ny", "nx"],
)
class TwoLayerKernel(ABC):
    def __init__(
        self,
        *,
        nz: int,
        ny: int,
        nx: int,
        rek: float = 0,
    ):
        # Store small, fundamental properties (others will be computed on demand)
        self.nz = nz
        self.ny = ny
        self.nx = nx
        self.rek = rek

    def get_full_state(
        self, state: states.State
    ) -> states.TempStates:
        def _empty_real():
            return jnp.zeros(
                self.get_grid().real_state_shape
            )

        def _empty_com():
            return jnp.zeros(
                self.get_grid().spectral_state_shape
            )

        self._state_shape_check(state)
        full_state = states.TempStates(
            state=state,
            ph=_empty_com(),
            u=_empty_real(),
            v=_empty_real(),
            dqhdt=_empty_com(),
        )
        full_state = self._invert(full_state)
        full_state = self._do_advection(full_state)
        full_state = self._do_friction(full_state)
        return full_state

    def get_updates(
        self, state: states.State
    ) -> states.State:
        """Get updates for time-stepping `state`.

        Parameters
        ----------
        state : State
            The state which will be time stepped using the computed updates.

        Returns
        -------
        State
            A new state object where each field corresponds to a
            time-stepping *update* to be applied.

        Note
        ----
        The object returned by this function has the same type of
        `state`, but contains *updates*. This is so the time-stepping
        can be done by mapping over the states and updates as JAX
        pytrees with the same structure.

        """
        full_state = self.get_full_state(state)
        return states.State(
            qh=full_state.dqhdt,
            _q_shape=self.get_grid().real_state_shape[-2:],
        )

    def postprocess_state(
        self, state: states.State) -> states.State:
        """Apply fixed filtering to `state`.

        This function should be called once on each new state after each time step.

        :class:`~pyqg_jax.steppers.SteppedModel` handles
        this internally.

        Parameters
        ----------
        state : State
            The state to be filtered.

        Returns
        -------
        State
            The filtered state.
        """
        return state.update(qh=jnp.expand_dims(self.filtr, 0) * state.qh)

    def create_initial_state(self, key=None) -> states.State:
        return states.State(
            qh=jnp.zeros(
                self.get_grid().spectral_state_shape
            ),
            _q_shape=self.get_grid().real_state_shape[-2:],
        )

    @abstractmethod
    def get_grid(self) -> Grid:
        pass

    def _state_shape_check(self, state):
        corr_shape = self.get_grid().spectral_state_shape
        corr_dims = len(corr_shape)
        dims = state.qh.ndim
        if dims != corr_dims:
            vmap_msg = " (use jax.vmap)" if dims > corr_dims else ""
            raise ValueError(
                f"state has {dims} dimensions, but should have {corr_dims}{vmap_msg}"
            )
        if state.qh.shape != corr_shape:
            raise ValueError(
                f"state.qh has wrong shape {state.qh.shape}, should be {corr_shape}"
            )

    @property
    def nl(self):
        return self.get_grid().nl

    @property
    def nk(self):
        return self.get_grid().nk

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

    @property
    def _k2l2(self) -> jax.Array:
        return (jnp.expand_dims(self.kk, 0) ** 2) + (jnp.expand_dims(self.ll, -1) ** 2)

    # Friction
    @property
    @abstractmethod
    def Ubg(self) -> jax.Array:
        pass

    @property
    @abstractmethod
    def filtr(self) -> jax.Array:
        pass

    @property
    @abstractmethod
    def Qy(self) -> jax.Array:
        pass

    @property
    def _ikQy(self):
        return 1j * (jnp.expand_dims(self.kk, 0) * jnp.expand_dims(self.Qy, -1))

    def _invert(
        self, state: states.TempStates
    ) -> states.TempStates:
        # Set ph to zero (skip, recompute fresh from sum below)
        # invert qh to find ph
        ph = self._apply_a_ph(state)
        # calculate spectral velocities
        uh = jnp.negative(jnp.expand_dims(self._il, (0, -1))) * ph
        vh = jnp.expand_dims(self._ik, (0, 1)) * ph
        # Update state values
        return state.update(ph=ph, uh=uh, vh=vh)

    def _do_advection(
        self, state: states.TempStates
    ) -> states.TempStates:
        # multiply to get advective flux in space
        uq = (state.u + jnp.expand_dims(self.Ubg[: self.nz], (-1, -2))) * state.q
        vq = state.v * state.q
        uqh = states._generic_rfftn(uq)
        vqh = states._generic_rfftn(vq)
        # spectral divergence
        dqhdt = jnp.negative(
            jnp.expand_dims(self._ik, (0, 1)) * uqh
            + jnp.expand_dims(self._il, (0, -1)) * vqh
            + jnp.expand_dims(self._ikQy[: self.nz], 1) * state.ph
        )
        return state.update(dqhdt=dqhdt)

    def _do_friction(
        self, state: states.TempStates
    ) -> states.TempStates:
        # Apply Beckman friction to lower layer tendency

        def compute_friction(state):
            dqhdt = jnp.concatenate(
                [
                    state.dqhdt[:-1],
                    jnp.expand_dims(
                        state.dqhdt[-1] + (self.rek * self._k2l2 * state.ph[-1]), 0
                    ),
                ],
                axis=0,
            )
            return state.update(dqhdt=dqhdt)

        return jax.lax.cond(
            self.rek != 0,
            compute_friction,
            lambda state: state,
            state,
        )

    @abstractmethod
    def _apply_a_ph(self, state: states.TempStates) -> jax.Array:
        pass

    def __repr__(self):
        return Pytree.auto_repr(self)
