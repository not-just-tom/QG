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
        # ensure single-layer qh has a leading layer axis: (layers, nl, nk)
        qh = jnp.expand_dims(qh, 0) # i just added this and im not sure if it will cause issues but it should make the shape consistent with two-layer states which always have a leading layer axis
        return states.State(qh=qh, _q_shape=(ny, nx))
        
    @abstractmethod
    def get_grid(self) -> Grid:
        pass

    @property
    def nl(self):
        return self.get_grid().nl

    @property
    def nk(self):
        return self.get_grid().nk

    @property
    @abstractmethod
    def kx(self) -> jax.Array:
        pass

    @property
    def _ik(self):
        return 1j * self.kx

    @property
    @abstractmethod
    def ky(self) -> jax.Array:
        pass

    @property
    @abstractmethod
    def Kmag(self) -> jax.Array:
        pass

    @property
    def _il(self):
        return 1j * self.ky
    
    @property
    def Qy(self) -> jax.Array:
        """Meridional gradient of background potential vorticity.
        For a single-layer model this is just the planetary vorticity gradient
        `beta` (constant in y). Return an array with length `nl` so it
        broadcasts correctly with spectral arrays.
        """
        return jnp.full((self.nl,), self.beta)

    @property
    def _ikQy(self):
        return 1j * (jnp.expand_dims(self.kx, 0) * jnp.expand_dims(self.Qy, -1))
    
    @property
    @abstractmethod
    def _dealias(self) -> jax.Array:
        """Dealias filter to be applied to spectral quantities."""
        pass
    
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
        # Set ph to zero (skip, recompute fresh from sum below)
        # invert qh to find ph
        ph = self._apply_a_ph(state)
        # calculate spectral velocities
        uh = jnp.negative(jnp.expand_dims(self._il, (0, -1))) * ph
        vh = jnp.expand_dims(self._ik, (0, 1)) * ph
        # Update state values
        return state.update(ph=ph, uh=uh, vh=vh)

    def _pseudo_random(self, key):
        # pseudo-random PV in spectral space
        key_r, key_i = jax.random.split(key)
        noise_real = 10*jax.random.normal(key_r, (self.nl, self.nk)) # ive got an arbitrary 10x multiplier here.
        noise_imag = 10*jax.random.normal(key_i, (self.nl, self.nk))
        qh = noise_real + 1j*noise_imag
        qh = qh.at[:, 0].set(jnp.real(qh[:, 0]))  
        qh = self._dealias*qh

        # then bandmask it
        band_mask = (self.Kmag >= self.kmin) & (self.Kmag <= self.kmax)
        qh = qh * band_mask
        qh = qh.at[:, 0].set(0.0)
        qh = self._dealias*qh
        return qh

    def _rhs_term(self, state: states.TempStates) -> states.TempStates:
        # multiply to get advective flux in space (single-layer)
        uq = state.u * state.q
        vq = state.v * state.q
        uqh = states._generic_rfftn(uq)
        vqh = states._generic_rfftn(vq)
        # apply dealias mask only to the nonlinear spectral products
        dmask = jnp.expand_dims(self._dealias, 0)
        uqh = uqh * dmask
        vqh = vqh * dmask

        # spectral divergence (include beta / Qy term)
        dqhdt = jnp.negative(
            jnp.expand_dims(self._ik, 0) * uqh
            + jnp.expand_dims(self._il, 1) * vqh
            + self._ikQy * state.ph
        )
        return state.update(dqhdt=dqhdt)
    
    def _apply_a_ph(self, state: states.TempStates) -> jax.Array:
        # Single-layer inversion: phi_hat = - q_hat / K^2, with k=0 mode set to 0
        qh = state.qh
        # build K^2 from kx, ky (shapes: (nk,), (nl,)) -> (nl, nk)
        K2 = (jnp.expand_dims(self.kx, 0) ** 2) + (jnp.expand_dims(self.ky, -1) ** 2)
        # avoid divide-by-zero at k=0 by masking; use safe divisor
        safe_K2 = jnp.where(K2 == 0.0, 1.0, K2)
        ph = -qh / safe_K2
        # enforce zero mean (k=0) to avoid singularity / drifting mean
        ph = ph.at[:, 0].set(0.0)
        return ph


@Pytree.register_pytree_class_attrs(
    children=["rek"],
    static_attrs=["nz", "ny", "nx", "kmin", "kmax"],
)
class TwoLayerKernel(ABC):
    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        nz: int,
        rek: float = 0,
        kmin: float = 3.0,
        kmax: float = 10,
    ):
        # Store small, fundamental properties (others will be computed on demand)
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.rek = rek
        self.kmin = kmin
        self.kmax = kmax

    def get_full_state(self, state: states.State) -> states.TempStates:
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

    def get_updates(self, state: states.State) -> states.State:
        full_state = self.get_full_state(state)
        return states.State(
            qh=full_state.dqhdt,
            _q_shape=self.get_grid().real_state_shape[-2:],
        )
    
    def initialise(self, seed) -> states.State:
        # pseudo-random initial condition
        key = jax.random.PRNGKey(seed)
        qh = self._pseudo_random(key)
        # Ensure _q_shape is Python ints, not JAX arrays
        ny, nx = int(self.ny), int(self.nx)
        # ensure single-layer qh has a leading layer axis: (layers, nl, nk)
        qh = jnp.expand_dims(qh, 0) # i just added this and im not sure if it will cause issues but it should make the shape consistent with two-layer states which always have a leading layer axis
        return states.State(qh=qh, _q_shape=(ny, nx))

    def _pseudo_random(self, key):
        # pseudo-random PV in spectral space (two-layer)
        key_r, key_i = jax.random.split(key)
        noise_real = 10*jax.random.normal(key_r, (self.nz, self.nl, self.nk))
        noise_imag = 10*jax.random.normal(key_i, (self.nz, self.nl, self.nk))
        qh = noise_real + 1j * noise_imag
        qh = qh.at[:, :, 0].set(jnp.real(qh[:, :, 0]))

        # apply filter and bandmask
        qh = jnp.expand_dims(self._dealias, 0) * qh
        band_mask = (self.Kmag >= self.kmin) & (self.Kmag <= self.kmax)
        qh = qh * jnp.expand_dims(band_mask, 0)
        qh = qh.at[:, :, 0].set(0.0)
        qh = jnp.expand_dims(self._dealias, 0) * qh
        return qh

    def postprocess_state(
        self, state: states.State) -> states.State:
        """ Check this im not sure I like it
        """
        return state.update(qh=jnp.expand_dims(self._dealias, 0) * state.qh)

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
    def kx(self) -> jax.Array:
        pass

    @property
    def _ik(self):
        return 1j * self.kx

    @property
    @abstractmethod
    def ky(self) -> jax.Array:
        pass

    @property
    @abstractmethod
    def Kmag(self) -> jax.Array:
        pass

    @property
    def _il(self):
        return 1j * self.ky

    @property
    def _k2l2(self) -> jax.Array:
        return (jnp.expand_dims(self.kx, 0) ** 2) + (jnp.expand_dims(self.ky, -1) ** 2)

    # Friction
    @property
    @abstractmethod
    def Ubg(self) -> jax.Array:
        pass

    @property
    @abstractmethod
    def _dealias(self) -> jax.Array:
        pass

    @property
    @abstractmethod
    def Qy(self) -> jax.Array:
        pass

    @property
    def _ikQy(self):
        return 1j * (jnp.expand_dims(self.kx, 0) * jnp.expand_dims(self.Qy, -1))

    def _invert(
        self, state: states.TempStates
    ) -> states.TempStates:
        # If kernel configured as single-layer (nz == 1), perform scalar inversion
        if getattr(self, "nz", None) == 1:
            qh = state.qh
            K2 = jnp.array(self.K2, dtype=qh.dtype)
            safe_K2 = jnp.where(K2 == 0, 1.0, K2)
            ph = -qh / safe_K2
            try:
                ph = ph.at[..., 0].set(0.0)
            except Exception:
                pass
            return state.update(ph=ph)

        # Existing two-layer inversion follows
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
        # apply dealias mask to nonlinear spectral products (broadcasts to layers)
        dmask = jnp.expand_dims(self._dealias, 0)
        uqh = uqh * dmask
        vqh = vqh * dmask

        # spectral divergence (two-layer); keep PV-gradient coupling if present
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
