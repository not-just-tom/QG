from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
import model.core.states as states
import model.utils.pytree as Pytree
from model.core.grid import Grid
import logging

# Module logger for Kernel
logger = logging.getLogger(__name__)

@Pytree.register_pytree_class_attrs(
    children=["rek", "forcing_amplitude"],
    static_attrs=["nz", "ny", "nx", "Lx", "Ly", "kmin", "kmax", "seed"],
)
class Kernel(ABC):
    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        nz: int,
        Lx: float,
        Ly: float,
        rek: float = 0,
        kmin: float = 3.0,
        kmax: float = 10,
        forcing_amplitude: float = 0.0,
        seed: int = 0,
        dt: float = 1.0,
    ):
        # params
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.Lx = Lx
        self.Ly = Ly
        self.rek = rek
        self.kmin = kmin*2*jnp.pi/Lx
        self.kmax = kmax*2*jnp.pi/Lx
        self.forcing_amplitude = forcing_amplitude
        self.seed = seed
        self.dt = dt


    def dealias(self, state: states.State) -> states.State:
        # describe this 
        return state.update(qh=self._dealias*state.qh)

    def spectral_to_real(self, field_h: jax.Array) -> jax.Array:
        """Transform a spectral field using this model's physical grid."""
        return states._generic_irfftn(
            field_h, shape=self.get_grid().real_state_shape
        )

    def real_to_spectral(self, field: jax.Array) -> jax.Array:
        """Transform a physical field using the model's FFT convention."""
        return states._generic_rfftn(field)

    def dealias_spectral(self, field_h: jax.Array) -> jax.Array:
        """Filter a spectral field after a nonlinear physical-space product.

        The leading dimensions may be layer and/or batch dimensions; the
        model's two-dimensional mask is broadcast over all of them.
        """
        mask = self._dealias.reshape(
            (1,) * (field_h.ndim - self._dealias.ndim) + self._dealias.shape
        )
        return field_h * mask

    def dealiased_product(self, field: jax.Array) -> jax.Array:
        """Transform and de-alias a nonlinear physical-space product."""
        return self.dealias_spectral(self.real_to_spectral(field))

    def apply_exact_step_filter(self, state: states.State) -> states.State:
        """Apply optional exact post-step spectral damping.

        Base kernels default to no-op; concrete models can override.
        """
        return state

    def get_full_state(
        self, state: states.State, forcing_key: jax.Array | None = None
    ) -> states.FullState:
        """Compute full state with all tendencies.
        
        Parameters
        ----------
        state : states.State
            The model state
        """
        def _empty_real():
            return jnp.zeros(
                self.get_grid().real_state_shape
            )

        def _empty_com():
            return jnp.zeros(
                self.get_grid().spectral_state_shape
            )

        self._state_shape_check(state)
        full_state = states.FullState(
            state=state,
            ph=_empty_com(),
            u=_empty_real(),
            v=_empty_real(),
            dqhdt=_empty_com(),
        )
        full_state = self._invert(full_state)
        full_state = self._do_advection(full_state)
        full_state = self._do_friction(full_state)
        full_state = self._do_stochastic_forcing(full_state, forcing_key)
        full_state = self._do_wind_forcing(full_state)
        return full_state

    def invert_pv(self, state: states.State) -> jax.Array:
        """Return streamfunction obtained from the model's PV inversion.

        This is intentionally narrower than :meth:`get_full_state`: closures
        that need only streamfunction do not need to also evaluate advection,
        friction, or stochastic forcing.
        """
        self._state_shape_check(state)
        real_dtype = jnp.real(state.qh).dtype
        full_state = states.FullState(
            state=state,
            ph=jnp.zeros_like(state.qh),
            u=jnp.zeros(self.get_grid().real_state_shape, dtype=real_dtype),
            v=jnp.zeros(self.get_grid().real_state_shape, dtype=real_dtype),
            dqhdt=jnp.zeros_like(state.qh),
        )
        return self._invert(full_state).ph

    def get_updates(
        self, state: states.State, forcing_key: jax.Array | None = None
    ) -> states.State:
        """Get tendency updates for time-stepping.
        
        Parameters
        ----------
        state : states.State
            The model state
        """
        full_state = self.get_full_state(state, forcing_key=forcing_key)
        return states.State(
            qh=full_state.dqhdt,
            _q_shape=self.get_grid().real_state_shape[-2:],
        )
    
    def initialise(self, key, n_jets) -> states.State:
        qh = self._pseudo_random(key, n_jets)
        return states.State(qh=qh, _q_shape=(self.ny, self.nx))

    def _pseudo_random(self, key, n_jets):
        # pseudo-random PV in spectral space
        key_r, key_i = jax.random.split(key)
        noise_real = jax.random.normal(key_r, (self.nz, self.nl, self.nk))
        noise_imag = jax.random.normal(key_i, (self.nz, self.nl, self.nk))
        qh = noise_real + 1j * noise_imag
        qh = qh.at[:, :, 0].set(jnp.real(qh[:, :, 0]))
        
        if n_jets is None:
            qh = jnp.expand_dims(self._dealias, 0) * qh
            qh = qh.at[:, 0, 0].set(0.0)
            return qh
        
        # band-limit around kR
        kR = 2 * jnp.pi * n_jets / self.get_grid().Ly
        band_mask = (self.Kmag >= kR / 2) & (self.Kmag <= 2 * kR)

        # masking
        qh = qh * band_mask[None, ...]
        qh = qh * self._dealias[None, ...]
        qh = qh.at[:, 0, 0].set(0.0)
        return qh

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
    @abstractmethod
    def ky(self) -> jax.Array:
        pass

    @property
    @abstractmethod
    def Kmag(self) -> jax.Array:
        pass

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

    def _invert(self, state: states.FullState) -> states.FullState:
        # If kernel configured as single-layer (nz == 1), perform scalar inversion
        if getattr(self, "nz", None) == 1:
            qh = state.qh
            K2 = jnp.array(self.K2, dtype=qh.dtype)
            K2 = jnp.where(K2 == 0, 1.0, K2)
            ph = -qh / K2
            ph = ph.at[..., 0].set(0.0)

            # ensure wavenumber arrays broadcast correctly to spectral shape
            uh = -jnp.expand_dims(1j * self.ky, (0, -1)) * ph
            vh =  jnp.expand_dims(1j * self.kx, (0, 1)) * ph

            return state.update(ph=ph, uh=uh, vh=vh)
        else:
            # Existing two-layer inversion follows
            ph = self._apply_a_ph(state)
            # calculate spectral velocities
            uh = jnp.negative(jnp.expand_dims(1j * self.ky, (0, -1))) * ph
            vh = jnp.expand_dims(1j * self.kx, (0, 1)) * ph
            # Update state values
            return state.update(ph=ph, uh=uh, vh=vh)

    def _do_advection(self, state: states.FullState) -> states.FullState:
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
            jnp.expand_dims(1j * self.kx, (0, 1)) * uqh
            + jnp.expand_dims(1j * self.ky, (0, -1)) * vqh
            + jnp.expand_dims(self._ikQy[: self.nz], 1) * state.ph
        )
        return state.update(dqhdt=dqhdt)

    def _do_friction(self, state: states.FullState) -> states.FullState:
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

    def _do_stochastic_forcing(self, state: states.FullState, forcing_key: jax.Array = None) -> states.FullState:
        """Apply stochastic forcing to the PV tendency.

        Annulus with inner radius kmin and outer radius kmax, with amplitude scaled to forcing_amplitude.
        If no PRNG key is supplied, the forcing is skipped so diagnostics and initialisation
        calls can safely inspect the state without triggering random-number generation.
                
        Parameters
        ----------
        state : states.FullState
            The current full state
        key : jax.Array, optional
            PRNG key for generating random forcing.
            
        Returns
        -------
        states.FullState
            State with forcing added to dqhdt
        """
        if self.forcing_amplitude == 0.0 or forcing_key is None:
            return state

        mask = (self.Kmag >= self.kmin) & (self.Kmag <= self.kmax)
        noise = jax.random.normal(forcing_key, self.get_grid().spectral_state_shape)
        forcing = self.forcing_amplitude * noise * mask / jnp.sqrt(self.dt)
        dqhdt = state.dqhdt + forcing 
        return state.update(dqhdt=dqhdt)
    
    def _do_wind_forcing(self, state: states.FullState):
        """Apply wind forcing to the PV tendency.

        Parameters
        ----------
        state : states.FullState
            The current full state
            
        Returns
        -------
        states.FullState
            State with forcing added to dqhdt
        """
        # Compute wind forcing based on model parameters
        wind_forcing = 0 # ill do this later 
        dqhdt = state.dqhdt + wind_forcing
        return state.update(dqhdt=dqhdt)
    

    @abstractmethod
    def _apply_a_ph(self, state: states.FullState) -> jax.Array:
        pass

    def __repr__(self):
        return Pytree.auto_repr(self)
