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
    static_attrs=["nz", "ny", "nx", "kmin", "kmax", "forcing_type", "forcing_center", "forcing_width", "forcing_seed"],
)
class Kernel(ABC):
    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        nz: int,
        rek: float = 0,
        kmin: float = 3.0,
        kmax: float = 10,
        forcing_type: str = "band",
        forcing_center: float = 6.5,
        forcing_width: float = 2.0,
        forcing_amplitude: float = 0.0,
        forcing_seed: int = 0,
    ):
        # Store small, fundamental properties (others will be computed on demand)
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.rek = rek
        self.kmin = kmin
        self.kmax = kmax
        self.forcing_type = forcing_type
        self.forcing_center = forcing_center
        self.forcing_width = forcing_width
        self.forcing_amplitude = forcing_amplitude
        self.forcing_seed = forcing_seed


    def dealias(self, state: states.State) -> states.State:
        # describe this 
        return state.update(qh=self._dealias*state.qh)

    def apply_exact_step_filter(self, state: states.State) -> states.State:
        """Apply optional exact post-step spectral damping.

        Base kernels default to no-op; concrete models can override.
        """
        return state

    def get_full_state(self, state: states.State, forcing_key: jax.Array = None) -> states.FullState:
        """Compute full state with all tendencies.
        
        Parameters
        ----------
        state : states.State
            The model state
        forcing_key : jax.Array, optional
            PRNG key for stochastic forcing. If None, no forcing applied.
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
        return full_state

    def get_updates(self, state: states.State, forcing_key: jax.Array = None) -> states.State:
        """Get tendency updates for time-stepping.
        
        Parameters
        ----------
        state : states.State
            The model state
        forcing_key : jax.Array, optional
            PRNG key for stochastic forcing. If None, no forcing applied.
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

        # combine all masks once
        qh = qh * jnp.expand_dims(band_mask, 0)
        qh = jnp.expand_dims(self._dealias, 0) * qh
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

    def _do_stochastic_forcing(self, state: states.FullState, key: jax.Array = None) -> states.FullState:
        """Apply stochastic forcing to the PV tendency.
        
        This method adds random forcing in spectral space weighted by the forcing_mask.
        For annulus forcing, the mask is a Gaussian envelope centered at forcing_center.
        For band forcing, it's a binary mask between kmin and kmax.
        
        Parameters
        ----------
        state : states.FullState
            The current full state
        key : jax.Array, optional
            PRNG key for generating random forcing. If None, no forcing is applied.
            
        Returns
        -------
        states.FullState
            State with forcing added to dqhdt
        """
        # Early return if no forcing
        if self.forcing_amplitude == 0:
            return state
        
        # Provide dummy key for JAX tracing if None
        if key is None:
            key = jax.random.PRNGKey(0)
            should_force = False
        else:
            should_force = True
        
        def apply_forcing(state_key_tuple):
            state, key = state_key_tuple
            # Generate complex white noise in spectral space
            key_real, key_imag = jax.random.split(key)
            noise_real = jax.random.normal(key_real, state.dqhdt.shape)
            noise_imag = jax.random.normal(key_imag, state.dqhdt.shape)
            noise = noise_real + 1j * noise_imag
            
            # Make k=0 mode real-valued (required for real-space output)
            noise = noise.at[..., :, 0].set(jnp.real(noise[..., :, 0]))
            
            # Apply forcing mask (broadcasts over layers)
            forcing_mask_broadcasted = jnp.expand_dims(self.forcing_mask, 0)
            forcing = self.forcing_amplitude * forcing_mask_broadcasted * noise
            
            # Add forcing to tendency
            dqhdt_forced = state.dqhdt + forcing
            return state.update(dqhdt=dqhdt_forced)
        
        def no_forcing(state_key_tuple):
            state, _ = state_key_tuple
            return state
        
        return jax.lax.cond(
            should_force,
            apply_forcing,
            no_forcing,
            (state, key),
        )

    @abstractmethod
    def _apply_a_ph(self, state: states.FullState) -> jax.Array:
        pass

    def __repr__(self):
        return Pytree.auto_repr(self)
