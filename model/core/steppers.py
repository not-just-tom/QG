import typing
import dataclasses
import abc
import jax
import jax.numpy as jnp
import model.utils.pytree as Pytree


P = typing.TypeVar("P")

@Pytree.register_pytree_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class StepperState(typing.Generic[P]):
    """Model state wrapped for time-stepping

    Attributes
    ----------

    state : PseudoSpectralState or ParameterizedModelState
        The inner state from the model being stepped forward. The
        actual type of `state` depends on the model being stepped.

    t : jax.numpy.float32
        The current model time

    tc : jax.numpy.uint32
        The current model timestep

    forcing_key : jax.Array
        Dedicated PRNG key used to generate stochastic forcing.
    """

    state: P
    t: jax.Array
    tc: jax.Array
    forcing_key: jax.Array

    def update(self, **kwargs):
        """Replace values stored in this state.

        This function produces a *new* state object, containing the
        replacement values.

        The keyword arguments may be any of `state`, `t`, `tc`, or `forcing_key`.

        The object this method is called on is not modified.

        Parameters
        ----------
        state : PseudoSpectralState or ParameterizedModelState, optional
            Replacement value for :attr:`state`.

        t : jax.numpy.float32, optional
            Replacement value for :attr:`t`.
            The current model time

        tc : jax.numpy.uint32, optional
            Replacement value for :attr:`tc`.

        Returns
        -------
        StepperState
            A copy of this object with the specified values replaced.
        """
        # Check that only valid updates are applied
        if extra_attrs := (kwargs.keys() - {"state", "t", "tc", "forcing_key"}):
            extra_attr_str = ", ".join(extra_attrs)
            raise ValueError(
            "invalid state updates, can only update state, t, tc, and forcing_key "
                f"(not {extra_attr_str})"
            )
        # Perform the update
        return dataclasses.replace(self, **kwargs)


@dataclasses.dataclass
class Stepper(abc.ABC):
    dt: float

    def initialise_stepper_state(self, state, forcing_key=None):
        """Wrap an existing `state` from a model in a
        :class:`StepperState` to prepare it for time stepping.

        This initialises a new :class:`StepperState` from a time of
        :pycode:`0`.

        Parameters
        ----------
        state
            The model state to wrap.

        Returns
        -------
        StepperState
            The wrapped state. Note this will be a subclass of
            :class:`StepperState` appropriate for this time stepper.
        """
        return StepperState(
            state=state,
            t=jnp.float32(0),
            tc=jnp.uint32(0),
            forcing_key=forcing_key,
        )

    @abc.abstractmethod
    def apply_updates(self, stepper_state, updates):
        pass


@Pytree.register_pytree_class_attrs(
    children=["model", "stepper"],
    static_attrs=[],
)
class SteppedModel:
    """Combine an inner model with a time stepper.

    This class simplifies the process of stepping a base model through
    time by handling the interactions between the model and time
    stepper.

    """

    def __init__(self, model, stepper):
        self.model = model
        self.stepper = stepper

    def initialise(self, key, *args, **kwargs):
        model_state = self.model.initialise(key, *args, **kwargs)
        return self.initialise_stepper_state(
            model_state, forcing_key=key
        )

    def initialise_stepper_state(self, state, forcing_key=None):
        if forcing_key==None:
            raise ValueError("Forcing_key needs to be prodived to initialise_stepper_state")
        return self.stepper.initialise_stepper_state(state, forcing_key=forcing_key)

    def _unwrap_state(self, state):
        unwrap_state = getattr(self.model, "unwrap_state", None)
        if unwrap_state is not None:
            return unwrap_state(state)
        return state

    def _apply_stochastic_forcing(self, state, *, forcing_key):
        apply_stochastic_forcing = getattr(self.model, "apply_stochastic_forcing", None)
        if apply_stochastic_forcing is not None:
            return apply_stochastic_forcing(state, forcing_key=forcing_key)

        inner_state = self._unwrap_state(state)
        stochastic_update = self.model.do_stochastic_forcing(
            inner_state,
            forcing_key=forcing_key,
        )
        return inner_state.update(qh=inner_state.qh + stochastic_update.qh)

    def _apply_postprocessing(self, state):
        apply_postprocessing = getattr(self.model, "apply_postprocessing", None)
        if apply_postprocessing is not None:
            return apply_postprocessing(state)

        postprocessed_state = self.model.dealias(state)
        return self.model.apply_exact_step_filter(postprocessed_state)

    def step_model(self, stepper_state, /, closure_params=None):
        # Apply model step
        forcing_key, next_forcing_key = jax.random.split(stepper_state.forcing_key)

        # all deterministic steps added and updates state
        if closure_params is None:
            updates = self.model.get_updates(stepper_state.state)
        else:
            updates = self.model.get_updates(
                stepper_state.state,
                closure_params=closure_params,
            )
        new_stepper_state = self.stepper.apply_updates(
            stepper_state,
            updates,
        )

        updated_state = self._apply_stochastic_forcing(
            new_stepper_state.state,
            forcing_key=forcing_key,
        )
        postprocessed_state = self._apply_postprocessing(updated_state)
        new_stepper_state = new_stepper_state.update(
            state=postprocessed_state,
            forcing_key=next_forcing_key,
        )

        return new_stepper_state

    def get_full_state(self, stepper_state):
        return self.model.get_full_state(
            stepper_state.state
        )


def _nostep_tree_map(func, tree, *rest):
    def wrap_nostep_update(leaf, update, *args, **kwargs):
        if isinstance(update, PassWeights):
            return update
        return func(leaf, update, *args, **kwargs)

    return jax.tree_util.tree_map(
        wrap_nostep_update,
        tree,
        *rest,
        is_leaf=(lambda l: isinstance(l, PassWeights)),
    )


def _dummy_step_init(state):
    def leaf_map(leaf):
        if isinstance(leaf, PassWeights):
            return PassWeights(None)
        return jnp.zeros_like(leaf)

    return jax.tree_util.tree_map(
        leaf_map, state, is_leaf=(lambda l: isinstance(l, PassWeights))
    )


def _map_state_remove_nostep(state):
    def leaf_map(leaf):
        if isinstance(leaf, PassWeights):
            return PassWeights(None)
        return leaf

    return jax.tree_util.tree_map(
        leaf_map, state, is_leaf=(lambda l: isinstance(l, PassWeights))
    )

@Pytree.register_pytree_dataclass
@dataclasses.dataclass(frozen=True, repr=False, kw_only=True)
class AB3State(StepperState[P]):
    _ablevel: jax.Array
    _updates: tuple[P, P]


@Pytree.register_pytree_dataclass
@dataclasses.dataclass(repr=False)
class AB3Stepper(Stepper):
    """Third-order Adams-Bashforth stepper.

    This is the same time stepping scheme as used in PyQG.

    This time-stepper bootstraps using lower order Adams-Bashforth
    schemes for the first two steps.

    Parameters
    ----------
    dt : float
        Numerical time step

    Attributes
    ----------
    dt : float
        Numerical time step
    """

    def initialise_stepper_state(self, state: P, forcing_key=None) -> AB3State[P]:
        """Wrap an existing `state` from a model in a
        :class:`StepperState` to prepare it for time stepping.

        This initialises a new :class:`StepperState` from a time of
        :pycode:`0`.

        Parameters
        ----------
        state
            The model state to wrap.

        Returns
        -------
        StepperState
            The wrapped state. Note this will be a subclass of
            :class:`StepperState` appropriate for this time stepper.
        """
        base_state = super().initialise_stepper_state(state, forcing_key=forcing_key)
        dummy_update: P = _dummy_step_init(state)
        return AB3State(
            state=base_state.state,
            t=base_state.t,
            tc=base_state.tc,
            forcing_key=base_state.forcing_key,
            _ablevel=jnp.uint8(0),
            _updates=(dummy_update, dummy_update),
        )

    def apply_updates(
        self,
        stepper_state: AB3State[P],
        updates: P,
    ) -> AB3State[P]:
        """Apply `updates` to the existing `stepper_state` producing
        the next step in time.

        `updates` should be provided by the model that produced
        :attr:`StepperState.state`.

        Parameters
        ----------
        stepper_state : StepperState
            The time-stepper wrapped state to be updated.

        updates : PseudoSpectralState or ParameterizedModelState
            The *unwrapped* updates to apply. The actual type of
            `updates` depends on the model being stepped.

        Returns
        -------
        StepperState
            The updated, wrapped state at the next time step.

        Note
        ----
        This method does not apply post-processing to the updated
        state.
        """
        new_ablevel, dt1, dt2, dt3 = jax.lax.switch(
            stepper_state._ablevel,
            [
                lambda: (jnp.uint8(1), 1.0, 0.0, 0.0),
                lambda: (jnp.uint8(2), 1.5, -0.5, 0.0),
                lambda: (jnp.uint8(2), (23 / 12), (-16 / 12), (5 / 12)),
            ],
        )

        def do_update(v, u, u_p, u_pp):
            dt = jnp.astype(self.dt, jax.eval_shape(jnp.real, v).dtype)
            return v + ((dt1 * dt) * u) + ((dt2 * dt) * u_p) + ((dt3 * dt) * u_pp)

        updates_p, updates_pp = stepper_state._updates
        new_state = _nostep_tree_map(
            do_update,
            stepper_state.state,
            updates,
            updates_p,
            updates_pp,
        )
        new_t = stepper_state.t + jnp.float32(self.dt)
        new_tc = stepper_state.tc + 1
        new_updates = (_map_state_remove_nostep(updates), updates_p)
        return AB3State(
            state=new_state,
            t=new_t,
            tc=new_tc,
            forcing_key=stepper_state.forcing_key,
            _ablevel=new_ablevel,
            _updates=new_updates,
        )
    


@Pytree.register_pytree_dataclass
@dataclasses.dataclass(repr=False)
class CNABStepper(Stepper):
    """Crank-Nicolson / Adams-Bashforth (CNAB) stepper.

    This implements the explicit AB2 treatment of the nonlinear term and
    provides a place to insert an implicit Crank-Nicolson solve for a
    linear operator in future (e.g. viscous Laplacian handled implicitly
    in spectral space). At present, when no implicit linear solve is
    available, it falls back to an explicit AB2 update for stability and
    API compatibility.

    Notes
    -----
    - The stepper stores the two most recent updates (u^n and u^{n-1}).
    - If you want full CNAB implicit solves, provide a helper that can be
      called from `apply_updates` to compute (I - dt/2 L)^{-1} acting on
      the RHS. That helper is not assumed here to keep the stepper pure
      and JAX-friendly without extra model hooks.
    """

    def initialise_stepper_state(self, state: P, forcing_key=None) -> AB3State[P]:
        base_state = super().initialise_stepper_state(state, forcing_key=forcing_key)
        dummy_update: P = _dummy_step_init(state)
        return AB3State(
            state=base_state.state,
            t=base_state.t,
            tc=base_state.tc,
            forcing_key=base_state.forcing_key,
            _ablevel=jnp.uint8(0),
            _updates=(dummy_update, dummy_update),
        )

    def apply_updates(
        self,
        stepper_state: AB3State[P],
        updates: P,
    ) -> AB3State[P]:
        """Apply CNAB/AB2 style update.

        If an implicit linear solver is available it may be inserted here by
        replacing the final assignment with a solve for (I - dt/2 L) q^{n+1}.
        """
        # AB2-like coefficients for explicit nonlinear term
        new_ablevel, coeff1, coeff2 = jax.lax.switch(
            stepper_state._ablevel,
            [
                lambda: (jnp.uint8(1), 1.0, 0.0),  # first step: forward Euler
                lambda: (jnp.uint8(2), 1.5, -0.5),  # second step: AB2 boot
                lambda: (jnp.uint8(2), 1.5, -0.5),  # steady AB2 thereafter
            ],
        )

        def do_update(v, u, u_p):
            dt = jnp.astype(self.dt, jax.eval_shape(jnp.real, v).dtype)
            # Explicit AB2 update for the non-linear part
            return v + ((coeff1 * dt) * u) + ((coeff2 * dt) * u_p)

        updates_p, updates_pp = stepper_state._updates
        new_state = _nostep_tree_map(
            do_update,
            stepper_state.state,
            updates,
            updates_p,
        )

        # NOTE: Placeholder for implicit CN solve. If the wrapped model exposes
        # a linear operator in spectral space and a solver, it can be applied
        # here to compute (I - dt/2 L)^{-1} acting on new_state.

        new_t = stepper_state.t + jnp.float32(self.dt)
        new_tc = stepper_state.tc + 1
        new_updates = (_map_state_remove_nostep(updates), updates_p)
        return AB3State(
            state=new_state,
            t=new_t,
            tc=new_tc,
            forcing_key=stepper_state.forcing_key,
            _ablevel=new_ablevel,
            _updates=new_updates,
        )
    

@Pytree.register_pytree_dataclass
@dataclasses.dataclass
class NoStepValue(typing.Generic[P]):
    """Shields contents from the provided time-steppers.

    When a time-stepper encounters a value wrapped in this class, it
    will skip its normal stepping computations and directly use the
    value from the updates. This allows a user to manually update an
    auxiliary value outside the normal time-stepping.

    For example, :func:`jax.random.key` values should not be
    time-stepped normally. Wrapping them in this class and manually
    :func:`updating them <jax.random.split>` can accomplish this.

    This class is used as part of :class:`ParameterizedModelState
    <pyqg_jax.parameterisations.ParameterizedModelState>`.

    Parameters
    ----------
    value : object
        The inner value to wrap. This can be an arbitrary JAX PyTree.

    Attributes
    ----------
    value
        The internal, wrapped value
    """

    value: P


@Pytree.register_pytree_dataclass
@dataclasses.dataclass
class PassWeights(typing.Generic[P]):
    """Shields contents from the provided time-steppers.

    When a time-stepper encounters a value wrapped in this class, it
    will skip its normal stepping computations and directly use the
    value from the updates. This allows a user to manually update an
    auxiliary value outside the normal time-stepping.
    """

    value: P
