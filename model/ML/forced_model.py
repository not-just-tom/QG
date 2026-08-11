import dataclasses
import types
import jax
from model.core import states, steppers
from model.utils import pytree as Pytree

@Pytree.register_pytree_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class ForcedModelState:
    """Wrapped model state for parameterised models.
    """

    model_state: states.State
    param_aux: steppers.PassWeights


def _init_none(init_state, model):
    return None


@Pytree.register_pytree_class_attrs(
    children=["model", "closure", "init_param_aux_func"],
    static_attrs=[],
)
class ForcedModel:
    """Model with defined parameterisation
    """

    def __init__(self, model, closure, init_param_aux_func=None):
        # closure(full_state, param_aux, model) -> full_state, param_aux
        # init_param_aux_func(model_state, model) -> param_aux
        # param_aux (often None) is used to carry parameterization state
        # between time steps, for example: a JAX PRNGKey, if needed
        self.model = model
        if isinstance(closure, types.FunctionType):
            # If this is a plain function, wrap in Partial
            # This ensures it is a pytree
            closure = jax.tree_util.Partial(closure)
        self.closure = closure
        if init_param_aux_func is None:
            init_param_aux_func = _init_none
        if isinstance(init_param_aux_func, types.FunctionType):
            init_param_aux_func = jax.tree_util.Partial(init_param_aux_func)
        self.init_param_aux_func = init_param_aux_func

    def unwrap_state(self, state):
        """Return the inner model state from a wrapped forced state."""
        if isinstance(state, ForcedModelState):
            return state.model_state
        return state

    def wrap_state(self, state, param_aux):
        """Wrap an inner model state back into a forced-state container."""
        return ForcedModelState(
            model_state=state,
            param_aux=steppers.PassWeights(param_aux),
        )

    def apply_stochastic_forcing(self, state, *, forcing_key=None):
        """Apply stochastic forcing while preserving the forced-state wrapper."""
        inner_state = self.unwrap_state(state)
        stochastic_update = self.model.do_stochastic_forcing(
            inner_state,
            forcing_key=forcing_key,
        )
        updated_inner_state = inner_state.update(
            qh=inner_state.qh + stochastic_update.qh
        )
        if isinstance(state, ForcedModelState):
            return ForcedModelState(
                model_state=updated_inner_state,
                param_aux=state.param_aux,
            )
        return updated_inner_state

    def apply_postprocessing(self, state):
        """Apply dealiasing and exact-step filtering while preserving the wrapper."""
        if isinstance(state, ForcedModelState):
            wrapped_state = state
        else:
            wrapped_state = self.wrap_state(state, None)
        postprocessed_state = self.dealias(wrapped_state)
        return self.apply_exact_step_filter(postprocessed_state)

    def get_full_state(self, state):
        """Expand a wrapped partial state into an *unwrapped* full
        state.
        """
        return self.model.get_full_state(state.model_state)

    def get_updates(self, state, *, forcing_key=None):
        """Get updates for time-stepping `state`.
        """
        param_updates, new_param_aux = self.closure(
            state.model_state, state.param_aux.value, self.model,
            forcing_key=forcing_key,
        )
        return ForcedModelState(
            model_state=param_updates,
            param_aux=steppers.PassWeights(new_param_aux),
        )

    def dealias(self, state):
        """Dealias the wrapped model state.
        """
        return ForcedModelState(
            model_state=self.model.dealias(state.model_state),
            param_aux=state.param_aux,
        )

    def apply_exact_step_filter(self, state):
        """Apply wrapped model's exact post-step spectral damping."""
        return ForcedModelState(
            model_state=self.model.apply_exact_step_filter(state.model_state),
            param_aux=state.param_aux,
        )

    def initialise(self, key, *args, **kwargs):
        return self.initialise_param_state(
            self.model.initialise(key=key), *args, **kwargs
        )

    def initialise_param_state(self, state, *args, **kwargs):
        init_param_state = self.init_param_aux_func(state, self.model, *args, **kwargs)
        return ForcedModelState(
            model_state=state,
            param_aux=steppers.PassWeights(init_param_state),
        )
