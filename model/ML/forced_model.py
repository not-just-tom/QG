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
    param_aux: steppers.NoStepValue


def _init_none(init_state, model):
    return None


@Pytree.register_pytree_class_attrs(
    children=["model", "param_func", "init_param_aux_func"],
    static_attrs=[],
)
class ForcedModel:
    """Model with defined parameterisation
    """

    def __init__(self, model, param_func, init_param_aux_func=None):
        # param_func(full_state, param_aux, model) -> full_state, param_aux
        # init_param_aux_func(model_state, model) -> param_aux
        # param_aux (often None) is used to carry parameterization state
        # between time steps, for example: a JAX PRNGKey, if needed
        self.model = model
        if isinstance(param_func, types.FunctionType):
            # If this is a plain function, wrap in Partial
            # This ensures it is a pytree
            param_func = jax.tree_util.Partial(param_func)
        self.param_func = param_func
        if init_param_aux_func is None:
            init_param_aux_func = _init_none
        if isinstance(init_param_aux_func, types.FunctionType):
            init_param_aux_func = jax.tree_util.Partial(init_param_aux_func)
        self.init_param_aux_func = init_param_aux_func

    def get_full_state(self, state):
        """Expand a wrapped partial state into an *unwrapped* full
        state.
        """
        return self.model.get_full_state(state.model_state)

    def get_updates(self, state):
        """Get updates for time-stepping `state`.

        `state` is a wrapped, partial :attr:`model` state. This
        function returns updates for time-stepping.

        This function makes use of :attr:`param_func`, applying the
        parameterization to the updates.

        Parameters
        ----------
        state : ParameterizedModelState
            The state which will be time stepped using the computed
            updates.

        Returns
        -------
        ParameterizedModelState
            A new state object where each field corresponds to a
            time-stepping *update* to be applied.

        Note
        ----
        The object returned by this function has the same type of
        `state`, but contains *updates*. This is so the time-stepping
        can be done by mapping over the states and updates as JAX
        pytrees with the same structure.
        """
        param_updates, new_param_aux = self.param_func(
            state.model_state, state.param_aux.value, self.model
        )
        return ForcedModelState(
            model_state=param_updates,
            param_aux=steppers.NoStepValue(new_param_aux),
        )

    def dealias(self, state):
        """Apply fixed filtering to `state`.

        This function should be called once on each new state after
        each time step.

        :class:`~pyqg_jax.steppers.SteppedModel` handles
        this internally.

        This function defers to :attr:`model` for the post-processing.

        Parameters
        ----------
        state : ParameterizedModelState
            The wrapped state to be filtered.

        Returns
        -------
        ParameterizedModelState
            The wrapped filtered state.
        """
        return ForcedModelState(
            model_state=self.model.dealias(state.model_state),
            param_aux=state.param_aux,
        )

    def initialise(self, key, *args, **kwargs):
        """Create a new wrapped initial state with random
        initialization.

        This function defers to :attr:`model` to initialize the inner
        state and makes use of :attr:`init_param_aux_func` to
        initialize the parameterization's auxiliary state.

        Parameters
        ----------
        key : jax.random.key
            The PRNG state used as the random key for initialization.

        *args
            Arbitrary additional arguments for :attr:`init_param_aux_func`

        **kwargs
            Arbitrary additional arguments for :attr:`init_param_aux_func`

        Returns
        -------
        ParameterizedModelState
            The new wrapped state with random initialization.
        """
        return self.initialise_param_state(
            self.model.initialise(key=key), *args, **kwargs
        )

    def initialise_param_state(self, state, *args, **kwargs):
        """Wrap an existing state from :attr:`model` in a
        :class:`ParameterizedModelState`.

        This function takes an existing inner model state and wraps it
        so that it can be used with the parameterized model.

        This function uses of :attr:`init_param_aux_func` to
        initialize the parameterization's auxiliary state.

        Parameters
        ----------
        state
            The inner model state to wrap. The type depends on
            :attr:`model` but is likely to be
            :class:`PseudoSpectralState
            <pyqg_jax.state.PseudoSpectralState>`.

        *args
            Arbitrary additional arguments for :attr:`init_param_aux_func`

        **kwargs
            Arbitrary additional arguments for :attr:`init_param_aux_func`

        Returns
        -------
        ParameterizedModelState
            A wrapped copy of `state`.
        """
        init_param_state = self.init_param_aux_func(state, self.model, *args, **kwargs)
        return ForcedModelState(
            model_state=state,
            param_aux=steppers.NoStepValue(init_param_state),
        )