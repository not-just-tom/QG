import functools
import model.core.states as states
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

def parameterization(param_func):
    """Decorator implementing parameterizations in terms of potential
    vorticity.

    The target function should take as its first three arguments
    :pycode:`(state, param_aux, model)` as with any other
    parameterization function. Additional arguments will be passed
    through unmodified.

    This function should then return two values: :pycode:`dq,
    param_aux`. These values will then be added to the model's
    original update value to form the parameterized update.

    The wrapped function is suitable for use with
    :class:`ParameterizedModel`.

    See also: :class:`pyqg.parameterizations.QParameterization`
    """

    @functools.wraps(param_func)
    def wrapped_q_param(state, param_aux, model, *args, **kwargs):
        dq, param_aux = param_func(state, param_aux, model, *args, **kwargs)
        dqh = states._generic_rfftn(dq)
        updates = model.get_updates(state)
        dqhdt = updates.qh + dqh
        return updates.update(qh=dqhdt), param_aux

    return wrapped_q_param

def param_to_single(param):
    try:
        dtype = getattr(param, "dtype", None)
        if dtype is None:
            return param
        # Handle numpy and jax dtypes robustly
        if np.issubdtype(dtype, np.floating) and dtype == np.dtype(np.float64):
            return param.astype(np.float32)
        if np.issubdtype(dtype, np.complexfloating) and dtype == np.dtype(np.complex128):
            return param.astype(np.complex64)
    except Exception:
        pass
    return param


def module_to_single(module):
    return jax.tree_util.tree_map(param_to_single, module)