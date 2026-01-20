import functools
import model.core.states as states

def q_parameterization(param_func):
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