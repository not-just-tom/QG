import jax.numpy as jnp
import jax
import functools
import equinox as eqx
import importlib
import model.ML.forced_model
import model.core.steppers
importlib.reload(model.core.steppers)
importlib.reload(model.ML.forced_model)
from model.ML.utils.utils import parameterization
from model.ML.forced_model import ForcedModel
from model.core.steppers import SteppedModel, AB3Stepper


def closure_combiner(state, closure_params, static_closure_obj=None):
    """Combine params and static closure, evaluate closure, return dq and params.
    """
    assert static_closure_obj is not None, "static_closure_obj must be provided"
    closure = eqx.combine(closure_params, static_closure_obj)
    q = state.q
    dq_closure = closure(q.astype(jnp.float32))
    return dq_closure.astype(q.dtype), closure_params

def load_forced_model(lr_model, closure, dt):
    '''Load forced model from provided closure'''

    closure_params, closure_static = eqx.partition(closure, eqx.is_array)
    init_param_func = lambda state, model, params: params

    def _param_adapter(state, param_aux, model, *args, **kwargs):
        return closure_combiner(state, param_aux, closure_static)

    closure_func = parameterization(_param_adapter)

    lr_stepper = AB3Stepper(dt=dt)
    forced_model = SteppedModel(
        model=ForcedModel(model=lr_model, closure=closure_func, init_param_aux_func=init_param_func),
        stepper=lr_stepper,
    )
    return forced_model, closure_params, closure_static

def roll_out(init_q, forced_model, nsteps, template_state, closure_params):
    """Roll out `forced_model` for `nsteps` model steps and return
    the sequence of physical-space `q` frames.
    """
    init_qh = jnp.fft.rfftn(init_q, axes=(-2, -1), norm='ortho').astype(template_state.qh.dtype)
    base_state = template_state.update(qh=init_qh)

    # This initializes the stepper state with '_params' (only arrays) in param_aux.
    init_state = forced_model.initialize_stepper_state(
        forced_model.model.initialise_param_state(base_state, closure_params)
    )

    def step(carry, _x):
        next_state = forced_model.step_model(carry)
        return next_state, next_state.state.model_state.q

    _, traj = jax.lax.scan(step, init_state, None, length=nsteps)
    return traj

def compute_traj_errors(target_traj, forced_model, template_state, closure_params):
    rolled_out = roll_out(
        init_q=target_traj[0],
        forced_model=forced_model,
        nsteps=target_traj.shape[0],
        template_state=template_state,
        closure_params=closure_params,
    )
    return rolled_out - target_traj

def make_train_epoch(lr_model, dt, optim):
    """Factory that returns a JIT-compiled `train_epoch` function bound to
    the provided low-resolution physics model `lr_model`, a step `dt` (low_res?), and optimizer.
    """
    # Prepare any template state that is static and can be captured
    template_state = lr_model.initialise(jax.random.PRNGKey(0))

    def _train_epoch(train_trajs, closure, optim_state):
        # Use the low-resolution physics model for training 
        forced_model, closure_params, static_closure_obj = load_forced_model(lr_model, closure, dt)

        def step_fn(carry, batch):
            closure_params, optim_state = carry

            def loss_fn(params, batch):
                err = jax.vmap(
                    functools.partial(compute_traj_errors,
                                      forced_model=forced_model,
                                      template_state=template_state,
                                      closure_params=params)
                )(batch)
                return jnp.mean(err**2)

            loss, grads = eqx.filter_value_and_grad(loss_fn)(closure_params, batch)
            updates, new_optim_state = optim.update(grads, optim_state, closure_params)
            new_closure_params = eqx.apply_updates(closure_params, updates)
            return (new_closure_params, new_optim_state), loss

        (final_closure_params, final_optim_state), losses = jax.lax.scan(
            step_fn, (closure_params, optim_state), train_trajs
        )
        return eqx.combine(final_closure_params, static_closure_obj), final_optim_state, losses

    return eqx.filter_jit(_train_epoch)

def make_test_epoch(lr_model, dt):
    """basically the same minus the optim update. 
    """
    # Prepare any template state that is static and can be captured
    template_state = lr_model.initialise(jax.random.PRNGKey(0))

    def _test_epoch(test_trajs, closure, optim_state):
        # Use the low-resolution physics model for testing 
        forced_model, closure_params, static_closure_obj = load_forced_model(lr_model, closure, dt)

        def step_fn(carry, batch):
            # carry is (closure_params, optim_state) but test epoch does not update
            closure_params, optim_state = carry

            def loss_fn(params, batch):
                err = jax.vmap(
                    functools.partial(compute_traj_errors,
                                      forced_model=forced_model,
                                      template_state=template_state,
                                      closure_params=params)
                )(batch)
                return jnp.mean(err ** 2)

            loss, _ = eqx.filter_value_and_grad(loss_fn)(closure_params, batch)
            # Return unchanged carry and the computed loss
            return (closure_params, optim_state), loss

        (final_closure_params, final_optim_state), losses = jax.lax.scan(
            step_fn, (closure_params, optim_state), test_trajs
        )
        return eqx.combine(final_closure_params, static_closure_obj), final_optim_state, losses

    return eqx.filter_jit(_test_epoch)


