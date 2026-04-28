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
import numpy as np


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
    """Memory-efficient rollout that operates in spectral space and returns only 
    the accumulated discrepancy. This avoids storing O(nsteps) large 4D arrays.
    """
    init_qh = jnp.fft.rfftn(init_q, axes=(-2, -1), norm='ortho').astype(template_state.qh.dtype)
    base_state = template_state.update(qh=init_qh)
    init_state = forced_model.initialize_stepper_state(
        forced_model.model.initialise_param_state(base_state, closure_params)
    )

    def step(carry, _x):
        # carry.state.model_state is the current State (spectral)
        # forced_model.step_model performs the actual AB3/RK step
        next_state = forced_model.step_model(carry)
        
        # next_state.state.model_state is the state AFTER the step
        # The tendency (dQ/dt * dt) is effectively the difference in spectral states
        dqh_total = next_state.state.model_state.qh - carry.state.model_state.qh
        
        # return next_state, and the spectral displacement
        return next_state, dqh_total

    # Scan returns the final state and the sequence of spectral displacements
    # Total memory: (nsteps, nz, ny, nx/2+1) complex, which is ~half the size of physical space
    _, traj_dqh = jax.lax.scan(step, init_state, None, length=nsteps)
    return traj_dqh

def compute_traj_errors(target_traj, forced_model, template_state, closure_params):
    # nsteps is number of intervals
    nsteps = target_traj.shape[0] - 1
    
    # traj_dqh: (nsteps, nz, ny, nx/2+1) spectral displacements
    traj_dqh = roll_out(
        init_q=target_traj[0],
        forced_model=forced_model,
        nsteps=nsteps,
        template_state=template_state,
        closure_params=closure_params,
    )
    
    # target_diff_h: true spectral displacement from coarsened high-res data
    target_qh = jax.vmap(lambda x: jnp.fft.rfftn(x, axes=(-2, -1), norm='ortho'))(target_traj)
    target_diff_h = target_qh[1:] - target_qh[:-1]

    # The discrepancy is entirely handled in spectral space to minimize iFFTs and memory
    # Error = True_Delta_Qh - Predicted_Delta_Qh
    residual_qh = target_diff_h - traj_dqh
    
    # Map back to physical space only at the very end for the loss
    # (nsteps, nz, ny, nx)
    residual_q = jax.vmap(lambda x: jnp.fft.irfftn(x, axes=(-2, -1), norm='ortho', s=target_traj.shape[-2:]))(residual_qh)
    
    return residual_q

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

            loss = loss_fn(closure_params, batch)
            # Return unchanged carry and the computed loss
            return (closure_params, optim_state), loss

        (final_closure_params, final_optim_state), losses = jax.lax.scan(
            step_fn, (closure_params, optim_state), test_trajs
        )
        return eqx.combine(final_closure_params, static_closure_obj), final_optim_state, losses

    return eqx.filter_jit(_test_epoch)


def make_validation_epoch(lr_model, dt):
    """Factory that returns a `validation_epoch` function
    """

    def _validation_epoch(traj, cfg, closure):
        """
        """

        # ensure traj is an array and has time dim
        traj = jnp.asarray(traj)
        if traj.ndim == 4:
            # (nt, nz, ny, nx) -> ok
            pass
        elif traj.ndim == 5:
            # batch provided; take first sample for validation convenience
            traj = traj[0]
        else:
            raise ValueError("Validation trajectory must have shape (nt, nz, ny, nx) or (batch, nt, nz, ny, nx)")

        # read parameters from cfg
        nsteps_cfg = int(getattr(cfg.plotting, "nsteps", traj.shape[0] - 1))
        seed = int(getattr(cfg.params, "seed", 0))

        # Prepare forced model and closure params
        forced_model, closure_params, _ = load_forced_model(lr_model, closure, dt)

        # limit nsteps to available trajectory length
        n_intervals = min(nsteps_cfg, int(traj.shape[0]) - 1)

        traj_dqh = roll_out(
            init_q=traj[0],
            forced_model=forced_model,
            nsteps=n_intervals,
            template_state=lr_model.initialise(jax.random.PRNGKey(seed)),
            closure_params=closure_params,
        )

        init_qh = jnp.fft.rfftn(traj[0], axes=(-2, -1), norm='ortho')
        qh_traj = jnp.concatenate([init_qh[None, ...], init_qh[None, ...] + jnp.cumsum(traj_dqh, axis=0)], axis=0)

        real_shape = traj.shape[-2:]
        pred_frames = jax.vmap(lambda x: jnp.fft.irfftn(x, axes=(-2, -1), norm='ortho', s=real_shape))(qh_traj)

        # compute sgs by applying closure to predicted frames (except final if desired)
        @jax.jit
        def _apply_closure(q):
            return closure(q.astype(jnp.float32)).astype(q.dtype)

        # we compute sgs for all pred frames except possibly the final step
        sgs_traj = jax.vmap(_apply_closure)(pred_frames)

        result = {
            "pred_frames": jax.device_get(pred_frames),
            "sgs": jax.device_get(sgs_traj),
            "qh": jax.device_get(qh_traj),
        }

        return result

    return _validation_epoch


