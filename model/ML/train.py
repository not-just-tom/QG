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
from model.core.steppers import SteppedModel, AB3Stepper, CNABStepper
import numpy as np


def _build_closure_scalers(trajs, closure_scale=1e-2, eps=1e-6):
    """Build simple per-layer scalers for closure input/output.

    The state scaler is used to normalize q before it is passed to the
    closure network. A separate, smaller output scaler is used for closure
    increments to reduce the risk of destabilizing large updates.
    """
    trajs = jnp.asarray(trajs)
    layer_axis = trajs.ndim - 3
    reduce_axes = tuple(i for i in range(trajs.ndim) if i != layer_axis)
    q_std = jnp.std(trajs, axis=reduce_axes)
    q_std = jnp.maximum(q_std, eps).reshape((-1, 1, 1))
    q_mean = jnp.zeros_like(q_std)

    dq_std = jnp.maximum(q_std * closure_scale, eps)
    dq_mean = jnp.zeros_like(dq_std)
    return q_mean, q_std, dq_mean, dq_std


def _build_closure_highpass_filter(lr_model, power=2.0):
    """Construct a smooth high-pass mask in spectral space.

    This suppresses low-wavenumber closure forcing and concentrates closure
    effects toward smaller scales, analogous to unresolved-scale gating.
    """
    kmag = jnp.asarray(lr_model.Kmag)
    kref = jnp.maximum(jnp.max(kmag), 1e-12)
    filt = (kmag / kref) ** power
    filt = jnp.where(kmag == 0, 0.0, filt)
    return filt


def closure_combiner(
    state,
    closure_params,
    static_closure_obj=None,
    q_mean=None,
    q_std=None,
    dq_mean=None,
    dq_std=None,
    closure_filter=None,
):
    """Combine params and static closure, evaluate closure, return dq and params.
    """
    assert static_closure_obj is not None, "static_closure_obj must be provided"
    closure = eqx.combine(closure_params, static_closure_obj)
    q = state.q
    if q_mean is None or q_std is None:
        q_in = q
    else:
        q_in = (q - q_mean) / (q_std + 1e-6)

    dq_closure = closure(q_in.astype(jnp.float32)).astype(q.dtype)

    if dq_mean is not None and dq_std is not None:
        dq_closure = (dq_closure * dq_std) + dq_mean

    if closure_filter is not None:
        dqh = jnp.fft.rfftn(dq_closure, axes=(-2, -1), norm='ortho')
        dqh = dqh * jnp.expand_dims(closure_filter, 0)
        dq_closure = jnp.fft.irfftn(
            dqh,
            axes=(-2, -1),
            norm='ortho',
            s=q.shape[-2:],
        ).astype(q.dtype)

    return dq_closure.astype(q.dtype), closure_params

def load_forced_model(
    lr_model,
    closure,
    dt,
    q_mean=None,
    q_std=None,
    dq_mean=None,
    dq_std=None,
    closure_filter=None,
):
    '''Load forced model from provided closure'''

    closure_params, closure_static = eqx.partition(closure, eqx.is_array)
    init_param_func = lambda state, model, params: params

    def _param_adapter(state, param_aux, model, *args, **kwargs):
        return closure_combiner(
            state,
            param_aux,
            closure_static,
            q_mean=q_mean,
            q_std=q_std,
            dq_mean=dq_mean,
            dq_std=dq_std,
            closure_filter=closure_filter,
        )

    closure_func = parameterization(_param_adapter)

    lr_stepper = CNABStepper(dt=dt)
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

def compute_traj_errors_and_cfl(target_traj, forced_model, template_state, closure_params, lr_model, dt):
    # nsteps is number of intervals
    nsteps = target_traj.shape[0] - 1

    init_qh = jnp.fft.rfftn(target_traj[0], axes=(-2, -1), norm='ortho').astype(template_state.qh.dtype)
    base_state = template_state.update(qh=init_qh)
    init_state = forced_model.initialize_stepper_state(
        forced_model.model.initialise_param_state(base_state, closure_params)
    )

    grid = lr_model.get_grid()
    dx = jnp.asarray(grid.dx, dtype=target_traj.dtype)
    dy = jnp.asarray(grid.dy, dtype=target_traj.dtype)
    dt_arr = jnp.asarray(dt, dtype=target_traj.dtype)

    def step(carry, _x):
        next_state = forced_model.step_model(carry)
        dqh_total = next_state.state.model_state.qh - carry.state.model_state.qh
        full = lr_model.get_full_state(next_state.state.model_state)
        cfl_val = (jnp.max(jnp.abs(full.u)) * dt_arr) / dx + (jnp.max(jnp.abs(full.v)) * dt_arr) / dy
        return next_state, (dqh_total, cfl_val)

    _, (traj_dqh, cfl_vals) = jax.lax.scan(step, init_state, None, length=nsteps)
    max_cfl = jnp.max(cfl_vals)

    target_qh = jax.vmap(lambda x: jnp.fft.rfftn(x, axes=(-2, -1), norm='ortho'))(target_traj)

    # pred_qh[t] = q(t0) + sum_{s=0}^{t} dqh[s]  (shape: nsteps, nz, ny, nx//2+1)
    pred_qh = init_qh[None] + jnp.cumsum(traj_dqh, axis=0)

    # State-level residual: truth trajectory vs predicted trajectory
    residual_qh = target_qh[1:] - pred_qh

    # Map back to physical space only at the very end for the loss
    # (nsteps, nz, ny, nx)
    residual_q = jax.vmap(lambda x: jnp.fft.irfftn(x, axes=(-2, -1), norm='ortho', s=target_traj.shape[-2:]))(residual_qh)

    return residual_q, max_cfl

def maddison_loss(residual_q, lr_model, beta=10.0, scale_factor=1e4):
    """Compute loss per Maddison (2026) eqn 7: spatial interior weighting, no boundary.
    
    residual_q: (nsteps, nz, ny, nx) error in physical space
    lr_model: QGM instance (provides grid spacing)
    beta: Rossby parameter
    scale_factor: front multiplier (10^4 in paper, can tune)
    """
    # With periodic (CIRCULAR) boundary conditions there is no true boundary,
    # so weight all grid points equally.
    ny, nx = residual_q.shape[-2:]
    interior_mask = jnp.ones((ny, nx))
    
    dx = lr_model.get_grid().dx
    L = float(lr_model.Lx)
    
    # Normalization: 1 / (beta^2 * L^2) * 1/(4*L^2) * scale_factor
    norm_factor = scale_factor / (beta**2 * L**2 * 4.0 * L**2)
    
    # Squared residual weighted by interior mask and grid spacing dx^2
    weighted_residual_sq = (residual_q ** 2) * interior_mask[None, None, :, :] * (dx**2)
    
    loss = norm_factor * jnp.mean(weighted_residual_sq)
    return loss

def make_train_epoch(lr_model, dt, optim, cfl_limit=1.0):
    """Factory that returns a JIT-compiled `train_epoch` function bound to
    the provided low-resolution physics model `lr_model`, a step `dt` (low_res?), and optimizer.
    """
    # Prepare any template state that is static and can be captured
    template_state = lr_model.initialise(jax.random.PRNGKey(0))

    def _train_epoch(train_trajs, closure, optim_state):
        q_mean, q_std, dq_mean, dq_std = _build_closure_scalers(train_trajs)
        closure_filter = _build_closure_highpass_filter(lr_model)
        # Use the low-resolution physics model for training 
        forced_model, closure_params, static_closure_obj = load_forced_model(
            lr_model,
            closure,
            dt,
            q_mean=q_mean,
            q_std=q_std,
            dq_mean=dq_mean,
            dq_std=dq_std,
            closure_filter=closure_filter,
        )

        def step_fn(carry, batch):
            closure_params, optim_state = carry

            def metrics_fn(params, batch):
                errs, max_cfl = jax.vmap(
                    functools.partial(
                        compute_traj_errors_and_cfl,
                        forced_model=forced_model,
                        template_state=template_state,
                        closure_params=params,
                        lr_model=lr_model,
                        dt=dt,
                    )
                )(batch)
                return errs, max_cfl

            def loss_fn(params, batch):
                err, _ = metrics_fn(params, batch)
                try:
                    return maddison_loss(err, lr_model, beta=float(lr_model.beta))
                except Exception:
                    # Fallback to simple MSE if Maddison loss fails
                    return jnp.mean(err**2)

            _, max_cfl = metrics_fn(closure_params, batch)
            discard = jnp.any(max_cfl > cfl_limit)

            def do_update(args):
                params, state = args
                loss, grads = eqx.filter_value_and_grad(loss_fn)(params, batch)
                updates, new_optim_state = optim.update(grads, state, params)
                new_closure_params = eqx.apply_updates(params, updates)
                return (new_closure_params, new_optim_state), (loss, jnp.bool_(False), jnp.max(max_cfl))

            def skip_update(args):
                params, state = args
                return (params, state), (jnp.asarray(jnp.nan, dtype=batch.dtype), jnp.bool_(True), jnp.max(max_cfl))

            return jax.lax.cond(discard, skip_update, do_update, (closure_params, optim_state))

        (final_closure_params, final_optim_state), (losses, discard_flags, max_cfls) = jax.lax.scan(
            step_fn, (closure_params, optim_state), train_trajs
        )
        return eqx.combine(final_closure_params, static_closure_obj), final_optim_state, losses, discard_flags, max_cfls

    return eqx.filter_jit(_train_epoch)

def make_test_epoch(lr_model, dt, cfl_limit=1.0):
    """basically the same minus the optim update. 
    """
    # Prepare any template state that is static and can be captured
    template_state = lr_model.initialise(jax.random.PRNGKey(0))

    def _test_epoch(test_trajs, closure, optim_state):
        q_mean, q_std, dq_mean, dq_std = _build_closure_scalers(test_trajs)
        closure_filter = _build_closure_highpass_filter(lr_model)
        # Use the low-resolution physics model for testing 
        forced_model, closure_params, static_closure_obj = load_forced_model(
            lr_model,
            closure,
            dt,
            q_mean=q_mean,
            q_std=q_std,
            dq_mean=dq_mean,
            dq_std=dq_std,
            closure_filter=closure_filter,
        )

        def step_fn(carry, batch):
            # carry is (closure_params, optim_state) but test epoch does not update
            closure_params, optim_state = carry

            def metrics_fn(params, batch):
                errs, max_cfl = jax.vmap(
                    functools.partial(
                        compute_traj_errors_and_cfl,
                        forced_model=forced_model,
                        template_state=template_state,
                        closure_params=params,
                        lr_model=lr_model,
                        dt=dt,
                    )
                )(batch)
                return errs, max_cfl

            def loss_fn(params, batch):
                err, _ = metrics_fn(params, batch)
                try:
                    return maddison_loss(err, lr_model, beta=float(lr_model.beta))
                except Exception:
                    # Fallback to simple MSE if Maddison loss fails
                    return jnp.mean(err**2)

            _, max_cfl = metrics_fn(closure_params, batch)
            discard = jnp.any(max_cfl > cfl_limit)
            loss = jax.lax.cond(
                discard,
                lambda _: jnp.asarray(jnp.nan, dtype=batch.dtype),
                lambda _: loss_fn(closure_params, batch),
                operand=None,
            )
            # Return unchanged carry and the computed loss
            return (closure_params, optim_state), (loss, discard, jnp.max(max_cfl))

        (final_closure_params, final_optim_state), (losses, discard_flags, max_cfls) = jax.lax.scan(
            step_fn, (closure_params, optim_state), test_trajs
        )
        return eqx.combine(final_closure_params, static_closure_obj), final_optim_state, losses, discard_flags, max_cfls

    return eqx.filter_jit(_test_epoch)


def make_validation_epoch(lr_model, dt):
    """Factory that returns a validation_epoch function with
    SGS diagnostics and target SGS computation.
    """

    def _validation_epoch(truth_traj, cfg, closure):

        truth_traj = jnp.asarray(truth_traj)

        if truth_traj.ndim == 4:
            pass
        elif truth_traj.ndim == 5:
            truth_traj = truth_traj[0]
        else:
            raise ValueError(
                "Validation trajectory must have shape "
                "(nt, nz, ny, nx) or (batch, nt, nz, ny, nx)"
            )

        nsteps_cfg = int(getattr(cfg.plotting, "nsteps", truth_traj.shape[0] - 1))
        seed = int(getattr(cfg.params, "seed", 0))

        n_intervals = min(nsteps_cfg, int(truth_traj.shape[0]) - 1)
        q_mean, q_std, dq_mean, dq_std = _build_closure_scalers(truth_traj)
        closure_filter = _build_closure_highpass_filter(lr_model)
        forced_model, closure_params, closure_static = load_forced_model(
            lr_model,
            closure,
            dt,
            q_mean=q_mean,
            q_std=q_std,
            dq_mean=dq_mean,
            dq_std=dq_std,
            closure_filter=closure_filter,
        )

        # Zero model baseline
        from model.ML.architectures.zero import ZeroModel
        zero_closure = ZeroModel()
        zero_model, zero_params, _ = load_forced_model(
            lr_model,
            zero_closure,
            dt,
            q_mean=q_mean,
            q_std=q_std,
            dq_mean=dq_mean,
            dq_std=dq_std,
            closure_filter=closure_filter,
        )

        template_state = lr_model.initialise(
            jax.random.PRNGKey(seed)
        )

        init_qh = jnp.fft.rfftn(
            truth_traj[0],
            axes=(-2, -1),
            norm='ortho'
        ).astype(template_state.qh.dtype)

        base_state = template_state.update(qh=init_qh)

        init_stepper_state = forced_model.initialize_stepper_state(
            forced_model.model.initialise_param_state(
                base_state,
                closure_params
            )
        )

        init_zero_state = zero_model.initialize_stepper_state(
            zero_model.model.initialise_param_state(
                base_state,
                zero_params
            )
        )

        real_shape = truth_traj.shape[-2:]

        # ML closure rollout
        def _step(carry, _x):

            dq_closure, _ = closure_combiner(
                carry.state.model_state,
                carry.state.param_aux.value,
                closure_static,
                q_mean=q_mean,
                q_std=q_std,
                dq_mean=dq_mean,
                dq_std=dq_std,
                closure_filter=closure_filter,
            )

            next_state = forced_model.step_model(carry)

            dqh_total = (
                next_state.state.model_state.qh
                - carry.state.model_state.qh
            )

            return next_state, (dqh_total, dq_closure)

        _, (traj_dqh, sgs_dq) = jax.lax.scan(
            _step,
            init_stepper_state,
            None,
            length=n_intervals,
        )

        def _zero_step(carry, _x):

            next_state = zero_model.step_model(carry)

            dqh_total = (
                next_state.state.model_state.qh
                - carry.state.model_state.qh
            )

            return next_state, dqh_total

        _, zero_dqh = jax.lax.scan(
            _zero_step,
            init_zero_state,
            None,
            length=n_intervals,
        )

        # Reconstruct trajectories in physical space
        qh0 = jnp.fft.rfftn(
            truth_traj[0],
            axes=(-2, -1),
            norm='ortho'
        )

        qh_traj = jnp.concatenate(
            [
                qh0[None, ...],
                qh0[None, ...] + jnp.cumsum(traj_dqh, axis=0),
            ],
            axis=0,
        )

        zero_qh_traj = jnp.concatenate(
            [
                qh0[None, ...],
                qh0[None, ...] + jnp.cumsum(zero_dqh, axis=0),
            ],
            axis=0,
        )

        pred_frames = jax.vmap(
            lambda x: jnp.fft.irfftn(
                x,
                axes=(-2, -1),
                norm='ortho',
                s=real_shape,
            )
        )(qh_traj)

        zero_frames = jax.vmap(
            lambda x: jnp.fft.irfftn(
                x,
                axes=(-2, -1),
                norm='ortho',
                s=real_shape,
            )
        )(zero_qh_traj)

        # SGS diagnostics
        sgs_increment = sgs_dq
        sgs_forcing = sgs_increment / dt # timestep-independent forcing form
        target_sgs = truth_traj[1:n_intervals + 1] - zero_frames[1:]
        target_sgs_forcing = target_sgs / dt

        result = {
            "pred_frames": jax.device_get(pred_frames),

            "zero_frames": jax.device_get(zero_frames),

            "sgs": jax.device_get(sgs_increment),

            "sgs_forcing": jax.device_get(sgs_forcing),

            "target_sgs": jax.device_get(target_sgs),

            "target_sgs_forcing": jax.device_get(target_sgs_forcing),

            "truth": jax.device_get(truth_traj),
        }
        return result

    return _validation_epoch


