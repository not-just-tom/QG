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


def _build_closure_scalers(trajs, closure_scale=5e-2, eps=1e-6):
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


def _spectrum_aux_loss(target_qh, pred_qh, lr_model, weight=0.1, power=1.0, eps=1e-12):
    """Compare target and predicted spectra in Fourier space.

    The penalty is shell-aware via a wavenumber weighting, and it operates on the
    predicted vs target trajectory spectra rather than on the physical-space residual.
    """
    target_spec = jnp.mean(jnp.abs(target_qh) ** 2, axis=(0, 1))
    pred_spec = jnp.mean(jnp.abs(pred_qh) ** 2, axis=(0, 1))

    kmag = jnp.asarray(lr_model.Kmag)
    kref = jnp.maximum(jnp.max(kmag), 1e-12)
    shell_weight = 1.0 + (kmag / kref) ** power

    log_spec_err = jnp.log(pred_spec + eps) - jnp.log(target_spec + eps)
    return weight * jnp.mean(shell_weight * (log_spec_err ** 2))


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
    """Evaluate closure and return per-step PV increment dQ plus params.
    """
    assert static_closure_obj is not None, "static_closure_obj must be provided"
    closure = eqx.combine(closure_params, static_closure_obj)
    q = state.q
    if q_mean is None or q_std is None:
        q_in = q
    else:
        q_in = (q - q_mean) / (q_std + 1e-6)

    dq_increment = closure(q_in.astype(jnp.float32)).astype(q.dtype)

    if dq_mean is not None and dq_std is not None:
        dq_increment = (dq_increment * dq_std) + dq_mean

    if closure_filter is not None:
        dqh_increment = jnp.fft.rfftn(dq_increment, axes=(-2, -1), norm='ortho')
        dqh_increment = dqh_increment * jnp.expand_dims(closure_filter, 0)
        dq_increment = jnp.fft.irfftn(
            dqh_increment,
            axes=(-2, -1),
            norm='ortho',
            s=q.shape[-2:],
        ).astype(q.dtype)

    return dq_increment.astype(q.dtype), closure_params

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
    dt_arr = jnp.asarray(dt)

    def _param_adapter(state, param_aux, model, *args, **kwargs):
        dq_increment, new_params = closure_combiner(
            state,
            param_aux,
            closure_static,
            q_mean=q_mean,
            q_std=q_std,
            dq_mean=dq_mean,
            dq_std=dq_std,
            closure_filter=closure_filter,
        )
        # Closure network predicts a per-step increment dQ; the parameterization
        # wrapper expects a tendency dQ/dt to add to model.get_updates(...).
        dq_forcing = dq_increment / dt_arr
        return dq_forcing, new_params

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

    return residual_q, max_cfl, target_qh[1:], pred_qh

def maddison_loss(residual_q, lr_model, beta=10.0, scale_factor=1e4):
    """Compute loss per Maddison (2026) eqn 7: spatial interior weighting, no boundary.
    scale_factor: front multiplier (10^4 in paper, very odd)
    """
    ny, nx = residual_q.shape[-2:]
    interior_mask = jnp.ones((ny, nx))
    dx = lr_model.get_grid().dx
    L = float(lr_model.Lx)
    
    # Normalization
    norm_factor = scale_factor / (beta**2 * L**2 * 4.0 * L**2)
    
    # Squared residual weighted by interior mask and grid spacing dx^2
    weighted_residual_sq = (residual_q ** 2) * interior_mask[None, None, :, :] * (dx**2)
    
    loss = norm_factor * jnp.mean(weighted_residual_sq)
    return loss

def make_train_epoch(lr_model, dt, optim, loss, cfl_limit=1.0):
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
                errs, max_cfl, target_qh, pred_qh = jax.vmap(
                    functools.partial(
                        compute_traj_errors_and_cfl,
                        forced_model=forced_model,
                        template_state=template_state,
                        closure_params=params,
                        lr_model=lr_model,
                        dt=dt,
                    )
                )(batch)
                return errs, max_cfl, target_qh, pred_qh

            def loss_fn(params, batch):
                err, _, target_qh, pred_qh = metrics_fn(params, batch)
                if loss == "maddison":
                    return maddison_loss(err, lr_model, beta=float(lr_model.beta))
                elif loss in {"maddison_spectral", "maddison+spectral"}:
                    return (
                        maddison_loss(err, lr_model, beta=float(lr_model.beta))
                        + _spectrum_aux_loss(target_qh, pred_qh, lr_model)
                    )
                elif loss == "mse":
                    return jnp.mean(err**2)
                else: raise ValueError(f"Unsupported loss type: {loss}")

            _, max_cfl, _, _ = metrics_fn(closure_params, batch)
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

def make_test_epoch(lr_model, dt, loss, cfl_limit=1.0):
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
                errs, max_cfl, target_qh, pred_qh = jax.vmap(
                    functools.partial(
                        compute_traj_errors_and_cfl,
                        forced_model=forced_model,
                        template_state=template_state,
                        closure_params=params,
                        lr_model=lr_model,
                        dt=dt,
                    )
                )(batch)
                return errs, max_cfl, target_qh, pred_qh

            def loss_fn(params, batch):
                err, _, target_qh, pred_qh = metrics_fn(params, batch)
                if loss == "maddison":
                    return maddison_loss(err, lr_model, beta=float(lr_model.beta))
                elif loss in {"maddison_spectral", "maddison+spectral"}:
                    return (
                        maddison_loss(err, lr_model, beta=float(lr_model.beta))
                        + _spectrum_aux_loss(target_qh, pred_qh, lr_model)
                    )
                elif loss == "mse":
                    return jnp.mean(err**2)
                else: 
                    raise ValueError(f"Unsupported loss type: {loss}")

            _, max_cfl, _, _ = metrics_fn(closure_params, batch)
            discard = jnp.any(max_cfl > cfl_limit)
            computed_loss = jax.lax.cond(
                discard,
                lambda _: jnp.asarray(jnp.nan, dtype=batch.dtype),
                lambda _: loss_fn(closure_params, batch),
                operand=None,
            )
            # Return unchanged carry and the computed loss
            return (closure_params, optim_state), (computed_loss, discard, jnp.max(max_cfl))

        (final_closure_params, final_optim_state), (losses, discard_flags, max_cfls) = jax.lax.scan(
            step_fn, (closure_params, optim_state), test_trajs
        )
        return eqx.combine(final_closure_params, static_closure_obj), final_optim_state, losses, discard_flags, max_cfls

    return eqx.filter_jit(_test_epoch)


def make_validation_epoch(lr_model, dt, loss):
    """Factory that returns a validation_epoch function with
    SGS diagnostics and target SGS computation.
    """

    def _validation_epoch(truth_traj, cfg, closure, zero_frames):

        truth_traj = jnp.asarray(truth_traj)
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

        real_shape = truth_traj.shape[-2:]

        # ML closure rollout
        def _step(carry, _x):

            dq_increment, _ = closure_combiner(
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

            return next_state, (dqh_total, dq_increment)

        _, (traj_dqh, sgs_increment_step) = jax.lax.scan(
            _step,
            init_stepper_state,
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

        pred_frames = jax.vmap(
            lambda x: jnp.fft.irfftn(
                x,
                axes=(-2, -1),
                norm='ortho',
                s=real_shape,
            )
        )(qh_traj)

        # Validation loss on rollout trajectory (per timestep)
        val_error = truth_traj[1:n_intervals + 1] - pred_frames[1:]
        if loss == "maddison":
            val_loss = jnp.array([
                maddison_loss(val_error[t:t + 1], lr_model, beta=float(lr_model.beta))
                for t in range(val_error.shape[0])
            ])
        elif loss in {"maddison_spectral", "maddison+spectral"}:
            target_qh = jax.vmap(
                lambda x: jnp.fft.rfftn(x, axes=(-2, -1), norm='ortho')
            )(truth_traj[1:n_intervals + 1])
            pred_qh = qh_traj[1:]
            val_loss = jnp.array([
                maddison_loss(val_error[t:t + 1], lr_model, beta=float(lr_model.beta))
                + _spectrum_aux_loss(target_qh[t:t + 1], pred_qh[t:t + 1], lr_model)
                for t in range(val_error.shape[0])
            ])
        elif loss == "mse":
            val_loss = jnp.mean(val_error ** 2, axis=(1, 2, 3))
        else:
            raise ValueError(f"Unsupported loss type: {loss}")

        # SGS diagnostics - compute ideal closure output by stepping physics from truth states
        
        # Build a physics-only model (no closure) for computing ideal targets
        from model.ML.architectures.zero import ZeroModel
        zero_closure_obj = ZeroModel()
        physics_only_model, physics_params, _ = load_forced_model(
            lr_model, zero_closure_obj, dt,
            q_mean=None, q_std=None, dq_mean=None, dq_std=None, closure_filter=None,
        )
        
        # Step physics forward from each truth state to see where physics alone would take us
        def step_physics_from_truth(q_truth):
            qh = jnp.fft.rfftn(q_truth, axes=(-2, -1), norm='ortho').astype(template_state.qh.dtype)
            state = template_state.update(qh=qh)
            init = physics_only_model.initialize_stepper_state(
                physics_only_model.model.initialise_param_state(state, physics_params)
            )
            next_state = physics_only_model.step_model(init)
            # Return the change in physical space
            q_next = jnp.fft.irfftn(
                next_state.state.model_state.qh,
                axes=(-2, -1), norm='ortho', s=real_shape
            )
            return q_next - q_truth
        
        # Compute where physics would take each truth state
        dq_physics_from_truth = jax.vmap(step_physics_from_truth)(truth_traj[:n_intervals])
        
        # Ideal closure output: what you'd need to add to physics to reach next truth state
        dq_truth_steps = jnp.diff(truth_traj[:n_intervals + 1], axis=0)
        ideal_closure_output = dq_truth_steps - dq_physics_from_truth
        
        # Teacher-forced: what does the model actually predict at truth states?
        def eval_closure_at_truth_state(q_truth):
            qh = jnp.fft.rfftn(q_truth, axes=(-2, -1), norm='ortho').astype(template_state.qh.dtype)
            state = template_state.update(qh=qh)
            dq_pred, _ = closure_combiner(
                state, closure_params, closure_static,
                q_mean=q_mean, q_std=q_std,
                dq_mean=dq_mean, dq_std=dq_std,
                closure_filter=closure_filter,
            )
            return dq_pred
        
        teacher_forced_sgs = jax.vmap(eval_closure_at_truth_state)(truth_traj[:n_intervals])
        
        # Rollout SGS (what was actually applied during deployment)
        sgs_increment = sgs_increment_step

        result = {
            "pred_frames": jax.device_get(pred_frames),

            "zero_frames": jax.device_get(zero_frames),

            "sgs": jax.device_get(sgs_increment),

            "target_sgs": jax.device_get(ideal_closure_output),
            
            "teacher_forced_sgs": jax.device_get(teacher_forced_sgs),

            "val_loss": jax.device_get(val_loss),

            "truth": jax.device_get(truth_traj),
        }
        return result

    return _validation_epoch


def zero_validation(lr_model, dt, truth_traj, cfg, loss):
    """Run zero model baseline and return zero_frames and zero_loss.
    
    Args:
        lr_model: Low-resolution physics model
        dt: Timestep
        truth_traj: Truth trajectory array (time, layers, ny, nx)
        cfg: Configuration object
        loss: Loss type string ('maddison' or 'mse')
    
    Returns:
        dict with 'zero_frames' and 'zero_loss' keys
    """
    truth_traj = jnp.asarray(truth_traj)
    
    nsteps_cfg = int(getattr(cfg.plotting, "nsteps", truth_traj.shape[0] - 1))
    seed = int(getattr(cfg.params, "seed", 0))
    n_intervals = min(nsteps_cfg, int(truth_traj.shape[0]) - 1)
    
    q_mean, q_std, dq_mean, dq_std = _build_closure_scalers(truth_traj)
    closure_filter = _build_closure_highpass_filter(lr_model)
    
    # Build zero model
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
    
    template_state = lr_model.initialise(jax.random.PRNGKey(seed))
    
    init_qh = jnp.fft.rfftn(
        truth_traj[0],
        axes=(-2, -1),
        norm='ortho'
    ).astype(template_state.qh.dtype)
    
    base_state = template_state.update(qh=init_qh)
    
    init_zero_state = zero_model.initialize_stepper_state(
        zero_model.model.initialise_param_state(
            base_state,
            zero_params
        )
    )
    
    real_shape = truth_traj.shape[-2:]
    
    # Zero model rollout
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
    
    # Reconstruct zero trajectory in physical space
    qh0 = jnp.fft.rfftn(
        truth_traj[0],
        axes=(-2, -1),
        norm='ortho'
    )
    
    zero_qh_traj = jnp.concatenate(
        [
            qh0[None, ...],
            qh0[None, ...] + jnp.cumsum(zero_dqh, axis=0),
        ],
        axis=0,
    )
    
    zero_frames = jax.vmap(
        lambda x: jnp.fft.irfftn(
            x,
            axes=(-2, -1),
            norm='ortho',
            s=real_shape,
        )
    )(zero_qh_traj)
    
    # Compute zero loss using the same loss function as training
    zero_error = truth_traj[1:n_intervals + 1] - zero_frames[1:]
    if loss == "maddison":
        zero_loss = jnp.array([
            maddison_loss(zero_error[t:t+1], lr_model, beta=float(lr_model.beta)) 
            for t in range(zero_error.shape[0])
        ])
    elif loss in {"maddison_spectral", "maddison+spectral"}:
        target_qh = jax.vmap(
            lambda x: jnp.fft.rfftn(x, axes=(-2, -1), norm='ortho')
        )(truth_traj[1:n_intervals + 1])
        pred_qh = zero_qh_traj[1:]
        zero_loss = jnp.array([
            maddison_loss(zero_error[t:t + 1], lr_model, beta=float(lr_model.beta))
            + _spectrum_aux_loss(target_qh[t:t + 1], pred_qh[t:t + 1], lr_model)
            for t in range(zero_error.shape[0])
        ])
    elif loss == 'mse':
        zero_loss = jnp.mean(zero_error ** 2, axis=(1, 2, 3))
    else:
        raise ValueError(f"Unsupported loss type: {loss}")
    
    return {
        'zero_frames': jax.device_get(zero_frames),
        'zero_loss': jax.device_get(zero_loss),
    }


