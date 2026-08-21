import jax.numpy as jnp
import jax
import functools
import equinox as eqx
import importlib
import model.ML.architectures.build_model
import model.ML.utils.dataloading
importlib.reload(model.ML.architectures.build_model)
importlib.reload(model.ML.utils.dataloading)
from model.ML.architectures.build_model import closure_combiner
from model.ML.utils.dataloading import load_forced_model

def rollout_traj_errors(target_traj, forced_model, template_state, closure_params, lr_model, dt, forcing_key=None):
    # nsteps is number of intervals
    nsteps = target_traj.shape[0] - 1

    init_qh = jnp.fft.rfftn(target_traj[0], axes=(-2, -1), norm='ortho').astype(template_state.qh.dtype)
    base_state = template_state.update(qh=init_qh)
    init_state = forced_model.initialise_stepper_state(
        forced_model.model.initialise_param_state(base_state, closure_params),
        forcing_key=forcing_key,
    )

    grid = lr_model.get_grid()
    dx = jnp.asarray(grid.dx, dtype=target_traj.dtype)
    dy = jnp.asarray(grid.dy, dtype=target_traj.dtype)
    dt_arr = jnp.asarray(dt, dtype=target_traj.dtype)

    def step(carry, _x):
        next_state = forced_model.step_model(carry, closure_params)
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

def make_train_epoch(lr_model, dt, optim, loss_fn, cfl_limit=1.0, closure_scale=0.1):
    """Factory that returns a JIT-compiled `train_epoch` function bound to
    the provided low-resolution physics model `lr_model`, a step `dt` (low_res?), and optimizer.
    
    loss_fn should take (errors, lr_model) and return scalar loss.
    """
    # Prepare any template state that is static and can be captured
    template_state = lr_model.initialise(jax.random.PRNGKey(0))

    def _train_epoch(train_trajs, closure, optim_state):
        # Use the low-resolution physics model for training 
        forced_model, closure_params, static_closure_obj, *_ = load_forced_model(
            lr_model,
            closure,
            dt,
            trajs=train_trajs,
            closure_scale=closure_scale,
        )

        def loss_rollout(params, batch):
            """Compute loss and CFL from rollout in one pass.
            Returns mean loss over batch and max CFL.
            """
            forcing_key = jax.random.PRNGKey(0) # fix: check this is ok
            errs, max_cfl, target_qh, pred_qh = jax.vmap(
                functools.partial(
                    rollout_traj_errors,
                    forced_model=forced_model,
                    template_state=template_state,
                    closure_params=params,
                    lr_model=lr_model,
                    dt=dt,
                    forcing_key=forcing_key,
                )
            )(batch)
            # loss_fn now returns per-sample losses (shape: batch_size)
            per_sample_losses = loss_fn(errs, lr_model)
            # Average over batch for gradient computation
            loss = jnp.mean(per_sample_losses)
            return loss, max_cfl

        def step_fn(carry, batch):
            closure_params, optim_state = carry
            
            # Compute loss and CFL in one rollout with gradients
            (loss, max_cfl), grads = eqx.filter_value_and_grad(
                loss_rollout, has_aux=True
            )(closure_params, batch)
            
            discard = jnp.any(max_cfl > cfl_limit)

            def do_update(args):
                params, state, grads, loss, max_cfl = args
                updates, new_optim_state = optim.update(grads, state, params)
                new_closure_params = eqx.apply_updates(params, updates)
                return (new_closure_params, new_optim_state), (loss, jnp.bool_(False), jnp.max(max_cfl))

            def skip_update(args):
                params, state, grads, loss, max_cfl = args
                return (params, state), (jnp.asarray(jnp.nan, dtype=batch.dtype), jnp.bool_(True), jnp.max(max_cfl))

            return jax.lax.cond(discard, skip_update, do_update, (closure_params, optim_state, grads, loss, max_cfl))

        (final_closure_params, final_optim_state), (losses, discard_flags, max_cfls) = jax.lax.scan(
            step_fn, (closure_params, optim_state), train_trajs
        )
        return eqx.combine(final_closure_params, static_closure_obj), final_optim_state, losses, discard_flags, max_cfls

    return eqx.filter_jit(_train_epoch)

def make_test_epoch(lr_model, dt, loss_fn, cfl_limit=1.0, closure_scale=0.1):
    """basically the same minus the optim update. I've moved the zero loss output on the same trajs to here 
    rather than having a split function and dealing with the random_keys matching up.
    """
    # Prepare any template state that is static and can be captured
    template_state = lr_model.initialise(jax.random.PRNGKey(0))

    def _test_epoch(test_trajs, closure, optim_state):
        # Use the low-resolution physics model for testing 
        forced_model, closure_params, static_closure_obj, *_ = load_forced_model(
            lr_model,
            closure,
            dt,
            trajs=test_trajs,
            closure_scale=closure_scale,
        )
        from model.ML.architectures.zero import ZeroModel
        zero_closure_obj = ZeroModel()
        zero_forced_model, zero_closure_params, zero_static_closure_obj, *_ = load_forced_model(
            lr_model,
            zero_closure_obj,
            dt,
            trajs=test_trajs,
            closure_scale=closure_scale,
        )

        def step_fn(carry, batch):
            # carry is (closure_params, optim_state) but test epoch does not update
            closure_params, optim_state = carry

            # Single rollout to get errors and CFL
            forcing_key = jax.random.PRNGKey(0)
            errs, max_cfl, _, _ = jax.vmap(
                functools.partial(
                    rollout_traj_errors,
                    forced_model=forced_model,
                    template_state=template_state,
                    closure_params=closure_params,
                    lr_model=lr_model,
                    dt=dt,
                    forcing_key=forcing_key,
                )
            )(batch)
            
            discard = jnp.any(max_cfl > cfl_limit)
            # loss_fn now returns per-sample losses; take mean for logging
            computed_loss = jax.lax.cond(
                discard,
                lambda _: jnp.asarray(jnp.nan, dtype=batch.dtype),
                lambda _: jnp.mean(loss_fn(errs, lr_model)),
                operand=None,
            )
            # Return unchanged carry and the computed loss
            return (closure_params, optim_state), (computed_loss, discard, jnp.max(max_cfl))

        def step_zero(carry, batch):
            # carry is (closure_params, optim_state) but test epoch does not update
            closure_params, optim_state = carry

            # Single rollout to get errors and CFL
            forcing_key = jax.random.PRNGKey(0)
            errs, max_cfl, _, _ = jax.vmap(
                functools.partial(
                    rollout_traj_errors,
                    forced_model=zero_forced_model,
                    template_state=template_state,
                    closure_params=closure_params,
                    lr_model=lr_model,
                    dt=dt,
                    forcing_key=forcing_key,
                )
            )(batch)
            
            discard = jnp.any(max_cfl > cfl_limit)
            # loss_fn now returns per-sample losses; take mean for logging
            computed_loss = jax.lax.cond(
                discard,
                lambda _: jnp.asarray(jnp.nan, dtype=batch.dtype),
                lambda _: jnp.mean(loss_fn(errs, lr_model)),
                operand=None,
            )
            # Return unchanged carry and the computed loss
            return (closure_params, optim_state), (computed_loss, discard, jnp.max(max_cfl))
        # test trajs with closure
        (final_closure_params, final_optim_state), (losses, discard_flags, max_cfls) = jax.lax.scan(
            step_fn, (closure_params, optim_state), test_trajs
        )

        # test traj with zero closure
        (final_zero_closure_params, final_zero_optim_state), (zero_losses, zero_discard_flags, zero_max_cfls) = jax.lax.scan(
            step_zero, (zero_closure_params, optim_state), test_trajs
        )
        return eqx.combine(final_closure_params, static_closure_obj), final_optim_state, losses, discard_flags, max_cfls, zero_losses

    return eqx.filter_jit(_test_epoch)


def make_validation_epoch(lr_model, dt, init_key, closure_scale=0.1):
    """Factory that returns a validation_epoch function with
    SGS diagnostics and target SGS computation.
    """

    def _validation_epoch(truth_traj, cfg, closure):

        truth_traj = jnp.asarray(truth_traj)
        nsteps_cfg = int(getattr(cfg.plotting, "nsteps", truth_traj.shape[0] - 1))
        n_intervals = min(nsteps_cfg, int(truth_traj.shape[0]) - 1)

        forced_model, closure_params, closure_static, q_mean, q_std, dq_mean, dq_std  = load_forced_model(
            lr_model,
            closure,
            dt,
            trajs=truth_traj,
            closure_scale=closure_scale,
        )

        template_state = lr_model.initialise(init_key)

        init_qh = jnp.fft.rfftn(
            truth_traj[0],
            axes=(-2, -1),
            norm='ortho'
        ).astype(template_state.qh.dtype)

        base_state = template_state.update(qh=init_qh)

        init_stepper_state = forced_model.initialise_stepper_state(
            forced_model.model.initialise_param_state(
                base_state,
                closure_params
            ),
            forcing_key=init_key,
        )

        real_shape = truth_traj.shape[-2:]

        # ML closure rollout
        def _step(carry, _x):
            dq, _ = closure_combiner(
                carry.state.model_state,
                closure_params,
                carry.state.param_aux.value,
                closure_static,
                q_mean=q_mean,
                q_std=q_std,
                dq_mean=dq_mean,
                dq_std=dq_std,
                dt=dt,
                model=lr_model,
            )

            next_state = forced_model.step_model(carry, closure_params)

            dqh_total = (
                next_state.state.model_state.qh
                - carry.state.model_state.qh
            )

            return next_state, (dqh_total, dq)

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

        pred = jax.vmap(
            lambda x: jnp.fft.irfftn(
                x,
                axes=(-2, -1),
                norm='ortho',
                s=real_shape,
            )
        )(qh_traj)

        
        # Build a physics-only model (no closure) for comparison
        from model.ML.architectures.zero import ZeroModel
        zero_closure = ZeroModel()
        zero_model, zero_params, *_ = load_forced_model(
            lr_model,
            zero_closure,
            dt,
            trajs=truth_traj,
            closure_scale=1.0,  # No scaling for zero model
        )
        
        init_zero_state = zero_model.initialise_stepper_state(
            zero_model.model.initialise_param_state(
                base_state,
                zero_params
            ),
            init_key
        )
        
        # Zero model rollout
        def _zero_step(carry, _x):
            next_state = zero_model.step_model(carry, zero_params)
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
        
        zero_qh_traj = jnp.concatenate(
            [
                qh0[None, ...],
                qh0[None, ...] + jnp.cumsum(zero_dqh, axis=0),
            ],
            axis=0,
        )
        
        zero = jax.vmap(
            lambda x: jnp.fft.irfftn(
                x,
                axes=(-2, -1),
                norm='ortho',
                s=real_shape,
            )
        )(zero_qh_traj)  
        
        # Rollout SGS (what was actually applied during deployment)
        sgs_increment = sgs_increment_step

        result = {
            "pred": jax.device_get(pred),

            "zero": jax.device_get(zero),

            "sgs": jax.device_get(sgs_increment),

            "truth": jax.device_get(truth_traj),
        }
        return result

    return _validation_epoch


