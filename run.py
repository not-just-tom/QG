import warnings
warnings.filterwarnings(
    "ignore",
    message=r"Explicitly requested dtype float64 requested in astype is not available.*",
    category=UserWarning,
    module=r"jax\\._src\\.numpy\\.array_methods",
)
import importlib 
import os
from omegaconf import OmegaConf

# Base repo paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CONFIG_DEFAULT_PATH = os.path.join(BASE_DIR, "config", "default.yaml")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, 'saved_closures')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Hydra will supply `cfg` into main(); set no global config here.
OUTDIR_OVERRIDE = None
import model.core.grid
import model.core.states
import model.core.kernel
import model.core.model
import model.core.steppers
import model.ML.generate_data
import model.ML.utils.coarsen
import model.ML.utils.loss
import model.ML.architectures.build_model
import model.ML.utils.dataloading
import model.ML.train
import model.utils.plotting
import model.utils.diagnostics
import model.ML.utils.utils
import model.ML.train
import model.ML.forced_model
importlib.reload(model.ML.utils.utils)
importlib.reload(model.ML.train)
importlib.reload(model.ML.forced_model)
importlib.reload(model.core.grid)
importlib.reload(model.core.states)
importlib.reload(model.core.kernel)
importlib.reload(model.core.model)
importlib.reload(model.core.steppers)
importlib.reload(model.ML.generate_data)
importlib.reload(model.ML.utils.coarsen)
importlib.reload(model.ML.utils.loss)
importlib.reload(model.ML.architectures.build_model)
importlib.reload(model.ML.utils.dataloading)
importlib.reload(model.ML.train)
importlib.reload(model.utils.diagnostics)
importlib.reload(model.utils.plotting)
from model.ML.train import make_train_epoch, make_test_epoch, make_validation_epoch, zero_validation, compute_zero_epoch_loss
from model.ML.architectures.build_model import build_closure
from model.ML.utils.coarsen import coarsen
from model.ML.utils.loss import build_loss
from model.ML.generate_data import generate_train_data
from model.ML.utils.dataloading import find_existing_closure, find_existing_data, ZarrDataLoader, checkpointer, prefetch_generator
from model.utils.logging import configure_logging
from model.utils.plotting import find_output_dir, gif_that, Plotter
from model.core.steppers import SteppedModel, AB3Stepper, CNABStepper
from model.core.model import QGM
import logging
import functools
import jax
import jax.numpy as jnp
import os
import json
import time
import numpy as np
import equinox as eqx
import optax
import gc


# =========================================
# Main loop to run from Command Line 
# =========================================
def run(cfg):
    # load values
    dt = cfg.plotting.dt
    njets= cfg.plotting.njets
    nsteps = int(getattr(cfg.plotting, 'nsteps', 0))
    cadence = int(getattr(cfg.plotting, 'cadence', 100))
    batch_size = cfg.ml.batch_size
    n_train = cfg.ml.n_train
    n_test = cfg.ml.n_test
    n_epochs = n_train + n_test
    params = dict(OmegaConf.to_container(cfg.params, resolve=True))
    seed = params.get("seed", 42)
    key = jax.random.PRNGKey(seed)
    ratio = params["hr_nx"]/params["nx"]
    cfl_limit = float(getattr(cfg.plotting, 'cfl', 1.0))

    use_float64 = cfg.ml.use_float64
    prefetch = cfg.ml.prefetch
    model_type = cfg.ml.model_type
    learning_rate = learning_rate = cfg['architectures'][model_type].get('learning_rate', 0)
    loss_fn = build_loss(cfg['architectures'][model_type].get('loss'))
    closure_scale = cfg['architectures'][model_type].get('closure_scale', 0.1)
    gc_every_batches = int(getattr(cfg.ml, 'gc_every_batches', 10))

    # curriculum stuff
    start_days        = cfg.ml.start_days
    end_days          = cfg.ml.end_days
    window_days       = list(range(start_days, end_days + 1))
    total_curriculum_epochs = len(window_days) * n_epochs

    logger = configure_logging(level=cfg.filepaths.log_level, out_file="../logs/run.log")
    logger = logging.getLogger(__name__)
    
    # GPU or CPU setup 
    device_type = (cfg.ml.device).lower()
    devices = jax.devices()
    gpu_devices = [d for d in devices if d.platform == "gpu"]
    if gpu_devices:
        jax.config.update("jax_platforms", "gpu")
        chosen = "gpu"
    else:
        jax.config.update("jax_platforms", "cpu")
        chosen = "cpu"

    logger.info(f"Requested device: {device_type}, using device: {chosen.upper()}")
    
    # === dataloading === #
    old_dt = dt
    if cfg.plotting.auto_dt == True:
        old_dt = dt
        logger.info("Auto-setting initial dt using CFL condition on a sample initial state.")
        raw_model = QGM({**params, "nx": params['hr_nx']})
        init_state = raw_model.initialise(key, tune=True, n_jets=njets, verbose=True)
        dt = float(raw_model.estimate_cfl_dt(init_state))

    # instantiate the model
    hr_model = SteppedModel(
        model=QGM({**params, "nx": params['hr_nx']}),
        stepper=AB3Stepper(dt=dt),
    )
    # build low-resolution physics model (coarsened from high-res physics)
    lr_model = coarsen(hr_model.model, params['nx'])
    low_res_dt = dt * ratio
    steps_per_day = int(24 * 3600 // low_res_dt)
    model_nsteps = int(end_days * 24 * 3600 // low_res_dt)
    requested_rollout_days = (nsteps * low_res_dt) / (24.0 * 3600.0)

    try:
        lr_init_state = lr_model.initialise(key, tune=True, n_jets=njets, verbose=False)
        lr_rhines_length, lr_u_rms = lr_model.rhines_length(lr_init_state)
        tau_eddy = lr_rhines_length / (lr_u_rms + 1e-12)
        logger.info(
            'Fine timestep is %.2gs and coarsened timestep is %.2gs. '
            'Training horizon is %.2f days, and validation rollout is %.2f days. '
            'Estimated eddy turnover time is %.2g days.',
            dt,
            low_res_dt,
            model_nsteps * low_res_dt / (24.0 * 3600.0),
            requested_rollout_days,
            float(tau_eddy) / (24.0 * 3600.0),
        )
    except Exception:
        logger.info(
            'Fine timestep is %.2gs and coarsened timestep is %.2gs. '
            'Training horizon is %d low-res steps (~%.2f days). '
            'Requested validation rollout is %d steps (~%.2f days).',
            dt,
            low_res_dt,
            model_nsteps,
            model_nsteps * low_res_dt / (24.0 * 3600.0),
            nsteps,
            requested_rollout_days,
        )

    timing_metadata = {
        'nsteps': int(model_nsteps),
        "dt (original)": float(old_dt),
        'auto_dt': bool(cfg.plotting.auto_dt),
        'final dt': float(dt),
    }

    training_metadata = {
        "training": {
            "optimiser": cfg.ml.optimiser,
            "prefetch": int(prefetch),
            "n_train": int(n_train),
            "n_test": int(n_test),
            "model_arch": OmegaConf.to_container(cfg['architectures'][model_type], resolve=True),
        }
    }

    # output dir 
    outbase = os.path.join(cfg.filepaths.out_dir)
    out_dir, found = find_output_dir(outbase, params, timing_metadata, model_type, training_metadata)
    if found:
        logger.info(f"Found existing output directory with matching parameters")
    else:
        os.makedirs(out_dir, exist_ok=True)

    run_dir, found = find_existing_data(DATA_DIR, params, timing_metadata)
    if found: 
        logger.info(f"Found existing data with matching parameters at {run_dir}, loading trajectories from there.")
        data_loader = ZarrDataLoader(run_dir)
    else:
        if cfg.ml.enabled == True:
            logger.info(f"No existing data found, generating new dataset at {run_dir}")
            os.makedirs(run_dir, exist_ok=False)
            generate_train_data(cfg, params, timing_metadata, hr_model, lr_model, run_dir)
            data_loader = ZarrDataLoader(run_dir)

    if os.environ.get('GENERATE_ONLY') == '1':
        logger.info("Generate-only flag set; exiting now.")
        return
    
    if cfg.ml.enabled == False:
        # just run the model and plot
        init_state = hr_model.initialise(key, tune=True, n_jets=njets, verbose=True)
        
        # Calculate eddy turnover time for high-resolution model
        hr_rhines_length, hr_u_rms = hr_model.model.rhines_length(init_state.state)
        tau_eddy = hr_rhines_length / (hr_u_rms + 1e-12)
        logger.info(
            'High-resolution model: Rhines length = %.2g km, U_rms = %.2g m/s, '
            'Estimated eddy turnover time = %.2f days',
            float(hr_rhines_length) / 1000.0,
            float(hr_u_rms),
            float(tau_eddy) / (24.0 * 3600.0),
        )
        
        @functools.partial(jax.jit, static_argnames=["nsteps", "cadence"])
        def rollout(state, nsteps, cadence):
            def loop_fn(carry, step):
                next_state = hr_model.step_model(carry)
                # record spectral qh every cadence steps 
                q_snapshot = jax.lax.cond(
                    step % cadence == 0,
                    lambda s: s.state.qh,
                    lambda s: jnp.zeros_like(s.state.qh),
                    next_state,
                )
                return next_state, q_snapshot

            steps = jnp.arange(nsteps)
            _final_carry, traj_steps = jax.lax.scan(loop_fn, state, steps)
            return _final_carry, traj_steps
        
        #spinup 
        spinup_days = cfg.plotting.spinup
        spinup = int((spinup_days * 24* 3600)// dt)
        rollout_days = nsteps * dt / (24 * 3600)
        logger.info(f"Spinup duration: {spinup_days:.1f} days ({spinup} steps)")
        logger.info(f"Rollout duration: {rollout_days:.1f} days ({nsteps} steps)")
        init_state, _ = rollout(init_state, spinup, cadence)
        
        _, q_traj_spectral = rollout(init_state, nsteps, cadence)
        q_traj_spectral = jax.device_get(q_traj_spectral)  # shape (nsteps, nz, ny, nx//2+1)

        # select only the frames recorded at cadence
        indices = np.arange(0, nsteps, cadence)
        q_traj_spectral = q_traj_spectral[indices]

        # Convert from spectral to physical space
        # q_traj_spectral shape: (n_frames, nz, ny, nx//2+1)
        from model.core.states import _generic_irfftn
        nt, nz, ny, nx_spectral = q_traj_spectral.shape
        nx = 2 * (nx_spectral - 1)
        physical_shape = (ny, nx)
        
        q_traj_list = []
        for t in range(nt):
            q_physical = _generic_irfftn(q_traj_spectral[t], shape=physical_shape)
            q_traj_list.append(np.asarray(q_physical))
        q_traj = np.stack(q_traj_list, axis=0)  # shape (n_frames, nz, ny, nx)

        outbase = os.path.join(cfg.filepaths.out_dir)
        out_dir, found = find_output_dir(outbase, params, timing_metadata, model_type, training_metadata)
        if found:
            logger.info(f"Found existing output directory with matching parameters, replacing the original.")
        else:
            # Ensure the output directory exists when creating a new run directory
            os.makedirs(out_dir, exist_ok=True)
        
        # Set up trajectories dict with truth data for PV diagnostic
        trajectories = {
            'truth': q_traj,
            'grid': hr_model.model.get_grid()
        }
        
        # Override config to plot only PV
        cfg_plot = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg_plot.plotting.plot = ['PV']
        
        Plotter(cfg_plot, trajectories=trajectories, out_dir=out_dir, cadence=cadence).plot()
        return


    # === closure building === 
    # Build training/sweep metadata to avoid accidentally reusing closures from different sweeps
    model_dir, found = find_existing_closure(MODEL_DIR, params, timing_metadata, model_type, training_metadata)
    start_epoch = 0
    if found:
        logger.info(f"Found existing {model_type} closure with matching parameters at {model_dir}, attempting to load checkpoint.")
        try:
            _, loaded_optim, ckpt_meta, loaded_loss_history = checkpointer(None, None, model_dir, save=False)
        except Exception:
            logger.exception("Failed to load checkpoint; will build a new closure")
            _, loaded_optim, ckpt_meta, loaded_loss_history = None, None, None, None

        saved_epoch = int(ckpt_meta.get('epoch', 0))
        saved_n_epochs = int(ckpt_meta.get('n_epochs', n_epochs))
        if saved_epoch >= saved_n_epochs:
            logger.info(f"Model at {model_dir} already trained for {saved_n_epochs} epochs; skipping training loop.")
        else:
            pass
        start_epoch = saved_epoch

    closure = build_closure(cfg)

    # Set up optimiser - might be needed to make more complex if we want to do things like learning rate scheduling
    if cfg.ml.optimiser=='Adam':
        optim = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))
    elif cfg.ml.optimiser=='AdamW':
        optim = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate))
    else:
        raise ValueError(f"Unsupported optimiser: {cfg.ml.optimiser}. Supported options are 'Adam' and 'AdamW'.")
    
    # Initialize optimiser state from template and, if available, map saved optimiser leaves into it
    template_optim_state = optim.init(eqx.filter(closure, eqx.is_array))
    if 'loaded_optim' in locals() and loaded_optim is not None:
        try:
            tpl_leaves, tpl_treedef = jax.tree_util.tree_flatten(template_optim_state)
            # loaded_optim is already a flat list of numpy arrays
            saved_leaves = loaded_optim
            if len(tpl_leaves) != len(saved_leaves):
                raise ValueError("Saved optimiser state does not match template structure")
            # cast and place saved leaves into template treedef
            new_leaves = []
            for tpl, sv in zip(tpl_leaves, saved_leaves):
                arr = np.asarray(sv)
                try:
                    arr = arr.astype(np.asarray(tpl).dtype)
                except Exception:
                    pass
                new_leaves.append(arr)
            optim_state = jax.tree_util.tree_unflatten(tpl_treedef, new_leaves)
            logger.info("Reconstructed optimiser state from checkpoint")
        except Exception:
            logger.exception("Failed to reconstruct optimiser state; using freshly initialised state")
            optim_state = template_optim_state
    else:
        optim_state = template_optim_state

    # Build training and test functions (JIT retraces automatically when batch_steps changes shape)
    train_epoch = make_train_epoch(lr_model, low_res_dt, optim, loss_fn, cfl_limit=cfl_limit, closure_scale=closure_scale)
    test_epoch = make_test_epoch(lr_model, low_res_dt, loss_fn, cfl_limit=cfl_limit, closure_scale=closure_scale)

    # Prepare trajectory indices
    all_traj_indices = list(range(len(data_loader)))
    if len(all_traj_indices) < n_epochs:
        raise ValueError(f"Not enough trajectories in dataset for requested train/test split.")

    # initialise loss history; if we loaded a saved history, continue it
    train_mean_losses = []
    test_mean_losses = []
    zero_mean_losses = []
    try:
        if 'loaded_loss_history' in locals() and loaded_loss_history is not None and ckpt_meta is not None:
            saved_epoch = int(ckpt_meta.get('epoch', 0))
            loaded_train = list(loaded_loss_history.get('train', []))
            loaded_test = list(loaded_loss_history.get('test', []))
            loaded_zero = list(loaded_loss_history.get('zero', []))
            if len(loaded_train) == saved_epoch:
                train_mean_losses = loaded_train
                test_mean_losses = loaded_test
                zero_mean_losses = loaded_zero
                logger.info(f"Loaded existing loss history: {len(train_mean_losses)} train entries, {len(test_mean_losses)} test entries, {len(zero_mean_losses)} zero entries")
            else:
                logger.warning(
                    "Ignoring loaded loss history: found %d train entries but checkpoint epoch=%d."
                    " This can happen after interrupted runs. Starting fresh.",
                    len(loaded_train), saved_epoch,
                )
    except Exception:
        logger.exception("Failed to restore loss history; starting fresh")

    # === Curriculum learning =============================================
    epoch_counter = int(start_epoch)
    if len(train_mean_losses) < epoch_counter:
        train_mean_losses.extend([float('nan')] * (epoch_counter - len(train_mean_losses)))
    if len(test_mean_losses) < epoch_counter:
        test_mean_losses.extend([float('nan')] * (epoch_counter - len(test_mean_losses)))
    if len(zero_mean_losses) < epoch_counter:
        zero_mean_losses.extend([float('nan')] * (epoch_counter - len(zero_mean_losses)))

    rng = jax.random.PRNGKey(seed + 1)
    start_time = time.time()

    for day_idx, current_days in enumerate(window_days):
        current_batch_steps = current_days * steps_per_day
        current_n_samples   = model_nsteps // current_batch_steps
        stage_batch_size = batch_size
        if current_days >= 14:
            stage_batch_size = max(1, batch_size // 4)
        elif current_days >= 7:
            stage_batch_size = max(1, batch_size // 2)
        if current_n_samples < 1:
            logger.warning(
                "Window %d days (%d steps) >= nsteps=%d; stopping curriculum early.",
                current_days, current_batch_steps, model_nsteps,
            )
            break

        stage_start_epoch = day_idx * n_epochs
        stage_resume_epoch = min(max(0, epoch_counter - stage_start_epoch), n_epochs)

        # Split once per stage for deterministic per-stage shuffle.
        window_rng, rng = jax.random.split(rng)
        shuffled = np.asarray(jax.random.permutation(window_rng, len(all_traj_indices))).tolist()

        # Fast-forward stage RNG/shuffle for already-completed sub-epochs.
        for _ in range(stage_resume_epoch):
            _, _, rng = jax.random.split(rng, 3)
            shuffled = shuffled[1:] + shuffled[:1]

        if stage_resume_epoch >= n_epochs:
            logger.info(
                "Skipping curriculum stage %d/%d : already completed.",
                day_idx + 1, len(window_days),
            )
            continue

        # Reinitialise optimiser only at fresh stage start.
        # For mid-stage resume, keep restored optimiser state from checkpoint.
        if stage_resume_epoch == 0:
            optim_state = optim.init(eqx.filter(closure, eqx.is_array))

        logger.info(
            "Curriculum stage %d/%d | window = %d days (%d steps, %d samples/traj, batch_size=%d) | sub-epoch %d/%d",
            day_idx + 1, len(window_days), current_days, current_batch_steps, current_n_samples, stage_batch_size,
            stage_resume_epoch + 1, n_epochs,
        )

        window_train_epoch_means = []
        window_test_epoch_means = []

        for stage_epoch in range(stage_resume_epoch, n_epochs):
            train_losses_accum = []
            test_losses_accum = []
            batch_counter = 0
            train_rng, test_rng, rng = jax.random.split(rng, 3)
            shuffled = shuffled[1:] + shuffled[:1] # move all indice in shuffled forward one
            
            train_idx = [all_traj_indices[i] for i in shuffled[:n_train]]
            test_idx = [all_traj_indices[i] for i in shuffled[n_train:n_epochs]]

            train_gen = data_loader.iterate_batches(
                traj_indices=train_idx,
                n_samples=current_n_samples,
                batch_steps=current_batch_steps,
                key=train_rng,
                batch_size=stage_batch_size,
            )
            for windows in prefetch_generator(train_gen, size=prefetch):
                batch_counter += 1
                windows = windows.astype(np.float32)
                chunk = windows.reshape((1, windows.shape[0], current_batch_steps) + windows.shape[2:])
                chunk = jax.device_put(chunk)
                closure, optim_state, losses, discard_flags, max_cfls = train_epoch(chunk, closure, optim_state)
                discard_flags = np.asarray(discard_flags).reshape(-1)
                losses = np.asarray(losses).reshape(-1)
                max_cfls = np.asarray(max_cfls).reshape(-1)
                if np.any(discard_flags):
                    n_discard = int(np.sum(discard_flags))
                    logger.warning(
                        "Discarded training batch: %d/%d samples; max rollout CFL=%.4f > limit %.4f",
                        n_discard, discard_flags.size, float(np.max(max_cfls)), cfl_limit,
                    )
                train_losses_accum.extend([float(loss) for loss, discarded in zip(losses, discard_flags) if not discarded])
                if batch_counter % gc_every_batches == 0:
                    gc.collect()


            test_gen = data_loader.iterate_batches(
                traj_indices=test_idx,
                n_samples=current_n_samples,
                batch_steps=current_batch_steps,
                key=test_rng,
                batch_size=stage_batch_size,
            )
            for windows in prefetch_generator(test_gen, size=prefetch):
                batch_counter += 1
                windows = windows.astype(np.float32)
                chunk = windows.reshape((1, windows.shape[0], current_batch_steps) + windows.shape[2:])
                chunk = jax.device_put(chunk)
                closure, optim_state, losses, discard_flags, max_cfls = test_epoch(chunk, closure, optim_state)
                discard_flags = np.asarray(discard_flags).reshape(-1)
                losses = np.asarray(losses).reshape(-1)
                max_cfls = np.asarray(max_cfls).reshape(-1)
                if np.any(discard_flags):
                    n_discard = int(np.sum(discard_flags))
                    logger.warning(
                        "Discarded test batch: %d/%d samples; max rollout CFL=%.4f > limit %.4f",
                        n_discard, discard_flags.size, float(np.max(max_cfls)), cfl_limit,
                    )
                test_losses_accum.extend([float(loss) for loss, discarded in zip(losses, discard_flags) if not discarded])
                if batch_counter % gc_every_batches == 0:
                    gc.collect()

            # Compute zero model baseline for current window size
            # Use one batch from test set for efficiency
            try:
                test_gen_zero = data_loader.iterate_batches(
                    traj_indices=test_idx[:min(4, len(test_idx))],  # Use first few test trajectories
                    n_samples=current_n_samples,
                    batch_steps=current_batch_steps+1,
                    key=test_rng,
                    batch_size=min(4, stage_batch_size),
                )
                zero_batch = next(test_gen_zero).astype(np.float32)
                zero_loss = compute_zero_epoch_loss(
                    lr_model, 
                    low_res_dt, 
                    zero_batch, 
                    loss_fn, 
                    current_batch_steps,
                    closure_scale=closure_scale
                )
            except Exception as e:
                logger.warning(f"Failed to compute zero loss for epoch: {e}")
                zero_loss = float('nan')

            train_mean = float(np.mean(train_losses_accum)) if train_losses_accum else float('nan')
            test_mean  = float(np.mean(test_losses_accum))  if test_losses_accum  else float('nan')
            train_mean_losses.append(train_mean)
            test_mean_losses.append(test_mean)
            zero_mean_losses.append(float(zero_loss))
            window_train_epoch_means.append(train_mean)
            window_test_epoch_means.append(test_mean)
            epoch_counter += 1

            logger.info(
                "Stage %d/%d | sub-epoch %d/%d | global epoch %d/%d | "
                "mean_train=%.4E | mean_test=%.4E | mean_zero=%.4E",
                day_idx + 1, len(window_days),
                stage_epoch + 1, n_epochs,
                epoch_counter, total_curriculum_epochs,
                train_mean, test_mean, float(zero_loss),
            )

            try:
                checkpointer(
                    closure, optim_state, model_dir, save=True,
                    epoch=epoch_counter,
                    n_epochs=total_curriculum_epochs,
                    losses={"train": train_mean_losses, "test": test_mean_losses, "zero": zero_mean_losses},
                )
                meta = {
                    "parameters": params,
                    "timing": timing_metadata,
                    "model_type": model_type,
                    "training": training_metadata.get("training", {}),
                    "curriculum": {
                        "steps_per_day": steps_per_day,
                        "start_days": start_days,
                        "end_days": end_days,
                        "n_epochs": n_epochs,
                        "current_day": current_days,
                        "stage_epoch": stage_epoch + 1,
                    },
                }
                with open(os.path.join(model_dir, "metadata.json"), "w") as f:
                    json.dump(meta, f, indent=4)
                logger.info(
                    "Saved curriculum checkpoint: stage %d/%d | sub-epoch %d/%d",
                    day_idx + 1, len(window_days), stage_epoch + 1, n_epochs,
                )
            except Exception:
                logger.exception(
                    "Failed to save checkpoint at stage %d sub-epoch %d", day_idx + 1, stage_epoch + 1
                )

        # Summarise results once the full window has completed.
        window_train_mean = float(np.mean(window_train_epoch_means)) if window_train_epoch_means else float('nan')
        window_test_mean = float(np.mean(window_test_epoch_means)) if window_test_epoch_means else float('nan')
        logger.info(
            "Completed window %d/%d: mean train=%.4E | mean test=%.4E over %d epochs",
            day_idx + 1,
            len(window_days),
            window_train_mean,
            window_test_mean,
            len(window_train_epoch_means),
        )
        gc.collect()
        if hasattr(jax, "clear_caches"):
            jax.clear_caches()

    #log time taken to train in metadata
    end_time = time.time()
    elapsed = end_time - start_time
    metadata_path = os.path.join(model_dir, "metadata.json")
    with open(metadata_path, "r") as f:
        meta = json.load(f)
    meta["time_to_train_seconds"] = elapsed
    with open(metadata_path, "w") as f:
        json.dump(meta, f, indent=4)


    # === validation & diagnostics ===
    truth_traj = data_loader.get_trajectory(n_epochs)  # shape (time, layers, ny, nx)
    available_rollout_steps = int(truth_traj.shape[0]) - 1
    available_rollout_days = available_rollout_steps * low_res_dt / (24.0 * 3600.0)
    effective_rollout_steps = min(nsteps, available_rollout_steps)
    effective_rollout_days = effective_rollout_steps * low_res_dt / (24.0 * 3600.0)
    if nsteps > available_rollout_steps:
        logger.warning(
            "Requested validation rollout is %d steps (~%.2f days), but data only provides %d steps (~%.2f days). "
            "Validation will use the shorter available rollout.",
            nsteps,
            requested_rollout_days,
            available_rollout_steps,
            available_rollout_days,
        )
    else:
        logger.info(
            "Validation rollout will run %d steps (~%.2f days).",
            effective_rollout_steps,
            effective_rollout_days,
        )
    trajectories = {}
    try:
        loaded_leaves, loaded_optim, ckpt_meta, loaded_loss_history = checkpointer(None, None, model_dir, save=False)
        closure = build_closure(cfg, loaded_leaves)
    except Exception:
        logger.exception("Failed to load trained model for testing.")

    # Compute zero model baseline for diagnostics (for zero_frames visualization only)
    try:
        zero_results = zero_validation(lr_model, low_res_dt, truth_traj, cfg, loss_fn)
        trajectories['zero_frames'] = zero_results['zero_frames']
        # Note: zero loss from per-epoch computation will be used for loss_history
    except Exception:
        raise RuntimeError("Failed to compute zero model baseline")

    # Build validation function and run it on a held-out trajectory
    validation_epoch = make_validation_epoch(lr_model, low_res_dt, loss_fn, closure_scale=closure_scale)
    validation_results = validation_epoch(truth_traj, cfg, closure, trajectories['zero_frames'])
    trajectories.update(validation_results)
    if 'val_loss' in validation_results:
        if 'loss_history' not in trajectories:
            trajectories['loss_history'] = {}
        trajectories['loss_history']['val'] = validation_results['val_loss']

    for k,v in trajectories.items():
        if isinstance(v, np.ndarray) and v.ndim > 1 and v.shape[0] > effective_rollout_steps:
            trajectories[k] = v[:effective_rollout_steps]

    # Use per-epoch zero losses from training loop
    trajectories["loss_history"] = {
        "train": train_mean_losses, 
        "test": test_mean_losses,
        "zero": zero_mean_losses,  # Per-epoch zero losses with curriculum progression
        'n_epochs': n_epochs,
    }
    trajectories["grid"] = lr_model.get_grid()
    
    if os.environ.get('HPC_RUN', '0') == '1':
        return print("hello you have skipped the plotting")

    Plotter(cfg, trajectories=trajectories, out_dir=out_dir, cadence=cadence).plot()


# ========================================================
# ========================================================

def main():
    import argparse
    from itertools import product
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=CONFIG_DEFAULT_PATH, help='Path to YAML config file')
    p.add_argument('--outdir', default=None, help='Optional output directory override')
    p.add_argument('--dry-run', action='store_true', help='Print sweep jobs but do not execute')
    p.add_argument('--generate-only', action='store_true', help='Only generate dataset then exit')
    p.add_argument('--hpc-run', action='store_true', help='HPC mode: no plotting; collect final validation MSE for sweep ranking')
    args = p.parse_args()

    # If requested, set an env var so `run()` can detect generate-only behavior.
    if args.generate_only:
        os.environ['GENERATE_ONLY'] = '1'

    # If requested, set an env var so `run()` can detect HPC mode and skip plotting.
    if getattr(args, 'hpc_run', False):
        os.environ['HPC_RUN'] = '1'

    base_cfg = OmegaConf.load(args.config)
    # apply optional outdir override
    if args.outdir is not None:
        if 'filepaths' not in base_cfg:
            base_cfg.filepaths = {}
        base_cfg.filepaths.out_dir = args.outdir

    # Convert to plain python containers for sweep detection (OmegaConf keeps ListConfig/DictConfig types)
    cfg_plain = OmegaConf.to_container(base_cfg, resolve=True)
    model_type = cfg_plain.get('ml', {}).get('model_type')
    archs = cfg_plain.get('architectures', {})
    arch_cfg = archs.get(model_type, {}) if archs else {}

    # Find sweeped keys whose value is a list with length > 1
    sweep_keys = [k for k, v in arch_cfg.items() if isinstance(v, list) and len(v) > 1]
    if not sweep_keys:
        # No sweep: run normally
        if args.dry_run:
            print('No sweep axes found; would run single job with provided config.')
            return
        return run(base_cfg)

    # Build lists for sweep axes
    lists = [arch_cfg[k] for k in sweep_keys]
    total = 1
    for l in lists:
        total *= max(1, len(l))
    print(f"Found sweep axes for {model_type}: {sweep_keys} -> {total} combinations")

    idx = 0
    results = []
    for combo in product(*lists):
        # deep-copy base config to plain dict then modify
        cfg_copy = OmegaConf.to_container(base_cfg, resolve=True)
        if 'architectures' not in cfg_copy:
            cfg_copy['architectures'] = {}
        if model_type not in cfg_copy['architectures']:
            cfg_copy['architectures'][model_type] = {}
        for k, val in zip(sweep_keys, combo):
            cfg_copy['architectures'][model_type][k] = val

        # set per-run outdir
        out_base = cfg_copy.get('filepaths', {}).get('out_dir', 'outputs')
        run_out = os.path.join(out_base, f'sweep_{model_type}', f'run_{idx+1}')
        if 'filepaths' not in cfg_copy:
            cfg_copy['filepaths'] = {}
        cfg_copy['filepaths']['out_dir'] = run_out

        cfg_run = OmegaConf.create(cfg_copy)

        print(f"Running sweep {idx+1}/{total}: {dict(zip(sweep_keys, combo))} -> {run_out}")
        if args.dry_run:
            idx += 1
            continue

        # run() will return a scalar metric when in HPC mode; otherwise None
        try:
            metric = run(cfg_run)
        except Exception:
            # If a single sweep job fails, log and continue to next combo
            logging.getLogger(__name__).exception("Sweep run failed for combo %s", dict(zip(sweep_keys, combo)))
            metric = None

        if os.environ.get('HPC_RUN', '0') == '1':
            results.append({
                'idx': idx + 1,
                'combo': dict(zip(sweep_keys, combo)),
                'out_dir': run_out,
                'mse': metric,
            })
        idx += 1

    # If HPC mode was requested, print top-5 runs by lowest validation MSE
    if os.environ.get('HPC_RUN', '0') == '1':
        # filter out failed runs (None mse)
        filtered = [r for r in results if r['mse'] is not None]
        filtered.sort(key=lambda r: r['mse'])
        print('\nTop 5 runs by validation MSE:')
        for i, r in enumerate(filtered[:5], 1):
            print(f"{i}. mse={r['mse']:.6E} -> {r['combo']} -> outdir={r['out_dir']}")



if __name__ == "__main__":
    main()
     
