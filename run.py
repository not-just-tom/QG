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
importlib.reload(model.ML.architectures.build_model)
importlib.reload(model.ML.utils.dataloading)
importlib.reload(model.ML.train)
importlib.reload(model.utils.diagnostics)
importlib.reload(model.utils.plotting)
from model.ML.train import make_train_epoch, make_test_epoch, make_validation_epoch, maddison_loss
from model.ML.architectures.build_model import build_closure
from model.ML.utils.coarsen import coarsen
from model.ML.generate_data import generate_train_data
from model.ML.utils.dataloading import find_existing_closure, find_existing_data, ZarrDataLoader, checkpointer, prefetch_generator
from model.utils.logging import configure_logging
from model.utils.plotting import find_output_dir, gif_that, Plotter
from model.core.steppers import SteppedModel, AB3Stepper
from model.core.model import QGM
import logging
import jax
import jax.numpy as jnp
import os
import json
import numpy as np
import equinox as eqx
import matplotlib.pyplot as plt
import optax


# =========================================
# Main loop to run from Command Line 
# =========================================
def run(cfg):
    # load values
    dt = cfg.plotting.dt
    njets= cfg.plotting.njets
    nsteps = cfg.plotting.nsteps
    batch_size = cfg.ml.batch_size
    n_train = cfg.ml.n_train
    n_test = cfg.ml.n_test
    n_epochs = n_train + n_test
    spinup = cfg.plotting.spinup
    params = dict(OmegaConf.to_container(cfg.params, resolve=True))
    seed = params.get("seed", 42)
    key = jax.random.PRNGKey(seed)
    ratio = params["hr_nx"]/params["nx"]
    use_float64 = cfg.ml.use_float64
    prefetch = cfg.ml.prefetch
    model_type = cfg.ml.model_type
    learning_rate = learning_rate = cfg['architectures'][model_type].get('learning_rate')

    # curriculum stuff
    steps_per_day     = cfg.ml.steps_per_day
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
    if cfg.plotting.auto_dt:
        old_dt = dt
        logger.info("Auto-setting initial dt using CFL condition on a sample initial state.")
        raw_model = QGM({**params, "nx": params['hr_nx']})
        init_state = raw_model.initialise(key, tune=True, n_jets=njets, verbose=True)
        dt = raw_model.estimate_cfl_dt(init_state)

    # instantiate the model
    hr_model = SteppedModel(
        model=QGM({**params, "nx": params['hr_nx']}),
        stepper=AB3Stepper(dt=dt),
    )
    # build low-resolution physics model (coarsened from high-res physics)
    lr_model = coarsen(hr_model.model, params['nx'])

    timing_metadata = {
        'spinup': int(spinup),
        'nsteps': int(nsteps),
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
        logger.info(f"No existing data found, generating new dataset at {run_dir}")
        os.makedirs(run_dir, exist_ok=False)
        generate_train_data(cfg, params, timing_metadata, hr_model, lr_model, run_dir)
        data_loader = ZarrDataLoader(run_dir)
    # If caller requested generate-only mode, stop after data creation to avoid
    # concurrently running heavy training workloads from multiple tasks.
    if os.environ.get('GENERATE_ONLY') == '1':
        logger.info("Generate-only flag set; exiting after data generation")
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
            logger.info(f"Resuming training from epoch {saved_epoch} (saved) out of {saved_n_epochs}")
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
    low_res_dt = dt*ratio
    cfl_limit = float(getattr(cfg.plotting, 'cfl', 1.0))
    train_epoch = make_train_epoch(lr_model, low_res_dt, optim, cfl_limit=cfl_limit)
    test_epoch = make_test_epoch(lr_model, low_res_dt, cfl_limit=cfl_limit)

    # Prepare trajectory indices
    all_traj_indices = list(range(len(data_loader)))
    if len(all_traj_indices) < n_epochs:
        raise ValueError(f"Not enough trajectories in dataset for requested train/test split.")

    # initialise loss history; if we loaded a saved history, continue it
    train_mean_losses = []
    test_mean_losses = []
    try:
        if 'loaded_loss_history' in locals() and loaded_loss_history is not None and ckpt_meta is not None:
            saved_epoch = int(ckpt_meta.get('epoch', 0))
            loaded_train = list(loaded_loss_history.get('train', []))
            loaded_test = list(loaded_loss_history.get('test', []))
            # Only accept loaded history if it matches the saved epoch length exactly.
            if len(loaded_train) == saved_epoch:
                train_mean_losses = loaded_train
                test_mean_losses = loaded_test
                logger.info(f"Loaded existing loss history: {len(train_mean_losses)} train entries, {len(test_mean_losses)} test entries")
            else:
                logger.warning(
                    "Ignoring loaded loss history: found %d train entries but checkpoint epoch=%d."
                    " This can happen after interrupted runs. Starting fresh.",
                    len(loaded_train), saved_epoch,
                )
    except Exception:
        logger.exception("Failed to restore loss history; starting fresh")

    # === Curriculum learning =============================================
    logger.info(
        "Runnning window %d→%d days (%d stages × %d epochs/stage = %d total epochs, %d steps/day)",
        start_days, end_days, len(window_days), n_epochs, total_curriculum_epochs, steps_per_day,
    )

    if start_epoch >= total_curriculum_epochs:
        logger.info(
            "Checkpoint already at/after final curriculum epoch (%d/%d); skipping training.",
            start_epoch, total_curriculum_epochs,
        )

    # Resume from checkpoint epoch, not just loss-history length.
    # If history is shorter (e.g. interrupted write), pad with NaNs to keep alignment.
    epoch_counter = int(start_epoch)
    if len(train_mean_losses) < epoch_counter:
        train_mean_losses.extend([float('nan')] * (epoch_counter - len(train_mean_losses)))
    if len(test_mean_losses) < epoch_counter:
        test_mean_losses.extend([float('nan')] * (epoch_counter - len(test_mean_losses)))

    rng = jax.random.PRNGKey(seed + 1)

    for day_idx, current_days in enumerate(window_days):
        current_batch_steps = current_days * steps_per_day
        current_n_samples   = nsteps // current_batch_steps
        if current_n_samples < 1:
            logger.warning(
                "Window %d days (%d steps) >= nsteps=%d; stopping curriculum early.",
                current_days, current_batch_steps, nsteps,
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
                "Skipping curriculum stage %d/%d (day=%d): already completed.",
                day_idx + 1, len(window_days), current_days,
            )
            continue

        # Reinitialise optimiser only at fresh stage start.
        # For mid-stage resume, keep restored optimiser state from checkpoint.
        if stage_resume_epoch == 0:
            optim_state = optim.init(eqx.filter(closure, eqx.is_array))

        logger.info(
            "Curriculum stage %d/%d | window = %d days (%d steps, %d samples/traj) | resuming sub-epoch %d/%d",
            day_idx + 1, len(window_days), current_days, current_batch_steps, current_n_samples,
            stage_resume_epoch + 1, n_epochs,
        )

        window_train_epoch_means = []
        window_test_epoch_means = []

        for stage_epoch in range(stage_resume_epoch, n_epochs):
            train_losses_accum = []
            test_losses_accum = []
            train_rng, test_rng, rng = jax.random.split(rng, 3)
            shuffled = shuffled[1:] + shuffled[:1] # move all indice in shuffled forward one
            
            train_idx = [all_traj_indices[i] for i in shuffled[:n_train]]
            test_idx = [all_traj_indices[i] for i in shuffled[n_train:n_epochs]]

            train_gen = data_loader.iterate_batches(
                traj_indices=train_idx,
                n_samples=current_n_samples,
                batch_steps=current_batch_steps,
                key=train_rng,
                batch_size=batch_size,
            )
            for windows in prefetch_generator(train_gen, size=prefetch):
                windows = windows.astype(np.float32)
                chunk = windows.reshape((1, windows.shape[0], current_batch_steps) + windows.shape[2:])
                chunk = jax.device_put(chunk)
                closure, optim_state, losses, discard_flags, max_cfls = train_epoch(chunk, closure, optim_state)
                discard_flags = np.asarray(discard_flags).reshape(-1)
                losses = np.asarray(losses).reshape(-1)
                max_cfls = np.asarray(max_cfls).reshape(-1)
                if np.any(discard_flags):
                    print(f"Discarded training batch: max rollout CFL={float(np.max(max_cfls)):.4f} > limit {cfl_limit:.4f}")
                train_losses_accum.extend([float(loss) for loss, discarded in zip(losses, discard_flags) if not discarded])


            test_gen = data_loader.iterate_batches(
                traj_indices=test_idx,
                n_samples=current_n_samples,
                batch_steps=current_batch_steps,
                key=test_rng,
                batch_size=batch_size,
            )
            for windows in prefetch_generator(test_gen, size=prefetch):
                windows = windows.astype(np.float32)
                chunk = windows.reshape((1, windows.shape[0], current_batch_steps) + windows.shape[2:])
                chunk = jax.device_put(chunk)
                closure, optim_state, losses, discard_flags, max_cfls = test_epoch(chunk, closure, optim_state)
                discard_flags = np.asarray(discard_flags).reshape(-1)
                losses = np.asarray(losses).reshape(-1)
                max_cfls = np.asarray(max_cfls).reshape(-1)
                if np.any(discard_flags):
                    print(f"Discarded test batch: max rollout CFL={float(np.max(max_cfls)):.4f} > limit {cfl_limit:.4f}")
                test_losses_accum.extend([float(loss) for loss, discarded in zip(losses, discard_flags) if not discarded])

            train_mean = float(np.mean(train_losses_accum)) if train_losses_accum else float('nan')
            test_mean  = float(np.mean(test_losses_accum))  if test_losses_accum  else float('nan')
            train_mean_losses.append(train_mean)
            test_mean_losses.append(test_mean)
            window_train_epoch_means.append(train_mean)
            window_test_epoch_means.append(test_mean)
            epoch_counter += 1

            logger.info(
                "Stage %d/%d (day=%d) | sub-epoch %d/%d | global epoch %d/%d | "
                "mean_train=%.4E | mean_test=%.4E",
                day_idx + 1, len(window_days), current_days,
                stage_epoch + 1, n_epochs,
                epoch_counter, total_curriculum_epochs,
                train_mean, test_mean,
            )

            try:
                checkpointer(
                    closure, optim_state, model_dir, save=True,
                    epoch=epoch_counter,
                    n_epochs=total_curriculum_epochs,
                    losses={"train": train_mean_losses, "test": test_mean_losses},
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
                    "Saved curriculum checkpoint: stage %d/%d sub-epoch %d/%d",
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
            "Completed window %d/%d (day=%d): mean train=%.4E | mean test=%.4E over %d epochs",
            day_idx + 1,
            len(window_days),
            current_days,
            window_train_mean,
            window_test_mean,
            len(window_train_epoch_means),
        )


    # === validation & diagnostics ===
    try:
        loaded_leaves, loaded_optim, ckpt_meta, loaded_loss_history = checkpointer(None, None, model_dir, save=False)
        closure = build_closure(cfg, loaded_leaves)
        eps = 1e-8

        _orig_closure = closure

        def projected_closure(q):
            out = _orig_closure(q)
            qh = jnp.fft.rfftn(q, axes=(-2,-1), norm='ortho')
            out_qh = jnp.fft.rfftn(out, axes=(-2,-1), norm='ortho')
            num = jnp.real(jnp.conj(qh) * out_qh)
            den = jnp.abs(qh)**2 + eps
            alpha = num / den
            out_qh_proj = out_qh - alpha * qh
            return jnp.fft.irfftn(out_qh_proj, axes=(-2,-1), norm='ortho', s=out.shape[-2:])
        closure = projected_closure
    except Exception:
        logger.exception("Failed to load trained model for testing.")

    # Build validation function and run it on a held-out trajectory
    validation_epoch = make_validation_epoch(lr_model, low_res_dt)
    truth_traj = data_loader.get_trajectory(n_epochs)  # shape (time, layers, ny, nx)
    cadence = int(getattr(cfg.plotting, 'cadence', 100))
    val_traj = validation_epoch(truth_traj, cfg, closure)

    if cfg.plotting.plotting_window != 0:
        window = cfg.plotting.plotting_window
    else:
        window = val_traj["pred_frames"].shape[0]
    pred_frames = np.asarray(val_traj["pred_frames"])[:window]  # (plotting_window, nz, ny, nx)
    sgs_traj = np.asarray(val_traj["sgs"])[:window]
    hr_frames = np.asarray(truth_traj)[:window]

    trajectories = {
        "pred": pred_frames,
        "truth": hr_frames,
        "sgs": sgs_traj,
        "loss_history": {"train": train_mean_losses, "test": test_mean_losses},
        "cadence": cadence,
    }
    # Compute zero-model baseline (low-res physics with no ML dynamics)
    try:
        from model.ML.architectures.zero import ZeroModel
        zero_closure = ZeroModel()
        zero_val = validation_epoch(truth_traj, cfg, zero_closure)
        zero_pred = np.asarray(zero_val["pred_frames"][:window])  # (nt, nz, ny, nx)
        # Compute per-timestep MSE consistent with MSEDiagnostic's averaging
        zero_mse = np.mean((zero_pred - hr_frames) ** 2, axis=(-2, -1))  # (nt, nz)
        zero_mse = np.mean(zero_mse, axis=1)  # (nt,)
        maddison = maddison_loss(zero_pred - hr_frames, lr_model, beta=float(lr_model.beta))
        trajectories["zero_loss"] = maddison
    except Exception as e:
        logger.exception("Failed to compute zero-model baseline: %s", str(e))
        

    # If running in HPC mode, skip all plotting and return a single scalar
    # metric: summed validation MSE across all timesteps (mean over layers)
    hpc_mode = os.environ.get('HPC_RUN', '0') == '1'
    if hpc_mode:
        # pred_frames, hr_frames shapes: (nt, nz, ny, nx)
        mse_per_t = np.mean((pred_frames - hr_frames) ** 2, axis=(-2, -1))  # (nt, nz)
        mse_per_t_mean = np.mean(mse_per_t, axis=1)  # (nt,)
        total_mse = float(np.sum(mse_per_t_mean))
        logging.getLogger(__name__).info(f"HPC mode: total validation MSE (summed over timesteps) = {total_mse:.6E}")
        return total_mse

    Plotter(cfg, trajectories=trajectories, out_dir=out_dir, cadence=cadence).plot()

    # ============================


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
     
