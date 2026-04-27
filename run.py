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
from model.ML.train import make_train_epoch, make_test_epoch, make_validation_epoch
from model.ML.architectures.build_model import build_closure
from model.ML.utils.coarsen import coarsen
from model.ML.generate_data import generate_train_data
from model.ML.utils.dataloading import find_existing_closure, find_existing_run, ZarrDataLoader, checkpointer, prefetch_generator
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
    batch_steps = cfg.ml.batch_steps
    n_train = cfg.ml.n_train
    n_test = cfg.ml.n_test
    n_epochs = n_train + n_test
    n_samples = nsteps//batch_steps
    spinup = cfg.plotting.spinup
    params = dict(OmegaConf.to_container(cfg.params, resolve=True))
    seed = params.get("seed", 42)
    key = jax.random.PRNGKey(seed)
    ratio = params["hr_nx"]/params["nx"]
    use_float64 = cfg.ml.use_float64
    minibatch_size = cfg.ml.minibatch_size
    prefetch = cfg.ml.prefetch
    model_type = cfg.ml.model_type
    learning_rate = learning_rate = cfg['architectures'][model_type].get('learning_rate')

    logger = configure_logging(level=cfg.filepaths.log_level, out_file="../logs/run.log")
    logger = logging.getLogger(__name__)
    
    # output dir 
    outbase = os.path.join(cfg.filepaths.out_dir)
    out_dir, found = find_output_dir(outbase, params, model_type)
    if found:
        logger.info(f"Found existing output directory with matching parameters")
    else:
        os.makedirs(out_dir, exist_ok=True)
    
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
        'batch_steps': int(batch_steps),
    }

    run_dir, found = find_existing_run(DATA_DIR, params, timing_metadata)
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

    # === ML training === 
    # Build training/sweep metadata to avoid accidentally reusing closures from different sweeps
    training_meta = {
        "training": {
            "optimiser": cfg.ml.optimiser,
            "prefetch": int(prefetch),
            "batch_steps": int(batch_steps),
            "n_train": int(n_train),
            "n_test": int(n_test),
            "model_arch": OmegaConf.to_container(cfg['architectures'][model_type], resolve=True),
        }
    }

    model_dir, found = find_existing_closure(MODEL_DIR, params, timing_metadata, model_type, extra_meta=training_meta)
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
        optim = optax.adam(learning_rate)
    elif cfg.ml.optimiser=='AdamW':
        optim = optax.adamw(learning_rate)
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

    # Build training and test functions
    low_res_dt = dt*ratio
    train_epoch = make_train_epoch(lr_model, low_res_dt, optim)
    test_epoch = make_test_epoch(lr_model, low_res_dt)

    # Prepare trajectory indices
    all_traj_indices = list(range(len(data_loader)))
    if len(all_traj_indices) < n_epochs:
        raise ValueError(f"Not enough trajectories in dataset for requested train/test split.")

    keys= jax.random.split(key, n_epochs + 2)
    more_keys = jax.random.split(keys[n_epochs], n_epochs) # very janky way of doing this, but no key reuse. 

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

    for epoch in range(start_epoch, n_epochs):
        if epoch == 0:
            logger.info(f"Starting training for {n_epochs} epochs with batch size: {batch_steps}, traj shape: {data_loader.traj_shape} and learning rate: {learning_rate}")
        # shuffle indices for train and test
        shuffled_indices = jax.random.permutation(keys[epoch], n_epochs)
        # this shuffle still doesnt guarantee a unique split every epoch
        train_indices = shuffled_indices[:n_train]
        test_indices = shuffled_indices[n_train:]

        # === train ===
        train_losses_accum = []
        # Convert JAX arrays to host-side python list of indices for zarr indexing
        # Map local shuffled positions into real trajectory IDs using all_traj_indices
        train_pos = list(np.asarray(jax.device_get(train_indices)).tolist())
        train_indices_host = [all_traj_indices[i] for i in train_pos]
        # Use the data loader's minibatch iterator and prefetch to overlap I/O
        train_gen = data_loader.iterate_minibatches(
            traj_indices=train_indices_host,
            n_samples=n_samples,
            batch_steps=batch_steps,
            key=more_keys[epoch],
            minibatch_size=minibatch_size,
        )
        train_prefetch = prefetch_generator(train_gen, size=prefetch)
        for windows in train_prefetch:
            windows = windows.astype(np.float32)
            # reshape to (n_batches=1, n_samples=minibatch_size, batch_steps, ...)
            chunk = windows.reshape((1, windows.shape[0], batch_steps) + windows.shape[2:])
            chunk = jax.device_put(chunk)
            closure, optim_state, losses = train_epoch(chunk, closure, optim_state)
            train_losses_accum.extend(list(np.asarray(losses).reshape(-1).tolist()))

        # === test ===
        test_losses_accum = []
        test_pos = list(np.asarray(jax.device_get(test_indices)).tolist())
        test_indices_host = [all_traj_indices[i] for i in test_pos]
        test_gen = data_loader.iterate_minibatches(
            traj_indices=test_indices_host,
            n_samples=n_samples,
            batch_steps=batch_steps,
            key=more_keys[epoch],
            minibatch_size=minibatch_size,
        )
        test_prefetch = prefetch_generator(test_gen, size=prefetch)
        for windows in test_prefetch:
            windows = windows.astype(np.float32)
            chunk = windows.reshape((1, windows.shape[0], batch_steps) + windows.shape[2:])
            chunk = jax.device_put(chunk)
            closure, optim_state, losses = test_epoch(chunk, closure, optim_state)
            test_losses_accum.extend(list(np.asarray(losses).reshape(-1).tolist()))

        # compute means and continue to checkpoint/save
        train_mean = float(np.mean(np.array(train_losses_accum))) if train_losses_accum else float('nan')
        test_mean = float(np.mean(np.array(test_losses_accum))) if test_losses_accum else float('nan')
        train_mean_losses.append(train_mean)
        test_mean_losses.append(test_mean)
        logger.info("Finished streaming epoch %d/%d | mean_train_loss=%.4E | mean_test_loss=%.4E", epoch + 1, n_epochs, train_mean, test_mean)
        # Save checkpoint after streaming epoch
        try:
            checkpointer(closure, optim_state, model_dir, save=True, epoch=epoch+1, n_epochs=n_epochs, losses={"train": train_mean_losses, "test": test_mean_losses})
            model_arch_resolved = OmegaConf.to_container(cfg['architectures'][model_type], resolve=True)
            meta = {
                "parameters": params,
                "timing": timing_metadata,
                "model_type": model_type,
                "model_arch": model_arch_resolved,
                "training": training_meta.get("training", {}),
            }
            with open(os.path.join(model_dir, "metadata.json"), "w") as f:
                json.dump(meta, f, indent=4)
            logger.info(f"Saved checkpoint for epoch {epoch+1} to {model_dir}")
        except Exception:
            logger.exception("Failed to save checkpoint after epoch %d", epoch + 1)


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
    cadence_used = int(getattr(cfg.plotting, 'cadence', 100))
    val_traj = validation_epoch(truth_traj, cfg, closure)

    pred_frames = np.asarray(val_traj["pred_frames"])  # (nt, nz, ny, nx)
    sgs_traj = np.asarray(val_traj["sgs"])
    hr_frames = np.asarray(truth_traj)

    trajectories = {
        "pred": pred_frames,
        "truth": hr_frames,
        "sgs": sgs_traj,
        "loss_history": {"train": train_mean_losses, "test": test_mean_losses},
        "cadence": cadence_used,
    }
    # Compute zero-model baseline (low-res physics with no ML dynamics)
    try:
        from model.ML.architectures.zero import ZeroModel
        zero_closure = ZeroModel()
        zero_val = validation_epoch(truth_traj, cfg, zero_closure)
        zero_pred = np.asarray(zero_val["pred_frames"])  # (nt, nz, ny, nx)
        # Compute per-timestep MSE consistent with MSEDiagnostic's averaging
        zero_mse = np.mean((zero_pred - hr_frames) ** 2, axis=(-2, -1))  # (nt, nz)
        zero_mse = np.mean(zero_mse, axis=1)  # (nt,)
        trajectories["zero_loss"] = zero_mse
    except Exception:
        # If baseline computation fails for any reason, continue without it
        pass

    Plotter(cfg, trajectories=trajectories, out_dir=out_dir).plot()

    # ============================


def main():
    import argparse
    from itertools import product
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=CONFIG_DEFAULT_PATH, help='Path to YAML config file')
    p.add_argument('--outdir', default=None, help='Optional output directory override')
    p.add_argument('--dry-run', action='store_true', help='Print sweep jobs but do not execute')
    p.add_argument('--generate-only', action='store_true', help='Only generate dataset then exit')
    args = p.parse_args()

    # If requested, set an env var so `run()` can detect generate-only behavior.
    if args.generate_only:
        os.environ['GENERATE_ONLY'] = '1'

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

        run(cfg_run)
        idx += 1



if __name__ == "__main__":
    main()
     
