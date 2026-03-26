import warnings
warnings.filterwarnings(
    "ignore",
    message=r"Explicitly requested dtype float64 requested in astype is not available.*",
    category=UserWarning,
    module=r"jax\\._src\\.numpy\\.array_methods",
)
import importlib 
import os
import argparse
from model.utils.config import Config

# Base repo paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CONFIG_DEFAULT_PATH = os.path.join(BASE_DIR, "config", "default.yaml")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, 'results', 'closures')

# Parse early CLI args (allow running with --config and --outdir)
_early_parser = argparse.ArgumentParser(add_help=False)
_early_parser.add_argument("--config", type=str, default=None)
_early_parser.add_argument("--outdir", type=str, default=None)
_early_args, _ = _early_parser.parse_known_args()
if _early_args.config:
    CONFIG_DEFAULT_PATH = os.path.abspath(_early_args.config)
OUTDIR_OVERRIDE = os.path.abspath(_early_args.outdir) if _early_args.outdir else None

# Load config and canonicalize common paths
cfg = Config.load_config(CONFIG_DEFAULT_PATH)
if hasattr(cfg, "filepaths") and hasattr(cfg.filepaths, "out_dir"):
    od = cfg.filepaths.out_dir
    if od and not os.path.isabs(od):
        cfg.filepaths.out_dir = os.path.abspath(os.path.join(BASE_DIR, od))

use_float64 = getattr(cfg.ml, "use_float64", False)
if use_float64:
    os.environ.setdefault("JAX_ENABLE_X64", "1") 
else:
    os.environ.setdefault("JAX_ENABLE_X64", "0")
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
from model.ML.train import make_train_epoch, make_test_epoch
from model.ML.architectures.build_model import build_closure
from model.ML.utils.coarsen import coarsen
from model.ML.generate_data import generate_train_data
from model.ML.utils.dataloading import find_existing_closure, find_existing_run, ZarrDataLoader, checkpointer, prefetch_generator
from model.utils.logging import configure_logging
from model.utils.plotting import find_output_dir, make_quad_gif, gif_that
from model.core.steppers import SteppedModel, AB3Stepper
from model.core.model import QGM
from model.ML.utils.utils import parameterization
from model.ML.train import roll_out_with_forced_model
from model.ML.forced_model import ForcedModel
import logging
import jax
import yaml
import os
import json
import numpy as np
import equinox as eqx
import matplotlib.pyplot as plt
import optax
import jax.numpy as jnp

# === just loading in params as dict === #
with open(CONFIG_DEFAULT_PATH) as f:
    cfg_dict = yaml.safe_load(f)

# load config values
params = dict(cfg_dict["params"])   # pure dict for JAX  




# =========================================
# Main loop to run from Command Line 
# =========================================
def main():
    # load values
    dt = cfg.plotting.dt
    njets= cfg.plotting.njets
    nsteps = cfg.plotting.nsteps
    learning_rate = cfg.ml.learning_rate
    batch_steps = cfg.ml.batch_steps
    n_train = cfg.ml.n_train
    n_test = cfg.ml.n_test
    n_epochs = n_train + n_test
    n_samples = nsteps//batch_steps
    spinup = cfg.plotting.spinup
    seed = params.get("seed", 42)
    key = jax.random.PRNGKey(seed)
    ratio = params["hr_nx"]/params["nx"]
    minibatch_size = cfg.ml.minibatch_size
    prefetch = cfg.ml.prefetch
    model_type = cfg.ml.model_type

    logger = configure_logging(level=cfg.filepaths.log_level, out_file="../logs/run.log")
    logger = logging.getLogger(__name__)
    
    # output dir 
    outbase = os.path.join(cfg.filepaths.out_dir)
    # allow overriding outdir from early CLI arg
    global OUTDIR_OVERRIDE
    if OUTDIR_OVERRIDE:
        outbase = OUTDIR_OVERRIDE
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
    logger.info(f"Running on {device_type.upper()} with x64_enabled={use_float64}")    

    if cfg.plotting.auto_dt:
        logger.info("Auto-setting initial dt using CFL condition on a sample initial state.")
        raw_model = QGM({**params, "nx": params['hr_nx']})
        init_state = raw_model.initialise(key, tune=True, n_jets=njets, verbose=True)
        dt = raw_model.estimate_cfl_dt(init_state)

    # instantiate the model
    hr_model = SteppedModel(
        model=QGM({**params, "nx": params['hr_nx']}),
        stepper=AB3Stepper(dt=dt),
    )
    # Build low-resolution physics model (coarsened from high-res physics)
    lr_model = coarsen(hr_model.model, params['nx'])

    # === dataloading === #
    timing_metadata = {
        'spinup': int(spinup),
        'nsteps': int(nsteps),
        "dt": float(dt),
        'batch_steps': int(batch_steps),
    }

    run_dir, found = find_existing_run(DATA_DIR, params, timing_metadata)
    if found: 
        logger.info(f"Found existing run with matching parameters at {run_dir}, loading data from there.")
        data_loader = ZarrDataLoader(run_dir)
    else:
        logger.info(f"No existing run found, generating new dataset at {run_dir}")
        os.makedirs(run_dir, exist_ok=False)
        generate_train_data(cfg, params, dt, hr_model, lr_model, run_dir)
        data_loader = ZarrDataLoader(run_dir)

    # === ML training === #
    model_dir, found = find_existing_closure(MODEL_DIR, params, timing_metadata, model_type)
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
            logger.info(f"Resuming training from epoch {start_epoch} (saved) out of {saved_n_epochs}")
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
            saved_leaves, _ = jax.tree_util.tree_flatten(loaded_optim)
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
    low_res_dt = dt/ratio
    train_epoch = make_train_epoch(lr_model, low_res_dt, optim)
    test_epoch = make_test_epoch(lr_model, low_res_dt)

    # Split trajectories into train and test sets
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
        train_indices_host = list(np.asarray(jax.device_get(train_indices)).tolist())
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
        test_indices_host = list(np.asarray(jax.device_get(test_indices)).tolist())
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
            meta = {
                "parameters": params,
                "timing": timing_metadata,
                "model_type": model_type,
            }
            with open(os.path.join(model_dir, "metadata.json"), "w") as f:
                json.dump(meta, f, indent=4)
            logger.info(f"Saved checkpoint for epoch {epoch+1} to {model_dir}")
        except Exception:
            logger.exception("Failed to save checkpoint after epoch %d", epoch + 1)


    # Initialize interactive plotting so the loss curve updates each epoch
    fig, ax = plt.subplots()
    ln1, = ax.plot([], [], label='train')
    ln2, = ax.plot([], [], label='test')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Step Loss")
    ax.grid(True)
    ax.set_title("Train and Test Loss Over Steps")
    ax.legend()
    plt.ion()
    ln1.set_data(np.arange(len(train_mean_losses)) + 1, train_mean_losses)
    ln2.set_data(np.arange(len(test_mean_losses)) + 1, test_mean_losses)
    ax.relim(); ax.autoscale_view()
    fig.canvas.draw(); fig.canvas.flush_events()
    plt.ioff()
    loss_history = os.path.join(out_dir, "loss_history.png")
    fig.savefig(loss_history)
    print(f"Saved loss plot to {loss_history}")

    try:
        loaded_leaves, loaded_optim, ckpt_meta, loaded_loss_history = checkpointer(None, None, model_dir, save=False)
    except Exception:
        logger.exception("Failed to load trained model for testing.")

    truth_traj = data_loader.get_trajectory(0)  # shape (time, layers, ny, nx)

    # Build closure template and reconstruct if checkpoint available
    closure = build_closure(cfg, loaded_leaves)

    # build HR model and coarsener
    dt = cfg.plotting.dt
    hr_model = SteppedModel(model=QGM({**params, "nx": params['hr_nx']}), stepper=AB3Stepper(dt=dt))
    coarse = coarsen(hr_model.model, params['nx'])

    # load a high-res trajectory (first available)
    data_dir, found_run = find_existing_run(os.path.join(BASE_DIR, "data"), params, timing_metadata)
    if not found_run:
        raise FileNotFoundError("No matching high-resolution data run found for provided parameters.")
    loader = ZarrDataLoader(data_dir)
    truth_traj = loader.get_trajectory(0)  # shape (time, layers, ny, nx)

    nsteps = truth_traj.shape[0]

    closure_params, closure_static = eqx.partition(closure, eqx.is_array)
    init_param_func = lambda state, model, params: params

    def _param_adapter(state, param_aux, model, *args, **kwargs):
        # param_aux holds closure params (arrays)
        # reuse closure_combiner behaviour: combine params + static to evaluate closure
        from model.ML.train import closure_combiner
        return closure_combiner(state, param_aux, closure_static)

    closure_func = parameterization(_param_adapter)
    forced_hr_static = SteppedModel(
        model=ForcedModel(model=coarse, closure=closure_func, init_param_aux_func=init_param_func),
        stepper=hr_model.stepper,
    )

    # template state for the forced model (low-res initialiser)
    template_state = coarse.initialise(jax.random.PRNGKey(0))

    # Use the first coarsened frame as init and roll out for full length
    init_q = jnp.asarray(truth_traj[0])
    pred_traj = roll_out_with_forced_model(init_q, forced_hr_static, template_state, nsteps, closure_params)
    pred_traj = np.asarray(pred_traj)


    @jax.jit
    def _ml_contrib(q):
        # closure expects float32, returns dq in same spatial shape
        return closure(q.astype(jnp.float32)).astype(q.dtype)

    sgs_traj = jax.vmap(_ml_contrib)(jnp.asarray(pred_traj))
    sgs_traj = np.asarray(sgs_traj)

    layer = 0
    hr_frames = np.asarray(truth_traj[:, layer])
    pred_frames = np.asarray(pred_traj[:, layer])
    sgs_frames = np.asarray(sgs_traj[:, layer])

    gif_out = os.path.join(out_dir, "PV.gif")
    quad_out = os.path.join(out_dir, "quad.gif")
    # make_quad_gif expects arrays with shape (nt, ny, nx)
    gif_that(hr_frames, out_file=gif_out, cadence=100)
    make_quad_gif(hr_frames, pred_frames, sgs_q=sgs_frames, out_file=quad_out, cadence=100)
    print(f"Saved comparison GIF to {gif_out}")

    try:
        mse_per_timestep = np.mean((pred_frames - hr_frames) ** 2, axis=(1, 2))
        mse_out = os.path.join(out_dir, "mse_per_timestep.png")
        plt.figure(figsize=(6, 3))
        plt.plot(np.arange(mse_per_timestep.size), mse_per_timestep, '-o')
        plt.xlabel('Timestep')
        plt.ylabel('MSE')
        plt.title('MSE per timestep: prediction vs truth')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(mse_out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved MSE plot to {mse_out}")
    except Exception:
        logger.exception("Failed to compute or save MSE plot")

    
    # ============================

if __name__ == "__main__":
    main()
     