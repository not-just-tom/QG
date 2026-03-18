import warnings
warnings.filterwarnings(
    "ignore",
    message=r"Explicitly requested dtype float64 requested in astype is not available.*",
    category=UserWarning,
    module=r"jax\\._src\\.numpy\\.array_methods",
)
import importlib 
import os
from model.utils.config import Config
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CONFIG_DEFAULT_PATH = os.path.join(BASE_DIR, "config", "default.yaml")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, 'results', 'closures')
cfg = Config.load_config(CONFIG_DEFAULT_PATH)
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
import model.ML.utils.utils
import model.ML.train
import model.utils.plotting
import model.utils.diagnostics
importlib.reload(model.core.grid)
importlib.reload(model.core.states)
importlib.reload(model.core.kernel)
importlib.reload(model.core.model)
importlib.reload(model.core.steppers)
importlib.reload(model.ML.generate_data)
importlib.reload(model.ML.utils.coarsen)
importlib.reload(model.ML.architectures.build_model)
importlib.reload(model.ML.utils.dataloading)
importlib.reload(model.ML.utils.utils)
importlib.reload(model.ML.train)
importlib.reload(model.utils.diagnostics)
importlib.reload(model.utils.plotting)
from model.ML.train import make_train_epoch, make_test_epoch
from model.ML.utils.utils import module_to_single
from model.ML.architectures.build_model import build_closure
from model.ML.utils.coarsen import Coarsen
from model.ML.generate_data import generate_train_data
from model.ML.utils.dataloading import find_existing_closure, find_existing_run, ZarrDataLoader, checkpointer
from model.utils.logging import configure_logging
from model.core.steppers import SteppedModel, AB3Stepper
from model.core.model import QGM
import logging
import jax
import yaml
import os
import json
import numpy as np
import equinox as eqx
import matplotlib.pyplot as plt
import optax

# === just loading in params as dict === #
with open(CONFIG_DEFAULT_PATH) as f:
    cfg_dict = yaml.safe_load(f)

params = dict(cfg_dict["params"])   # pure dict for JAX  




# =========================================
# Main loop to run from Command Line 
# =========================================
def main():
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

    logger.info(f"Running on {device_type.upper()} with x64_enabled={use_float64}")

    # load config values
    dt = cfg.plotting.dt
    njets= cfg.plotting.njets
    nsteps = cfg.plotting.nsteps
    learning_rate = cfg.ml.learning_rate
    batch_steps = cfg.ml.batch_steps
    n_train = cfg.ml.n_train
    n_test = cfg.ml.n_test

    n_epochs = n_train + n_test
    n_samples = nsteps//batch_steps
    seed = params.get("seed", 42)
    key = jax.random.PRNGKey(seed)

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
    coarse = Coarsen(hr_model, params['nx'])

    # === dataloading === #
    timing_metadata = {
        'spinup': int(cfg.plotting.spinup),
        'nsteps': int(cfg.plotting.nsteps),
        "dt": float(cfg.plotting.dt),
        'batch_steps': int(cfg.ml.batch_steps),
    }

    run_dir, found = find_existing_run(DATA_DIR, params, timing_metadata)
    if found: 
        logger.info(f"Found existing run with matching parameters at {run_dir}, loading data from there.")
        data_loader = ZarrDataLoader(run_dir)
    else:
        logger.info(f"No existing run found, generating new dataset at {run_dir}")
        os.makedirs(run_dir, exist_ok=False)
        # generate and load
        generate_train_data(cfg, params, hr_model, coarse, run_dir)
        data_loader = ZarrDataLoader(run_dir)

    # === ML training === #
    model_dir, found = find_existing_closure(MODEL_DIR, params, timing_metadata, cfg.ml.model_type)
    loaded_params = None
    loaded_optim = None
    ckpt_meta = None
    start_epoch = 0
    if found:
        logger.info(f"Found existing closure with matching parameters at {model_dir}, attempting to load checkpoint.")
        try:
            loaded_leaves, loaded_optim, ckpt_meta, loaded_loss_history = checkpointer(None, None, model_dir, save=False)
        except Exception:
            logger.exception("Failed to load checkpoint; will build a new closure")
            loaded_leaves, loaded_optim, ckpt_meta, loaded_loss_history = None, None, None, None

        # If there's no checkpoint metadata, avoid overwriting the existing run: create a new model_dir
        if ckpt_meta is None:
            # compute next available index for this model prefix
            hr_nx = params['hr_nx']
            lr_nx = params['nx']
            prefix = f"{cfg.ml.model_type}_hr{hr_nx}_nx{lr_nx}_"
            existing = [n for n in os.listdir(MODEL_DIR) if n.startswith(prefix)]
            # extract numeric suffixes
            idxs = []
            for n in existing:
                try:
                    idx = int(n.rsplit("_", 1)[-1])
                    idxs.append(idx)
                except Exception:
                    continue
            next_idx = max(idxs, default=0) + 1
            new_run_name = f"{prefix}{next_idx:02d}"
            model_dir = os.path.join(MODEL_DIR, new_run_name)
            os.makedirs(model_dir, exist_ok=True)
            logger.info(f"No checkpoint found in existing closure dir; creating new model_dir {model_dir}")
        else:
            # If checkpoint metadata exists, decide whether to resume or error if already finished
            saved_epoch = int(ckpt_meta.get('epoch', 0))
            saved_n_epochs = int(ckpt_meta.get('n_epochs', n_epochs))
            if saved_epoch >= saved_n_epochs:
                raise ValueError(f"Model at {model_dir} already trained for {saved_n_epochs} epochs; nothing to resume.")
            start_epoch = saved_epoch
            logger.info(f"Resuming training from epoch {start_epoch} (saved) out of {saved_n_epochs}")

    # Build a fresh closure template and, if params were loaded, combine them
    closure_template = build_closure(cfg)
    if 'loaded_leaves' in locals() and loaded_leaves is not None:
        try:
            template_params, template_static = eqx.partition(closure_template, eqx.is_array)
            tpl_leaves, tpl_treedef = jax.tree_util.tree_flatten(template_params)
            if len(tpl_leaves) != len(loaded_leaves):
                raise ValueError(f"Loaded params length {len(loaded_leaves)} does not match template {len(tpl_leaves)}")

            # cast loaded leaves to template dtypes and build new param pytree
            new_leaves = []
            for tpl, ld in zip(tpl_leaves, loaded_leaves):
                arr = np.asarray(ld)
                # ensure dtype matches template leaf
                try:
                    arr = arr.astype(np.asarray(tpl).dtype)
                except Exception:
                    pass
                new_leaves.append(jax.device_put(arr))

            new_params = jax.tree_util.tree_unflatten(tpl_treedef, new_leaves)
            closure_model = eqx.combine(new_params, template_static)
            logger.info("Reconstructed closure from loaded params")
        except Exception:
            logger.exception("Failed to reconstruct closure from params; falling back to fresh closure")
            closure_model = closure_template
    else:
        closure_model = closure_template

    closure = module_to_single(closure_model)
    if cfg.ml.optimiser=='Adam':
        optim = optax.adam(learning_rate)
    elif cfg.ml.optimiser=='AdamW':
        optim = optax.adamw(learning_rate)
    else:
        raise ValueError(f"Unsupported optimiser: {cfg.ml.optimiser}. Supported options are 'Adam' and 'AdamW'.")
    # Initialize optimizer state from template and, if available, map saved optimizer leaves into it
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
    train_epoch = make_train_epoch(coarse, hr_model, optim)
    test_epoch = make_test_epoch(coarse, hr_model, optim)

    logger.info(
        f"Training with chunked windows from Zarr: n_traj={len(data_loader)}, "
        f"traj_shape={data_loader.traj_shape}, batch_steps={batch_steps}"
    )

    # Split trajectories into train and test sets
    all_traj_indices = list(range(len(data_loader)))
    if len(all_traj_indices) < n_epochs:
        raise ValueError(f"Not enough trajectories in dataset for requested train/test split.")

    keys= jax.random.split(key, n_epochs + 2)
    more_keys = jax.random.split(keys[n_epochs], n_epochs) # very janky way of doing this, but no key reuse. 
    all_batch_losses = []
    # initialise loss history; if we loaded a saved history, continue it
    train_mean_losses = []
    test_mean_losses = []
    try:
        if 'loaded_loss_history' in locals() and loaded_loss_history is not None:
            train_mean_losses = list(loaded_loss_history.get('train', []))
            test_mean_losses = list(loaded_loss_history.get('test', []))
            logger.info(f"Loaded existing loss history: {len(train_mean_losses)} train entries, {len(test_mean_losses)} test entries")
    except Exception:
        logger.exception("Failed to restore loss history; starting fresh")

    for epoch in range(start_epoch, n_epochs):
        # shuffle indices for train and test
        shuffled_indices = jax.random.permutation(keys[epoch], n_epochs)
        # this shuffle still doesnt guarantee a unique split every epoch
        train_indices = shuffled_indices[:n_train]
        test_indices = shuffled_indices[n_train:]
        
        train_trajs = data_loader.sample_windows(
            n_samples=n_samples,
            batch_steps=batch_steps,
            key=more_keys[epoch],
            traj_indices=train_indices,
        ).astype(np.float32)

        test_trajs = data_loader.sample_windows(
            n_samples=n_samples,
            batch_steps=batch_steps,
            key=more_keys[epoch],
            traj_indices=test_indices,
        ).astype(np.float32)

        #print(train_trajs.shape)
                
        # Reshape to (n_batches, batch_size, window_size, ...) # this is stupid but leave it for now 
        train_trajs = train_trajs.reshape(
            (n_train, n_samples, batch_steps) + train_trajs.shape[2:]
        )
        # and for test. 
        test_trajs = test_trajs.reshape(
            (n_test, n_samples, batch_steps) + test_trajs.shape[2:]
        )
        
        # Explicitly move to device
        train_trajs = jax.device_put(train_trajs)

        logger.info("Executing epoch %d/%d.", epoch + 1, n_epochs)
        closure, optim_state, train_loss = train_epoch(train_trajs, closure, optim_state)
        closure, optim_state, test_loss = test_epoch(test_trajs, closure, optim_state)
      

        train_mean = float(np.mean(np.array(train_loss)))
        test_mean = float(np.mean(np.array(test_loss)))
        train_mean_losses.append(train_mean)
        test_mean_losses.append(test_mean)
        logger.info("Finished epoch %d/%d | mean_train_loss=%.4E | mean_test_loss=%.4E", epoch + 1, n_epochs, train_mean, test_mean)
        # Save checkpoint after every epoch
        try:
            checkpointer(closure, optim_state, model_dir, save=True, epoch=epoch+1, n_epochs=n_epochs, losses={"train": train_mean_losses, "test": test_mean_losses})
            meta = {
                "parameters": params,
                "timing": timing_metadata,
                "model_type": cfg.ml.model_type,
            }
            with open(os.path.join(model_dir, "metadata.json"), "w") as f:
                json.dump(meta, f, indent=4)
            logger.info(f"Saved checkpoint for epoch {epoch+1} to {model_dir}")
        except Exception:
            logger.exception("Failed to save checkpoint after epoch %d", epoch + 1)

    plt.plot(np.arange(len(train_mean_losses)) + 1, train_mean_losses)
    plt.plot(np.arange(len(test_mean_losses)) + 1, test_mean_losses)
    plt.xlabel("Step")
    plt.ylabel("Step Loss")
    plt.grid(True)
    plt.title("Train and Test Loss Over Steps")
    plt.legend()
    # Save checkpoint for this trained closure
    try:
        checkpointer(closure_model, optim_state, model_dir, save=True)
        # write metadata for the saved model so future runs can find it
        meta = {
            "parameters": params,
            "timing": timing_metadata,
            "model_type": cfg.ml.model_type,
        }
        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=4)
        logger.info(f"Saved checkpoint and metadata to {model_dir}")
    except Exception:
        logger.exception("Failed to save model checkpoint")

    
    # ============================

if __name__ == "__main__":
    main()
     