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
MODEL_DIR = os.path.join(DATA_DIR, "saved_models")
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
from model.ML.train import make_train_epoch
from model.ML.utils.utils import module_to_single
from model.ML.architectures.build_model import build_closure
from model.ML.utils.coarsen import Coarsen
from model.ML.generate_data import generate_train_data
from model.ML.utils.dataloading import find_existing_closure, find_existing_run, ZarrDataLoader
from model.utils.logging import configure_logging
from model.core.steppers import SteppedModel, AB3Stepper
from model.core.model import QGM
import logging
import jax
import yaml
import os
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
    learning_rate = cfg.ml.learning_rate
    batch_size = cfg.ml.batch_size
    batch_steps = cfg.ml.batch_steps
    n_batches = getattr(cfg.ml, "n_batches", 100)
    n_epochs = cfg.ml.n_epochs
    n_train = cfg.ml.n_train
    n_test = cfg.ml.n_test
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
        "dt": float(cfg.plotting.dt),
        'steps': cfg.plotting.nsteps,
        "cadence": int(cfg.plotting.cadence if hasattr(cfg.plotting, 'cadence') else 1),    
        'batch_size': int(batch_size),
        'batch_steps': int(batch_steps),
    }

    run_dir, found = find_existing_run(DATA_DIR, params['hr_nx'], params['nx'], params, timing_metadata)
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
    found = find_existing_closure(MODEL_DIR, cfg)
    if found:
        raise NotImplementedError("Model loading is not implemented yet")
    else:
        closure_model = build_closure(cfg)
    closure = module_to_single(closure_model)
    if cfg.ml.optimiser=='Adam':
        optim = optax.adam(learning_rate)
    elif cfg.ml.optimiser=='AdamW':
        optim = optax.adamw(learning_rate)
    else:
        raise ValueError(f"Unsupported optimiser: {cfg.ml.optimiser}. Supported options are 'Adam' and 'AdamW'.")
    optim_state = optim.init(eqx.filter(closure, eqx.is_array))

    # Build training function
    train_epoch = make_train_epoch(coarse, hr_model, optim)

    logger.info(
        f"Training with chunked windows from Zarr: n_traj={len(data_loader)}, "
        f"traj_shape={data_loader.traj_shape}, batch_size={batch_size}, batch_steps={batch_steps}"
    )

    # Split trajectories into train and test sets
    all_traj_indices = list(range(len(data_loader)))
    train_indices = all_traj_indices[:n_train]
    test_indices = all_traj_indices[n_train:n_train + n_test]
    if len(all_traj_indices) < n_train + n_test:
        raise ValueError(f"Not enough trajectories in dataset for requested train/test split.")
    logger.info(f"Train indices: {train_indices}, Test indices: {test_indices}")

    np_rng = np.random.default_rng(seed=seed)
    all_batch_losses = []
    epoch_mean_losses = []

    for epoch in range(n_epochs):        
        epoch_batches = data_loader.sample_windows(
            n_samples=n_batches * batch_size,
            window_size=batch_steps,
            rng=np_rng,
            subset_traj_indices=train_indices,
        ).astype(np.float32)
        
        # Reshape to (n_batches, batch_size, window_size, ...)
        epoch_batches = epoch_batches.reshape(
            (n_batches, batch_size, batch_steps) + epoch_batches.shape[2:]
        )
        
        # Explicitly move to device
        epoch_batches = jax.device_put(epoch_batches)

        logger.info("Executing epoch %d/%d. Total batches: %d, batch size: %d, batch steps: %d", epoch + 1, n_epochs, n_batches, batch_size, batch_steps)
        closure, optim_state, epoch_losses_jax = train_epoch(epoch_batches, closure, optim_state)
        
        epoch_losses = np.array(epoch_losses_jax)
        all_batch_losses.extend(epoch_losses.tolist())

        epoch_mean = float(np.mean(epoch_losses))
        epoch_mean_losses.append(epoch_mean)
        logger.info("Finished epoch %d/%d | mean_loss=%.4E", epoch + 1, n_epochs, epoch_mean)

    plt.plot(np.arange(len(all_batch_losses)) + 1, all_batch_losses)
    plt.xlabel("Step")
    plt.ylabel("Step Loss")
    plt.grid(True)
    plt.title("Training Loss Over Steps")

    # ============================

if __name__ == "__main__":
    main()
     