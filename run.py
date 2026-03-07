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
        "dt": float(cfg.plotting.dt),
        'steps': cfg.plotting.nsteps, 
        'batch_steps': int(batch_steps),
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
    train_mean_losses = []
    test_mean_losses = []

    for epoch in range(n_epochs):
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

    plt.plot(np.arange(len(train_mean_losses)) + 1, train_mean_losses)
    plt.plot(np.arange(len(test_mean_losses)) + 1, test_mean_losses)
    plt.xlabel("Step")
    plt.ylabel("Step Loss")
    plt.grid(True)
    plt.title("Train and Test Loss Over Steps")
    plt.legend()

    # ============================

if __name__ == "__main__":
    main()
     