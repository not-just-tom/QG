import importlib 
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
import model.ML.forced_model
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
importlib.reload(model.ML.forced_model)
importlib.reload(model.ML.train)
importlib.reload(model.utils.diagnostics)
importlib.reload(model.utils.plotting)
from model.ML.train import train_epoch
from model.ML.utils.utils import parameterization, module_to_single
from model.ML.architectures.build_model import build_closure
from model.ML.utils.coarsen import Coarsen
from model.ML.generate_data import generate_train_data
from model.ML.utils.dataloading import find_existing_closure, find_existing_run, ZarrDataLoader
from model.utils.config import Config
from model.utils.logging import configure_logging
from model.ML.forced_model import ForcedModel
from model.core.steppers import SteppedModel, AB3Stepper
from model.core.model import QGM
import logging
import jax
import jax.numpy as jnp
import functools
import yaml
import os
import numpy as np
import equinox as eqx
import matplotlib.pyplot as plt
import optax

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CONFIG_DEFAULT_PATH = os.path.join(BASE_DIR, "config", "default.yaml")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(DATA_DIR, "saved_models")

# === just loading in params as dict === #
with open(CONFIG_DEFAULT_PATH) as f:
    cfg_dict = yaml.safe_load(f)

params = dict(cfg_dict["params"])   # pure dict for JAX  

# =========================================
# Main loop to run from Command Line 
# =========================================
def main():
    cfg = Config.load_config(CONFIG_DEFAULT_PATH)
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

    use_float64 = getattr(cfg.ml, "use_float64", False)
    jax.config.update("jax_enable_x64", use_float64)

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
        logger.info("Auto-estimating initial dt using CFL condition on a sample initial state...")
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

    # === online ML training === #
    closure = module_to_single(closure_model)

    optim = optax.adam(learning_rate)
    optim_state = optim.init(eqx.filter(closure, eqx.is_array))

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

    def closure_parameterization(state, closure_params, static_closure_obj=None):
        """Combine params and static closure, evaluate closure, return dq and params.

        This function is NOT the parameterization wrapper — it simply implements
        the combination logic. It will be wrapped later with `parameterization`
        so that it receives `(state, param_aux, model)` when called by
        `ForcedModel.get_updates`.
        """
        assert static_closure_obj is not None, "static_closure_obj must be provided"
        # Combine the dynamic parameters with the static structure
        closure = eqx.combine(closure_params, static_closure_obj)

        q = state.q
        dq_closure = closure(q.astype(jnp.float32))
        return dq_closure.astype(q.dtype), closure_params

    # Use the low-resolution template state (trajectories are coarsened)
    template_state = coarse.lr_model.initialise(jax.random.PRNGKey(0))

    def roll_out_with_forced_model(init_q, forced_hr_model, nsteps, closure_params):
        init_qh = jnp.fft.rfftn(init_q, axes=(-2, -1), norm='ortho').astype(template_state.qh.dtype)
        base_state = template_state.update(qh=init_qh)
        
        # This initializes the stepper state with '_params' (only arrays) in param_aux.
        init_state = forced_hr_model.initialize_stepper_state(
            forced_hr_model.model.initialise_param_state(base_state, closure_params)
        )

        def step(carry, _x):
            next_state = forced_hr_model.step_model(carry)
            return next_state, next_state.state.model_state.q

        _, traj = jax.lax.scan(step, init_state, None, length=nsteps)
        return traj

    def compute_traj_errors(target_traj, forced_hr_model, closure_params):
        rolled_out = roll_out_with_forced_model(
            init_q=target_traj[0],
            forced_hr_model=forced_hr_model,
            nsteps=target_traj.shape[0],
            closure_params=closure_params,
        )
        return rolled_out - target_traj

    @eqx.filter_jit
    def train_epoch(epoch_batches, closure, optim_state):
        # Use the low-resolution physics model for training (data are coarsened)
        lr_base_model = coarse.lr_model
        stepper_obj = hr_model.stepper
        
        # Partition the closure into dynamics arrays and static structure
        closure_params, static_closure_obj = eqx.partition(closure, eqx.is_array)

        init_param_func = lambda state, model, params: params
        def _param_func(state, param_aux, model, *args, **kwargs):
            # param_aux holds the dynamic closure parameters
            return closure_parameterization(state, param_aux, static_closure_obj)

        # Wrap with the `parameterization` decorator so it returns updates
        # in the form expected by ForcedModel.
        closure_func = parameterization(_param_func)
        forced_hr_static = SteppedModel(
            model=ForcedModel(
                model=lr_base_model,
                closure=closure_func,
                init_param_aux_func=init_param_func,
            ),
            stepper=stepper_obj,
        )

        def step_fn(carry, batch):
            closure_params, optim_state = carry

            def loss_fn(params, batch):
                err = jax.vmap(
                    functools.partial(compute_traj_errors, 
                                      forced_hr_model=forced_hr_static,
                                      closure_params=params)
                )(batch)
                return jnp.mean(err ** 2)

            loss, grads = eqx.filter_value_and_grad(loss_fn)(closure_params, batch)
            updates, new_optim_state = optim.update(grads, optim_state, closure_params)
            new_closure_params = eqx.apply_updates(closure_params, updates)
            return (new_closure_params, new_optim_state), loss

        (final_closure_params, final_optim_state), losses = jax.lax.scan(
            step_fn, (closure_params, optim_state), epoch_batches
        )
        return eqx.combine(final_closure_params, static_closure_obj), final_optim_state, losses

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
     