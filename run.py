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
importlib.reload(model.utils.diagnostics)
importlib.reload(model.utils.plotting)
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
    
    # === JAX Device and Precision configuration === #
    device_type = getattr(cfg.ml, "device", "cpu").lower()
    
    # Use a more robust check to avoid RuntimeError with unknown backend names like 'rocm'
    if device_type == "gpu":
        # Try to find an available GPU-like backend
        backend_order = ["cuda", "gpu"] 
        # Don't include 'rocm' by default unless you're on a Linux AMD system where it's known
        found_gpu = False
        for backend in backend_order:
            try:
                jax.devices(backend)
                jax.config.update("jax_platforms", backend)
                found_gpu = True
                break
            except Exception:
                continue
        if not found_gpu:
            jax.config.update("jax_platforms", "cpu")
            device_type = "cpu (fallback)"
    else:
        jax.config.update("jax_platforms", "cpu")

    use_float64 = getattr(cfg.ml, "use_float64", False)
    jax.config.update("jax_enable_x64", use_float64)




    # setup logging
    logger = configure_logging(level=cfg.filepaths.log_level, out_file="../logs/run.log")
    logger = logging.getLogger(__name__)

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
        "dt": float(dt),
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
    net = module_to_single(closure_model)

    optim = optax.adam(learning_rate)
    optim_state = optim.init(eqx.filter(net, eqx.is_array))

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

    @parameterization
    def net_parameterization(state, net_params, model, static_net_obj=None):
        assert static_net_obj is not None, "static_net_obj must be provided"
        # Combine the dynamic parameters with the static structure
        net = eqx.combine(net_params, static_net_obj)
        
        q = state.q
        dq_closure = net(q.astype(jnp.float32))
        return dq_closure.astype(q.dtype), net_params

    template_state = hr_model.model.initialise(jax.random.PRNGKey(0))

    def roll_out_with_forced_model(init_q, forced_hr_model, nsteps, net_params):
        init_qh = jnp.fft.rfftn(init_q, axes=(-2, -1), norm='ortho').astype(template_state.qh.dtype)
        base_state = template_state.update(qh=init_qh)
        
        # This initializes the stepper state with 'net_params' (only arrays) in param_aux.
        init_state = forced_hr_model.initialize_stepper_state(
            forced_hr_model.model.initialise_param_state(base_state, net_params)
        )

        def step(carry, _x):
            next_state = forced_hr_model.step_model(carry)
            return next_state, next_state.state.model_state.q

        _final_step, traj = jax.lax.scan(step, init_state, None, length=nsteps)
        return traj

    def compute_traj_errors(target_traj, forced_hr_model, net_params):
        rolled_out = roll_out_with_forced_model(
            init_q=target_traj[0],
            forced_hr_model=forced_hr_model,
            nsteps=target_traj.shape[0],
            net_params=net_params,
        )
        return rolled_out - target_traj

    @eqx.filter_jit
    def train_epoch(epoch_batches, net, optim_state):
        hr_base_model = hr_model.model
        stepper_obj = hr_model.stepper
        
        # Partition the net into dynamic parameters and static structure
        net_params, static_net_obj = eqx.partition(net, eqx.is_array)

        # Identity function to pass net_params into the parameterization
        init_param_func = lambda state, model, params: params

        # Build model with static structure captured
        param_func = functools.partial(net_parameterization, static_net_obj=static_net_obj)
        forced_hr_static = SteppedModel(
            model=ForcedModel(
                model=hr_base_model,
                param_func=param_func,
                init_param_aux_func=init_param_func,
            ),
            stepper=stepper_obj,
        )

        def step_fn(carry, batch):
            net_params, optim_state = carry

            def loss_fn(params, batch):
                err = jax.vmap(
                    functools.partial(compute_traj_errors, 
                                      forced_hr_model=forced_hr_static,
                                      net_params=params)
                )(batch)
                return jnp.mean(err ** 2)

            loss, grads = eqx.filter_value_and_grad(loss_fn)(net_params, batch)
            updates, new_optim_state = optim.update(grads, optim_state, net_params)
            new_net_params = eqx.apply_updates(net_params, updates)
            return (new_net_params, new_optim_state), loss

        (final_net_params, final_optim_state), losses = jax.lax.scan(
            step_fn, (net_params, optim_state), epoch_batches
        )
        return eqx.combine(final_net_params, static_net_obj), final_optim_state, losses

    np_rng = np.random.default_rng(seed=seed)
    all_batch_losses = []
    epoch_mean_losses = []

    for epoch in range(n_epochs):
        logger.info("Starting epoch %d of %d", epoch + 1, n_epochs)
        
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
        epoch_batches_gpu = jax.device_put(epoch_batches)

        logger.info("Executing epoch %d/%d. Total batches: %d, batch size: %d, batch steps: %d", epoch + 1, n_epochs, n_batches, batch_size, batch_steps)
        net, optim_state, epoch_losses_jax = train_epoch(
            epoch_batches_gpu, net, optim_state
        )
        
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
     