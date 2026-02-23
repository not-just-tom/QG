import importlib 
import model.core.grid
import model.core.states
import model.core.kernel
import model.core.model
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
    
    # setup logging
    logger = configure_logging(level=cfg.filepaths.log_level, out_file="../logs/run.log")
    logger = logging.getLogger(__name__)

    # load config values
    dt = cfg.plotting.dt
    learning_rate = cfg.ml.learning_rate
    batch_size = cfg.ml.batch_size
    batch_steps = cfg.ml.batch_steps
    n_batches = getattr(cfg.ml, "n_batches", 100)
    n_epochs = cfg.ml.n_epochs
    n_train = cfg.ml.n_train
    n_test = cfg.ml.n_test
    seed = params.get("seed", 42)
 

    # instantiate the model
    hr_model = SteppedModel(
        model=QGM({**params, "nx": params['hr_nx']}),
        stepper=AB3Stepper(dt),
    )
    coarse = Coarsen(hr_model, params['nx'])

    # === dataloading === #
    run_dir, found = find_existing_run(DATA_DIR, params['hr_nx'], params['nx'], params)
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
    logger.info(f"Train indices: {train_indices}, Test indices: {test_indices}")

    def sample_batch_windows(train_order, rng):
        """Sample windows from randomly ordered training trajectories."""
        windows = []
        for _ in range(batch_size):
            traj_idx = train_order[rng.integers(0, len(train_order))]
            window = data_loader.sample_windows(
                n_samples=1,
                window_size=batch_steps,
                rng=rng,
                fixed_traj_idx=traj_idx,
            )
            windows.append(window)
        return np.concatenate(windows, axis=0).astype(np.float32)

    @parameterization
    def net_parameterization(state, param_aux, model, net):
        assert param_aux is None
        q = state.q
        dq_closure = net(q.astype(jnp.float32))
        return dq_closure.astype(q.dtype), None

    template_state = hr_model.model.initialise(jax.random.PRNGKey(0))

    def roll_out_with_forced_model(init_q, forced_hr_model, nsteps):
        init_qh = jnp.fft.rfftn(init_q, axes=(-2, -1), norm='ortho').astype(template_state.qh.dtype)
        base_state = template_state.update(qh=init_qh)
        init_state = forced_hr_model.initialize_stepper_state(
            forced_hr_model.model.initialise_param_state(base_state)
        )

        def step(carry, _x):
            next_state = forced_hr_model.step_model(carry)
            return next_state, next_state.state.model_state.q

        _final_step, traj = jax.lax.scan(step, init_state, None, length=nsteps)
        return traj

    def compute_traj_errors(target_traj, forced_hr_model):
        rolled_out = roll_out_with_forced_model(
            init_q=target_traj[0],
            forced_hr_model=forced_hr_model,
            nsteps=target_traj.shape[0],
        )
        return rolled_out - target_traj

    @eqx.filter_jit
    def train_batch(batch, net, optim_state):
        batch = jnp.asarray(batch)

        def loss_fn(net, batch):
            forced_hr_model = SteppedModel(
                model=ForcedModel(
                    model=hr_model.model,
                    param_func=functools.partial(net_parameterization, net=net),
                ),
                stepper=AB3Stepper(dt=dt),
            )
            err = jax.vmap(
                functools.partial(compute_traj_errors, forced_hr_model=forced_hr_model)
            )(batch)
            return jnp.mean(err ** 2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(net, batch)
        updates, new_optim_state = optim.update(grads, optim_state, net)
        new_net = eqx.apply_updates(net, updates)
        return loss, new_net, new_optim_state

    np_rng = np.random.default_rng(seed=seed)
    all_batch_losses = []
    epoch_mean_losses = []

    for epoch in range(n_epochs):
        logger.info("Starting epoch %d of %d", epoch + 1, n_epochs)
        epoch_losses = []

        # Randomize train trajectory order for this epoch
        train_order = np_rng.permutation(train_indices)

        for batch_i in range(n_batches):
            batch = sample_batch_windows(train_order, np_rng)

            loss, net, optim_state = train_batch(batch, net, optim_state)
            loss_val = float(loss)
            epoch_losses.append(loss_val)
            all_batch_losses.append(loss_val)

            if (batch_i + 1) % 10 == 0:
                logger.info(
                    "Epoch %d/%d | Batch %d/%d | loss=%.4E",
                    epoch + 1,
                    n_epochs,
                    batch_i + 1,
                    n_batches,
                    loss_val,
                )

        epoch_mean = float(np.mean(epoch_losses))
        epoch_mean_losses.append(epoch_mean)
        logger.info("Finished epoch %d/%d | mean_loss=%.4E", epoch + 1, n_epochs, epoch_mean)

    plt.plot(np.arange(len(all_batch_losses)) + 1, all_batch_losses)
    plt.xlabel("Step")
    plt.ylabel("Step Loss")
    plt.grid(True)
    

    # ============================

if __name__ == "__main__":
    main()
     