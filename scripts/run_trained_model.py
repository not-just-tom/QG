import os
import sys
# Base repo paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
CONFIG_DEFAULT_PATH = os.path.join(BASE_DIR, "config", "default.yaml")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, 'saved_closures')
import importlib 
import model.core.model
import model.ML.utils.coarsen
import model.utils.plotting
importlib.reload(model.utils.plotting)
importlib.reload(model.core.model)
importlib.reload(model.ML.utils.coarsen)
import logging
import pathlib
import yaml
import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from model.utils.config import Config
from model.utils.logging import configure_logging
from model.core.steppers import SteppedModel, AB3Stepper
from model.utils.plotting import make_quad_gif, gif_that
from model.ML.utils.dataloading import find_existing_closure, find_existing_run, checkpointer, ZarrDataLoader
from model.ML.architectures.build_model import build_closure
from model.ML.train import roll_out, load_forced_model
from model.ML.forced_model import ForcedModel
from model.core.model import QGM
from model.ML.utils.coarsen import coarsen


# === just loading in params as dict === #
with open(CONFIG_DEFAULT_PATH) as f:
    cfg_dict = yaml.safe_load(f)

params = dict(cfg_dict["params"])

def run():
    cfg = Config.load_config(CONFIG_DEFAULT_PATH)
    out_dir = pathlib.Path(cfg.filepaths.out_dir)
    if out_dir.is_file():
        raise ValueError(f"Path must be a directory, not a file: {cfg.filepaths.out_dir}")
    out_dir.mkdir(exist_ok=True)
    configure_logging(level=cfg.filepaths.log_level, out_file="../logs/run.log") # return to this to put numbers on it 
    logger = logging.getLogger("main")
    

    # load config values
    dt = cfg.plotting.dt
    nsteps = cfg.plotting.nsteps
    key = jax.random.PRNGKey(cfg.params.seed)
    njets = cfg.plotting.njets
    ratio = params["hr_nx"]/params["nx"]
    low_res_dt = dt/ratio

    if cfg.plotting.auto_dt:
        logger.info("Auto-setting initial dt using CFL condition on a sample initial state.")
        raw_model = QGM({**params, "nx": params['hr_nx']})
        init_state = raw_model.initialise(key, tune=True, n_jets=njets, verbose=True)
        dt = raw_model.estimate_cfl_dt(init_state)

    # locate model checkpoint
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
        logger.error('No data found!')

    model_dir, found = find_existing_closure(MODEL_DIR, params, timing_metadata, cfg.ml.model_type)
    loaded_leaves, loaded_optim, ckpt_meta, loaded_loss_history = None, None, None, None
    if found:
        logger.info(f"Found existing closure with matching parameters at {model_dir}, attempting to load checkpoint.")
        try:
            loaded_leaves, loaded_optim, ckpt_meta, loaded_loss_history = checkpointer(None, None, model_dir, save=False)
        except Exception:
            logger.exception("Failed to load checkpoint.")

    # Build closure template and reconstruct if checkpoint available
    closure = build_closure(cfg, loaded_leaves)

    # build HR model and coarsener
    dt = cfg.plotting.dt
    # instantiate the model
    hr_model = SteppedModel(
        model=QGM({**params, "nx": params['hr_nx']}),
        stepper=AB3Stepper(dt=dt),
    )
    # build low-resolution physics model (coarsened from high-res physics)
    lr_model = coarsen(hr_model.model, params['nx'])

    # === test the trained closure now === #
    try:
        loaded_leaves, loaded_optim, ckpt_meta, loaded_loss_history = checkpointer(None, None, model_dir, save=False)
        closure = build_closure(cfg, loaded_leaves)
    except Exception:
        logger.exception("Failed to load trained model for testing.")

    forced_model, closure_params, closure_static = load_forced_model(lr_model, closure, low_res_dt)

    truth_traj = data_loader.get_trajectory(0)  # shape (time, layers, ny, nx) 
    nsteps = truth_traj.shape[0]

    # template state for the forced model (low-res initialiser)
    template_state = lr_model.initialise(jax.random.PRNGKey(0))

    pred_traj_full = roll_out(truth_traj[0], forced_model, nsteps, template_state, closure_params)
    pred_traj = np.asarray(pred_traj_full)

    # separate the sgs forcing from the baseline model
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

    #pred_frames = pred_frames[:60]
    #hr_frames = hr_frames[:60]

    gif_out = os.path.join(out_dir, "PV.gif")
    quad_out = os.path.join(out_dir, "quad.gif")
    # make_quad_gif expects arrays with shape (nt, ny, nx)
    #gif_that(pred_frames, out_file=gif_out, cadence=100)
    make_quad_gif(hr_frames, pred_frames, sgs_q=sgs_frames, out_file=quad_out, cadence=10)
    print(f"Saved comparison GIF to {quad_out}")

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
    run()