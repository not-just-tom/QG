import os
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
CONFIG_DEFAULT_PATH = os.path.join(BASE_DIR,"config", "default.yaml")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
MODEL_DIR = os.path.join(BASE_DIR, 'results', 'closures')
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
from model.utils.plotting import make_triple_gif, gif_that
from model.ML.utils.dataloading import find_existing_closure, find_existing_run, ZarrDataLoader, checkpointer
from model.ML.architectures.build_model import build_closure
from model.ML.utils.utils import module_to_single, parameterization
from model.ML.train import roll_out_with_forced_model
from model.ML.forced_model import ForcedModel
from model.core.model import QGM
from model.ML.utils.coarsen import Coarsen


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
    cadence = cfg.plotting.cadence
    outname = getattr(cfg.filepaths, "outname", "../outputs/qg_closure_comparison.gif")

    # locate model checkpoint
    timing_metadata = {
        'spinup': int(cfg.plotting.spinup),
        'nsteps': int(cfg.plotting.nsteps),
        "dt": float(cfg.plotting.dt), 
        'batch_steps': int(cfg.ml.batch_steps),
    }

    model_dir, found = find_existing_closure(MODEL_DIR, params, timing_metadata, cfg.ml.model_type)
    loaded_leaves, loaded_optim, ckpt_meta, loaded_loss_history = None, None, None, None
    if found:
        logger.info(f"Found existing closure with matching parameters at {model_dir}, attempting to load checkpoint.")
        try:
            loaded_leaves, loaded_optim, ckpt_meta, loaded_loss_history = checkpointer(None, None, model_dir, save=False)
        except Exception:
            logger.exception("Failed to load checkpoint.")

    # Build closure template and reconstruct if checkpoint available
    closure_template = build_closure(cfg)
    if loaded_leaves is not None:
        try:
            template_params, template_static = eqx.partition(closure_template, eqx.is_array)
            tpl_leaves, tpl_treedef = jax.tree_util.tree_flatten(template_params)
            if len(tpl_leaves) != len(loaded_leaves):
                raise ValueError("Checkpoint parameter count mismatch with template")
            new_leaves = []
            for tpl, ld in zip(tpl_leaves, loaded_leaves):
                arr = np.asarray(ld)
                try:
                    arr = arr.astype(np.asarray(tpl).dtype)
                except Exception:
                    pass
                new_leaves.append(jax.device_put(arr))
            new_params = jax.tree_util.tree_unflatten(tpl_treedef, new_leaves)
            closure_model = eqx.combine(new_params, template_static)
            logger.info("Reconstructed closure from checkpoint parameters")
        except Exception:
            logger.exception("Failed to reconstruct closure from checkpoint; using fresh template")
            closure_model = closure_template
    else:
        closure_model = closure_template

    # make single-precision closure suitable for inference
    closure = module_to_single(closure_model)

    # build HR model and coarsener
    dt = cfg.plotting.dt
    hr_model = SteppedModel(model=QGM({**params, "nx": params['hr_nx']}), stepper=AB3Stepper(dt=dt))
    coarse = Coarsen(hr_model, params['nx'])

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
        model=ForcedModel(model=coarse.lr_model, closure=closure_func, init_param_aux_func=init_param_func),
        stepper=hr_model.stepper,
    )

    # template state for the forced model (low-res initialiser)
    template_state = coarse.lr_model.initialise(jax.random.PRNGKey(0))

    # Use the first coarsened frame as init and roll out for full length
    init_q = jnp.asarray(truth_traj[0])
    pred_traj = roll_out_with_forced_model(init_q, forced_hr_static, template_state, nsteps, closure_params)
    pred_traj = np.asarray(pred_traj)

    # Create GIF comparing coarsened truth, ML-predicted, and their difference
    layer = 0
    hr_frames = np.asarray(truth_traj[:, layer])
    pred_frames = np.asarray(pred_traj[:, layer])
    diff_frames = pred_frames - hr_frames

    gif_out = os.path.join("..", "outputs", "PV.gif")
    # make_triple_gif expects arrays with shape (nt, ny, nx)
    gif_that(hr_frames, out_file=gif_out, cadence=100)
    make_triple_gif(hr_frames, pred_frames, diff_frames, out_file='triple.gif', cadence=100)
    logger.info(f"Saved comparison GIF to {gif_out}")

    try:
        mse_per_timestep = np.mean((pred_frames - hr_frames) ** 2, axis=(1, 2))
        mse_out = os.path.join("..", "outputs", "mse_per_timestep.png")
        plt.figure(figsize=(6, 3))
        plt.plot(np.arange(mse_per_timestep.size), mse_per_timestep, '-o')
        plt.xlabel('Timestep')
        plt.ylabel('MSE')
        plt.title('MSE per timestep: prediction vs truth')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(mse_out, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved MSE plot to {mse_out}")
    except Exception:
        logger.exception("Failed to compute or save MSE plot")


if __name__ == "__main__":
    run()