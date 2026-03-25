import warnings
warnings.filterwarnings(
    "ignore",
    message=r"Explicitly requested dtype float64 requested in astype is not available.*",
    category=UserWarning,
)
import os
import importlib
from model.utils.config import Config
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
CONFIG_DEFAULT_PATH = os.path.join(BASE_DIR, "config", "default.yaml")
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
import model.core.model
import model.core.steppers
import model.utils.plotting
import model.utils.diagnostics
importlib.reload(model.core.grid)
importlib.reload(model.core.states)
importlib.reload(model.core.kernel)
importlib.reload(model.core.model)
importlib.reload(model.core.model)
importlib.reload(model.core.steppers)
importlib.reload(model.utils.diagnostics)
importlib.reload(model.utils.plotting)
from model.utils.logging import configure_logging
from model.core.steppers import SteppedModel, AB3Stepper
from model.core.model import QGM
from model.utils.diagnostics import Animator
from model.utils.plotting import find_output_dir
import logging
import jax
import jax.numpy as jnp
import functools
import yaml
import numpy as np

# === just loading in params as dict === #
with open(CONFIG_DEFAULT_PATH) as f:
    cfg_dict = yaml.safe_load(f)

params = dict(cfg_dict["params"])   # pure dict for JAX  

# =========================================
# Main loop to run from Command Line 
# =========================================
def main():  
    # setup logging
    logger = configure_logging(level=cfg.filepaths.log_level, out_file="../logs/run.log")
    logger = logging.getLogger(__name__)

    # load config values
    nsteps = cfg.plotting.nsteps
    cadence = cfg.plotting.cadence
    njets = cfg.plotting.njets
    spinup = cfg.plotting.spinup
    key = jax.random.PRNGKey(params['seed'])
    model = QGM(params)
    # Instantiate the model from configs using factory
    if cfg.plotting.auto_dt:
        logger.info("Auto-setting initial dt using CFL condition on a sample initial state.")
        init_state = model.initialise(key, tune=True, n_jets=njets, verbose=True)
        dt = model.estimate_cfl_dt(init_state)

    sm = SteppedModel(model=model, stepper=AB3Stepper(dt))
    init_state = sm.initialise(key)


    @functools.partial(jax.jit, static_argnames=["nsteps", "cadence"])
    def rollout(state, nsteps, cadence):
        def loop_fn(carry, step):
            next_state = sm.step_model(carry)
            # record spectral qh every cadence steps 
            q_snapshot = jax.lax.cond(
                step % cadence == 0,
                lambda s: s.state.qh,
                lambda s: jnp.zeros_like(s.state.qh),
                next_state,
            )
            return next_state, q_snapshot

        steps = jnp.arange(nsteps)
        _final_carry, traj_steps = jax.lax.scan(loop_fn, state, steps)
        return _final_carry, traj_steps

    if spinup != 0:
        logger.info(f"Running spinup for {spinup} steps.")
        init_state, _ = rollout(init_state, spinup, cadence)
    _, q_traj = rollout(init_state, nsteps, cadence)
    q_traj = jax.device_get(q_traj)  # shape (nsteps, nz, nl, nk)

    # select only the frames recorded at cadence
    indices = np.arange(0, nsteps, cadence)
    q_traj = q_traj[indices]

    # Animator and animation
    outbase = os.path.join(cfg.filepaths.out_dir)
    run_dir, found = find_output_dir(outbase, params)
    if found:
        logger.info(f"Found existing output directory with matching parameters, replacing the original.")
    else:
        # Ensure the output directory exists when creating a new run directory
        os.makedirs(run_dir, exist_ok=True)

    Animator = Animator(cfg, sm)
    Animator.animate(cfg, run_dir, q_traj)
    Animator.plot_final(run_dir)

    # ============================

if __name__ == "__main__":
    main()
     