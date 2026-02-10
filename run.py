import importlib 
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
from model.utils.config import Config
from model.utils.logging import configure_logging
from model.core.steppers import SteppedModel, build_stepper
from model.core.model import QGM
from model.utils.diagnostics import Recorder
import logging
import jax
import jax.numpy as jnp
import functools
import yaml
import cmocean as cmo
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
CONFIG_DEFAULT_PATH = os.path.join(BASE_DIR, "QG", "config", "default.yaml")

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
    nsteps = cfg.plotting.nsteps
    cadence = cfg.plotting.cadence
    cfg_stepper = cfg.plotting.stepper
    
    # Instantiate the model from configs using factory
    model = QGM(params)
    stepper = build_stepper(cfg_stepper, dt) # not used really, maybe in future
    sm = SteppedModel(model=model, stepper=stepper)
    init_state = sm.initialise(params['seed'], tune=True, n_jets=16)
    recorder = Recorder(cfg, sm)

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

    _, q_traj = rollout(init_state, nsteps, cadence)
    q_traj = jax.device_get(q_traj)  # shape (nsteps, nz, nl, nk)

    # select only the frames recorded at cadence
    indices = np.arange(0, nsteps, cadence)
    q_traj = q_traj[indices]

    # recorder and animation 
    recorder.finalize_from_spectral(q_traj)
    recorder.animate(cfg, q_traj)
    recorder.plot_final()

    # ============================

if __name__ == "__main__":
    main()
     