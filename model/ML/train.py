"""Small demo training loop for spectral correction model.

This provides a convenience function `train_spectral_demo` that synthesizes a
"truth" spectral correction and fits the spectral parameters to it using SGD.
It is intentionally simple and meant as a starting point for more advanced
training (e.g., differentiable time integration, multi-step losses, etc.).
"""
import os
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
CONFIG_DEFAULT_PATH = os.path.join(BASE_DIR,"config", "default.yaml")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import importlib 
import model.core.model
import model.ML.utils.coarsen
importlib.reload(model.core.model)
importlib.reload(model.ML.utils.coarsen)
from model.core.model import QGM
from model.utils.diagnostics import Recorder
from model.ML.utils.coarsen import Coarsener
import logging
import pathlib
import yaml
import jax
import functools
import cmocean.cm as cmo
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from model.utils.config import Config
from model.utils.logging import configure_logging
from model.core.steppers import SteppedModel, build_stepper
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from model.utils.plotting import make_triple_gif


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
    configure_logging(level=cfg.filepaths.log_level, out_file="../logs/run.log") #return to this to put numbers on it 
    logger = logging.getLogger("main")

    # load config values
    dt = cfg.plotting.dt
    nsteps = cfg.plotting.nsteps
    cadence = cfg.plotting.cadence
    cfg_stepper = cfg.plotting.stepper
    outname = getattr(cfg.filepaths, "outname", "../outputs/qg.gif")

    # Instantiate the model from configs
    model = QGM(params=params)
    stepper = build_stepper(cfg_stepper, dt)
    sm = SteppedModel(model=model, stepper=stepper)
    recorder = Recorder(cfg, grid=model.get_grid())
    init_state = sm.initialise(params['seed'])
    coarsener = Coarsener(hr_model=model, n_lr=32)
    
   # === jax.jit functionality === #
    @functools.partial(jax.jit, static_argnames=["nsteps"])
    def roll_out_state(state, nsteps):

        def loop_fn(carry, _x):
            current_state = carry
            next_state = sm.step_model(current_state)
            # Note: we output the current state for ys
            # This includes the starting step in the trajectory
            return next_state, current_state

        _final_carry, traj_steps = jax.lax.scan(
            loop_fn, state, None, length=nsteps
        )
        return traj_steps

    final_state = roll_out_state(
       init_state, nsteps=nsteps
    ).state

    @jax.jit
    def compute_small(state):
        return coarsener.coarsen_state(state), coarsener.sgs_forcing(state)
    
    lr_state, sgs_forcing = compute_small(final_state)

    q_vmax = max(jnp.abs(s.q[0]).max() for s in [final_state, lr_state])
    f_vmax = max(jnp.abs(f[0]).max() for f in [sgs_forcing])

    fig = plt.figure(layout="tight")
    gs = gridspec.GridSpec(1, 3)

    # Plot large image
    ax = fig.add_subplot(gs[0])
    ax.imshow(final_state.q[7500], cmap=cmo.balance)#, vmin=-q_vmax, vmax=q_vmax)
    ax.set_title("High Res State")

    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(lr_state.q[7500], cmap=cmo.balance)#, vmin=-q_vmax, vmax=q_vmax)
    ax1.set_title("Low Res State")

    ax2 = fig.add_subplot(gs[2])
    ax2.imshow(sgs_forcing[7500], cmap=cmo.curl, vmin=-f_vmax, vmax=f_vmax)
    ax2.set_title("SGS Forcing")

    make_triple_gif(final_state.q[::cadence], lr_state.q[::cadence], sgs_forcing[::cadence], out_file="../outputs/q_triple.gif", cadence=100)


if __name__ == "__main__":
    run()