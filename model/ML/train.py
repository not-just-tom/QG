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
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

def make_triple_gif(hr_q, lr_q, sgs_q, out_file="q_triple.gif", interval=50):
    """
    Create a GIF with 3 panels side by side:
    - hr_q: high-res field, shape (nt, ny, nx)
    - lr_q: low-res field, shape (nt, ny, nx)
    - sgs_q: SGS forcing, shape (nt, ny, nx)
    """
    nt = hr_q.shape[0]

    # Determine global vmin/vmax for HR and LR
    hr_vmax = jnp.max(jnp.abs(hr_q))
    lr_vmax = jnp.max(jnp.abs(lr_q))
    sgs_vmax = jnp.max(jnp.abs(sgs_q))

    fig = plt.figure(figsize=(12, 4), constrained_layout=True)
    gs = gridspec.GridSpec(1, 3, figure=fig)

    ax_hr = fig.add_subplot(gs[0])
    ax_lr = fig.add_subplot(gs[1])
    ax_sgs = fig.add_subplot(gs[2])

    im_hr = ax_hr.imshow(hr_q[0], cmap=cmo.balance, vmin=-hr_vmax, vmax=hr_vmax)
    ax_hr.set_title("High-Res")
    ax_hr.axis("off")

    im_lr = ax_lr.imshow(lr_q[0], cmap=cmo.balance, vmin=-lr_vmax, vmax=lr_vmax)
    ax_lr.set_title("Low-Res")
    ax_lr.axis("off")

    im_sgs = ax_sgs.imshow(sgs_q[0], cmap=cmo.curl, vmin=-sgs_vmax, vmax=sgs_vmax)
    ax_sgs.set_title("SGS Forcing")
    ax_sgs.axis("off")

    def update(frame):
        im_hr.set_array(hr_q[frame])
        im_lr.set_array(lr_q[frame])
        im_sgs.set_array(sgs_q[frame])
        ax_hr.set_title(f"High-Res t={frame}")
        ax_lr.set_title(f"Low-Res t={frame}")
        ax_sgs.set_title(f"SGS t={frame}")
        return [im_hr, im_lr, im_sgs]

    ani = animation.FuncAnimation(
        fig, update, frames=nt, interval=interval, blit=False
    )
    ani.save(out_file, fps=10)
    plt.close(fig)
    print(f"Saved GIF to {out_file}")


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
    @functools.partial(jax.jit, static_argnames=["num_steps"])
    def roll_out_state(state, num_steps):

        def loop_fn(carry, _x):
            current_state = carry
            next_state = sm.step_model(current_state)
            # Note: we output the current state for ys
            # This includes the starting step in the trajectory
            return next_state, current_state

        _final_carry, traj_steps = jax.lax.scan(
            loop_fn, state, None, length=num_steps
        )
        return traj_steps

    final_state = roll_out_state(
       init_state, num_steps=7500
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



    make_triple_gif(final_state.q[::cadence], lr_state.q[::cadence], sgs_forcing[::cadence], out_file="../outputs/q_triple.gif", interval=200)


if __name__ == "__main__":
    run()