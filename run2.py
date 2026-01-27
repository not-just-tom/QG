import importlib 
import model.core.model
import model.core.states
import model.core.grid
import model.core.steppers
import model.utils.plotting
import model.utils.diagnostics
importlib.reload(model.core.states)
importlib.reload(model.core.model)
importlib.reload(model.core.grid)
importlib.reload(model.core.steppers)
importlib.reload(model.utils.diagnostics)
importlib.reload(model.utils.plotting)
from model.utils.plotting import animate
from model.utils.config import Config
from model.utils.logging import configure_logging
from model.core.steppers import SteppedModel, build_stepper
from model.core.model import QGM
from model.utils.diagnostics import Recorder
import logging
import jax
import time
import functools
import yaml
import os

#delete
import operator
import functools
import matplotlib.pyplot as plt
import cmocean.cm as cmo
import jax
import jax.numpy as jnp


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
    outname = getattr(cfg.filepaths, "outname", "../outputs/qg.gif")

    

    # Instantiate the model from configs
    model = QGM(params=params)
    stepper = build_stepper(cfg_stepper, dt)
    sm = SteppedModel(model=model, stepper=stepper)
    recorder = Recorder(cfg, grid=model.get_grid())
    init_state = sm.initialise(params['seed'])

    @jax.jit
    def step_only(state):
        return sm.step_model(state)


    def rollout(state, nsteps, cadence):

        def loop_fn(carry, step):
            next_state = step_only(carry)

            # store only every `cadence` steps
            q_snapshot = jax.lax.cond(
                step % cadence == 0,
                lambda s: s.state.q,   # cheap: already in real space
                lambda _: jnp.zeros_like(carry.state.q),
                next_state,
            )

            return next_state, q_snapshot

        final_state, q_traj = jax.lax.scan(
            loop_fn, state, jnp.arange(nsteps)
        )

        return q_traj

    q_traj = rollout(init_state, nsteps, cadence)
    q_traj = jax.device_get(q_traj)
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 5))

    vmax = np.max(np.abs(q_traj))
    im = ax.imshow(
        q_traj[0],
        origin="lower",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        animated=True,
    )

    ax.set_title("PV evolution")
    plt.colorbar(im, ax=ax)

    def update(frame):
        im.set_data(q_traj[frame])
        ax.set_title(f"PV evolution (frame {frame})")
        return (im,)
    
    writer = FFMpegWriter(
        fps=30,
        metadata=dict(artist="you"),
        bitrate=1800,
    )

    ani = FuncAnimation(
        fig,
        update,
        frames=len(q_traj),
        blit=True,
    )

    ani.save("pv_evolution.mp4", writer=writer)
    plt.close(fig)




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

    # Create initial state and set recorder
    logger.info(type(sm.stepper))

    traj = roll_out_state(init_state, num_steps=70)
    # pick the last step in the trajectory
    final_state = jax.tree.map(operator.itemgetter(-1), traj)  # this works because traj is a list of StepperState-like objects

    # get full real-space state
    full_final = sm.get_full_state(final_state)
    final_q = full_final.q  # shape now (ny, nx)

    # plot
    fig, ax = plt.subplots(layout="constrained")
    vmax = jnp.abs(final_q).max()
    ax.set_title("Final PV")
    ax.imshow(final_q, cmap=cmo.balance, vmin=-vmax, vmax=vmax)
    plt.show()

    # ============================

     