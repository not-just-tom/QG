import importlib 
import model.core.grid
import model.core.states
import model.core.kernels
import model.core.TwoLayer
import model.core.model
import model.core.steppers
import model.utils.plotting
import model.utils.diagnostics
importlib.reload(model.core.grid)
importlib.reload(model.core.states)
importlib.reload(model.core.kernels)
importlib.reload(model.core.TwoLayer)
importlib.reload(model.core.model)
importlib.reload(model.core.steppers)
importlib.reload(model.utils.diagnostics)
importlib.reload(model.utils.plotting)
from model.utils.plotting import animate
from model.utils.config import Config
from model.utils.logging import configure_logging
from model.core.steppers import SteppedModel, build_stepper
from model.core.model import create_model
from model.utils.diagnostics import Recorder
import logging
import jax
import jax.numpy as jnp
import time
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
    outname = getattr(cfg.filepaths, "outname", "../outputs/qg.gif")

    
    # Instantiate the model from configs using factory
    n_layers = params.pop('n_layers', 1)  # Extract n_layers, default to 1
    model = create_model(params, n_layers=n_layers)
    stepper = build_stepper(cfg_stepper, dt)
    sm = SteppedModel(model=model, stepper=stepper)
    recorder = Recorder(cfg, grid=model.get_grid()) # basically depreciated at this point....
    init_state = sm.initialise(params['seed'])

    print(init_state.state.qh.shape)


    @functools.partial(jax.jit, static_argnames=["nsteps", "cadence"])
    def rollout(state, nsteps, cadence):
        def loop_fn(carry, step):
            next_state = sm.step_model(carry)
            # record spectral qh every `cadence` steps; keep same shape
            q_snapshot = jax.lax.cond(
                step % cadence == 0,
                lambda s: s.state.qh,
                lambda s: jnp.zeros_like(s.state.qh),
                next_state,
            )
            return next_state, q_snapshot

        steps = jnp.arange(nsteps)
        _final_carry, traj_steps = jax.lax.scan(loop_fn, state, steps)
        return traj_steps

    q_traj = rollout(init_state, nsteps, cadence)
    q_traj = jax.device_get(q_traj)  # shape (nsteps, nl, nk)

    # select only the frames recorded at cadence
    indices = np.arange(0, nsteps, cadence)
    q_traj = q_traj[indices]

    # q_traj is spectral with shape (n_frames, nl, nk). Convert to physical (n_frames, ny, nx).
    ny, nx = init_state.state._q_shape
    phys_frames = []
    for i in range(q_traj.shape[0]):
        frame = np.fft.irfftn(q_traj[i], s=(ny, nx))
        phys_frames.append(frame)
    q_traj = np.stack(phys_frames)

    fig, ax = plt.subplots(figsize=(6, 5))

    vmax = np.max(np.abs(q_traj))
    im = ax.imshow(
        q_traj[0, 0],
        origin="lower",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        animated=True,
    )

    ax.set_title("PV evolution")
    plt.colorbar(im, ax=ax)

    def update(frame):
        im.set_data(q_traj[frame, 0, :, :])
        ax.set_title(f"PV evolution (Step {frame*cadence})")
        return (im,)
    
    writer = PillowWriter(fps=10)

    ani = FuncAnimation(
        fig,
        update,
        frames=len(q_traj[:,0,:,:]),
        blit=True,
    )

    ani.save("pv_evolution.gif", writer=writer)
    plt.close(fig)

    # ============================

if __name__ == "__main__":
    main()
     