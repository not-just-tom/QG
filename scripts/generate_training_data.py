"""
Generate a single trajectory (coarsened) according to default configs, saves to disk, and plots it. 
"""
import os
from model.core.steppers import SteppedModel, AB3Stepper
from model.core.model import QGM
from model.utils.logging import configure_logging
from model.ML.utils.coarsen import coarsen
from omegaconf import OmegaConf
import functools
import jax
import jax.numpy as jnp
import logging
import matplotlib.animation as animation 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
#go to parent dir:
PARENT_DIR = os.path.dirname(BASE_DIR)
CONFIG_DEFAULT_PATH = os.path.join(PARENT_DIR, "config", "default.yaml")
    

def main(cfg):
    # load values
    dt = cfg.plotting.dt
    spinup = cfg.plotting.spinup
    nsteps = int(getattr(cfg.plotting, 'nsteps', 0))
    cadence = int(getattr(cfg.plotting, 'cadence', 100))
    validation_rollout = int(getattr(cfg.plotting, 'validation_rollout', 1000))
    params = dict(OmegaConf.to_container(cfg.params, resolve=True))
    params["dt"] = float(dt)
    n_jets = params.get('n_jets', 4)
    seed = params.get("seed", 42)
    key = jax.random.PRNGKey(seed)
    ratio = params["hr_nx"]/params["nx"]

    logger = configure_logging(level=cfg.filepaths.log_level, out_file="../logs/run.log")
    logger = logging.getLogger(__name__)
    
    # GPU or CPU setup 
    device_type = (cfg.ml.device).lower()
    devices = jax.devices()
    gpu_devices = [d for d in devices if d.platform == "gpu"]
    if gpu_devices:
        jax.config.update("jax_platforms", "gpu")
        chosen = "gpu"
    else:
        jax.config.update("jax_platforms", "cpu")
        chosen = "cpu"

    logger.info(f"Requested device: {device_type}, using device: {chosen.upper()}")
    
    # === dataloading === #
    old_dt = dt
    if cfg.plotting.auto_dt == True:
        old_dt = dt
        logger.info("Auto-setting initial dt using CFL condition on a sample initial state.")
        raw_model = QGM({**params, "nx": params['hr_nx']})
        init_state = raw_model.initialise(key, n_jets=n_jets, verbose=True)
        dt = float(raw_model.estimate_cfl_dt(init_state))
        params["dt"] = dt * raw_model.time_scale

    # instantiate the model
    hr_physics_model = QGM({**params, "nx": params['hr_nx']})
    dt = float(hr_physics_model.dt)
    hr_model = SteppedModel(
        model=hr_physics_model,
        stepper=AB3Stepper(dt=dt),
    )
    # build low-resolution physics model (coarsened from high-res physics)
    lr_model = coarsen(hr_model.model, params['nx'])
    low_res_dt = dt * ratio

    lr_init_state = lr_model.initialise(key, n_jets=n_jets, verbose=False)
    lr_rhines_length, lr_u_rms = lr_model.rhines_length(lr_init_state)
    tau_eddy = lr_rhines_length / (lr_u_rms + 1e-12)
    logger.info(
        'Fine timestep is %.2gs and coarsened timestep is %.2gs. '
        'Model spinup for %.2f days (~%.2g eddy turnover times). '
        'Training horizon is %d low-res steps (~%.2f days). '
        'Validation plotting window is %d steps (~%.2f days).',
        dt * hr_physics_model.time_scale,
        low_res_dt * hr_physics_model.time_scale,
        spinup,
        float(tau_eddy * hr_physics_model.time_scale) / (24.0 * 3600.0),
        nsteps,
        nsteps * low_res_dt * hr_physics_model.time_scale / (24.0 * 3600.0),
        validation_rollout,
        validation_rollout * low_res_dt * hr_physics_model.time_scale / (24.0 * 3600.0),
    )

    timing_metadata = {
        'nsteps': int(nsteps),
        "dt (original)": float(old_dt),
        'auto_dt': bool(cfg.plotting.auto_dt),
        'final dt': float(dt),
    }

    @functools.partial(jax.jit, static_argnames=["nsteps"])
    def generate_trajectory(init_state, nsteps):
        """Generate one coarsened trajectory."""
        def _coarsen_state(step_state):
            state = step_state.state

            nk = lr_template.qh.shape[-2] // 2
            trunc = jnp.concatenate(
                [
                    state.qh[:, :nk, :nk + 1],
                    state.qh[:, -nk:, :nk + 1],
                ],
                axis=-2,
            )

            filtered = trunc * lr_model._dealias / ratio
            lr_state = lr_template.update(qh=filtered)

            return lr_state.q

        def step(carry, _):
            def _hr_step(inner_carry, _):
                return hr_model.step_model(inner_carry), None

            next_state, _ = jax.lax.scan(
                _hr_step,
                carry,
                None,
                length=ratio,
            )

            return next_state, _coarsen_state(next_state)

        _, traj_q = jax.lax.scan(
            step,
            init_state,
            None,
            length=nsteps,
        )

        return traj_q
    dummy_key = jax.random.PRNGKey(0)
    lr_template = lr_model.initialise(dummy_key)

    ratio = int(hr_model.model.nx / lr_model.nx)

    nsteps = max(
        int(timing_metadata["nsteps"]),
        cfg.plotting.nsteps,
    )

    spinup_time = hr_model.model.seconds_to_model_time(
        cfg.plotting.spinup * 24 * 3600
    )
    print(f"Spinup time in model units: {spinup_time}")
    spinup = int(spinup_time // hr_model.stepper.dt)
    print(f"model hr dt: {hr_model.stepper.dt}")
    print(f"Spinup in steps: {spinup}")

    # Initialise ONE trajectory
    seed = int(params.get("seed", 0))
    key = jax.random.PRNGKey(seed)

    init_kwargs = {}
    n_jets = params.get("n_jets")
    if n_jets is not None:
        init_kwargs = {
            "n_jets": int(n_jets),
            "pseudo": True,
        }

    state = hr_model.initialise(key, **init_kwargs)

    # Spin up ONE trajectory
    if spinup > 0:
        @functools.partial(jax.jit, static_argnames=["spinup"])
        def spinup_state(state, spinup):
            def step(carry, _):
                return hr_model.step_model(carry), None

            state, _ = jax.lax.scan(
                step,
                state,
                None,
                length=spinup,
            )
            return state

        state = spinup_state(state, spinup)

    # Generate ONE trajectory
    q_traj = generate_trajectory(state, nsteps)

    # Only transfer the final result
    q_traj = jax.device_get(q_traj)

    return q_traj, cadence

def gif_that(q_state, out_file='plotting.gif', cadence=100):
    'just a simple plotting'
    q_state = q_state[::cadence] 
    nt = q_state.shape[0]

    # Determine global vmin/vmax for HR and LR
    fig = plt.figure(figsize=(4, 4), constrained_layout=True)
    gs = gridspec.GridSpec(1, 1, figure=fig)

    ax = fig.add_subplot(gs[0])

    im = ax.imshow(q_state[0, 0], cmap="RdBu_r")#, vmin=-hr_vmax, vmax=hr_vmax)
    ax.set_title("State")
    ax.axis("off")

    def update(frame):
        im.set_array(q_state[frame, 0])
        ax.set_title(f"PV state, step={frame*cadence}")
        return im

    ani = animation.FuncAnimation(
        fig, update, frames=nt, blit=False
    )
    ani.save(out_file, fps=10)


if __name__ == "__main__":
    cfg = OmegaConf.load(CONFIG_DEFAULT_PATH)
    q_traj, cadence = main(cfg)
    gif_that(q_traj, out_file=os.path.join(cfg.filepaths.out_dir, "training_data_plot.png"), cadence=cadence)