import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
CONFIG_DEFAULT_PATH = os.path.join(BASE_DIR, "QG", "config", "default.yaml")

import importlib 
import model.core.model
importlib.reload(model.core.model)
import model.utils.diagnostics
importlib.reload(model.utils.diagnostics)
import model.utils.plotting
importlib.reload(model.utils.plotting)
from model.core.grid import Grid
from model.utils.config import load_config
from model.utils.diagnostics import Recorder
from model.utils.logging import get_logger
from model.core.solver import Solver
from model.core.model import QGM
from model.utils.plotting import animate_model

import sys
import time
import jax
import jax.numpy as jnp
from jax.numpy.fft import irfftn

logger = get_logger(__name__)


def _params_from_config(cfg):
    params = {}
    # grid parameters (Lx, Ly, nx, ny)
    params.update(cfg.grid)
    # time step
    try:
        params["dt"] = float(cfg.time.dt)
    except Exception:
        params["dt"] = 1e-3
    # physics / forcing (support SimpleNamespace or dict)
    phys = getattr(cfg, "physics", None)
    if phys is None:
        params["beta"] = 1.0
    else:
        params["beta"] = getattr(phys, "beta", 1.0)

    forcing = getattr(cfg, "forcing", None)
    params["k_f"] = getattr(forcing, "k_f", 8.0)
    params["k_width"] = getattr(forcing, "k_width", 2.0)
    params["epsilon"] = getattr(forcing, "epsilon", 1e-3)
    return params


def main():
    cfg = load_config(CONFIG_DEFAULT_PATH)

    params = _params_from_config(cfg)

    # build grid + load params for initial condition # not complete - just bare min to get psuedo-rand goin
    grid = Grid(params)
    forcing = getattr(cfg, "forcing")

    # generate initial state
    if custom_init := getattr(cfg, "initial_condition", None):
        logger.info("Using custom initial condition from config")
        # === i dont use this yet but i need a check to make sure it fits the grid ===
        raise NotImplementedError("Custom not implemented yet, needs grid shape check")
    elif forcing is None:
        logger.info("Using random initial, as forcing absent")
        key = jax.random.PRNGKey(0) 
        key, k1, k2 = jax.random.split(key)
        noise_real = jax.random.normal(k1, (grid.ny, grid.nx // 2 + 1)) 
        noise_imag = jax.random.normal(k2, (grid.ny, grid.nx // 2 + 1)) 
        qh = noise_real + 1j * noise_imag 
        qh = qh.at[:, 0].set(jnp.real(qh[:, 0]))  
        initial = Solver.dealias(qh, grid, s=8) # dealiasing
    else:
        logger.info("Using pseudo-random initial generated from forcing wavenumber")
        k_f = getattr(forcing, 'k_f', 8.0)
        kmin = getattr(forcing, 'kmin', 4.0)
        kmax = getattr(forcing, 'kmax', 10.0)
        key = jax.random.PRNGKey(0) 
        key, k1 = jax.random.split(key)

        # pseudo-random initial spectrum peaked at k_f
        qh = Solver.pseudo_randomiser(grid, k_f, k1)
        qh = Solver.dealias(qh, grid, s=8)

        # --- band-pass filter ---
        band_mask = (grid.Kmag >= kmin) & (grid.Kmag <= kmax)
        qh = qh * band_mask
        qh = qh.at[:, 0].set(0.0)

        #  physical-space initial field
        initial = irfftn(qh, axes=(-2, -1), norm='ortho').real
    model = QGM(params, initial, grid)

    # initialize spectral state
    seed = getattr(forcing, "seed", 0) if hasattr(cfg, 'forcing') else 0 # check here I don't want to move the seed to elsewhere and if you're looking at this in the future then check you haven't and its defaulting 
    key = jax.random.PRNGKey(int(seed))
    key, key0 = jax.random.split(key)
    qh = Solver.pseudo_randomiser(grid, params.get("k_f", 8.0), key0)


    dt = float(params["dt"])
    tmax = float(getattr(getattr(cfg, "timestep", {}), 'tmax', 5.0) or 5.0)
    cadence = int(getattr(getattr(cfg, "diagnostics", {}), 'cadence', 100) or 100)
    steps = int(float(tmax) / float(dt))
    
    recorder = Recorder(cfg)

    # === loop time ===
    n = 0
    logger.info("Starting run: dt=%s, steps=%d", float(dt), steps)
    start = time.time()
    model.initialize()

    for _ in range((steps // cadence)+1):
        model.steps(cadence)
        # sample diagnostics 
        recorder.sample(model)
        if n % 1000 == 0:
            logger.info("Completed step %d / %d", n, steps)
        n += cadence

    logger.info("Finished run in %.2fs", time.time() - start)

    # final sample to ensure buffers populated
    recorder.sample(model)

    # Run animation if requested in config
    animate_list = list(getattr(getattr(cfg, "diagnostics", {}), "animate", []))
    if animate_list:
        outname = getattr(getattr(cfg, "diagnostics", {}), "outname", "../outputs/qg.gif")
        animate_model(model, recorder, nsteps=steps, frame_interval=cadence, outname=outname, plots=animate_list)
    logger.info("Plotting complete at time %.2fs", time.time() - start)

if __name__ == "__main__":
    main()