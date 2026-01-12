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
from model.utils.config import Config
from model.utils.diagnostics import Recorder
from model.utils.logging import configure_logging
from model.core.solver import Solver
from model.core.model import QGM
from model.utils.plotting import animate_model

import logging
import time
import jax
import jax.numpy as jnp
from jax.numpy.fft import irfftn





def main():
    cfg = Config.load_config(CONFIG_DEFAULT_PATH)

    logger = configure_logging(level=cfg.filepaths.log_level, out_file="../logs/run.log")
    logger = logging.getLogger(__name__)

    # build grid + load params for initial condition # not complete - just bare min to get psuedo-rand goin
    grid = Grid(cfg)
    
    
    # === build the ml closure === # 
    if cfg.ml.enabled is True:
        import model.ML.architectures
        importlib.reload(model.ML.architectures)
        from model.ML.architectures import build_net

        net_factory = build_net(cfg.ml.model)
        key = jax.random.PRNGKey(int(cfg.params.seed))
        net = net_factory(key=key) # yeah this definitely needs so be standardised 

        #  this also might just need to be in the actual net definition later?
        def ml_closure(qh, net=net, grid=grid):
            q = irfftn(qh, axes=(-2, -1), norm='ortho').real
            out = net(q)
            out_h = jax.numpy.fft.rfftn(out, axes=(-2, -1), norm='ortho')
            out_h = Solver.dealias(out_h, grid, s=8)
            return out_h
    else:
        ml_closure = None

    # ===== build the model ===== #
    model = QGM(cfg, ml_closure=ml_closure)

    dt = float(cfg.params.dt)
    tmax = float(cfg.params.tmax or 50.0)
    cadence = int(getattr(getattr(cfg, "diagnostics", {}), 'cadence', 100) or 100)
    steps = int(float(tmax) / float(dt))
    
    recorder = Recorder(cfg)

    # === loop over time ===
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