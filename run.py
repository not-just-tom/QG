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
from model.core.model import create_model
from model.utils.diagnostics import Recorder
import logging
import jax
import time
import functools
import yaml
import os

jax.config.update("jax_enable_x64", True)


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
    state = sm.initialise(params['seed'])


    # Time loop
    logger.info("Starting run: dt=%s, steps=%d", float(dt), nsteps)
    start = time.time()
    for n in range(nsteps+1):
        state = sm.step_model(state) 
        if n % cadence == 0:
            full = sm.get_full_state(state)
            recorder.sample(full)
        if n % (10*cadence) == 0:
            logger.info("Completed step %d / %d", n, nsteps)

    # Run animation if requested in config
    animate_list = list(getattr(getattr(cfg, "diagnostics", {}), "animate", []))
    animate(recorder, model.get_grid(), cadence=cadence, outname=outname, plots=animate_list)
    logger.info("Plotting complete at time %.2fs", time.time() - start)

    

if __name__ == "__main__":
    main()


    