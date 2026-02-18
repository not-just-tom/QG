import importlib 
import model.core.grid
import model.core.states
import model.core.kernel
import model.core.model
import model.core.model
import model.core.steppers
import model.ML.generate_data
import model.ML.utils.coarsen
import model.utils.plotting
import model.utils.diagnostics
import model.utils.dataloading
importlib.reload(model.core.grid)
importlib.reload(model.core.states)
importlib.reload(model.core.kernel)
importlib.reload(model.core.model)
importlib.reload(model.core.model)
importlib.reload(model.core.steppers)
importlib.reload(model.ML.generate_data)
importlib.reload(model.ML.utils.coarsen)
importlib.reload(model.utils.dataloading)
importlib.reload(model.utils.diagnostics)
importlib.reload(model.utils.plotting)
from model.ML.utils.coarsen import coarsen_state
from model.ML.generate_data import generate_train_data
from model.utils.dataloading import metadata_matches, find_existing_run
from model.utils.config import Config
from model.utils.logging import configure_logging
from model.core.steppers import SteppedModel, AB3Stepper
from model.core.model import QGM
from model.utils.diagnostics import Recorder
import logging
import json
import jax
import jax.numpy as jnp
import functools
import yaml
import os
import numpy as np
import random
import string
import hashlib

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CONFIG_DEFAULT_PATH = os.path.join(BASE_DIR, "config", "default.yaml")
DATA_DIR = os.path.join(BASE_DIR, "data")

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
    
    # Instantiate the model from configs using factory
    model = QGM(params)
    stepper = AB3Stepper(dt)
    sm = SteppedModel(model=model, stepper=stepper)

    # I have to initialise with a hr to downsample, so generate this too 
    hr_model = QGM({**params, "nx": params['hr_nx']})
    hr_init_state = hr_model.initialise(params['seed'], tune=True, n_jets=16)
    init_state = model.set_initial(hr_init_state.qh)


    # === dataloading === #
    params_json = json.dumps(params, sort_keys=True, separators=(',', ':'))
    params_hash = hashlib.sha256(params_json.encode('utf-8')).hexdigest()[:8]

    # Try to find an existing run with the same params
    found_dir, found_meta_path, found_meta = find_existing_run(DATA_DIR, params['hr_nx'], params['nx'], params_hash)
    if found_dir is not None:
        metadata_path = found_meta_path
        stored_meta = found_meta
        if not metadata_matches(params, stored_meta):
            raise RuntimeError(f"Metadata mismatch for {found_dir} - check {metadata_path}")
        # =================================
        # dataloading will go here
        # =================================
    else:
        suffix = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(2))
        dirname = f"data_hr{params['hr_nx']}_nx{params['nx']}_{suffix}"
        hr_dir = os.path.join(DATA_DIR, dirname)
        logger.info(f"No existing run found, generating new train data at {hr_dir}.zarr")

        # Now generate the training data
        generate_train_data(params, hr_init_state, hr_dir)
        logger.info(f"Finished generating training data at {hr_dir}")


        # =================================
        # THEN same dataloading will go here
        # =================================



        
    
    '''I left it here, looking at the best method to coarsen the data
    got weird with the coarsener wrapper bs i might have to figure out
    '''



    

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

    #init_state, _ = rollout(init_state, 100000, cadence)
    _, q_traj = rollout(init_state, nsteps, cadence)
    q_traj = jax.device_get(q_traj)  # shape (nsteps, nz, nl, nk)

    # select only the frames recorded at cadence
    indices = np.arange(0, nsteps, cadence)
    q_traj = q_traj[indices]

    # recorder and animation
    recorder = Recorder(cfg, sm)
    recorder.animate(cfg, q_traj)
    outbase = os.path.join(cfg.filepaths.out_dir, "diagnostics")
    recorder.plot_final(outbase)

    # ============================

if __name__ == "__main__":
    main()
     