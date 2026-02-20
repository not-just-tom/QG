import jax
import jax.numpy as jnp
import zarr
import os 
import json
import functools
import logging
import math
import numpy as np
import datetime
from zarr.codecs import BloscCodec

logger = logging.getLogger(__name__)

def generate_train_data(cfg, params, hr_model, coarse, hr_dir):
    '''aiming for this to generate the zarr file for hr training data, and 
    also save the metadata for the run in a json file.'''

    os.makedirs(hr_dir, exist_ok=True)

    
    # Timing parameters
    dt = cfg.plotting.dt
    num_steps = cfg.plotting.nsteps
    cadence = cfg.plotting.cadence if hasattr(cfg.plotting, 'cadence') else 1
    batch_size = getattr(cfg.ml, "batch_size", 5)
    
    logger.info(f"Generating {cfg.ml.n_trajs} trajectories with {num_steps} steps "
                f"(cadence: {cadence})")
    
    # JIT the trajectory generation
    @functools.partial(jax.jit, static_argnames=["nsteps", "cadence"])
    def generate_trajectory(init_state, nsteps, cadence):
        """Generate coarsened trajectory with subsampling."""
        def step(carry, _x):
            next_state = hr_model.step_model(carry)
            lr_state = coarse.coarsen_state(next_state.state)
            return next_state, lr_state.q
        
        _, traj_q = jax.lax.scan(step, init_state, None, length=nsteps)
        # cadence: keep every cadence-th step
        return traj_q[::cadence]
    
    # Vectorize over trajectories
    batched_traj = jax.jit(
        jax.vmap(generate_trajectory, in_axes=(0, None, None)),
        static_argnums=(1, 2),
    )
    
    # Metadata and HDF5 setup
    metadata = {
        "parameters": params,
        "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "dt": float(dt),
        "cadence": int(cadence),
    }
    
    metadata_path = os.path.join(hr_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    # zarr setup
    zarr_path = os.path.join(hr_dir, "trajectories.zarr")
    z_root = zarr.open_group(zarr_path, mode="w")
    z_root.attrs.update(metadata)

    traj_group = z_root.create_group("trajectories")

    # Zarr v3 codec (use list of codecs)
    compressor = BloscCodec(
        cname="zstd",
        clevel=3,  # lower level = faster
        shuffle="bitshuffle",
    )
    
    # trajectory loop 
    rng = jax.random.key(params["seed"])
    n_total = cfg.ml.n_trajs
    n_generated = 0

    while n_generated < n_total:

        current_batch = min(batch_size, n_total - n_generated)

        rng, subkey = jax.random.split(rng)
        keys = jax.random.split(subkey, current_batch)

        # Vectorized initialisation
        init_states = []
        for k in keys:
            state = hr_model.initialise(k, tune=True, n_jets=16)
            init_states.append(state)

        init_states = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs),
            *init_states
        )
        logger.info(f"Initialised batch of {current_batch} trajectories")

        # Generate batch
        traj_batch = batched_traj(init_states, num_steps, cadence)

        logger.info(f"Generated batch of {current_batch} trajectories, shape: {traj_batch.shape}")
        # Transfer once per batch
        traj_batch = jax.device_get(traj_batch)

        for i in range(current_batch):

            logger.info(f"Processing trajectory {n_generated+i+1}/{n_total}")
            q_traj = traj_batch[i]

            if not np.all(np.isfinite(q_traj)):
                logger.warning(f"NaN detected in trajectory {n_generated+i}")
                continue

            traj_group.create_array(
                f"traj_{n_generated+i:05d}",
                data=q_traj.astype(np.float32),
                chunks=(1, q_traj.shape[1], q_traj.shape[2], q_traj.shape[3]),
                compressors=[compressor],  # v3 uses list of compressors
            )

        n_generated += current_batch
        logger.info(f"Generated {n_generated}/{n_total} trajectories")

    logger.info("Finished generating all trajectories")
    logger.info(f"Saved to {zarr_path}")
    

    