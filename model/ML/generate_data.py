import jax
import jax.numpy as jnp
import zarr
import os 
import json
import functools
import logging
import numpy as np
import datetime
from zarr.codecs import BloscCodec

logger = logging.getLogger(__name__)

def generate_train_data(cfg, params, dt, hr_model, lr_model, hr_dir):
    '''Generate zarr training data from the high-res `hr_model` and coarsen
    on-the-fly using `lr_model` as the low-resolution physics template, and lower res dt.
    Saves metadata and trajectories into `hr_dir`.
    '''
    os.makedirs(hr_dir, exist_ok=True)

    # Timing parameters
    try:
        nsteps = cfg.plotting.nsteps
        batch_size = cfg.ml.batch_size
        batch_steps = cfg.ml.batch_steps
        spinup = int(cfg.plotting.spinup)
    except AttributeError as e:
        logger.error("Missing required configuration parameters: %s", e)
        raise ValueError("Configuration must include plotting.nsteps, ml.batch_size, ml.batch_steps, and plotting.spinup") from e

    logger.info(f"Generating %d trajectories with %d steps.", cfg.ml.n_train + cfg.ml.n_test, nsteps)
    
    # Prepare low-resolution template and ratio for coarsening
    dummy_key = jax.random.PRNGKey(0)
    lr_template = lr_model.initialise(dummy_key)
    ratio = int(float(hr_model.model.nx) / float(lr_model.nx))

    # JIT the trajectory generation; closure captures `lr_template`, `lr_model._dealias`, and `ratio`.
    @functools.partial(jax.jit, static_argnames=["nsteps"])
    def generate_trajectory(init_state, nsteps):
        """Generate coarsened trajectory with subsampling."""
        def step(carry, _x):
            next_state = hr_model.step_model(carry)
            state = next_state.state
            # Galerkin truncation to low-res spectral coefficients
            nk = lr_template.qh.shape[-2] // 2
            trunc = jnp.concatenate(
                [
                    state.qh[:, :nk, :nk + 1],
                    state.qh[:, -nk:, :nk + 1],
                ],
                axis=-2,
            )
            filtered = trunc * lr_model._dealias / (ratio ** 2)
            lr_state = lr_template.update(qh=filtered)
            return next_state, lr_state.q

        _, traj_q = jax.lax.scan(step, init_state, None, length=nsteps)
        return traj_q[::ratio]  # Subsample in time to match low-res steps
    
    # Vectorize over trajectories
    batched_traj = jax.jit(
        jax.vmap(generate_trajectory, in_axes=(0, None)),
        static_argnums=(1,),
    )

    timing_metadata = {
        'spinup': int(cfg.plotting.spinup),
        'nsteps': int(nsteps),
        "dt": float(dt),
        'batch_steps': int(cfg.ml.batch_steps),
    }

    metadata = {
        'model_type': cfg.ml.model_type,
        "parameters": params,
        'timing': timing_metadata,
        "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
    }
    
    metadata_path = os.path.join(hr_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    # zarr setup
    zarr_path = os.path.join(hr_dir, "trajectories.zarr")
    z_root = zarr.open_group(zarr_path, mode="w")
    z_root.attrs.update(metadata)

    traj_group = z_root.create_group("trajectories")

    # Zarr v3 codec
    compressor = BloscCodec(
        cname="zstd",
        clevel=3,  # lower level = faster
        shuffle="bitshuffle",
    )
    

    rng = jax.random.PRNGKey(int(params.get("seed", 0)))
    n_total = cfg.ml.n_train + cfg.ml.n_test
    n_generated = 0

    # If spinup>0, define a jitted routine to step the high-res model
    if spinup > 0:
        @functools.partial(jax.jit, static_argnames=["spinup"])
        def _spinup_state(init_state, spinup):
            def _step(carry, _x):
                next_state = hr_model.step_model(carry)
                return next_state, None
            final_state, _ = jax.lax.scan(_step, init_state, None, length=spinup)
            return final_state

        # Vectorize the spinup across the batch; `_spinup_state` already
        # has `spinup` as a static arg via `static_argnames`, so a plain
        # `vmap` over the batch axis is sufficient.
        _spinup_batched = jax.vmap(_spinup_state, in_axes=(0, None))

    while n_generated < n_total:

        current_batch = min(batch_size, n_total - n_generated)

        rng, subkey = jax.random.split(rng)
        keys = jax.random.split(subkey, current_batch)

        init_states = jax.vmap(functools.partial(hr_model.initialise))(keys)
        
        logger.info(f"Initialised batch of {current_batch} trajectories")

        # Run spinup on each initial state if requested
        if spinup > 0:
            logger.info(f"Running spinup of {spinup} steps for the batch")
            init_states = _spinup_batched(init_states, spinup)

        # Generate batch
        traj_batch = batched_traj(init_states, nsteps*ratio) # multiplied so after sumsampling we get nsteps

        logger.info(f"Generating current batch of {current_batch} trajectories, shape: {traj_batch.shape}")
        # Transfer once per batch
        traj_batch = jax.device_get(traj_batch)

        for i in range(current_batch):

            logger.info(f"Processing trajectory {n_generated+i+1}/{n_total}")
            q_traj = traj_batch[i]

            if not np.all(np.isfinite(q_traj)):
                logger.warning(f"NaN detected in trajectory {n_generated+i}")
                continue

            # Optimize chunking by aligning with training window size (batch_steps)
            time_chunk = min(q_traj.shape[0], batch_steps)
            traj_group.create_array(
                f"traj_{n_generated+i:05d}",
                data=q_traj.astype(np.float32),
                chunks=(time_chunk, q_traj.shape[1], q_traj.shape[2], q_traj.shape[3]),
                compressors=[compressor],
            )

        n_generated += current_batch
        logger.info(f"Generated {n_generated}/{n_total} trajectories")

    logger.info("Finished generating all trajectories")
    logger.info(f"Saved to {zarr_path}")
    

    