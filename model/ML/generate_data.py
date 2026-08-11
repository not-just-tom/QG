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

def generate_train_data(cfg, params, timing_metadata, hr_model, lr_model, hr_dir):
    '''Generate zarr training data from the high-res `hr_model` and coarsen
    on-the-fly using `lr_model` as the low-resolution physics template, and lower res dt.
    Saves metadata and trajectories into `hr_dir`.
    '''

    # Timing parameters
    n_total = cfg.ml.n_train + cfg.ml.n_test + 1 # one for validation
    batch_size = 11 # hardcoded bc it was confusing me. It's just the trajs generated in batches
    spinup = int(cfg.plotting.spinup * 24 * 3600 // hr_model.stepper.dt)
    # Prepare low-resolution template and ratio for coarsening
    dummy_key = jax.random.PRNGKey(0)
    lr_template = lr_model.initialise(dummy_key)
    ratio = int(float(hr_model.model.nx) / float(lr_model.nx))
    nsteps = max(int(timing_metadata.get("nsteps")), cfg.plotting.nsteps) # ensure we generate at least as many steps as needed for plotting diagnostics, even if timing metadata is shorter.
    logger.info(
        "Generating %d trajectories with %d low-res steps.",
        n_total,
        nsteps,
    )

    @functools.partial(jax.jit, static_argnames=["nsteps"])
    def generate_trajectory(init_state, nsteps):
        """Generate coarsened trajectory with one coarsen per coarse sample."""
        def _coarsen_state(step_state):
            state = step_state.state
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
            return lr_state.q

        def step(carry, _x):
            # Advance ratio high-res steps, then emit one low-res sample.
            def _hr_step(inner_carry, _):
                return hr_model.step_model(inner_carry), None

            next_state, _ = jax.lax.scan(_hr_step, carry, None, length=ratio)
            return next_state, _coarsen_state(next_state)

        _, traj_q = jax.lax.scan(step, init_state, None, length=nsteps)
        return traj_q
    
    # Vectorize over trajectories
    batched_traj = jax.jit(
        jax.vmap(generate_trajectory, in_axes=(0, None)),
        static_argnums=(1,),
    )

    metadata = {
        "parameters": params,
        'timing': timing_metadata,
        "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
    }
    
    metadata_path = os.path.join(hr_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    # zarr setup
    zarr_path = os.path.join(hr_dir, "trajectories.zarr")
    z_root = zarr.open_group(zarr_path, mode='a')
    z_root.attrs.update(metadata)

    traj_group = z_root.require_group("trajectories")

    # Zarr v3 codec
    compressor = BloscCodec(
        cname="zstd",
        clevel=3,  # lower level = faster
        shuffle="bitshuffle",
    )
    

    rng = jax.random.PRNGKey(int(params.get("seed", 0)))

    existing = list(traj_group.array_keys())
    if existing:
        n_generated = (
            max(int(name.split("_")[1]) for name in existing)
            + 1
        )
    else:
        n_generated = 0

    logger.info(
        f"Found {len(existing)} existing trajectories. "
        f"Starting from index {n_generated}"
    )

    # If spinup>0, define a jitted routine to step the high-res model
    if spinup > 0:
        @functools.partial(jax.jit, static_argnames=["spinup"])
        def _spinup_state(init_state, spinup):
            def _step(carry, _x):
                next_state = hr_model.step_model(carry)
                return next_state, None
            final_state, _ = jax.lax.scan(_step, init_state, None, length=spinup)
            return final_state

        # Vectorise the spinup across the batch; `_spinup_state` already
        # has `spinup` as a static arg via `static_argnames`, so a plain
        # `vmap` over the batch axis is sufficient.
        _spinup_batched = jax.vmap(_spinup_state, in_axes=(0, None))

    # Prefer balanced, band-limited initial conditions when available.
    n_jets = getattr(cfg.plotting, "njets", None)
    if n_jets is not None:
        init_kwargs = {"n_jets": int(n_jets), "pseudo": True, "tune": True}
        logger.info("Using tuned jet initialisation for data generation (n_jets=%s)", n_jets)
    else:
        logger.info("Using default random initialisation for data generation")
    
    while n_generated < n_total+len(existing):
        
        current_batch = min(batch_size, n_total+len(existing) - n_generated)
        rng, subkey = jax.random.split(rng)
        keys = jax.random.split(subkey, current_batch)

        init_states = jax.vmap(functools.partial(hr_model.initialise, **init_kwargs))(keys)
        
        logger.info(f"Initialised batch of {current_batch} trajectories")

        # Run spinup on each initial state if requested
        if spinup > 0:
            init_states = _spinup_batched(init_states, spinup)

        # Generate batch: one coarsened sample per coarse step.
        traj_batch = batched_traj(init_states, nsteps)

        logger.info(f"Generating current batch of {current_batch} trajectories, shape: {traj_batch.shape}")
        # Transfer once per batch
        traj_batch = jax.device_get(traj_batch)

        for i in range(current_batch):

            logger.info(f"Processing trajectory {n_generated+i+1}/{n_total+len(existing)}")
            q_traj = traj_batch[i]

            if not np.all(np.isfinite(q_traj)):
                logger.warning(f"NaN detected in trajectory {n_generated+i}")
                continue

            traj_group.create_array(
                f"traj_{n_generated+i:05d}",
                data=q_traj.astype(np.float32),
                chunks=(1000, q_traj.shape[1], q_traj.shape[2], q_traj.shape[3]),
                compressors=[compressor],
                attributes={
                    "init_key": keys[i].tolist(), # save the initialisation key for reproducibility (just in case)
                },
            )

        n_generated += current_batch
        logger.info(f"Generated {n_generated}/{n_total+len(existing)} trajectories")

    logger.info("Finished generating all trajectories")
    logger.info(f"Saved to {zarr_path}")
    

    