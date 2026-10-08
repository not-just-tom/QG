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

def generate_train_data(cfg, params, tau_eddy, timing_metadata, hr_model, lr_model, hr_dir):
    '''Generate zarr training data from the high-res `hr_model` and coarsen
    on-the-fly using `lr_model` as the low-resolution physics template, and lower res dt.
    Saves metadata and trajectories into `hr_dir`.
    '''
    # Timing parameters
    n_total = cfg.ml.n_train + cfg.ml.n_test
    batch_size = 21 # hardcoded bc it was confusing me. It's just the number of trajs generated in batches
    # Prepare low-resolution template and ratio for coarsening
    dummy_key = jax.random.PRNGKey(0)
    lr_template = lr_model.initialise(
        dummy_key,
        n_jets=params.get("n_jets"),
        pseudo=(params.get("n_jets") is not None),
    )
    spinup_time = cfg.plotting.spinup * tau_eddy
    spinup = int(spinup_time // hr_model.stepper.dt)
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
            filtered = trunc * lr_model._dealias / ratio
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

    # Prefer balanced, band-limited initial conditions when available.
    n_jets = params.get('n_jets', None)
    if n_jets is not None:
        init_kwargs = {"n_jets": int(n_jets), "pseudo": True}
        logger.info("Using tuned jet initialisation for data generation (n_jets=%s)", n_jets)
    else:
        logger.info("Using default random initialisation for data generation")
    
    while n_generated < n_total+len(existing):
        
        current_batch = min(batch_size, n_total+len(existing) - n_generated)
        rng, subkey = jax.random.split(rng)
        keys = jax.random.split(subkey, current_batch)

        init_states = jax.vmap(functools.partial(hr_model.model.initialise, **init_kwargs))(keys)
        
        logger.info(f"Initialised batch of {current_batch} trajectories")

        # Run spinup on each initial state if requested
        if spinup > 0:
            init_states = hr_model.spinup(init_states, spinup)

        # Generate batch: one coarsened sample per coarse step.
        traj_batch = batched_traj(init_states, nsteps)

        logger.info(f"Generating current batch of {current_batch} trajectories, shape: {traj_batch.shape}")
        # Transfer once per batch
        traj_batch = jax.device_get(traj_batch)

        failed_runs = 0
        for i in range(current_batch):

            logger.info(f"Processing trajectory {n_generated+i+1}/{n_total+len(existing)}")
            q_traj = traj_batch[i]

            if not np.all(np.isfinite(q_traj)):
                logger.warning(f"NaN detected in trajectory {n_generated+i}")
                failed_runs += 1
                continue

            traj_group.create_array(
                f"traj_{n_generated+i:05d}",
                data=q_traj.astype(np.float32),
                chunks=(128, q_traj.shape[1], q_traj.shape[2], q_traj.shape[3]),
                compressors=[compressor],
                attributes={
                    "init_key": keys[i].tolist(), # save the initialisation key for reproducibility (just in case)
                },
            )

        n_generated += current_batch - failed_runs
        logger.info(f"Generated {n_generated}/{n_total+len(existing)} trajectories")

    logger.info("Finished generating all trajectories")
    logger.info(f"Saved to {zarr_path}")


def _generate_validation_trajectory(cfg, params, tau_eddy, hr_model, lr_model, init_key):
    """Generate one deterministic validation trajectory."""
    validation_rollout = int(cfg.plotting.validation_rollout)
    if validation_rollout <= 0:
        raise ValueError("plotting.validation_rollout must be positive")

    dummy_key = jax.random.PRNGKey(0)
    lr_template = lr_model.initialise(
        dummy_key,
        n_jets=params.get("n_jets"),
        pseudo=(params.get("n_jets") is not None),
    )
    ratio = int(float(hr_model.model.nx) / float(lr_model.nx))
    if ratio <= 0 or ratio * lr_model.nx != hr_model.model.nx:
        raise ValueError("High- and low-resolution model sizes must have an integer ratio")

    @functools.partial(jax.jit, static_argnames=["validation_rollout"])
    def generate_trajectory(init_state, validation_rollout):
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
            filtered = trunc * lr_model._dealias / ratio
            lr_state = lr_template.update(qh=filtered)
            return lr_state.q

        def step(carry, _x):
            # Advance ratio high-res steps, then emit one low-res sample.
            def _hr_step(inner_carry, _):
                return hr_model.step_model(inner_carry), None

            next_state, _ = jax.lax.scan(_hr_step, carry, None, length=ratio)
            return next_state, _coarsen_state(next_state)

        _, traj_q = jax.lax.scan(step, init_state, None, length=validation_rollout)
        return traj_q

    n_jets = params.get("n_jets")
    init_kwargs = {"n_jets": int(n_jets), "pseudo": True} if n_jets is not None else {}
    init_states = jax.vmap(
        functools.partial(hr_model.model.initialise, **init_kwargs)
    )(jax.random.split(init_key, 1))
    spinup_time = cfg.plotting.spinup * tau_eddy
    spinup_steps = int(spinup_time // hr_model.stepper.dt)
    spun_up_states = hr_model.spinup(init_states, spinup_steps)
    init_state = jax.tree_util.tree_map(lambda value: value[0], spun_up_states)
    logger.info("Spun up validation initial state for %d high-res steps", spinup_steps)
    trajectory = jax.device_get(generate_trajectory(init_state, validation_rollout))
    if not np.all(np.isfinite(trajectory)):
        raise ValueError("Generated validation trajectory contains non-finite values")
    return np.asarray(trajectory, dtype=np.float32)


def _validation_init_key(params):
    """Use a stable key independent of the training trajectory keys."""
    seed = int(params.get("seed", 0))
    return jax.random.fold_in(jax.random.PRNGKey(seed), 0x5A17)


def generate_validation(cfg, params, tau_eddy, hr_model, lr_model, hr_dir):
    """Create the dataset's separate, fixed validation trajectory."""
    zarr_path = os.path.join(hr_dir, "trajectories.zarr")
    z_root = zarr.open_group(zarr_path, mode="a")
    if "validation_trajectory" in z_root:
        raise ValueError(
            "Validation trajectory already exists; use extend_validation to lengthen it"
        )

    init_key = _validation_init_key(params)
    trajectory = _generate_validation_trajectory(
        cfg, params, tau_eddy, hr_model, lr_model, init_key
    )
    compressor = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")
    z_root.create_array(
        "validation_trajectory",
        data=trajectory,
        chunks=(min(128, trajectory.shape[0]), *trajectory.shape[1:]),
        compressors=[compressor],
        attributes={"init_key": np.asarray(init_key).tolist()},
    )
    logger.info(
        "Saved validation trajectory with %d steps to %s",
        trajectory.shape[0],
        zarr_path,
    )


def extend_validation(cfg, params, tau_eddy, hr_model, lr_model, hr_dir):
    """Extend the fixed validation trajectory without changing its existing prefix."""
    zarr_path = os.path.join(hr_dir, "trajectories.zarr")
    z_root = zarr.open_group(zarr_path, mode="a")
    if "validation_trajectory" not in z_root:
        return generate_validation(cfg, params, tau_eddy, hr_model, lr_model, hr_dir)

    validation_array = z_root["validation_trajectory"]
    target_steps = int(cfg.plotting.validation_rollout)
    current_steps = validation_array.shape[0]
    if target_steps <= current_steps:
        return
    if target_steps <= 0:
        raise ValueError("plotting.validation_rollout must be positive")

    attrs = dict(validation_array.attrs)
    if "init_key" not in attrs:
        raise ValueError(
            "Cannot extend validation trajectory because its initialization key is missing"
        )
    init_key = jnp.asarray(attrs["init_key"], dtype=jnp.uint32)
    full_trajectory = _generate_validation_trajectory(
        cfg, params, tau_eddy, hr_model, lr_model, init_key
    )
    if full_trajectory.shape[0] != target_steps:
        raise RuntimeError(
            f"Generated {full_trajectory.shape[0]} validation steps; expected {target_steps}"
        )
    existing_prefix = np.asarray(validation_array[:])
    if not np.allclose(
        full_trajectory[:current_steps], existing_prefix, rtol=1e-6, atol=1e-7
    ):
        raise RuntimeError(
            "Regenerated validation trajectory does not match its saved prefix; "
            "refusing to change the held-out data"
        )

    validation_array.resize((target_steps, *validation_array.shape[1:]))
    validation_array[current_steps:] = full_trajectory[current_steps:]
    logger.info(
        "Extended validation trajectory from %d to %d steps",
        current_steps,
        target_steps,
    )