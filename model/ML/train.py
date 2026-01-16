"""Small demo training loop for spectral correction model.

This provides a convenience function `train_spectral_demo` that synthesizes a
"truth" spectral correction and fits the spectral parameters to it using SGD.
It is intentionally simple and meant as a starting point for more advanced
training (e.g., differentiable time integration, multi-step losses, etc.).
"""
from __future__ import annotations

import time
import jax
import jax.numpy as jnp
import pathlib
import os
import sys
import itertools
import logging
import json 
import platform
import math
import contextlib
import numpy as np
import equinox as eqx
import optax
import re
import importlib
import model.utils
import model.ML
importlib.reload(model.utils)
importlib.reload(model.ML)
from model.utils.logging import configure_logging
from model.ML.utils import build_network, make_json_serializable, load_network_continue


def run(config):
    out_dir = pathlib.Path(config.filepaths.out_dir)
    if out_dir.is_file():
        raise ValueError(f"Path must be a directory, not a file: {config.filepaths.out_dir}")
    out_dir.mkdir(exist_ok=True)
    configure_logging(level=config.filepaths.log_level, out_file="logs/run.log") #return to this to put numbers on it 
    logger = logging.getLogger("main")
    logger.info("Arguments: %s", vars(config))

    # Select seed
    if config.seed is None:
        logger.info("No seed provided")
    else:
        seed = config.params.seed # this needs to be changed
    logger.info("Using seed %d", seed)
    np_rng = np.random.default_rng(seed=seed)# what does this do

    # Configure required elements for training
    rng_ctr = jax.random.PRNGKey(seed=np_rng.integers(2**32).item())
    train_path = (pathlib.Path(config.train_set) / "shuffled.hdf5").resolve()
    val_path = (pathlib.Path(config.val_set) / "data.hdf5").resolve()
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(exist_ok=True)


    # Create data normalizer and its inverse
    model_params = load_model_params(train_path)
    coarse_op_name = model_params.scale_operator
    # Construct neural net
    model = config.ml.model
    rng, rng_ctr = jax.random.split(rng_ctr, 2)
    logger.info("Training network: %s", model)
    state, network_info = build_network(
        architecture=model,
        lr=config.ml.lr,
        rng=rng,
        input_channels=config.ml.input_channels, #this needs to change w model so come back to this
        output_channels=config.ml.output_channels,
        train_path=train_path,
        optim_type=config.ml.optimizer,
        num_epochs=config.ml.num_epochs,
        batches_per_epoch=config.ml.batches_per_epoch,
        end_lr=config.ml.end_lr,
        schedule_type=config.ml.lr_schedule,
        coarse_op_name=coarse_op_name,
        arch_config={
            "zero_mean": config.network_zero_mean,
        },
        channel_coarsen_type=config.channel_coarsen_type,
    )
    net_aux = network_info["net_aux"]
    if config.continue_from_checkpoint is True:
        logger.info("CONTINUING NETWORK: %s", config.net_weight_continue)
        # Load network from file, wrap in train state
        state, network_info = load_network_continue(
            config.net_weight_continue,
            state,
            network_info,
        )
        logger.info("Loaded trained network weights")
    # Store network info
    with utils.rename_save_file(weights_dir / "network_info.json", "w", encoding="utf8") as net_info_file:
        json.dump(network_info, net_info_file)
    # Store run details
    with utils.rename_save_file(out_dir / "cli_info.json", "w", encoding="utf8") as cli_info_file:
        cli_info = {
                "argv": sys.argv,
                "parsed_config": dict(vars(config)),
                "environ": dict(os.environ),
                "node": platform.node(),
        }
        if git_info is not None:
            cli_info["git_info"] = {
                "commit": git_info.hash,
                "clean_worktree": git_info.clean_worktree
            }
        else:
            cli_info["git_info"] = None
        json.dump(cli_info, cli_info_file)

    # Process noise_spec
    noise_spec = {}
    for spec in config.noise_specs:
        spec = spec.strip()
        if not spec:
            continue
        chan_name, var = spec.split("=")
        noise_spec[chan_name.strip()] = np.array([float(v.strip()) for v in var.strip().split(",")])

    # Open data files
    with contextlib.ExitStack() as train_context:
        # Open data files
        train_loader = train_context.enter_context(
            make_train_loader(
                train_path=train_path,
                batch_size=config.batch_size,
                loader_chunk_size=config.loader_chunk_size,
                base_logger=logger.getChild("train_loaders"),
                np_rng=np_rng,
                required_fields=required_fields,
            )
        )
        val_loader = train_context.enter_context(
            make_val_loader(
                file_path=val_path,
                required_fields=required_fields,
                base_logger=logger.getChild("val_loaders"),
            )
        )

        noisy_batch_config = {
            "mode": config.noisy_batch_mode,
            "simple-prob-clean": {
                "prob": config.simple_prob_clean,
                "start_epoch": config.simple_prob_clean_start_epoch,
            },
        }

        # Training functions
        train_batch_fn = eqx.filter_jit(
            make_batch_computer(
                input_channels=input_channels,
                output_channels=output_channels,
                model_params=model_params,
                processing_size=processing_size,
                noise_spec=noise_spec,
                net_aux=net_aux,
                noisy_batch_config=noisy_batch_config,
            ),
            donate="all",
        )
        # Determine fixed validation samples
        val_samp_rng = np.random.default_rng(seed=config.val_sample_seed)
        val_traj_idxs = val_samp_rng.integers(low=0, high=val_loader.num_trajs, size=config.num_val_samples, dtype=np.uint64)
        val_step_idxs = val_samp_rng.integers(low=0, high=val_loader.num_steps, size=config.num_val_samples, dtype=np.uint64)
        val_stats_fn = eqx.filter_jit(
            make_validation_stats_function(
                input_channels=input_channels,
                output_channels=output_channels,
                model_params=model_params,
                processing_size=processing_size,
                net_aux=net_aux,
            )
        )

        # Running statistics
        min_mean_loss = None
        min_val_loss = None

        # Training loop
        epoch_reports = []
        save_names_written = set()
        save_names_permanent = set()
        save_names_mapping = {}
        for epoch in range(1, config.num_epochs + 1):
            logger.info("Starting epoch %d of %d", epoch, config.num_epochs)
            # Training step
            with contextlib.closing(train_loader.iter_batches()) as train_batch_iter:
                state, epoch_stats, rng_ctr = do_epoch(
                    train_state=state,
                    batch_iter=itertools.islice(train_batch_iter, config.ml.batches_per_epoch),
                    batch_fn=train_batch_fn,
                    logger=logger.getChild(f"{epoch:05d}_train"),
                    rng_ctr=rng_ctr,
                    epoch=epoch,
                    noisy_batch_config=noisy_batch_config,
                )
            mean_loss = epoch_stats["mean_loss"]

            # Validation step
            val_stat_report = None
            val_loss = None
            if epoch % config.val_interval == 0:
                logger.info("Starting validation for epoch %d", epoch)
                val_stat_report = do_validation(
                    train_state=state,
                    loader=val_loader,
                    sample_stat_fn=val_stats_fn,
                    traj=val_traj_idxs,
                    step=val_step_idxs,
                    logger=logger.getChild(f"{epoch:05d}_val"),
                )
                val_loss = val_stat_report["standard_mse"]
                logger.info("Finished validation for epoch %d", epoch)

            # Save snapshots
            saved_names = []
            # Save the network after each epoch
            epoch_name = f"epoch{epoch:04d}"
            epoch_file = weights_dir / f"{epoch_name}.eqx"
            save_network(epoch_name, output_dir=weights_dir, state=state, base_logger=logger)
            save_names_written.add(epoch_name)
            # Link checkpoint
            utils.atomic_symlink(epoch_file, weights_dir / "checkpoint.eqx")
            save_names_mapping["checkpoint"] = epoch_name
            saved_names.append("checkpoint")
            # Link best loss (maybe)
            if min_mean_loss is None or (math.isfinite(mean_loss) and mean_loss <= min_mean_loss):
                min_mean_loss = mean_loss
                utils.atomic_symlink(epoch_file, weights_dir / "best_loss.eqx")
                save_names_mapping["best_loss"] = epoch_name
                saved_names.append("best_loss")
            if val_loss is not None and (min_val_loss is None or (math.isfinite(val_loss) and val_loss <= min_val_loss)):
                min_val_loss = val_loss
                utils.atomic_symlink(epoch_file, weights_dir / "best_val_loss.eqx")
                save_names_mapping["best_val_loss"] = epoch_name
                saved_names.append("best_val_loss")
            # Save interval
            if epoch % config.save_interval == 0:
                utils.atomic_symlink(epoch_file, weights_dir / "interval.eqx")
                save_names_mapping["interval"] = epoch_name
                saved_names.append("interval")
            # Permanently fix epoch (if requested)
            if (epoch % config.save_interval == 0) or (epoch == config.num_epochs):
                save_names_permanent.add(epoch_name)
                saved_names.append(epoch_name)
            # Save the final epoch with a special name
            if epoch == config.num_epochs:
                utils.atomic_symlink(epoch_file, weights_dir / "final_snapshot.eqx")
                save_names_mapping["final_snapshot"] = epoch_name
                saved_names.append("final_snapshot")
            logger.info("Wrote file and link names: %s", saved_names)
            # Clean up any now unlinked files
            save_names_to_remove = (save_names_written - save_names_permanent) - {v for v in save_names_mapping.values() if v is not None}
            for name_to_remove in save_names_to_remove:
                try:
                    logger.debug("Removing weights file %s", name_to_remove)
                    os.remove(weights_dir / f"{name_to_remove}.eqx")
                    save_names_written.discard(name_to_remove)
                except FileNotFoundError:
                    logger.warning("Tried to remove missing weights file %s", name_to_remove)

            epoch_reports.append(
                {
                    "epoch": epoch,
                    "train_stats": epoch_stats,
                    "val_stats": val_stat_report,
                    "saved_names": saved_names,
                }
            )
            with utils.rename_save_file(out_dir / "train_report.json", "w", encoding="utf8") as train_report_file:
                json.dump(epoch_reports, train_report_file)

            logger.info("Finished epoch %d", epoch)

    # End of training loop
    logger.info("Finished training")

