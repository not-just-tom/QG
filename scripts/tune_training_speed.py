#!/usr/bin/env python3
"""Short, bounded search for faster QG training-loop settings.

Only the requested number of synthetic minibatch updates are run for each
candidate. The script never launches a full training job or touches datasets
and checkpoints. It measures minibatch sizes and explicit Python GC cadence,
then estimates configured training time for optional per-epoch batch caps.

The estimate is directional: synthetic inputs, a short probe rollout, and
linear scaling do not capture data loading, test/validation work, compilation
for every curriculum shape, or training-quality differences.

Examples:
    python scripts/tune_training_speed.py
    python scripts/tune_training_speed.py --probe-steps 2 --probe-updates 1
    python scripts/tune_training_speed.py --batch-sizes 2,5,10 --gc-every 0,10,50
    python scripts/tune_training_speed.py --batch-limits all,5,10
"""

import argparse
import gc
import json
import math
import sys
import statistics
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from omegaconf import OmegaConf

from model.ML.train import rollout_traj_errors
from model.ML.utils.dataloading import load_forced_model
from model.ML.utils.loss import build_loss
from model.core.model import QGM
from model.ML.utils.coarsen import coarsen
from model.ML.architectures.build_model import build_closure


DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"


def _positive_int_list(value, option):
    try:
        numbers = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{option} must be comma-separated integers") from exc
    if not numbers or any(number <= 0 for number in numbers):
        raise argparse.ArgumentTypeError(f"{option} values must all be positive")
    return list(dict.fromkeys(numbers))


def _gc_intervals(value):
    try:
        numbers = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--gc-every must be comma-separated integers") from exc
    if not numbers or any(number < 0 for number in numbers):
        raise argparse.ArgumentTypeError("--gc-every values must be nonnegative")
    return list(dict.fromkeys(numbers))


def _batch_limits(value):
    limits = []
    for item in value.split(","):
        item = item.strip().lower()
        if item == "all":
            limits.append(None)
        else:
            try:
                number = int(item)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    "--batch-limits values must be positive integers or 'all'"
                ) from exc
            if number <= 0:
                raise argparse.ArgumentTypeError(
                    "--batch-limits values must be positive integers or 'all'"
                )
            limits.append(number)
    if not limits:
        raise argparse.ArgumentTypeError("--batch-limits cannot be empty")
    return list(dict.fromkeys(limits))


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-type")
    parser.add_argument("--nx", type=int)
    parser.add_argument("--probe-steps", type=int, default=2)
    parser.add_argument("--probe-updates", type=int, default=1)
    parser.add_argument("--batch-sizes", default="1,4,10")
    parser.add_argument(
        "--observed-run-minutes",
        type=float,
        help=(
            "Optional wall time for the configured full run, used to calibrate "
            "relative candidate estimates (for example, 43)"
        ),
    )
    parser.add_argument(
        "--gc-every",
        default="0,10,50",
        help="Explicit gc.collect cadence in updates; 0 means no explicit collections",
    )
    parser.add_argument(
        "--batch-limits",
        default="all,5",
        help="Maximum training updates per epoch to estimate; 'all' uses every window",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-probe-windows",
        type=int,
        default=100,
        help="Safety limit: sum(batch sizes) * probe updates",
    )
    parser.add_argument(
        "--memory-limit-gib",
        type=float,
        default=8.0,
        help="Skip estimates projected above this compiler temporary-memory estimate; 0 disables",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.probe_steps <= 0 or args.probe_updates <= 0 or args.max_probe_windows <= 0:
        parser.error("--probe-steps, --probe-updates, and --max-probe-windows must be positive")
    if args.nx is not None and args.nx <= 0:
        parser.error("--nx must be positive")
    if args.memory_limit_gib < 0:
        parser.error("--memory-limit-gib must be nonnegative")
    if args.observed_run_minutes is not None and args.observed_run_minutes <= 0:
        parser.error("--observed-run-minutes must be positive")
    if not args.config.is_file():
        parser.error(f"Config file does not exist: {args.config}")
    try:
        args.batch_sizes = _positive_int_list(args.batch_sizes, "--batch-sizes")
        args.gc_every = _gc_intervals(args.gc_every)
        args.batch_limits = _batch_limits(args.batch_limits)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    probe_count = sum(args.batch_sizes) * args.probe_updates
    if probe_count > args.max_probe_windows:
        parser.error(
            f"This search would execute {probe_count} probe windows, exceeding "
            f"--max-probe-windows={args.max_probe_windows}. Reduce candidates or "
            "raise the limit deliberately."
        )
    return args


def _block(tree):
    return jax.tree_util.tree_map(
        lambda value: value.block_until_ready()
        if hasattr(value, "block_until_ready")
        else value,
        tree,
    )


def _get_config(args):
    cfg = OmegaConf.load(args.config)
    configured_type = OmegaConf.to_container(cfg.ml.model_type, resolve=True)
    if args.model_type:
        model_type = args.model_type.lower()
    elif isinstance(configured_type, (list, tuple)):
        model_type = str(configured_type[0]).lower()
    else:
        model_type = str(configured_type).lower()
    cfg.ml.model_type = model_type
    params = dict(OmegaConf.to_container(cfg.params, resolve=True))
    high_resolution_nx = int(params.get("hr_nx", params["nx"]))
    low_resolution_nx = args.nx or int(params["nx"])
    if low_resolution_nx > high_resolution_nx:
        raise ValueError(
            f"Low-resolution nx ({low_resolution_nx}) exceeds high-resolution nx "
            f"({high_resolution_nx})"
        )
    params["dt"] = float(cfg.plotting.dt)
    hr_model = QGM({**params, "nx": high_resolution_nx})
    lr_model = coarsen(hr_model, low_resolution_nx)
    return cfg, model_type, params, lr_model


def _make_update(
    *,
    forced_model,
    template_state,
    lr_model,
    dt,
    optimizer,
    cfl_limit,
    loss_fn,
    forcing_key,
):
    def update(parameters, optimizer_state, batch):
        def loss_rollout(current_parameters):
            errors, max_cfl, _, _ = jax.vmap(
                lambda trajectory: rollout_traj_errors(
                    trajectory,
                    forced_model,
                    template_state,
                    current_parameters,
                    lr_model,
                    dt,
                    forcing_key,
                )
            )(batch)
            return jnp.mean(loss_fn(errors, lr_model)), max_cfl

        (loss, max_cfl), gradients = eqx.filter_value_and_grad(
            loss_rollout, has_aux=True
        )(parameters)
        discard = jnp.any(max_cfl > cfl_limit)

        def apply_update(args):
            current_params, current_state, grads = args
            updates, next_state = optimizer.update(grads, current_state, current_params)
            return eqx.apply_updates(current_params, updates), next_state

        def skip_update(args):
            current_params, current_state, _ = args
            return current_params, current_state

        next_parameters, next_optimizer_state = jax.lax.cond(
            discard,
            skip_update,
            apply_update,
            (parameters, optimizer_state, gradients),
        )
        return next_parameters, next_optimizer_state, loss, discard

    return update


def _stage_description(cfg):
    nsteps = int(cfg.plotting.nsteps)
    steps_per_curriculum = int(cfg.ml.steps_per_curriculum)
    start = int(cfg.ml.start_curriculum)
    end = int(cfg.ml.end_curriculum)
    training_trajectories = int(cfg.ml.n_train)
    epoch_count = int(cfg.ml.n_train) + int(cfg.ml.n_test)
    stages = []
    for curriculum in range(start, end + 1):
        rollout_steps = curriculum * steps_per_curriculum
        windows_per_trajectory = nsteps // rollout_steps
        if windows_per_trajectory <= 0:
            continue
        stages.append(
            {
                "curriculum": curriculum,
                "rollout_steps": rollout_steps,
                "windows_per_epoch": training_trajectories * windows_per_trajectory,
                "epochs": epoch_count,
            }
        )
    return stages


def _measure_candidate(
    args,
    batch_size,
    gc_intervals,
    *,
    closure_params,
    initial_optimizer_state,
    optimizer,
    forced_model,
    template_state,
    lr_model,
    dt,
    loss_fn,
    cfl_limit,
    sample_batches,
    forcing_key,
):
    update_fn = _make_update(
        forced_model=forced_model,
        template_state=template_state,
        lr_model=lr_model,
        dt=dt,
        optimizer=optimizer,
        cfl_limit=cfl_limit,
        loss_fn=loss_fn,
        forcing_key=forcing_key,
    )
    batch = jnp.stack(
        [
            jnp.stack(
                [
                    sample_batches[(i + update_index * batch_size) % len(sample_batches)]
                    for i in range(batch_size)
                ]
            )
            for update_index in range(args.probe_updates)
        ]
    )

    start = time.perf_counter()
    lowered = jax.jit(update_fn).lower(
        closure_params, initial_optimizer_state, batch[0]
    )
    lowering_seconds = time.perf_counter() - start
    start = time.perf_counter()
    compiled = lowered.compile()
    compile_seconds = time.perf_counter() - start

    params = closure_params
    state = initial_optimizer_state
    start = time.perf_counter()
    params, state, _, _ = _block(compiled(params, state, batch[0]))
    first_execution_seconds = time.perf_counter() - start
    del params, state

    params = closure_params
    state = initial_optimizer_state
    start = time.perf_counter()
    for update_index in range(args.probe_updates):
        params, state, loss, discarded = compiled(
            params, state, batch[update_index]
        )
        _block((loss, discarded))
    update_elapsed = time.perf_counter() - start
    seconds_per_update = update_elapsed / args.probe_updates
    del params, state

    gc_timings = {}
    for interval in gc_intervals:
        collection_timings = []
        if interval:
            for _ in range(3):
                gc_start = time.perf_counter()
                gc.collect()
                collection_timings.append(time.perf_counter() - gc_start)
        gc_seconds_per_collection = (
            float(statistics.median(collection_timings))
            if collection_timings
            else 0.0
        )
        gc_timings[str(interval)] = {
            "seconds_per_update": seconds_per_update,
            "gc_seconds_per_collection": gc_seconds_per_collection,
            "gc_samples_seconds": collection_timings,
        }

    try:
        memory = compiled.memory_analysis()
        temp_bytes = getattr(memory, "temp_size_in_bytes", None)
    except (AttributeError, NotImplementedError, ValueError):
        temp_bytes = None
    cost = lowered.cost_analysis()
    flops = cost.get("flops") if isinstance(cost, dict) else None

    return {
        "batch_size": batch_size,
        "probe_steps": args.probe_steps,
        "probe_updates": args.probe_updates,
        "measured_update_seconds_total": update_elapsed,
        "lowering_seconds": lowering_seconds,
        "compile_seconds": compile_seconds,
        "first_execution_seconds": first_execution_seconds,
        "estimated_temp_bytes_at_probe_shape": temp_bytes,
        "estimated_flops_per_update_at_probe_shape": flops,
        "gc_settings": gc_timings,
    }


def _full_run_estimates(cfg, batch_size, gc_every, batch_limit, timing, args):
    stages = _stage_description(cfg)
    probe_steps = args.probe_steps
    measurements = timing["gc_settings"][str(gc_every)]
    measured_seconds_per_update = measurements["seconds_per_update"]
    estimated_updates = 0
    estimated_processed_windows = 0
    estimated_full_windows = 0
    estimated_train_seconds = 0.0
    estimated_gc_collections = 0
    gc_seconds_per_collection = measurements["gc_seconds_per_collection"]
    max_estimated_temp_bytes = timing["estimated_temp_bytes_at_probe_shape"]

    for stage in stages:
        estimated_full_windows += stage["windows_per_epoch"] * stage["epochs"]
        updates_per_epoch = math.ceil(stage["windows_per_epoch"] / batch_size)
        if batch_limit is not None:
            updates_per_epoch = min(updates_per_epoch, batch_limit)
        estimated_processed_windows += (
            min(stage["windows_per_epoch"], updates_per_epoch * batch_size)
            * stage["epochs"]
        )
        update_count = updates_per_epoch * stage["epochs"]
        estimated_updates += update_count
        horizon_multiplier = stage["rollout_steps"] / probe_steps
        estimated_train_seconds += (
            update_count * measured_seconds_per_update * horizon_multiplier
        )
        if gc_every:
            estimated_gc_collections += (
                updates_per_epoch // gc_every
            ) * stage["epochs"]

    estimated_gc_seconds = (
        estimated_gc_collections * gc_seconds_per_collection
    )
    estimated_compile_seconds = timing["compile_seconds"] * len(stages)
    return {
        "batch_size": batch_size,
        "gc_every": gc_every,
        "max_batches_per_epoch": batch_limit,
        "estimated_updates": estimated_updates,
        "estimated_processed_windows": estimated_processed_windows,
        "estimated_full_windows": estimated_full_windows,
        "fraction_of_full_windows": (
            estimated_processed_windows / estimated_full_windows
            if estimated_full_windows
            else 0.0
        ),
        "estimated_gc_collections": estimated_gc_collections,
        "estimated_train_seconds": estimated_train_seconds,
        "estimated_gc_seconds": estimated_gc_seconds,
        "estimated_compile_seconds": estimated_compile_seconds,
        "estimated_total_seconds": (
            estimated_train_seconds
            + estimated_gc_seconds
            + estimated_compile_seconds
        ),
        "max_projected_temp_bytes": (
            max_estimated_temp_bytes
            * max((stage["rollout_steps"] / probe_steps for stage in stages), default=1)
            if max_estimated_temp_bytes is not None
            else None
        ),
    }


def _format_seconds(value):
    if value < 60:
        return f"{value:.1f}s"
    if value < 3600:
        return f"{value / 60:.1f}m"
    return f"{value / 3600:.2f}h"


def _run(args):
    cfg, model_type, params, lr_model = _get_config(args)
    seed_key = jax.random.PRNGKey(args.seed)
    closure_key, data_key, forcing_key = jax.random.split(seed_key, 3)
    closure = build_closure(cfg)
    closure_params = eqx.filter(closure, eqx.is_array)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(float(cfg.architectures[model_type].get("learning_rate", 1e-4))),
    )
    optimizer_state = optimizer.init(closure_params)
    loss_fn = build_loss(str(cfg.architectures[model_type].get("loss", "mae")))
    template_state = lr_model.initialise(closure_key, n_jets=params.get("n_jets"))
    nz = int(params.get("nz", 1))
    nx = int(lr_model.nx)
    probe_trajectories = jax.random.normal(
        data_key,
        (1, max(args.batch_sizes), args.probe_steps + 1, nz, nx, nx),
        dtype=jnp.float32,
    )
    scale = jnp.maximum(jnp.std(template_state.q), 1e-6)
    probe_trajectories = probe_trajectories * scale
    closure_scale = float(
        cfg.architectures[model_type].get("closure_scale", 0.1)
    )
    forced_model, closure_params, _, _ = load_forced_model(
        lr_model,
        closure,
        float(lr_model.dt),
        trajs=probe_trajectories,
        closure_scale=closure_scale,
    )
    sample_batches = list(probe_trajectories[0])

    report = {
        "config": str(args.config.resolve()),
        "model_type": model_type,
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "device": [str(device) for device in jax.devices()],
        "probe_steps": args.probe_steps,
        "probe_updates": args.probe_updates,
        "batch_sizes": args.batch_sizes,
        "gc_every_candidates": args.gc_every,
        "batch_limit_candidates": [
            "all" if value is None else value for value in args.batch_limits
        ],
        "note": (
            "Synthetic short-rollout probes. Estimates scale probe time linearly "
            "with rollout horizon and omit test/validation, data loading, checkpointing, "
            "and training-quality effects."
        ),
        "measurements": [],
        "estimates": [],
    }

    print(
        f"Device: {jax.default_backend()} | model: {model_type} | nx={nx} | "
        f"probe: {args.probe_steps} steps, {args.probe_updates} update(s)"
    )
    print(
        "This is a bounded synthetic search, not a full training run. "
        f"Maximum measured probe windows: "
        f"{sum(args.batch_sizes) * args.probe_updates}."
    )

    cfl_limit = float(getattr(cfg.plotting, "cfl", 1.0))
    for batch_size in args.batch_sizes:
        print(f"\n[compile/measure] minibatch={batch_size}")
        measurement = _measure_candidate(
            args,
            batch_size,
            args.gc_every,
            closure_params=closure_params,
            initial_optimizer_state=optimizer_state,
            optimizer=optimizer,
            forced_model=forced_model,
            template_state=template_state,
            lr_model=lr_model,
            dt=float(lr_model.dt),
            loss_fn=loss_fn,
            cfl_limit=cfl_limit,
            sample_batches=sample_batches,
            forcing_key=forcing_key,
        )
        report["measurements"].append(measurement)
        print(
            f"  compile {measurement['compile_seconds']:.2f}s; "
            f"one update after compile {measurement['first_execution_seconds']:.2f}s"
        )
        for gc_every, result in measurement["gc_settings"].items():
            print(
                f"  gc every {gc_every:>3}: "
                f"{result['seconds_per_update']:.3f}s/update, "
                    f"GC median {result['gc_seconds_per_collection']:.5f}s/collection"
            )

        for gc_every in args.gc_every:
            for batch_limit in args.batch_limits:
                estimate = _full_run_estimates(
                    cfg,
                    batch_size,
                    gc_every,
                    batch_limit,
                    measurement,
                    args,
                )
                estimate["compile_seconds_probe_shape"] = measurement["compile_seconds"]
                estimate["probe_steps"] = args.probe_steps
                estimate["probe_seconds_per_update"] = measurement["gc_settings"][
                    str(gc_every)
                ]["seconds_per_update"]
                report["estimates"].append(estimate)

    baseline_batch = int(cfg.ml.batch_size)
    baseline_gc = int(getattr(cfg.ml, "gc_every_batches", 50))
    baseline = [
        estimate
        for estimate in report["estimates"]
        if estimate["batch_size"] == baseline_batch
        and estimate["gc_every"] == baseline_gc
        and estimate["max_batches_per_epoch"] is None
    ]
    if args.observed_run_minutes is not None and baseline:
        baseline_estimated_seconds = baseline[0]["estimated_total_seconds"]
        for estimate in report["estimates"]:
            estimate["calibrated_full_run_seconds"] = (
                args.observed_run_minutes
                * 60
                * estimate["estimated_total_seconds"]
                / max(baseline_estimated_seconds, 1e-12)
            )

    estimates = report["estimates"]
    if args.memory_limit_gib > 0:
        maximum_bytes = args.memory_limit_gib * (1024**3)
        before_filter_count = len(estimates)
        estimates = [
            estimate
            for estimate in estimates
            if estimate["max_projected_temp_bytes"] is None
            or estimate["max_projected_temp_bytes"] <= maximum_bytes
        ]
        filtered_count = before_filter_count - len(estimates)
        if filtered_count:
            print(
                f"\nOmitted {filtered_count} configurations whose projected XLA "
                f"temporary-memory estimate exceeds {args.memory_limit_gib:.1f} GiB. "
                "This is only a compiler estimate."
            )
    estimates.sort(key=lambda result: result["estimated_total_seconds"])

    def print_ranking(title, choices):
        print(f"\n=== {title} (fastest first) ===")
        print(
            f"{'Minibatch':>10} {'GC every':>9} {'Max batches':>12} "
            f"{'Windows':>10} {'Data kept':>10} {'Est. train':>12} "
            f"{'GC est.':>10} {'Temp est.':>12} {'Total est.':>12}"
        )
        print("-" * 94)
        for estimate in choices[:12]:
            limit_label = (
                "all"
                if estimate["max_batches_per_epoch"] is None
                else str(estimate["max_batches_per_epoch"])
            )
            temp = estimate["max_projected_temp_bytes"]
            temp_label = (
                "n/a"
                if temp is None
                else f"{temp / (1024**3):.2f} GiB"
            )
            calibrated_seconds = estimate.get("calibrated_full_run_seconds")
            wall_label = (
                "n/a"
                if calibrated_seconds is None
                else _format_seconds(calibrated_seconds)
            )
            print(
                f"{estimate['batch_size']:>10} "
                f"{estimate['gc_every']:>9} "
                f"{limit_label:>12} "
                f"{estimate['estimated_processed_windows']:>10} "
                f"{estimate['fraction_of_full_windows']:>9.1%} "
                f"{_format_seconds(estimate['estimated_total_seconds']):>12} "
                f"{_format_seconds(estimate['estimated_gc_seconds']):>10} "
                f"{temp_label:>12} "
                f"{wall_label:>12}"
            )

    print_ranking(
        "Full-data candidates (minibatch/GC changes only)",
        sorted(
            (
                estimate
                for estimate in estimates
                if estimate["max_batches_per_epoch"] is None
            ),
            key=lambda result: result["estimated_total_seconds"],
        ),
    )
    print_ranking(
        "Reduced-work candidates (batch cap changes data exposure)",
        sorted(
            (
                estimate
                for estimate in estimates
                if estimate["max_batches_per_epoch"] is not None
            ),
            key=lambda result: result["estimated_total_seconds"],
        ),
    )

    if baseline:
        print(
            f"\nConfigured full-data baseline estimate "
            f"(minibatch={baseline_batch}, gc_every={baseline_gc}): "
            f"{_format_seconds(baseline[0]['estimated_total_seconds'])} "
            "training plus estimated compilation."
        )
    elif args.observed_run_minutes is not None:
        print(
            "\nCould not calibrate to observed runtime: include the configured "
            "batch size and GC interval in --batch-sizes and --gc-every."
        )
    print(
        "\nImportant: a batch cap processes fewer training windows and changes "
        "the training schedule/results. Larger minibatches also change update "
        "frequency. The ranking is for speed, not equivalent model quality. "
        "Probe-time estimates are approximate; validate the best few settings "
        "with a short real-data run before committing to a full run."
    )
    return report


def main():
    args = parse_args()
    report = _run(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"\nWrote detailed candidate report to {args.output.resolve()}")


if __name__ == "__main__":
    main()
