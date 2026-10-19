#!/usr/bin/env python3
"""Measure JAX compile time and warmed execution costs for QG model workloads.

This read-only diagnostic uses synthetic inputs; it does not load or modify
training data or checkpoints. Each workload reports tracing/lowering time,
backend compilation time, first execution time, steady-state latency, and
available XLA cost/memory estimates.

Examples:
    python scripts/profile_jax_bottlenecks.py
    python scripts/profile_jax_bottlenecks.py --model-type cnn --nx 64 --steps 8
    python scripts/profile_jax_bottlenecks.py --workloads physics closure rollout train
    python scripts/profile_jax_bottlenecks.py --trace-dir profile-trace
"""

import argparse
import functools
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from omegaconf import OmegaConf

from model.ML.architectures.build_model import build_closure
from model.ML.train import rollout_traj_errors
from model.ML.utils.dataloading import load_forced_model
from model.ML.utils.loss import build_loss
from model.core.model import QGM
from model.ML.utils.coarsen import coarsen


DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"
WORKLOADS = ("physics", "closure", "rollout", "train")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-type", help="Override ml.model_type in the config")
    parser.add_argument("--nx", type=int, help="Override the low-resolution grid width")
    parser.add_argument("--steps", type=int, default=4, help="Rollout intervals")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--train-warmup",
        type=int,
        default=0,
        help="Extra full training runs after the first timed run (can be costly)",
    )
    parser.add_argument(
        "--train-repeats",
        type=int,
        default=0,
        help="Extra measured full training runs; 0 keeps train to a single run",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--workloads", nargs="+", choices=WORKLOADS, default=list(WORKLOADS)
    )
    parser.add_argument("--trace-dir", type=Path, help="Write a JAX profiler trace")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    if (
        args.steps <= 0
        or args.batch_size <= 0
        or args.repeats <= 0
        or args.warmup < 0
        or args.train_warmup < 0
        or args.train_repeats < 0
    ):
        parser.error(
            "--steps, --batch-size, and --repeats must be positive; "
            "warm-up and train-repeat counts must be nonnegative"
        )
    if not args.config.is_file():
        parser.error(f"Config file does not exist: {args.config}")
    return args


def _block(tree):
    return jax.tree_util.tree_map(
        lambda value: value.block_until_ready()
        if hasattr(value, "block_until_ready")
        else value,
        tree,
    )


def _memory_report(compiled):
    try:
        analysis = compiled.memory_analysis()
    except (AttributeError, NotImplementedError, ValueError) as exc:
        return {"unavailable": str(exc)}
    if analysis is None:
        return None
    fields = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
        "alias_size_in_bytes",
    )
    return {
        name: getattr(analysis, name)
        for name in fields
        if hasattr(analysis, name)
    }


def _cost_report(lowered):
    try:
        cost = lowered.cost_analysis()
    except (AttributeError, NotImplementedError, ValueError) as exc:
        return {"unavailable": str(exc)}
    if isinstance(cost, dict):
        return {
            str(key): float(value) if isinstance(value, (int, float)) else str(value)
            for key, value in cost.items()
        }
    return str(cost)


def _profile(name, function, args, warmup, repeats):
    compiled_function = jax.jit(function)
    start = time.perf_counter()
    lowered = compiled_function.lower(*args)
    lowering_seconds = time.perf_counter() - start

    start = time.perf_counter()
    compiled = lowered.compile()
    compile_seconds = time.perf_counter() - start

    start = time.perf_counter()
    result = _block(compiled(*args))
    first_execution_seconds = time.perf_counter() - start
    del result

    for _ in range(warmup):
        _block(compiled(*args))

    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        _block(compiled(*args))
        timings.append(time.perf_counter() - start)

    median_seconds = statistics.median(timings) if timings else None
    report = {
        "lowering_seconds": lowering_seconds,
        "compile_seconds": compile_seconds,
        "first_execution_seconds": first_execution_seconds,
        "steady_seconds_median": median_seconds,
        "steady_seconds_p90": float(np.percentile(timings, 90)) if timings else None,
        "steady_seconds_min": min(timings) if timings else None,
        "steady_sample_count": len(timings),
        "compile_to_steady_ratio": (
            compile_seconds / max(median_seconds, 1e-12)
            if median_seconds is not None
            else None
        ),
        "cost_analysis": _cost_report(lowered),
        "memory_analysis_bytes": _memory_report(compiled),
        "input_signature": _input_signature(args),
    }
    line = (
        f"[{name}] lower={lowering_seconds:.3f}s compile={compile_seconds:.3f}s "
        f"first={first_execution_seconds:.3f}s"
    )
    if median_seconds is not None:
        line += (
            f" steady-med={median_seconds:.6f}s "
            f"compile/steady={report['compile_to_steady_ratio']:.1f}x"
        )
    else:
        line += " (no extra full-run timing)"
    print(line)
    return report


def _input_signature(tree):
    counts = {}
    for leaf in jax.tree_util.tree_leaves(tree):
        if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
            continue
        signature = (tuple(leaf.shape), str(leaf.dtype), str(leaf.device))
        counts[signature] = counts.get(signature, 0) + 1
    return [
        {
            "shape": list(shape),
            "dtype": dtype,
            "device": device,
            "count": count,
        }
        for (shape, dtype, device), count in sorted(counts.items(), key=repr)
    ]


def _as_reportable_error(exc):
    return {"error": f"{type(exc).__name__}: {exc}"}


def _human_bytes(value):
    if value is None:
        return "n/a"
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if amount < 1024 or unit == "GiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} GiB"


def _print_summary(report):
    print("\n=== JAX bottleneck summary ===")
    print(
        f"Backend: {report['backend']} | JAX: {report['jax_version']} | "
        f"Grid: {report['grid']['nx']}x{report['grid']['nx']} "
        f"(nz={report['grid']['nz']}) | rollout intervals: "
        f"{report['rollout_intervals']} | batch: {report['batch_size']}"
    )
    print(
        f"{'Workload':<12} {'Lower':>9} {'Compile':>10} {'First run':>10} "
        f"{'Warm median':>12} {'Compile/warm':>13} {'Temp memory':>13} {'FLOPs*':>14}"
    )
    print("-" * 100)
    for name, result in report["workloads"].items():
        if "error" in result:
            print(f"{name:<12} FAILED: {result['error'].splitlines()[0]}")
            continue
        memory = result.get("memory_analysis_bytes") or {}
        cost = result.get("cost_analysis") or {}
        ratio = result.get("compile_to_steady_ratio")
        warm = result.get("steady_seconds_median")
        flops = cost.get("flops") if isinstance(cost, dict) else None
        print(
            f"{name:<12} "
            f"{result['lowering_seconds']:>8.3f}s "
            f"{result['compile_seconds']:>9.3f}s "
            f"{result['first_execution_seconds']:>9.3f}s "
            f"{(f'{warm:.6f}s' if warm is not None else 'n/a'):>12} "
            f"{(f'{ratio:.1f}x' if ratio is not None else 'n/a'):>12} "
            f"{_human_bytes(memory.get('temp_size_in_bytes')):>13} "
            f"{flops if flops is not None else 'n/a':>14}"
        )
    print(
        "\nHow to read this:\n"
        "  Lower = tracing/lowering; Compile = backend compilation; First run = "
        "first execution after compile.\n"
        "  Warm median = synchronized steady-state execution time. Compile/warm is "
        "a ratio, not a claim of repeated compilation.\n"
        "  Temp memory and FLOPs are compiler estimates, not measured peak "
        "device-memory use or wall-clock cost.\n"
        "  *FLOPs are shown as the compiler reports them; compare only workloads "
        "on the same backend and similar shapes.\n"
        "For train, only the first complete update is timed by default; "
        "extra full-run repeats are opt-in.\n"
        "Full timing, cost, memory, and grouped input-shape details follow as JSON."
    )


def _config_and_models(args):
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
    if low_resolution_nx <= 0 or low_resolution_nx > high_resolution_nx:
        raise ValueError(
            f"Low-resolution nx must be in 1..{high_resolution_nx}, got {low_resolution_nx}"
        )
    dt = float(cfg.plotting.dt)
    params["dt"] = dt
    high_resolution_model = QGM({**params, "nx": high_resolution_nx})
    low_resolution_model = coarsen(high_resolution_model, low_resolution_nx)
    return cfg, model_type, params, low_resolution_model


def _initialise_inputs(args, cfg, model_type, params, lr_model):
    key = jax.random.PRNGKey(args.seed)
    key, closure_key, data_key = jax.random.split(key, 3)
    closure = build_closure(cfg)
    template_state = lr_model.initialise(key, n_jets=params.get("n_jets"))
    nz = int(params.get("nz", 1))
    nx = int(lr_model.nx)
    time_length = args.steps + 1
    noise = jax.random.normal(
        data_key,
        (1, args.batch_size, time_length, nz, nx, nx),
        dtype=jnp.float32,
    )
    trajectory_scale = jnp.maximum(jnp.std(template_state.q), 1e-6)
    trajectories = noise * trajectory_scale

    dt = float(lr_model.dt)
    closure_scale = float(cfg.architectures[model_type].get("closure_scale", 0.1))
    forced_model, closure_params, closure_static, _ = load_forced_model(
        lr_model,
        closure,
        dt,
        trajs=trajectories,
        closure_scale=closure_scale,
    )

    init_qh = jnp.fft.rfftn(trajectories[0, 0, 0], axes=(-2, -1), norm="ortho")
    base_state = template_state.update(qh=init_qh.astype(template_state.qh.dtype))
    init_state = forced_model.initialise_stepper_state(
        forced_model.model.initialise_param_state(base_state, closure_params),
        forcing_key=closure_key,
    )
    return (
        closure,
        closure_params,
        template_state,
        trajectories,
        forced_model,
        closure_static,
        init_state,
    )


def _run(args):
    cfg, model_type, params, lr_model = _config_and_models(args)
    (
        closure,
        closure_params,
        template_state,
        trajectories,
        forced_model,
        closure_static,
        init_state,
    ) = _initialise_inputs(args, cfg, model_type, params, lr_model)

    report = {
        "config": str(args.config.resolve()),
        "model_type": model_type,
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "grid": {"nx": int(lr_model.nx), "nz": int(params.get("nz", 1))},
        "rollout_intervals": args.steps,
        "batch_size": args.batch_size,
        "workloads": {},
    }
    jobs = {}
    if "physics" in args.workloads:
        jobs["physics"] = (
            lambda state: lr_model.get_updates(state),
            (template_state,),
        )

    closure_is_stateful = bool(getattr(closure, "stateful", False))
    if "closure" in args.workloads:
        latent_state = init_state.state.param_aux.value

        def closure_call(parameters, field, latent):
            module = eqx.combine(parameters, closure_static)
            if closure_is_stateful:
                return module(field, latent)
            return module(field)

        jobs["closure"] = (
            closure_call,
            (closure_params, trajectories[0, 0, 0], latent_state),
        )

    if "rollout" in args.workloads:
        def rollout_call(state):
            def step(carry, _):
                next_state = forced_model.step_model(carry, closure_params)
                return next_state, next_state.state.model_state.qh

            return jax.lax.scan(step, state, None, length=args.steps)

        jobs["rollout"] = (rollout_call, (init_state,))

    if "train" in args.workloads:
        learning_rate = float(cfg.architectures[model_type].get("learning_rate", 1e-4))
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), optax.adam(learning_rate)
        )
        optim_state = optimizer.init(eqx.filter(closure, eqx.is_array))
        loss_fn = build_loss("mae")
        cfl_limit = float(getattr(cfg.plotting, "cfl", 1.0))
        train_forcing_key = jax.random.PRNGKey(args.seed)

        def train_workload(parameters, opt_state, batches):
            def loss_rollout(current_params, batch):
                errors, max_cfl, _, _ = jax.vmap(
                    functools.partial(
                        rollout_traj_errors,
                        forced_model=forced_model,
                        template_state=template_state,
                        closure_params=current_params,
                        lr_model=lr_model,
                        dt=float(lr_model.dt),
                        forcing_key=train_forcing_key,
                    )
                )(batch)
                return jnp.mean(loss_fn(errors, lr_model)), max_cfl

            def step(carry, batch):
                current_params, current_opt_state = carry
                (loss, max_cfl), grads = eqx.filter_value_and_grad(
                    loss_rollout, has_aux=True
                )(current_params, batch)
                discard = jnp.any(max_cfl > cfl_limit)

                def update(values):
                    params, state, gradients = values
                    updates, new_state = optimizer.update(gradients, state, params)
                    return eqx.apply_updates(params, updates), new_state

                def skip(values):
                    params, state, _ = values
                    return params, state

                next_params, next_opt_state = jax.lax.cond(
                    discard,
                    skip,
                    update,
                    (current_params, current_opt_state, grads),
                )
                return (next_params, next_opt_state), (loss, discard)

            return jax.lax.scan(step, (parameters, opt_state), batches)

        jobs["train"] = (
            train_workload,
            (closure_params, optim_state, trajectories),
        )

    trace_started = False
    if args.trace_dir:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
        jax.profiler.start_trace(str(args.trace_dir.resolve()))
        trace_started = True
        print(f"[trace] recording profiler data to {args.trace_dir.resolve()}")

    try:
        for name, (function, function_args) in jobs.items():
            print(f"[profile] {name}")
            try:
                if name == "train":
                    warmup, repeats = args.train_warmup, args.train_repeats
                else:
                    warmup, repeats = args.warmup, args.repeats
                report["workloads"][name] = _profile(
                    name, function, function_args, warmup, repeats
                )
            except Exception as exc:
                report["workloads"][name] = _as_reportable_error(exc)
                print(f"[{name}] FAILED: {type(exc).__name__}: {exc}")
    finally:
        if trace_started:
            jax.profiler.stop_trace()

    return report


def main():
    args = parse_args()
    try:
        report = _run(args)
    except Exception as exc:
        raise SystemExit(f"Profiler setup failed: {type(exc).__name__}: {exc}") from exc

    rendered = json.dumps(report, indent=2, default=str)
    _print_summary(report)
    print("\n=== Detailed workload report (JSON) ===")
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"\nWrote report to {args.output.resolve()}")


if __name__ == "__main__":
    main()
