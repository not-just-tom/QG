#!/usr/bin/env python3
"""Plot data-only diagnostics for saved trajectory datasets."""

import argparse
import json
from pathlib import Path

import jax
import matplotlib
import numpy as np
import zarr
from omegaconf import OmegaConf

matplotlib.use("Agg")

from model.core.model import QGM
from model.ML.utils.coarsen import coarsen
from model.utils.diagnostics import build_diagnostic


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "data_hr512_nx32_13"
DEFAULT_DIAGNOSTICS = ("ke_spectrum", "pv_spectrum", "domain", "zonal")


def _has_group_metadata(path: Path) -> bool:
    return (path / "zarr.json").is_file() or (path / ".zgroup").is_file()


def _is_array_store(path: Path) -> bool:
    zarr_metadata = path / "zarr.json"
    if zarr_metadata.is_file():
        with zarr_metadata.open(encoding="utf-8") as metadata_file:
            return json.load(metadata_file).get("node_type") == "array"
    return (path / ".zarray").is_file()


def discover_trajectory_arrays(zarr_path: Path):
    """Open trajectory arrays in a regular group store or a leaf-only Zarr store."""
    if not zarr_path.is_dir():
        raise FileNotFoundError(f"Zarr store directory not found: {zarr_path}")

    trajectories_path = zarr_path / "trajectories"
    if _has_group_metadata(zarr_path) and _has_group_metadata(trajectories_path):
        root = zarr.open_group(str(zarr_path), mode="r")
        trajectories = root["trajectories"]
        names = sorted(trajectories.array_keys())
        if names:
            return [(name, trajectories[name]) for name in names]

    if not trajectories_path.is_dir():
        raise FileNotFoundError(
            f"No trajectories group or direct trajectory arrays found in {zarr_path}"
        )

    arrays = []
    for array_path in sorted(trajectories_path.iterdir()):
        if array_path.is_dir() and _is_array_store(array_path):
            arrays.append((array_path.name, zarr.open_array(str(array_path), mode="r")))

    if not arrays:
        raise ValueError(
            f"No readable trajectory arrays found under {trajectories_path}"
        )
    return arrays


def load_config_and_params(config_path: Path, data_dir: Path):
    cfg = OmegaConf.load(config_path)
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Dataset metadata file not found: {metadata_path}")
    with metadata_path.open(encoding="utf-8") as metadata_file:
        dataset_metadata = json.load(metadata_file)

    params = dict(OmegaConf.to_container(cfg.params, resolve=True))
    params.update(dataset_metadata.get("parameters", {}))
    timing = dataset_metadata.get("timing", {})
    dt = float(timing.get("dt (original)", params.get("dt", cfg.plotting.dt)))
    auto_dt = bool(timing.get("auto_dt", cfg.plotting.auto_dt))
    return cfg, params, dt, auto_dt


def build_diagnostic_models(params, dt, auto_dt, cfl):
    hr_nx = int(params["hr_nx"])
    lr_nx = int(params["nx"])
    hr_params = {**params, "nx": hr_nx}

    if auto_dt:
        timestep_model = QGM(hr_params)
        key = jax.random.PRNGKey(int(params.get("seed", 0)))
        n_jets = params.get("n_jets")
        init_kwargs = (
            {"n_jets": int(n_jets), "pseudo": True}
            if n_jets is not None
            else {}
        )
        init_state = timestep_model.initialise(key, **init_kwargs)
        dt = float(timestep_model.estimate_cfl_dt(init_state, cfl=cfl))
        dt *= timestep_model.time_scale

    params["dt"] = dt
    hr_model = QGM({**params, "nx": hr_nx})
    return params, coarsen(hr_model, lr_nx)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load saved trajectories and create data-only diagnostics."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (defaults under config.filepaths.out_dir).",
    )
    parser.add_argument(
        "--trajectory-index",
        type=int,
        help="Process only this zero-based trajectory index; default is all.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Maximum frames per trajectory; defaults to plotting.nsteps, 0 means all.",
    )
    parser.add_argument(
        "--diagnostics",
        nargs="+",
        choices=DEFAULT_DIAGNOSTICS,
        default=list(DEFAULT_DIAGNOSTICS),
        help="Data-only diagnostics to generate.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config.resolve()
    data_dir = args.data_dir.resolve()
    cfg, params, dt, auto_dt = load_config_and_params(config_path, data_dir)
    params, lr_model = build_diagnostic_models(
        params, dt, auto_dt, float(cfg.plotting.cfl)
    )

    if args.output_dir is None:
        output_root = Path(str(cfg.filepaths.out_dir))
        if not output_root.is_absolute():
            output_root = PROJECT_ROOT / output_root
        output_root = output_root.resolve() / "trajectory_diagnostics" / data_dir.name
    else:
        output_root = args.output_dir.resolve()

    trajectories = discover_trajectory_arrays(data_dir / "trajectories.zarr")
    if args.trajectory_index is not None:
        if not 0 <= args.trajectory_index < len(trajectories):
            raise IndexError(
                f"Trajectory index {args.trajectory_index} is out of range "
                f"for {len(trajectories)} trajectories"
            )
        trajectories = [trajectories[args.trajectory_index]]

    configured_steps = int(cfg.plotting.nsteps)
    max_steps = configured_steps if args.max_steps is None else args.max_steps
    if max_steps < 0:
        raise ValueError("--max-steps must be non-negative")
    cadence = int(cfg.plotting.cadence)

    print(f"Dataset: {data_dir}")
    print(f"Diagnostics: {', '.join(args.diagnostics)}")
    print(f"Output directory: {output_root}")

    for trajectory_name, trajectory_array in trajectories:
        n_steps = trajectory_array.shape[0]
        if max_steps:
            n_steps = min(n_steps, max_steps)
        q_traj = np.asarray(trajectory_array[:n_steps], dtype=np.float32)
        if q_traj.ndim != 4:
            raise ValueError(
                f"{trajectory_name} must have shape (time, layer, y, x), "
                f"got {q_traj.shape}"
            )

        trajectory_data = {
            "truth": q_traj,
            "grid": lr_model.get_grid(),
            "params": params,
            "dt": float(lr_model.dt),
        }
        trajectory_output = output_root / trajectory_name
        trajectory_output.mkdir(parents=True, exist_ok=True)
        print(f"Processing {trajectory_name}: {q_traj.shape}")

        for diagnostic_name in args.diagnostics:
            diagnostic = build_diagnostic(diagnostic_name)
            output_path = trajectory_output / (
                f"{trajectory_name}_{diagnostic_name}.{diagnostic.output}"
            )
            diagnostic.run(trajectory_data, str(output_path), cadence)
            print(f"Saved {output_path}")

        del q_traj, trajectory_data


if __name__ == "__main__":
    main()
