"""Probe FNO spectral coverage, initialization balance, and block signal flow.

Example:
    python scripts/diagnose_fno.py \
        --metadata outputs/sweep_fno/run_1/fno/512_to_32/74/metadata.json \
        --data-dir data/data_hr512_nx32_08

Pass --checkpoint-dir pointing at a trained closure directory to diagnose its
weights instead of a fresh initialization. The script is read-only.
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from model.ML.architectures.build_model import build_closure
from model.ML.utils.dataloading import ZarrDataLoader, checkpointer


def rms(value):
    return jnp.sqrt(jnp.mean(jnp.square(jnp.abs(value))))


def old_and_current_masks(height, width, xmodes, ymodes, old_xmodes, old_ymodes):
    current = np.zeros((height, width // 2 + 1), dtype=bool)
    current_y = min(int(ymodes), height // 2)
    current_x = min(int(xmodes), width // 2 + 1)
    if current_y and current_x:
        current[:current_y, :current_x] = True
        current[-current_y:, :current_x] = True

    legacy = np.zeros_like(current)
    legacy_y = min(int(old_xmodes), height)
    legacy_x = min(int(old_ymodes), width // 2 + 1)
    if legacy_y and legacy_x:
        if legacy_y >= height:
            legacy[:, :legacy_x] = True
        else:
            legacy[:legacy_y, :legacy_x] = True
            legacy[-legacy_y:, :legacy_x] = True
    return current, legacy


def independent_spectral_requested(layer, x):
    """Return coefficients from independent positive/negative-y weights."""
    height, width = x.shape[-2:]
    x_ft = jnp.fft.rfft2(x[None, ...], axes=(-2, -1))
    out_ft = jnp.zeros(
        (1, layer.out_channels, height, width // 2 + 1),
        dtype=jnp.complex64,
    )
    y_modes = min(layer.ymodes, height // 2)
    x_modes = min(layer.xmodes, width // 2 + 1)
    if y_modes and x_modes:
        out_ft = out_ft.at[:, :, :y_modes, :x_modes].set(
            layer.compl_mul2d(
                x_ft[:, :, :y_modes, :x_modes],
                layer.weights1[:, :, :y_modes, :x_modes],
            )
        )
        out_ft = out_ft.at[:, :, -y_modes:, :x_modes].set(
            layer.compl_mul2d(
                x_ft[:, :, -y_modes:, :x_modes],
                layer.weights2[:, :, :y_modes, :x_modes],
            )
        )
    return out_ft


def boundary_projection_stats(layer, x):
    independent = independent_spectral_requested(layer, x)
    constrained = layer._spectral_coefficients(x)
    reconstructed = jnp.fft.rfft2(
        jnp.fft.irfft2(constrained, s=x.shape[-2:], axes=(-2, -1)),
        axes=(-2, -1),
    )
    boundary_columns = [0]
    if x.shape[-1] % 2 == 0 and layer.xmodes > x.shape[-1] // 2:
        boundary_columns.append(x.shape[-1] // 2)

    independent_boundary = independent[..., boundary_columns]
    constrained_boundary = constrained[..., boundary_columns]
    relative_boundary_adjustment = rms(
        independent_boundary - constrained_boundary
    ) / (
        rms(independent_boundary) + 1e-12
    )
    relative_reconstruction_error = rms(constrained - reconstructed) / (
        rms(constrained) + 1e-12
    )
    relative_total_adjustment = rms(independent - constrained) / (
        rms(independent) + 1e-12
    )
    boundary_energy_share = jnp.sum(jnp.abs(independent_boundary) ** 2) / (
        jnp.sum(jnp.abs(independent) ** 2) + 1e-12
    )
    return {
        "boundary_columns": boundary_columns,
        "boundary_relative_adjustment_from_independent_weights": float(
            relative_boundary_adjustment
        ),
        "all_modes_relative_adjustment_from_independent_weights": float(
            relative_total_adjustment
        ),
        "constrained_coefficients_relative_reconstruction_error": float(
            relative_reconstruction_error
        ),
        "boundary_share_of_independent_spectral_energy": float(boundary_energy_share),
    }


def parameter_gradient_stats(module, input_value):
    params, static = eqx.partition(module, eqx.is_array)

    def energy(current_params):
        current_module = eqx.combine(current_params, static)
        return 0.5 * jnp.mean(current_module(input_value) ** 2)

    gradients = eqx.filter_grad(energy)(params)
    leaves = jax.tree_util.tree_leaves(eqx.filter(gradients, eqx.is_array))
    squared_norm = sum(jnp.sum(jnp.abs(leaf) ** 2) for leaf in leaves)
    parameter_count = sum(leaf.size for leaf in leaves)
    return {
        "l2_norm": float(jnp.sqrt(squared_norm)),
        "rms_per_parameter": float(jnp.sqrt(squared_norm / max(parameter_count, 1))),
        "parameter_count": int(parameter_count),
    }


def block_signal_stats(model, inputs):
    per_sample_stats = []
    first_input = inputs[0]
    first_x = model.input_proj(model.prepare_inputs(first_input))
    first_spectral = model.spec_layers[0](first_x)
    first_local = model.w_layers[0](first_x)
    spectral_rms = float(rms(first_spectral))
    local_rms = float(rms(first_local))

    for input_value in inputs:
        x = model.input_proj(model.prepare_inputs(input_value))
        block_outputs = []
        spectral_values = []
        local_values = []
        for spectral, local in zip(model.spec_layers, model.w_layers):
            spectral_value = spectral(x)
            local_value = local(x)
            x = x + model.activation(spectral_value + local_value)
            block_outputs.append(x)
            spectral_values.append(rms(spectral_value))
            local_values.append(rms(local_value))

        def continue_from(value, block_index, original_input):
            for next_index in range(block_index + 1, model.n_layers):
                value = value + model.activation(
                    model.spec_layers[next_index](value)
                    + model.w_layers[next_index](value)
                )
            value = model.proj1(value)
            value = model.activation(value)
            return model.proj2(value) + model.skip_proj(original_input)

        block_grad_rms = []
        for block_index, block_value in enumerate(block_outputs):
            gradient = jax.grad(
                lambda value: jnp.mean(
                    continue_from(value, block_index, input_value) ** 2
                )
            )(block_value)
            block_grad_rms.append(rms(gradient))

        network_output = continue_from(
            block_outputs[-1], model.n_layers - 1, input_value
        )
        per_sample_stats.append(
            {
                "spectral_branch_rms": [float(value) for value in spectral_values],
                "local_branch_rms": [float(value) for value in local_values],
                "block_output_rms": [float(rms(value)) for value in block_outputs],
                "output_rms": float(rms(network_output)),
                "output_spatial_std": float(jnp.std(network_output)),
                "gradient_rms_at_block_output": [float(value) for value in block_grad_rms],
            }
        )

    def mean_by_block(field):
        values = np.asarray([sample[field] for sample in per_sample_stats], dtype=float)
        return np.mean(values, axis=0).tolist()

    first_gradients = {
        "spectral_branch": parameter_gradient_stats(model.spec_layers[0], first_x),
        "local_branch": parameter_gradient_stats(model.w_layers[0], first_x),
    }
    return {
        "sample_count": len(inputs),
        "mean_spectral_branch_rms_by_block": mean_by_block("spectral_branch_rms"),
        "mean_local_branch_rms_by_block": mean_by_block("local_branch_rms"),
        "mean_block_output_rms": mean_by_block("block_output_rms"),
        "mean_gradient_rms_at_block_output": mean_by_block(
            "gradient_rms_at_block_output"
        ),
        "mean_network_output_rms": float(
            np.mean([sample["output_rms"] for sample in per_sample_stats])
        ),
        "mean_network_output_spatial_std": float(
            np.mean([sample["output_spatial_std"] for sample in per_sample_stats])
        ),
        "first_block_branch_rms_ratio_spectral_to_local": spectral_rms
        / (local_rms + 1e-12),
        "first_block_spectral_multiplier_to_match_local_rms": local_rms
        / (spectral_rms + 1e-12),
        "first_block_branch_parameter_gradients_for_half_mean_square_branch_output": first_gradients,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path, help="Run metadata JSON")
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Directory containing the training trajectories.zarr",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Optional trained closure directory containing closure_ckpt.npz",
    )
    parser.add_argument("--max-trajectories", type=int, default=3)
    parser.add_argument("--frames-per-trajectory", type=int, default=8)
    parser.add_argument(
        "--legacy-xmodes",
        type=int,
        default=32,
        help="xmodes from the pre-axis-correction implementation",
    )
    parser.add_argument(
        "--legacy-ymodes",
        type=int,
        default=16,
        help="ymodes from the pre-axis-correction implementation",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_trajectories <= 0 or args.frames_per_trajectory <= 0:
        raise ValueError("Trajectory and frame counts must be positive")

    metadata_path = args.metadata.resolve()
    data_dir = args.data_dir.resolve()
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)

    parameters = metadata["parameters"]
    model_arch = metadata["training"]["model_arch"]
    if metadata.get("model_type") != "fno":
        raise ValueError(f"Expected FNO metadata, got {metadata.get('model_type')!r}")

    cfg = OmegaConf.create(
        {
            "params": parameters,
            "ml": {"model_type": "fno"},
            "architectures": {"fno": model_arch},
        }
    )
    loaded_leaves = None
    if args.checkpoint_dir is not None:
        loaded_leaves, _, _, _ = checkpointer(
            None, None, str(args.checkpoint_dir.resolve()), save=False
        )
        if loaded_leaves is None:
            raise ValueError(f"No trained closure checkpoint found in {args.checkpoint_dir}")
        template = build_closure(cfg)
        template_params, _ = eqx.partition(template, eqx.is_array)
        template_leaves = jax.tree_util.tree_leaves(template_params)
        loaded_shapes = [np.shape(leaf) for leaf in loaded_leaves]
        template_shapes = [leaf.shape for leaf in template_leaves]
        if loaded_shapes != template_shapes:
            raise ValueError(
                "Checkpoint parameter shapes do not match the architecture in the run metadata"
            )
    model = build_closure(cfg, loaded_leaves)

    data = ZarrDataLoader(str(data_dir))
    trajectory_count = min(args.max_trajectories, data.n_trajectories)
    if trajectory_count == 0:
        raise ValueError(f"No training trajectories found in {data_dir}")

    samples = []
    for trajectory_index in range(trajectory_count):
        trajectory, _ = data.get_trajectory(trajectory_index)
        indices = np.linspace(
            0, trajectory.shape[0] - 1, args.frames_per_trajectory, dtype=int
        )
        samples.extend(trajectory[indices])
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 4:
        raise ValueError(f"Expected sample fields shaped (N, C, H, W), got {samples.shape}")
    if samples.shape[-2:] != (int(parameters["nx"]), int(parameters["nx"])):
        raise ValueError(
            f"Training data grid {samples.shape[-2:]} does not match metadata nx={parameters['nx']}"
        )

    layer_axis = samples.ndim - 3
    reduce_axes = tuple(axis for axis in range(samples.ndim) if axis != layer_axis)
    q_mean = np.mean(samples, axis=reduce_axes, keepdims=True)
    q_std = np.maximum(np.std(samples, axis=reduce_axes, keepdims=True), 1e-6)
    normalized_samples = jnp.asarray((samples - q_mean) / (q_std + 1e-6))

    height, width = samples.shape[-2:]
    current_mask, legacy_mask = old_and_current_masks(
        height,
        width,
        model.spec_layers[0].xmodes,
        model.spec_layers[0].ymodes,
        args.legacy_xmodes,
        args.legacy_ymodes,
    )
    mean_spectrum = np.mean(
        np.abs(
            np.fft.rfftn(np.asarray(normalized_samples), axes=(-2, -1), norm="ortho")
        )
        ** 2,
        axis=(0, 1),
    )
    total_spectral_energy = np.sum(mean_spectrum) + 1e-12
    current_energy_share = float(np.sum(mean_spectrum[current_mask]) / total_spectral_energy)
    legacy_energy_share = float(np.sum(mean_spectrum[legacy_mask]) / total_spectral_energy)

    initial_layer_input = model.input_proj(
        model.prepare_inputs(normalized_samples[0])
    )
    boundary_stats = boundary_projection_stats(model.spec_layers[0], initial_layer_input)
    signal_stats = block_signal_stats(model, normalized_samples)
    report = {
        "metadata": str(metadata_path),
        "data_dir": str(data_dir),
        "checkpoint_dir": str(args.checkpoint_dir.resolve()) if args.checkpoint_dir else None,
        "samples": {
            "trajectory_count": trajectory_count,
            "frames_per_trajectory": args.frames_per_trajectory,
            "shape": list(samples.shape),
        },
        "spectral_coverage": {
            "grid_shape_y_x": [height, width],
            "current_corrected_modes": {
                "ymodes": model.spec_layers[0].ymodes,
                "xmodes": model.spec_layers[0].xmodes,
                "retained_mode_count": int(np.sum(current_mask)),
                "input_spectral_energy_share": current_energy_share,
            },
            "legacy_axis_semantics": {
                "old_xmodes": args.legacy_xmodes,
                "old_ymodes": args.legacy_ymodes,
                "retained_mode_count": int(np.sum(legacy_mask)),
                "input_spectral_energy_share": legacy_energy_share,
            },
            "old_only_input_spectral_energy_share": float(
                np.sum(mean_spectrum[legacy_mask & ~current_mask]) / total_spectral_energy
            ),
        },
        "first_spectral_layer_boundary_projection": boundary_stats,
        "signal_and_gradient_scales": signal_stats,
    }

    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + os.linesep, encoding="utf-8")
        print(f"\nWrote diagnostic report to {output_path}")


if __name__ == "__main__":
    main()
