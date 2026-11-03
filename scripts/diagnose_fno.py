"""Probe FNO spectral coverage, initialization balance, and block signal flow.

Example:
    python scripts/diagnose_fno.py \
        --metadata outputs/sweep_fno/run_1/fno/512_to_32/74/metadata.json \
        --data-dir data/data_hr512_nx32_08 \
        --schedule-steps 1000

Pass --checkpoint-dir pointing at a trained closure directory to diagnose its
weights instead of a fresh initialization. The script is read-only and probes
the production FNO's spectral branches, gradients, rolling history, and
optimizer schedule.
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
import optax
from omegaconf import OmegaConf

from model.ML.architectures.build_model import build_closure
from model.ML.architectures.memory import TemporalMemory, TemporalMemoryState
from model.ML.utils.dataloading import ZarrDataLoader, checkpointer


def rms(value):
    return jnp.sqrt(jnp.mean(jnp.square(jnp.abs(value))))


def old_and_current_masks(height, width, xmodes, ymodes, old_xmodes, old_ymodes):
    current = np.zeros((height, width // 2 + 1), dtype=bool)
    current_y = min(int(ymodes), (height - 1) // 2)
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
    y_modes = min(layer.ymodes, (height - 1) // 2)
    x_modes = min(layer.xmodes, width // 2 + 1)
    if y_modes and x_modes:
        out_ft = out_ft.at[:, :, :y_modes, :x_modes].set(
            jnp.einsum(
                "bixy,ioxy->boxy",
                x_ft[:, :, :y_modes, :x_modes],
                layer.weights1[:, :, :y_modes, :x_modes],
            )
        )
        out_ft = out_ft.at[:, :, -y_modes:, :x_modes].set(
            jnp.einsum(
                "bixy,ioxy->boxy",
                x_ft[:, :, -y_modes:, :x_modes],
                layer.weights2[:, :, :y_modes, :x_modes],
            )
        )
    return out_ft


def project_real_fft_boundaries(coefficients, width):
    """Project rFFT zero/Nyquist columns onto Hermitian symmetry in y."""
    boundary_columns = [0]
    if width % 2 == 0:
        boundary_columns.append(width // 2)
    y_partner = (-jnp.arange(coefficients.shape[-2])) % coefficients.shape[-2]
    boundary = coefficients[..., boundary_columns]
    paired_boundary = jnp.take(boundary, y_partner, axis=-2)
    return coefficients.at[..., boundary_columns].set(
        0.5 * (boundary + jnp.conj(paired_boundary))
    )


def boundary_projection_stats(layer, x):
    independent = independent_spectral_requested(layer, x)
    expected = project_real_fft_boundaries(independent, x.shape[-1])
    actual = jnp.fft.rfft2(
        layer(x),
        axes=(-2, -1),
    )
    boundary_columns = [0]
    if x.shape[-1] % 2 == 0 and layer.xmodes > x.shape[-1] // 2:
        boundary_columns.append(x.shape[-1] // 2)

    independent_boundary = independent[..., boundary_columns]
    expected_boundary = expected[..., boundary_columns]
    actual_boundary = actual[..., boundary_columns]
    boundary_pair_indices = (-jnp.arange(x.shape[-2])) % x.shape[-2]
    boundary_hermitian_error = rms(
        actual_boundary
        - jnp.conj(jnp.take(actual_boundary, boundary_pair_indices, axis=-2))
    ) / (
        rms(actual_boundary) + 1e-12
    )
    relative_boundary_error = rms(expected_boundary - actual_boundary) / (
        rms(expected_boundary) + 1e-12
    )
    relative_total_error = rms(expected - actual) / (
        rms(expected) + 1e-12
    )
    relative_projection_size = rms(independent - expected) / (
        rms(independent) + 1e-12
    )
    boundary_energy_share = jnp.sum(jnp.abs(independent_boundary) ** 2) / (
        jnp.sum(jnp.abs(independent) ** 2) + 1e-12
    )
    return {
        "boundary_columns": boundary_columns,
        "boundary_relative_error_vs_hermitian_projected_weights": float(
            relative_boundary_error
        ),
        "all_modes_relative_error_vs_hermitian_projected_weights": float(
            relative_total_error
        ),
        "boundary_hermitian_symmetry_error": float(
            boundary_hermitian_error
        ),
        "relative_size_of_required_boundary_projection": float(
            relative_projection_size
        ),
        "boundary_share_of_independent_spectral_energy": float(boundary_energy_share),
    }


def block_signal_stats(model, inputs):
    per_sample_stats = []
    first_input = inputs[0]
    first_x = model.input_proj(first_input)
    first_spectral = model.blocks[0].spectral(first_x)
    first_local = model.blocks[0].local(first_x)
    spectral_rms = float(rms(first_spectral))
    local_rms = float(rms(first_local))

    for input_value in inputs:
        x = model.input_proj(input_value)
        block_outputs = []
        spectral_values = []
        local_values = []
        for block in model.blocks:
            spectral_value = block.spectral(x)
            local_value = block.local(x)
            x = block(x)
            block_outputs.append(x)
            spectral_values.append(rms(spectral_value))
            local_values.append(rms(local_value))

        def continue_from(value, block_index):
            for block in model.blocks[block_index + 1 :]:
                value = block(value)
            return model.proj2(jax.nn.gelu(model.proj1(value)))

        block_grad_rms = []
        for block_index, block_value in enumerate(block_outputs):
            gradient = jax.grad(
                lambda value: jnp.mean(
                    continue_from(value, block_index) ** 2
                )
            )(block_value)
            block_grad_rms.append(rms(gradient))

        network_output = continue_from(block_outputs[-1], len(model.blocks) - 1)
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
    }


def build_input_windows(sequences, field_channels, history_length=9):
    windows = []
    for sequence in sequences:
        for frame_index, current in enumerate(sequence):
            previous = sequence[max(0, frame_index - history_length):frame_index]
            padding_count = history_length - len(previous)
            if padding_count:
                padding = np.repeat(sequence[:1], padding_count, axis=0)
                previous = np.concatenate((padding, previous), axis=0)
            window = np.concatenate((previous, current[None]), axis=0)
            windows.append(
                window.reshape(
                    (history_length + 1) * field_channels,
                    *current.shape[-2:],
                )
            )
    return np.asarray(windows, dtype=np.float32)


def memory_history_stats(closure, input_window):
    if not isinstance(closure, TemporalMemory):
        return {"enabled": False}

    channels = closure.field_channels
    history_length = closure.history_length
    current = input_window[-channels:]
    stored_history = input_window[:-channels]
    zero_history = jnp.zeros_like(stored_history)
    repeated_history = jnp.concatenate([current] * history_length, axis=0)

    def predict(history):
        state = TemporalMemoryState(
            history=history,
            initialized=jnp.asarray(True),
            inner_state=None,
        )
        prediction, _ = closure(current, state)
        return prediction

    predictions = {
        "sampled_history": predict(stored_history),
        "zero_history": predict(zero_history),
        "repeated_current_history": predict(repeated_history),
    }
    reference = predictions["sampled_history"]
    return {
        "enabled": True,
        "current_field_rms": float(rms(current)),
        "sampled_history_rms": float(rms(stored_history)),
        "prediction_rms_by_history": {
            name: float(rms(prediction))
            for name, prediction in predictions.items()
        },
        "relative_prediction_difference_from_sampled_history": {
            name: float(
                rms(prediction - reference) / (rms(reference) + 1e-12)
            )
            for name, prediction in predictions.items()
        },
    }


def laplacian_targets(input_windows, field_channels):
    height, width = input_windows.shape[-2:]
    current = input_windows[:, -field_channels:]
    ky = jnp.fft.fftfreq(height) * height
    kx = jnp.fft.rfftfreq(width) * width
    ky_grid, kx_grid = jnp.meshgrid(ky, kx, indexing="ij")
    current_hat = jnp.fft.rfft2(current, axes=(-2, -1))
    laplacian = jnp.fft.irfft2(
        -(ky_grid**2 + kx_grid**2)[None, None] * current_hat,
        s=(height, width),
        axes=(-2, -1),
    )
    target_scale = rms(current) / (rms(laplacian) + 1e-12)
    return current, laplacian * target_scale


def gradient_group_stats(model, inputs, targets):
    def loss_fn(current_model):
        predictions = jax.vmap(current_model)(inputs)
        return jnp.mean((predictions - targets) ** 2)

    gradients = eqx.filter_grad(loss_fn)(model)

    def stats(tree):
        leaves = [
            leaf
            for leaf in jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_array))
            if leaf is not None
        ]
        squared_norm = sum(jnp.sum(jnp.abs(leaf) ** 2) for leaf in leaves)
        parameter_count = sum(leaf.size for leaf in leaves)
        return {
            "l2_norm": float(jnp.sqrt(squared_norm)),
            "rms_per_parameter": float(
                jnp.sqrt(squared_norm / max(parameter_count, 1))
            ),
            "parameter_count": int(parameter_count),
        }

    projection_gradients = (
        gradients.input_proj,
        gradients.proj1,
        gradients.proj2,
    )
    return {
        "synthetic_target": "normalized Laplacian of current input frame",
        "total": stats(gradients),
        "input_and_output_projections": stats(projection_gradients),
        "spectral_by_block": [
            stats(block.spectral) for block in gradients.blocks
        ],
        "local_by_block": [stats(block.local) for block in gradients.blocks],
    }


def optimizer_schedule_stats(model, metadata, schedule_steps):
    training = metadata.get("training", {})
    optimizer_name = training.get("optimiser", "AdamW")
    if optimizer_name not in ("Adam", "AdamW"):
        raise ValueError(f"Unsupported optimizer in metadata: {optimizer_name!r}")

    learning_rate = float(metadata["training"]["model_arch"].get("learning_rate", 0))
    if learning_rate <= 0:
        raise ValueError("FNO learning_rate in metadata must be positive")

    spectral_multiplier = float(
        metadata["training"]["model_arch"].get("spectral_lr_multiplier", 5.0)
    )
    if spectral_multiplier <= 0:
        raise ValueError("FNO spectral_lr_multiplier in metadata must be positive")

    def label_parameters(params):
        return jax.tree_util.tree_map_with_path(
            lambda path, leaf: (
                "spectral"
                if eqx.is_array(leaf)
                and any(
                    getattr(key, "name", None) == "spectral"
                    for key in path
                )
                else "base" if eqx.is_array(leaf) else None
            ),
            params,
        )

    labels = label_parameters(model)
    counts = {"base": 0, "spectral": 0}
    for label in jax.tree_util.tree_leaves(labels):
        if label in counts:
            counts[label] += 1

    if counts["spectral"] == 0:
        raise ValueError("FNO spectral parameter labels are empty")
    if counts["base"] == 0:
        raise ValueError("FNO base parameter labels are empty")

    report = {
        "optimizer": optimizer_name,
        "base_learning_rate": learning_rate,
        "spectral_learning_rate": learning_rate * spectral_multiplier,
        "spectral_lr_multiplier": spectral_multiplier,
        "parameter_leaf_counts": counts,
    }
    if schedule_steps is not None:
        schedule = optax.cosine_decay_schedule(
            init_value=1.0,
            decay_steps=schedule_steps,
        )
        probe_steps = sorted({0, schedule_steps // 2, schedule_steps})
        report["cosine_schedule"] = {
            "decay_steps": schedule_steps,
            "progress": [
                {
                    "step": step,
                    "learning_rate_scale": float(schedule(step)),
                    "base_effective_rate": float(schedule(step)) * learning_rate,
                    "spectral_effective_rate": (
                        float(schedule(step)) * learning_rate * spectral_multiplier
                    ),
                }
                for step in probe_steps
            ],
        }
    return report


def spectral_mode_stats(layer, height, width):
    requested_modes = [
        (1, 1),
        (max(0, layer.ymodes - 1), max(0, layer.xmodes - 1)),
        (layer.ymodes, 1),
        (1, layer.xmodes),
        (height // 2, 1),
    ]
    cases = []
    yy = jnp.arange(height)[:, None]
    xx = jnp.arange(width)[None, :]
    for ky, kx in dict.fromkeys(requested_modes):
        field = jnp.sin(
            2 * jnp.pi * (ky * yy / height + kx * xx / width)
        )
        probe = jnp.zeros((layer.in_channels, height, width))
        probe = probe.at[0].set(field)
        output = layer(probe)
        active = (
            ky < min(layer.ymodes, (height - 1) // 2)
            and kx < min(layer.xmodes, width // 2 + 1)
        )
        cases.append(
            {
                "ky": ky,
                "kx": kx,
                "expected_active": bool(active),
                "output_to_input_rms": float(rms(output) / (rms(probe) + 1e-12)),
            }
        )
    return cases


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
        "--gradient-samples",
        type=int,
        default=4,
        help="Number of real trajectory windows used for synthetic-target gradient probes",
    )
    parser.add_argument(
        "--schedule-steps",
        type=int,
        help="Optional planned update count for checking cosine-schedule progress",
    )
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
    if (
        args.max_trajectories <= 0
        or args.frames_per_trajectory <= 0
        or args.gradient_samples <= 0
    ):
        raise ValueError("Trajectory, frame, and gradient sample counts must be positive")
    if args.schedule_steps is not None and args.schedule_steps <= 0:
        raise ValueError("--schedule-steps must be positive")
    if args.legacy_xmodes < 0 or args.legacy_ymodes < 0:
        raise ValueError("Legacy mode counts must be non-negative")

    metadata_path = args.metadata.resolve()
    data_dir = args.data_dir.resolve()
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)

    parameters = metadata["parameters"]
    model_arch = metadata["training"]["model_arch"]
    if metadata.get("model_type") != "fno":
        raise ValueError(f"Expected FNO metadata, got {metadata.get('model_type')!r}")

    memory_cfg = metadata.get("training", {}).get("memory", {})
    cfg = OmegaConf.create(
        {
            "params": parameters,
            "ml": {"model_type": "fno", "memory": memory_cfg},
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

    sequences = []
    for trajectory_index in range(trajectory_count):
        trajectory, _ = data.get_trajectory(trajectory_index)
        indices = np.linspace(
            0, trajectory.shape[0] - 1, args.frames_per_trajectory, dtype=int
        )
        sequences.append(np.asarray(trajectory[indices], dtype=np.float32))
    samples = np.concatenate(sequences, axis=0)
    if samples.ndim != 4:
        raise ValueError(f"Expected sample fields shaped (N, C, H, W), got {samples.shape}")
    if samples.shape[1] != int(parameters["nz"]):
        raise ValueError(
            f"Training data has {samples.shape[1]} channels but metadata nz="
            f"{parameters['nz']}"
        )
    if samples.shape[-2:] != (int(parameters["nx"]), int(parameters["nx"])):
        raise ValueError(
            f"Training data grid {samples.shape[-2:]} does not match metadata nx={parameters['nx']}"
        )

    layer_axis = samples.ndim - 3
    reduce_axes = tuple(axis for axis in range(samples.ndim) if axis != layer_axis)
    q_mean = np.mean(samples, axis=reduce_axes, keepdims=True)
    q_std = np.maximum(np.std(samples, axis=reduce_axes, keepdims=True), 1e-6)
    normalized_samples = jnp.asarray((samples - q_mean) / (q_std + 1e-6))
    normalized_sequences = [
        (sequence - q_mean) / (q_std + 1e-6)
        for sequence in sequences
    ]
    configured_memory = cfg.ml.memory
    history_length = (
        int(configured_memory.history_length)
        if configured_memory.get("enabled", False)
        else 9
    )
    input_windows = build_input_windows(
        normalized_sequences,
        int(parameters["nz"]),
        history_length=history_length,
    )

    height, width = samples.shape[-2:]
    base_closure = model.closure if isinstance(model, TemporalMemory) else model
    fno = base_closure.fno
    spectral_layer = fno.blocks[0].spectral
    current_mask, legacy_mask = old_and_current_masks(
        height,
        width,
        spectral_layer.xmodes,
        spectral_layer.ymodes,
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

    current_inputs = jnp.asarray(input_windows[:, -int(parameters["nz"]) :])
    initial_layer_input = fno.input_proj(current_inputs[0])
    boundary_stats = boundary_projection_stats(spectral_layer, initial_layer_input)
    signal_stats = block_signal_stats(
        fno,
        current_inputs[: min(len(current_inputs), args.gradient_samples)],
    )
    gradient_inputs = current_inputs[: args.gradient_samples]
    _, gradient_targets = laplacian_targets(
        gradient_inputs,
        int(parameters["nz"]),
    )
    gradient_stats = gradient_group_stats(
        fno,
        gradient_inputs,
        gradient_targets,
    )
    memory_stats = memory_history_stats(model, jnp.asarray(input_windows[0]))
    optimizer_stats = optimizer_schedule_stats(
        fno,
        metadata,
        args.schedule_steps,
    )
    report = {
        "metadata": str(metadata_path),
        "data_dir": str(data_dir),
        "checkpoint_dir": str(args.checkpoint_dir.resolve()) if args.checkpoint_dir else None,
        "samples": {
            "trajectory_count": trajectory_count,
            "frames_per_trajectory": args.frames_per_trajectory,
            "shape": list(samples.shape),
            "history_window_shape": list(input_windows.shape),
        },
        "spectral_coverage": {
            "grid_shape_y_x": [height, width],
            "current_corrected_modes": {
                "ymodes": spectral_layer.ymodes,
                "xmodes": spectral_layer.xmodes,
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
        "first_spectral_layer_requested_vs_actual_response": boundary_stats,
        "single_mode_response": spectral_mode_stats(
            spectral_layer,
            height,
            width,
        ),
        "signal_and_gradient_scales": signal_stats,
        "parameter_gradients_on_synthetic_laplacian": gradient_stats,
        "rolling_history_sensitivity": memory_stats,
        "fno_optimizer_and_cosine_schedule": optimizer_stats,
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
