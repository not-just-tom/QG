'''
Script to compare the performance of different models trained on the same dataset.
Loads and compares models on a given validation dataset
'''

import argparse
import os
import sys
import jax
import json
import numpy as np
import jax.numpy as jnp
import importlib
import zarr
from omegaconf import OmegaConf
import model.ML.train
import model.ML.architectures.build_model
import model.ML.utils.dataloading
import model.core.model
import model.core.grid
import model.utils.plotting
import model.utils.config
importlib.reload(model.ML.train)
importlib.reload(model.ML.architectures.build_model)
importlib.reload(model.ML.utils.dataloading)
importlib.reload(model.core.model)
importlib.reload(model.core.grid)
importlib.reload(model.utils.plotting)
importlib.reload(model.utils.config)
from model.ML.train import rollout_traj_errors
from model.ML.architectures.build_model import build_closure
from model.ML.utils.dataloading import checkpointer, load_forced_model
from model.utils.plotting import Plotter
from model.utils.config import Config
from model.core.steppers import SteppedModel, AB3Stepper, CNABStepper
from model.core.model import QGM
from model.ML.utils.coarsen import coarsen
from model.core.grid import build_grid


def load_model_from_closure_dir(closure_dir):
    """Load a trained model from a saved closure directory.
    
    Args:
        closure_dir: Path to directory containing closure_ckpt.npz and metadata.json
        
    Returns:
        dict with keys: closure, metadata, loss_history, model_name
    """
    # go to parent directory of closure_dir
    closure_dir = os.path.abspath(closure_dir)
    print(f"Loading model from: {closure_dir}")
    
    # Load metadata
    metadata_path = os.path.join(closure_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.json not found in {closure_dir}")
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    # Load checkpoint data
    loaded_params_leaves, _, ckpt_meta, loss_history = checkpointer(
        None, None, closure_dir, save=False
    )
    
    if loaded_params_leaves is None:
        raise ValueError(f"Failed to load closure parameters from {closure_dir}")
    
    # Extract model info from metadata
    model_type = metadata.get("model_type", "unknown")
    arch_params = metadata.get("training", {}).get("model_arch", {})
    
    # Build closure with loaded parameters
    closure = build_closure(
        cfg=None,
        loaded_leaves=loaded_params_leaves,
        model_type=model_type,
        arch_params=arch_params
    )
    
    # Create a friendly model name from the closure directory
    model_name = os.path.basename(os.path.dirname(os.path.dirname(closure_dir)))
    model_name += f"_{os.path.basename(os.path.dirname(closure_dir))}"
    model_name += f"_{os.path.basename(closure_dir)}"
    
    print(f"  Model type: {model_type}")
    print(f"  Model name: {model_name}")
    print(f"  Parameters: nx={metadata['parameters']['nx']}, hr_nx={metadata['parameters']['hr_nx']}")
    
    return {
        "closure": closure,
        "metadata": metadata,
        "loss_history": loss_history,
        "model_name": model_name,
        "model_type": model_type,
        "arch_params": arch_params,
    }


def load_validation_data(data_path):
    """Load validation trajectories from zarr.

    Returns
    -------
    np.ndarray
        Shape: (n_trajs, nt, nz, ny, nx)
    """
    print(f"Loading validation data from: {data_path}")

    zarr_path = os.path.join(
        data_path,
        "trajectories.zarr",
        "trajectories"
    )

    if not os.path.exists(zarr_path):
        raise FileNotFoundError(
            f"Trajectories not found at {zarr_path}"
        )

    group = zarr.open(zarr_path, mode="r")

    # Ensure deterministic ordering
    traj_keys = sorted(group.keys())

    trajs = np.stack(
        [np.array(group[k]) for k in traj_keys],
        axis=0
    )

    print(f"  Loaded trajectories shape: {trajs.shape}")
    print(f"  dtype: {trajs.dtype}")

    return trajs


def build_config_from_metadata(metadata):
    """Build a Config object from metadata.
    
    Args:
        metadata: Metadata dict from saved model
        
    Returns:
        Config object
    """
    # Create a minimal config dictionary
    cfg_dict = {
        "params": metadata["parameters"],
        "timing": metadata["timing"],
        "ml": {
            "model_type": metadata["model_type"],
            "enabled": True,
        },
        "architectures": {
            metadata["model_type"]: metadata.get("training", {}).get("model_arch", {})
        },
        "filepaths": {
            "out_dir": ".",
        },
        "plotting": {
            "cadence": 10,
            "plot": [],
        }
    }
    
    return cfg_dict


def rollout_model(model_info, validation_trajs, n_trajs=5):
    """Rollout a model on validation trajectories.
    
    Args:
        model_info: Dict from load_model_from_closure_dir
        validation_trajs: Array of validation trajectories
        n_trajs: Number of trajectories to evaluate
        
    Returns:
        dict with rollout results
    """
    print(f"\nRolling out {model_info['model_name']}...")
    
    closure = model_info["closure"]
    metadata = model_info["metadata"]
    
    # Build config from metadata
    cfg = build_config_from_metadata(metadata)
    params = cfg["params"]
    dt = cfg["timing"].get("final dt", cfg["timing"].get("dt (original)", 0.01))
    
    # Build low-res model
    # instantiate the model
    hr_model = SteppedModel(
        model=QGM({**params, "nx": params['hr_nx']}),
        stepper=AB3Stepper(dt=dt),
    )
    # build low-resolution physics model (coarsened from high-res physics)
    lr_model = coarsen(hr_model.model, params['nx'])
    template_state = lr_model.initialise(jax.random.PRNGKey(0))
    
    # Get dt from metadata
    dt = metadata["timing"].get("final dt", metadata["timing"].get("dt (original)", 0.01))
    
    # Select subset of validation trajectories
    eval_trajs = validation_trajs[:n_trajs]
    
    
    # Build forced model with closure
    forced_model, closure_params, closure_static, q_mean, q_std, dq_mean, dq_std = load_forced_model(
        lr_model,
        closure,
        dt,
        trajs=eval_trajs,
        closure_scale=metadata.get("training", {}).get("model_arch", {}).get("closure_scale", 0.2),
    )
    
    # Rollout each trajectory
    all_residuals = []
    all_pred_qh = []
    all_target_qh = []
    max_cfls = []
    
    for i, traj in enumerate(eval_trajs):
        residual_q, max_cfl, target_qh, pred_qh = rollout_traj_errors(
            traj, forced_model, template_state, closure_params, lr_model, dt
        )
        all_residuals.append(residual_q)
        all_pred_qh.append(pred_qh)
        all_target_qh.append(target_qh)
        max_cfls.append(max_cfl)
    
    # Stack results
    all_residuals = jnp.stack(all_residuals, axis=0)
    all_pred_qh = jnp.stack(all_pred_qh, axis=0)
    all_target_qh = jnp.stack(all_target_qh, axis=0)
    max_cfls = jnp.array(max_cfls)
    
    # Compute metrics
    mse_loss = jnp.mean(all_residuals ** 2)
    mae_loss = jnp.mean(jnp.abs(all_residuals))
    avg_cfl = jnp.mean(max_cfls)
    
    print(f"  MSE Loss: {mse_loss:.6f}")
    print(f"  MAE Loss: {mae_loss:.6f}")
    print(f"  Avg Max CFL: {avg_cfl:.6f}")
    
    # Convert predictions back to physical space for plotting
    pred_frames = jax.vmap(lambda x: jnp.fft.irfftn(x, axes=(-2, -1), norm='ortho', s=eval_trajs.shape[-2:]))(
        all_pred_qh
    )
    
    return {
        "residuals": np.array(all_residuals),
        "pred_qh": np.array(all_pred_qh),
        "target_qh": np.array(all_target_qh),
        "pred_frames": np.array(pred_frames),
        "mse_loss": float(mse_loss),
        "mae_loss": float(mae_loss),
        "max_cfls": np.array(max_cfls),
        "avg_cfl": float(avg_cfl),
    }


def compare_models(folder_dir, validation_data_path, output_dir, n_trajs=5):
    """Compare multiple models on validation data.
    
    Args:
        model_dirs: List of paths to saved closure directories
        validation_data_path: Path to validation data directory
        output_dir: Directory to save comparison results
        n_trajs: Number of trajectories to evaluate
    """
    os.makedirs(output_dir, exist_ok=True)
    model_dirs = [
        os.path.join(folder_dir, d)
        for d in os.listdir(folder_dir)
        if os.path.isdir(os.path.join(folder_dir, d))
    ]
    # Load all models
    models = []
    for model_dir in model_dirs:
        try:
            model_info = load_model_from_closure_dir(model_dir)
            models.append(model_info)
        except Exception as e:
            print(f"Error loading model from {model_dir}: {e}")
            continue
    
    if not models:
        raise ValueError("No models successfully loaded")
    
    # Load validation data
    validation_trajs = load_validation_data(validation_data_path)
    
    # Rollout each model
    results = {}
    for model_info in models:
        model_name = model_info["model_name"]
        try:
            rollout_results = rollout_model(model_info, validation_trajs, n_trajs=n_trajs)
            results[model_name] = {
                **rollout_results,
                "loss_history": model_info["loss_history"],
                "metadata": model_info["metadata"],
            }
        except Exception as e:
            print(f"Error rolling out {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not results:
        raise ValueError("No models successfully evaluated")
    
    # Prepare trajectories dict for Plotter
    # Use the first model's metadata for grid construction
    first_model = models[0]
    cfg = build_config_from_metadata(first_model["metadata"])
    
    trajs_dict = {
        "truth": validation_trajs[:n_trajs],
        "grid": build_grid(params=cfg["params"]),
    }
    
    # Add each model's predictions and loss history
    for model_name, result in results.items():
        trajs_dict[f"pred_{model_name}"] = result["pred_frames"]
        trajs_dict[f"loss_history_{model_name}"] = result["loss_history"]
    
    # Create comparison plots using Plotter
    cfg_dict = first_model["metadata"].copy()
    cfg_dict["plotting"] = {
        "cadence": 10,
        "plot": ["pareto_validation", "multi_model_comparison", "multi_model_loss"],
    }
    cfg = Config(cfg_dict)
    
    # Store model names for the plotter
    trajs_dict["model_names"] = list(results.keys())
    trajs_dict["model_results"] = results
    
    plotter = Plotter(cfg, trajectories=trajs_dict, out_dir=output_dir)
    plotter.plot()
    
    # Save comparison metrics to JSON
    metrics = {
        model_name: {
            "mse_loss": result["mse_loss"],
            "mae_loss": result["mae_loss"],
            "avg_cfl": result["avg_cfl"],
        }
        for model_name, result in results.items()
    }
    
    metrics_path = os.path.join(output_dir, "comparison_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\nComparison complete! Results saved to {output_dir}")
    print(f"Metrics saved to {metrics_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare multiple trained models")
    parser.add_argument(
        "--folder_dir",
        required=True,
        help="Directory containing model directories"
    )
    parser.add_argument(
        "--validation_data",
        required=True,
        help="Path to validation data directory (e.g., data/data_hr128_nx32_01)"
    )
    parser.add_argument(
        "--output_dir",
        default="../outputs/model_comparison",
        help="Directory to save comparison results"
    )
    parser.add_argument(
        "--n_trajs",
        type=int,
        default=5,
        help="Number of trajectories to evaluate"
    )
    
    args = parser.parse_args()
    
    compare_models(
        args.folder_dir,
        args.validation_data,
        args.output_dir,
        args.n_trajs
    )

