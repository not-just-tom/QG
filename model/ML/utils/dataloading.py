import os
import jax
import json
import re
import zarr
import numpy as np
import equinox as eqx
from typing import Optional, List
import logging
logger = logging.getLogger(__name__)

# Match either data or model run directories, e.g.:
#  - data_hr128_nx32_01
#  - cnn_hr128_nx32_01
RUN_RE = re.compile(r".*_hr(?P<hr>\d+)_nx(?P<lr>\d+)_(?P<idx>\d{2})")

def _save_pytree(obj, basepath: str):
    import pickle
    leaves, treedef = jax.tree_util.tree_flatten(obj)
    # convert leaves to numpy for saving
    leaves_np = [np.asarray(x) for x in leaves]
    # save leaves as compressed npz
    np.savez_compressed(basepath + ".npz", *leaves_np)
    # save treedef separately
    with open(basepath + ".treedef", "wb") as f:
        pickle.dump(treedef, f)


def _load_pytree(basepath: str):
    import pickle
    if not os.path.exists(basepath + ".npz") or not os.path.exists(basepath + ".treedef"):
        raise FileNotFoundError(f"Checkpoint files not found for {basepath}")
    arrs = np.load(basepath + ".npz")
    leaves = [arrs[f"arr_{i}"] for i in range(len(arrs.files))]
    with open(basepath + ".treedef", "rb") as f:
        treedef = pickle.load(f)
    return jax.tree_util.tree_unflatten(treedef, leaves)


def _load_leaves(basepath: str):
    """Return the raw saved leaves (numpy arrays) for a checkpointed pytree."""
    if not os.path.exists(basepath + ".npz"):
        raise FileNotFoundError(f"Checkpoint npz not found for {basepath}")
    arrs = np.load(basepath + ".npz")
    leaves = [arrs[f"arr_{i}"] for i in range(len(arrs.files))]
    return leaves


def checkpointer(closure_obj=None, optim_state=None, model_dir: str = None, save: bool = False, epoch: int = None, n_epochs: int = None, losses: dict = None):
    '''
    Save or load a checkpoint (closure module + optimizer state) to/from `model_dir`.

    Usage:
      - To load:  `closure_obj, optim_state = checkpointer(None, None, model_dir, save=False)`
      - To save:  `checkpointer(closure_obj, optim_state, model_dir, save=True)`
    '''
    if model_dir is None:
        raise ValueError("model_dir must be provided")

    os.makedirs(model_dir, exist_ok=True)
    closure_base = os.path.join(model_dir, "closure_ckpt")
    optim_base = os.path.join(model_dir, "optim_ckpt")

    if save:
        if closure_obj is None or optim_state is None:
            raise ValueError("Both closure_obj and optim_state required for saving")
        # Partition closure into arrays (params) and static
        try:
            closure_params, _static = eqx.partition(closure_obj, eqx.is_array)
        except Exception:
            logger.exception("Failed to partition closure for saving")
            raise

        # Save closure params and optimizer state (array pytrees)
        try:
            _save_pytree(closure_params, closure_base)
        except Exception:
            logger.exception("Failed to save closure checkpoint")
            raise
        try:
            _save_pytree(optim_state, optim_base)
        except Exception:
            logger.exception("Failed to save optimizer checkpoint")
            raise

        # Write a minimal checkpoint metadata file (include epoch info if provided)
        meta = {"saved_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z"}
        if epoch is not None:
            meta["epoch"] = int(epoch)
        if n_epochs is not None:
            meta["n_epochs"] = int(n_epochs)
        try:
            with open(os.path.join(model_dir, "checkpoint_meta.json"), "w") as f:
                json.dump(meta, f, indent=4)
        except Exception:
            logger.exception("Failed to write checkpoint meta file")
        # Optionally save loss history as JSON
        if losses is not None:
            try:
                lh_path = os.path.join(model_dir, "loss_history.json")
                with open(lh_path, "w") as f:
                    json.dump({"train": list(losses.get("train", [])), "test": list(losses.get("test", []))}, f, indent=4)
            except Exception:
                logger.exception("Failed to write loss history")
        return True
    else:
        # load params and optimizer state (may be None if missing)
        try:
            # load raw leaves for params so we can reinsert them into a fresh template
            loaded_params_leaves = _load_leaves(closure_base)
        except Exception:
            logger.exception("Failed to load closure checkpoint leaves")
            loaded_params_leaves = None
        try:
            loaded_optim = _load_pytree(optim_base)
        except Exception:
            logger.exception("Failed to load optimizer checkpoint")
            loaded_optim = None
        # attempt to read checkpoint metadata
        ckpt_meta = None
        try:
            meta_path = os.path.join(model_dir, "checkpoint_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    ckpt_meta = json.load(f)
        except Exception:
            logger.exception("Failed to read checkpoint metadata")

        # attempt to read loss history
        loss_history = None
        try:
            lh_path = os.path.join(model_dir, "loss_history.json")
            if os.path.exists(lh_path):
                with open(lh_path) as f:
                    loss_history = json.load(f)
        except Exception:
            logger.exception("Failed to read loss history")

        return loaded_params_leaves, loaded_optim, ckpt_meta, loss_history

def metadata_matches(requested: dict, stored: dict) -> bool:
    return canonicalize(requested) == canonicalize(stored)

def canonicalize(params: dict) -> dict:
    def round_floats(x):
        if isinstance(x, float):
            return round(x, 12)
        if isinstance(x, dict):
            return {k: round_floats(v) for k, v in sorted(x.items())}
        if isinstance(x, list):
            return [round_floats(v) for v in x]
        return x

    return round_floats(params)

def find_existing_closure(model_dir, params, timing_metadata, model_type):
    hr_nx = params['hr_nx']
    lr_nx = params['nx']
    prefix = f"{model_type}_hr{hr_nx}_nx{lr_nx}_"
    candidates = []

    for name in os.listdir(model_dir):
        m = RUN_RE.fullmatch(name)
        if m is None:
            continue
        if int(m["hr"]) != hr_nx or int(m["lr"]) != lr_nx:
            continue

        run_dir = os.path.join(model_dir, name)
        meta_path = os.path.join(run_dir, "metadata.json")
        if not os.path.exists(meta_path):
            continue

        try:
            with open(meta_path) as f:
                stored_meta = json.load(f)
        except Exception:
            continue

        # Exact metadata match
        if (metadata_matches(params, stored_meta["parameters"])) and (metadata_matches(timing_metadata, stored_meta['timing'])) and (model_type == stored_meta['model_type']):
            return run_dir, True
        candidates.append(int(m["idx"]))

    # No match found
    next_idx = max(candidates, default=0) + 1
    run_name = f"{prefix}{next_idx:02d}"
    run_dir = os.path.join(model_dir, run_name)
    return run_dir, False


def find_existing_run(base_dir, params, timing_metadata):
    hr_nx = params['hr_nx']
    lr_nx = params['nx']
    prefix = f"data_hr{hr_nx}_nx{lr_nx}_"
    candidates = []

    for name in os.listdir(base_dir):
        m = RUN_RE.fullmatch(name)
        if m is None:
            continue
        if int(m["hr"]) != hr_nx or int(m["lr"]) != lr_nx:
            continue

        run_dir = os.path.join(base_dir, name)
        meta_path = os.path.join(run_dir, "metadata.json")
        if not os.path.exists(meta_path):
            continue

        try:
            with open(meta_path) as f:
                stored_meta = json.load(f)
        except Exception:
            continue

        # Exact metadata match
        if (metadata_matches(params, stored_meta["parameters"])) and (metadata_matches(timing_metadata, stored_meta['timing'])):
            return run_dir, True
        candidates.append(int(m["idx"]))

    # No match found
    next_idx = max(candidates, default=0) + 1
    run_name = f"{prefix}{next_idx:02d}"
    run_dir = os.path.join(base_dir, run_name)
    return run_dir, False

class ZarrDataLoader:
    """Lazy data loader for chunked Zarr trajectory data.
    
    Designed for efficient, shuffled training with large datasets that don't fit in memory.
    Only loads data when requested, leveraging Zarr's chunked storage.
    
    Attributes
    ----------
    zarr_path : str
        Path to the .zarr directory
    traj_group : zarr.Group
        The trajectories group containing individual trajectory arrays
    n_trajectories : int
        Total number of trajectories in the dataset
    traj_names : list[str]
        List of trajectory array names (e.g., ['traj_00000', 'traj_00001', ...])
    metadata : dict
        Metadata from the Zarr store attributes
    """
    
    def __init__(self, run_dir: str):
        """Initialize the data loader.
        
        Parameters
        ----------
        run_dir : str
            Path to the run directory containing trajectories.zarr
        """
        self.zarr_path = os.path.join(run_dir, "trajectories.zarr")
        if not os.path.exists(self.zarr_path):
            raise FileNotFoundError(f"Zarr store not found at {self.zarr_path}")
        
        # Open in read-only mode
        self.z_root = zarr.open_group(self.zarr_path, mode="r")
        self.traj_group = self.z_root["trajectories"]
        self.metadata = dict(self.z_root.attrs)
        
        # Get list of trajectory names
        self.traj_names = sorted([k for k in self.traj_group.keys()])
        self.n_trajectories = len(self.traj_names)
        
        if self.n_trajectories == 0:
            raise ValueError(f"No trajectories found in {self.zarr_path}")
        
        # Get shape info from first trajectory
        first_traj = self.traj_group[self.traj_names[0]]
        self.traj_shape = first_traj.shape  # (time_steps, layers, ny, nx)
        self.dtype = first_traj.dtype
        
    def __len__(self) -> int:
        """Return number of trajectories."""
        return self.n_trajectories
    
    def get_trajectory(self, idx: int) -> np.ndarray:
        """Load a complete trajectory by index.
        
        Parameters
        ----------
        idx : int
            Trajectory index (0 to n_trajectories-1)
            
        Returns
        -------
        np.ndarray
            Shape (time_steps, layers, ny, nx)
        """
        if idx < 0 or idx >= self.n_trajectories:
            raise IndexError(f"Trajectory index {idx} out of range [0, {self.n_trajectories})")
        
        traj_name = self.traj_names[idx]
        return self.traj_group[traj_name][:]
    
    def get_trajectory_window(
        self, 
        traj_idx: int, 
        start_time: int, 
        batch_steps: int
    ) -> np.ndarray:
        """Load a time window from a trajectory.
        
        This is efficient because Zarr only loads the requested chunks.
        
        Parameters
        ----------
        traj_idx : int
            Trajectory index
        start_time : int
            Starting time step
        batch_steps : int
            Number of time steps to load
            
        Returns
        -------
        np.ndarray
            Shape (batch_steps, layers, ny, nx)
        """
        traj_name = self.traj_names[traj_idx]
        traj = self.traj_group[traj_name]
        
        end_time = start_time + batch_steps
        if end_time > traj.shape[0]:
            raise ValueError(
                f"Window [{start_time}:{end_time}] exceeds trajectory length {traj.shape[0]}"
            )
        
        return traj[start_time:end_time]
    
    def sample_windows(self, n_samples, batch_steps, key, traj_indices):
        """Sample random time windows from trajectories.
        """
        start_times = jax.random.permutation(key, n_samples) * batch_steps 
        windows = []
        for traj_idx in traj_indices: 
            for start_time in start_times:
                # ^^^ this means that right now this has the same order of times for each traj
                window = self.get_trajectory_window(traj_idx, start_time, batch_steps)
                windows.append(window)
        
        windows = np.stack(windows, axis=0)
        return windows
    
    def __repr__(self) -> str:
        return (
            f"ZarrDataLoader(\n"
            f"  path={self.zarr_path}\n"
            f"  n_trajectories={self.n_trajectories}\n"
            f"  traj_shape={self.traj_shape}\n"
            f"  dtype={self.dtype}\n"
            f")"
        )