import os
import json
import re
import zarr
import numpy as np
from typing import Optional, Tuple, List
import logging
logger = logging.getLogger(__name__)

RUN_RE = re.compile(r"data_hr(?P<hr>\d+)_nx(?P<lr>\d+)_(?P<idx>\d{2})")

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

def find_existing_closure(dir, cfg):
    '''Search not implemented yet'''
    return False

def find_existing_run(base_dir, hr_nx, lr_nx, params, timing_metadata):
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
        if (metadata_matches(params, stored_meta["parameters"])):
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
        window_size: int
    ) -> np.ndarray:
        """Load a time window from a trajectory.
        
        This is efficient because Zarr only loads the requested chunks.
        
        Parameters
        ----------
        traj_idx : int
            Trajectory index
        start_time : int
            Starting time step
        window_size : int
            Number of time steps to load
            
        Returns
        -------
        np.ndarray
            Shape (window_size, layers, ny, nx)
        """
        traj_name = self.traj_names[traj_idx]
        traj = self.traj_group[traj_name]
        
        end_time = start_time + window_size
        if end_time > traj.shape[0]:
            raise ValueError(
                f"Window [{start_time}:{end_time}] exceeds trajectory length {traj.shape[0]}"
            )
        
        return traj[start_time:end_time]
    
    def sample_windows(
        self,
        n_samples: int,
        window_size: int,
        rng: Optional[np.random.Generator] = None,
        fixed_traj_idx: Optional[int] = None,
        subset_traj_indices: Optional[List[int]] = None,
        return_indices: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Sample random time windows from trajectories.

        Parameters
        ----------
        n_samples : int
            Number of windows to sample
        window_size : int
            Length of each time window
        rng : np.random.Generator, optional
            Random number generator. If None, creates a new one.
        fixed_traj_idx : int, optional
            If provided, only sample from this trajectory. Otherwise sample from all.
        subset_traj_indices : list of int, optional
            If provided, only sample from these trajectory indices.
        return_indices : bool, default=False
            If True, also return (traj_idx, start_time) for each sample
            
        Returns
        -------
        windows : np.ndarray
            Shape (n_samples, window_size, layers, ny, nx)
        indices : list of (traj_idx, start_time), optional
            Only returned if return_indices=True
        """
        if rng is None:
            rng = np.random.default_rng()
        
        max_start_time = self.traj_shape[0] - window_size
        if max_start_time < 0:
            raise ValueError(
                f"Window size {window_size} exceeds trajectory length {self.traj_shape[0]}"
            )
        
        # Sample start times
        start_times = rng.integers(0, max_start_time + 1, size=n_samples)
        
        # Sample trajectories (fixed or random)
        if fixed_traj_idx is not None:
            traj_indices = np.full(n_samples, fixed_traj_idx, dtype=int)
        elif subset_traj_indices is not None:
            traj_indices = rng.choice(subset_traj_indices, size=n_samples)
        else:
            traj_indices = rng.integers(0, self.n_trajectories, size=n_samples)
        
        # Load the windows
        windows = []
        indices = []
        for traj_idx, start_time in zip(traj_indices, start_times):
            window = self.get_trajectory_window(traj_idx, start_time, window_size)
            windows.append(window)
            indices.append((int(traj_idx), int(start_time)))
        
        windows = np.stack(windows, axis=0)
        
        if return_indices:
            return windows, indices
        return windows
    
    def get_all_trajectory_indices(self) -> List[Tuple[int, int]]:
        """Get all valid (traj_idx, time_idx) pairs.
        
        Useful for creating epoch-based training with full data coverage.
        
        Returns
        -------
        list of (traj_idx, time_idx)
            All valid combinations
        """
        indices = []
        for traj_idx in range(self.n_trajectories):
            for time_idx in range(self.traj_shape[0]):
                indices.append((traj_idx, time_idx))
        return indices
    
    def __repr__(self) -> str:
        return (
            f"ZarrDataLoader(\n"
            f"  path={self.zarr_path}\n"
            f"  n_trajectories={self.n_trajectories}\n"
            f"  traj_shape={self.traj_shape}\n"
            f"  dtype={self.dtype}\n"
            f")"
        )