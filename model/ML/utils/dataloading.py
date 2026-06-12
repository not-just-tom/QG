import os
import jax
import json
import re
import zarr
import numpy as np
import jax.numpy as jnp
import equinox as eqx
import logging
import threading
import queue
from model.ML.utils.utils import parameterization
from model.ML.forced_model import ForcedModel
from model.core.steppers import SteppedModel, AB3Stepper, CNABStepper
from model.ML.architectures.build_model import closure_combiner
logger = logging.getLogger(__name__)

# Match either data or model run directories, e.g.:
#  - data_hr128_nx32_01
#  - cnn_hr128_nx32_01
RUN_RE = re.compile(r".*_hr(?P<hr>\d+)_nx(?P<lr>\d+)_(?P<idx>\d{2})")
CLOSURE_RE = re.compile(r"(?P<hr>\d+)_to_(?P<lr>\d+)")

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


def _load_leaves(basepath: str):
    """Return the raw saved leaves (numpy arrays) for a checkpointed pytree."""
    if not os.path.exists(basepath + ".npz"):
        raise FileNotFoundError(f"Checkpoint npz not found for {basepath}")
    arrs = np.load(basepath + ".npz")
    leaves = [arrs[f"arr_{i}"] for i in range(len(arrs.files))]
    return leaves

def load_forced_model(
    lr_model,
    closure,
    dt,
    trajs=None,
    closure_scale=0.1,
):
    '''Load forced model from provided closure.
    
    If trajs is provided, automatically computes scalers and filter.
    Otherwise uses provided scalers/filter (useful for inference).
    '''
    trajs_arr = jnp.asarray(trajs)
    eps = 1e-6
    layer_axis = trajs_arr.ndim - 3
    reduce_axes = tuple(i for i in range(trajs_arr.ndim) if i != layer_axis)
    q_std = jnp.std(trajs_arr, axis=reduce_axes)
    q_std = jnp.maximum(q_std, eps).reshape((-1, 1, 1))
    q_mean = jnp.zeros_like(q_std)
    dq_std = jnp.maximum(q_std * closure_scale, eps)
    dq_mean = jnp.zeros_like(dq_std)


    closure_params, closure_static = eqx.partition(closure, eqx.is_array)
    init_param_func = lambda state, model, params: params
    dt_arr = jnp.asarray(dt)

    def _param_adapter(state, param_aux, model, *args, **kwargs):
        dq_increment, new_params = closure_combiner(
            state,
            param_aux,
            closure_static,
            q_mean=q_mean,
            q_std=q_std,
            dq_mean=dq_mean,
            dq_std=dq_std,
        )
        # Closure network predicts a per-step increment dQ; the parameterization
        # wrapper expects a tendency dQ/dt to add to model.get_updates(...).
        dq_forcing = dq_increment / dt_arr
        return dq_forcing, new_params

    closure_func = parameterization(_param_adapter)

    lr_stepper = AB3Stepper(dt=dt)
    forced_model = SteppedModel(
        model=ForcedModel(model=lr_model, closure=closure_func, init_param_aux_func=init_param_func),
        stepper=lr_stepper,
    )
    
    # Return preprocessing params for diagnostics
    return forced_model, closure_params, closure_static, q_mean, q_std, dq_mean, dq_std

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

        # Save closure params and optimizer state (array pytrees).
        # Write to temporary files and atomically replace so only the most
        # recent complete checkpoint is present on disk.
        try:
            # write to temp bases in same directory to allow atomic replace
            tmp_closure_base = closure_base + ".tmp"
            tmp_optim_base = optim_base + ".tmp"
            _save_pytree(closure_params, tmp_closure_base)
            # replace existing files atomically
            for ext in (".npz", ".treedef"):
                src = tmp_closure_base + ext
                dst = closure_base + ext
                try:
                    os.replace(src, dst)
                except Exception:
                    # if replace fails, attempt move
                    os.rename(src, dst)
            # Save only the flat leaves for the optimizer — no pickle treedef,
            # so this survives JAX/equinox version changes.  Reconstruction
            # uses the template treedef built from a freshly-initialised optim.
            optim_leaves, _ = jax.tree_util.tree_flatten(optim_state)
            optim_leaves_np = [np.asarray(x) for x in optim_leaves]
            np.savez_compressed(tmp_optim_base + ".npz", *optim_leaves_np)
            src = tmp_optim_base + ".npz"
            dst = optim_base + ".npz"
            try:
                os.replace(src, dst)
            except Exception:
                os.rename(src, dst)
        except Exception:
            logger.exception("Failed to save closure or optimizer checkpoint")
            raise

        # checkpoint metadata
        meta = {"saved_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z"}
        if epoch is not None:
            meta["epoch"] = int(epoch)
        if n_epochs is not None:
            meta["n_epochs"] = int(n_epochs)
        try:
            # write metadata atomically
            meta_path = os.path.join(model_dir, "checkpoint_meta.json")
            tmp_meta = meta_path + ".tmp"
            with open(tmp_meta, "w") as f:
                json.dump(meta, f, indent=4)
            try:
                os.replace(tmp_meta, meta_path)
            except Exception:
                os.rename(tmp_meta, meta_path)
        except Exception:
            logger.exception("Failed to write checkpoint meta file")
        # Optionally save loss history as JSON
        if losses is not None:
            try:
                lh_path = os.path.join(model_dir, "loss_history.json")
                tmp_lh = lh_path + ".tmp"
                with open(tmp_lh, "w") as f:
                    json.dump({"train": list(losses.get("train", [])), "test": list(losses.get("test", [])), 'zero': list(losses.get('zero', []))}, f, indent=4)
                try:
                    os.replace(tmp_lh, lh_path)
                except Exception:
                    os.rename(tmp_lh, lh_path)
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
            # Load raw leaves only; caller reconstructs using template treedef.
            loaded_optim = _load_leaves(optim_base)
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
    # Keys to ignore in metadata comparison
    IGNORE_KEYS = {"auto_dt", "created_utc", "saved_utc", "dt (original)", 'nsteps'}
    
    def round_floats(x):
        if isinstance(x, dict):
            return {k: round_floats(v) for k, v in sorted(x.items()) if k not in IGNORE_KEYS}
        if isinstance(x, list):
            return [round_floats(v) for v in x]
        return x

    return round_floats(params)

def find_existing_closure(model_dir, params, timing_metadata, model_type, training_metadata):
    hr_nx = params['hr_nx']
    lr_nx = params['nx']
    folder = f"{hr_nx}_to_{lr_nx}"
    model_base = os.path.join(model_dir, model_type, folder)
    os.makedirs(model_base, exist_ok=True)


    candidates = []

    for name in os.listdir(model_base):        
        run_dir = os.path.join(model_base,  name)
        meta_path = os.path.join(run_dir, "metadata.json")

        # If metadata exists, check for exact parameter+timing match
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    stored_meta = json.load(f)
            except Exception:
                continue

            # Basic parameter+timing+model_type match
            params_match = (metadata_matches(params, stored_meta.get("parameters", {}))) and (metadata_matches(timing_metadata, stored_meta.get('timing', {}))) and (model_type == stored_meta.get('model_type'))

            # If training_metadata supplied (e.g. training / sweep info), require it to match as well
            training_match = True
            for k, v in training_metadata.items():
                stored_section = stored_meta.get(k, {})
                if not metadata_matches(v, stored_section):
                    training_match = False
                    break

            if params_match and training_match:
                return run_dir, True, name[:2]

        # Keep the index as a candidate (even if metadata missing or mismatched)
        candidates.append(int(name))

    # No exact match found; select next index within model_type namespace
    next_idx = max(candidates, default=0) + 1
    run_name = f"{next_idx:02d}"
    run_dir = os.path.join(model_base, run_name)
    return run_dir, False, next_idx


def find_existing_data(base_dir, params, timing_metadata):
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

        def timing_conditions(timing_metadata, stored_timing):
            # Check nsteps condition
            requested_nsteps = int(timing_metadata.get("nsteps", 0))
            stored_nsteps = int(stored_timing.get("nsteps", 0))
            if metadata_matches(timing_metadata, stored_timing):
                return True, None
            if stored_nsteps < requested_nsteps:
                return False, 'nsteps'
            else:
                return False, None
        timing_match, fail_reason = timing_conditions(timing_metadata, stored_meta.get("timing", {}))
        # Exact metadata match
        if (metadata_matches(params, stored_meta["parameters"])) and timing_match:
            return run_dir, True
        if fail_reason == 'nsteps':
            return run_dir, 'nsteps'
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
    
    def save_trajectory(self, traj_idx: int, traj_data: np.ndarray):
        """Save a trajectory to the Zarr store.
        
        Parameters
        ----------
        traj_idx : int
            Trajectory index
        traj_data : np.ndarray
            Trajectory data to save, shape (time_steps, layers, ny, nx)
        """
        traj_name = f"traj_{traj_idx:05d}"
        if traj_name in self.traj_group:
            raise ValueError(f"Trajectory {traj_name} already exists in Zarr store")
        
        self.traj_group.create_array(
            traj_name,
            data=traj_data.astype(self.dtype),
            chunks=(traj_data.shape[0], traj_data.shape[1], traj_data.shape[2], traj_data.shape[3]),
            compressor=self.traj_group.compressor,
        )

    
    def sample_windows(self, n_samples, batch_steps, key, traj_indices):
        """Sample random time windows from trajectories.
        """
        # Generate permuted start times on the device, then convert to
        # host-side Python ints for indexing zarr arrays (which expect
        # native Python/numpy types, not JAX DeviceArrays).
        perm = jax.random.permutation(key, n_samples)
        perm_host = np.asarray(jax.device_get(perm))
        start_times = [int(x * batch_steps) for x in perm_host]

        windows = []
        for traj_idx in traj_indices:
            for start_time in start_times:
                # Use Python int for zarr indexing
                window = self.get_trajectory_window(int(traj_idx), int(start_time), batch_steps)
                windows.append(window)

        windows = np.stack(windows, axis=0)
        return windows

    def iterate_batches(self, *, traj_indices, n_samples, batch_steps, key, batch_size=1):
        """Stream batches of windows without materialising entire epoch.

        Yields numpy arrays shaped `(batch_size, batch_steps, layers, ny, nx)`.

        Parameters
        ----------
        traj_indices : sequence[int]
            Indices of trajectories to sample from.
        n_samples : int
            Number of start times to sample per-trajectory (will be permuted).
        batch_steps : int
            Number of timesteps in each window.
        key : jax.random.PRNGKey
            PRNG key used to generate permutation of start times.
        batch_size : int
            Number of windows per yielded batch.
        """
        # Build start times (device permutation -> host ints)
        perm = jax.random.permutation(key, n_samples)
        perm_host = np.asarray(jax.device_get(perm))
        start_times = [int(x * batch_steps) for x in perm_host]

        batch = []
        for traj_idx in traj_indices:
            for start_time in start_times:
                window = self.get_trajectory_window(int(traj_idx), int(start_time), batch_steps)
                batch.append(window)
                if len(batch) == batch_size:
                    yield np.stack(batch, axis=0)
                    batch = []

        # yield remainder
        if batch:
            yield np.stack(batch, axis=0)

    def iterate_batches_device(
        self,
        *,
        traj_indices,
        n_samples,
        batch_steps,
        key,
        batch_size=1,
        batch_prefetch: int = 2,
        device=None,
    ):
        """Like `iterate_batches` but yields JAX device arrays (already
        transferred to the specified device). This reduces host->device
        round-trips in training loops.
        """
        gen = self.iterate_batches(
            traj_indices=traj_indices,
            n_samples=n_samples,
            batch_steps=batch_steps,
            key=key,
            batch_size=batch_size,
        )
        if batch_prefetch and batch_prefetch > 0:
            gen = prefetch_generator(gen, size=batch_prefetch)

        for np_batch in gen:
            # convert to JAX array and push to device once per batch
            jax_batch = jax.device_put(jnp.asarray(np_batch), device)
            yield jax_batch


def prefetch_generator(generator, size: int = 2):
    """Prefetch items from a blocking generator into a small queue using a background thread.

    Usage:
      prefetch_gen = prefetch_generator(my_generator(), size=4)
      for batch in prefetch_gen:
          process(batch)

    This is thread-safe for zarr/numpy reads and reduces I/O stalls.
    """
    q: "queue.Queue[object]" = queue.Queue(maxsize=size)
    sentinel = object()

    def _worker():
        try:
            for item in generator:
                q.put(item)
        finally:
            q.put(sentinel)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    while True:
        item = q.get()
        if item is sentinel:
            return
        yield item
