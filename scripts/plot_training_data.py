"""Plot coarsened-HR truth vs LR stepped simulation for data_hr128_nx32_01.

Four rows:
  1) Coarsened-HR truth (zarr data — what the LR world should look like)
  2) LR stepped model (no closure) — this IS the per-step input to the closure
  3) Long-horizon difference: truth - LR (cumulative drift)
  4) Per-step SGS target: (dq_truth - dq_lr) at each frame — what a perfect
     closure would output at that step to correct the LR physics tendency

dt_lr = dt_hr * (hr_nx / lr_nx)
"""

import argparse
import json
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import zarr

from model.core.model import QGM
from model.core.steppers import AB3Stepper, SteppedModel, CNABStepper
from model.ML.utils.coarsen import coarsen


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATASET_DIR = os.path.join(BASE_DIR, "data", "data_hr128_nx32_01")
DEFAULT_OUT = os.path.join(BASE_DIR, "outputs", "data_diagnostics", "hr_vs_lr_stepped_data_hr128_nx32_01.png")


def load_data(dataset_dir: str, traj_idx: int) -> tuple[np.ndarray, dict]:
    zarr_path = os.path.join(dataset_dir, "trajectories.zarr")
    root = zarr.open_group(zarr_path, mode="r")
    trajs = root["trajectories"]
    keys = sorted(trajs.keys())
    if not keys:
        raise ValueError(f"No trajectories found in {zarr_path}")
    if traj_idx >= len(keys):
        traj_idx = 0
    q = np.asarray(trajs[keys[traj_idx]][:], dtype=np.float32)

    meta_path = os.path.join(dataset_dir, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    return q, meta


def infer_dt_hr(meta: dict) -> float:
    timing = meta.get("timing", {})
    for key in ("final dt", "dt", "dt (original)"):
        if key in timing:
            return float(timing[key])
    raise KeyError("Could not infer dt from metadata timing section")


def simulate_lr_rollout(q_truth: np.ndarray, meta: dict) -> tuple[np.ndarray, float]:
    params = dict(meta.get("parameters", {}))
    if "hr_nx" not in params or "nx" not in params:
        raise KeyError("metadata.parameters must contain hr_nx and nx")

    hr_nx = int(params["hr_nx"])
    lr_nx = int(params["nx"])
    ratio = hr_nx / lr_nx
    if abs(ratio - round(ratio)) > 1e-12:
        raise ValueError(f"hr_nx/lr_nx must be integer. Got {hr_nx}/{lr_nx}")
    ratio = int(round(ratio))

    dt_hr = infer_dt_hr(meta)
    dt_lr = dt_hr * ratio

    hr_model = QGM({**params, "nx": hr_nx, "ny": hr_nx})
    lr_model = coarsen(hr_model, lr_nx)
    stepped_lr = SteppedModel(model=lr_model, stepper=CNABStepper(dt=dt_lr))

    nsteps = q_truth.shape[0]
    q_pred = np.zeros_like(q_truth, dtype=np.float32)

    q0 = jnp.asarray(q_truth[0], dtype=jnp.float32)
    q0h = jnp.fft.rfftn(q0, axes=(-2, -1), norm="ortho")
    lr_state0 = lr_model.set_initial(qh=q0h, _q_shape=q0.shape[-2:])
    step_state = stepped_lr.initialize_stepper_state(lr_state0)

    q_pred[0] = np.asarray(lr_state0.q, dtype=np.float32)
    for i in range(1, nsteps):
        step_state = stepped_lr.step_model(step_state)
        q_pred[i] = np.asarray(step_state.state.q, dtype=np.float32)

    return q_pred, dt_lr


def compute_sgs_target(q_truth: np.ndarray, q_lr: np.ndarray) -> np.ndarray:
    """Per-step SGS forcing target: what a perfect closure would add at each step.

    dq_sgs[t] = (q_truth[t+1] - q_truth[t]) - (q_lr[t+1] - q_lr[t])

    This is the step-by-step spectral residual that the closure is trained to
    reproduce. Shape: (nsteps-1, nz, ny, nx).
    """
    dq_truth = np.diff(q_truth, axis=0)   # (T-1, nz, ny, nx)
    dq_lr    = np.diff(q_lr,    axis=0)
    return (dq_truth - dq_lr).astype(np.float32)


def make_plot(
    q_truth: np.ndarray,
    q_lr: np.ndarray,
    dq_sgs: np.ndarray,
    out_path: str,
    n_frames: int,
):
    """Four-row figure.

    Row 1: coarsened-HR truth   — the target trajectory
    Row 2: LR stepped (no closure) — per-step input to the closure
    Row 3: long-horizon difference truth - LR  — cumulative drift
    Row 4: per-step SGS target dq_sgs = dq_truth - dq_lr  — closure output target
    """
    truth_2d = q_truth[:, 0]
    lr_2d    = q_lr[:, 0]
    diff_2d  = truth_2d - lr_2d           # cumulative, same length as truth
    sgs_2d   = dq_sgs[:, 0]              # per-step increments, length T-1

    # Frame indices into the full trajectory for rows 1-3
    frame_idx = np.linspace(0, q_truth.shape[0] - 1, n_frames, dtype=int)
    # For the SGS row use the same time points but clamp to T-2 (valid range)
    sgs_frame_idx = np.clip(frame_idx, 0, dq_sgs.shape[0] - 1)

    vmax_q   = np.percentile(np.abs(truth_2d[frame_idx]), 99)
    vmax_d   = np.percentile(np.abs(diff_2d[frame_idx]),  99)
    vmax_sgs = np.percentile(np.abs(sgs_2d[sgs_frame_idx]), 99)

    fig, axes = plt.subplots(4, n_frames, figsize=(3.5 * n_frames, 11.5))
    fig.suptitle(
        "data_hr128_nx32_01  |  row 1: HR truth  |  row 2: LR (closure input)  "
        "|  row 3: truth−LR (drift)  |  row 4: SGS target dq_truth−dq_LR (closure output target)",
        fontsize=9,
    )

    for col, tidx in enumerate(frame_idx):
        axes[0, col].imshow(truth_2d[tidx], origin="lower", cmap="RdBu_r", vmin=-vmax_q, vmax=vmax_q)
        axes[0, col].set_title(f"t={tidx}", fontsize=9)
        axes[0, col].set_axis_off()

        axes[1, col].imshow(lr_2d[tidx], origin="lower", cmap="RdBu_r", vmin=-vmax_q, vmax=vmax_q)
        axes[1, col].set_axis_off()

        axes[2, col].imshow(diff_2d[tidx], origin="lower", cmap="seismic", vmin=-vmax_d, vmax=vmax_d)
        axes[2, col].set_axis_off()

    for col, tidx in enumerate(sgs_frame_idx):
        axes[3, col].imshow(sgs_2d[tidx], origin="lower", cmap="seismic", vmin=-vmax_sgs, vmax=vmax_sgs)
        axes[3, col].set_title(f"dq_sgs t={tidx}", fontsize=9)
        axes[3, col].set_axis_off()

    axes[0, 0].set_ylabel("HR truth", fontsize=9)
    axes[1, 0].set_ylabel("LR stepped\n(closure input q)", fontsize=9)
    axes[2, 0].set_ylabel("truth − LR\n(long-horizon drift)", fontsize=9)
    axes[3, 0].set_ylabel("SGS target\n(closure output target)", fontsize=9)

    sm_q = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-vmax_q, vmax_q))
    sm_q.set_array([])
    fig.colorbar(sm_q, ax=axes[0:2, :], fraction=0.02, pad=0.01, label="q")

    sm_d = plt.cm.ScalarMappable(cmap="seismic", norm=plt.Normalize(-vmax_d, vmax_d))
    sm_d.set_array([])
    fig.colorbar(sm_d, ax=axes[2, :], fraction=0.02, pad=0.01, label="truth − LR")

    sm_sgs = plt.cm.ScalarMappable(cmap="seismic", norm=plt.Normalize(-vmax_sgs, vmax_sgs))
    sm_sgs.set_array([])
    fig.colorbar(sm_sgs, ax=axes[3, :], fraction=0.02, pad=0.01, label="dq_sgs")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--traj_idx", type=int, default=0)
    p.add_argument("--n_frames", type=int, default=4)
    p.add_argument("--out_path", default=DEFAULT_OUT)
    return p.parse_args()


def main():
    args = parse_args()
    q_truth, meta = load_data(DATASET_DIR, args.traj_idx)
    q_lr, dt_lr = simulate_lr_rollout(q_truth, meta)
    dq_sgs = compute_sgs_target(q_truth, q_lr)
    make_plot(q_truth, q_lr, dq_sgs, args.out_path, args.n_frames)
    print(f"Saved: {args.out_path}")
    print(f"Trajectory shape: {q_truth.shape}")
    print(f"LR timestep: {dt_lr:.6f}")
    print(f"SGS target shape: {dq_sgs.shape}")


if __name__ == "__main__":
    # Keep CPU default unless user has configured GPU JAX explicitly.
    jax.config.update("jax_platforms", os.environ.get("JAX_PLATFORMS", "cpu"))
    main()
