"""
plot_training_data.py
---------------------
Visualise the training data stored in the zarr archives under data/.

Three figure types are produced:

  1. hr_vs_lr_<tag>.png
     Side-by-side of the coarsened-HR field (zarr truth) and a naively
     doubly-coarsened version at half the resolution, plus their difference.
     This is the visual answer to "what would be lost without a closure?".

  2. training_increments_<tag>.png
     Step-to-step vorticity increments  dq = q[t+1] - q[t]  for several
     frames.  This is the quantity the ML closure is trained to reproduce.

  3. ke_spectra_<tag>.png
     Isotropic kinetic-energy spectra for every dataset found in data/,
     overlaid on one axes so you can compare how energy is distributed
     across resolutions.

  4. temporal_variance_<tag>.png
     Spatial RMS of q as a function of time — gives a quick sanity check
     that the trajectory is statistically stationary after spin-up.

Usage
-----
  python scripts/plot_training_data.py                  # auto-discover all datasets
  python scripts/plot_training_data.py --data_dir data/data_hr128_nx32_01
  python scripts/plot_training_data.py --traj_idx 2 --out_dir outputs/data_diagnostics
"""

import argparse
import json
import os
import re
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import zarr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_ROOT = os.path.join(BASE_DIR, "data")
DEFAULT_OUT = os.path.join(BASE_DIR, "outputs", "data_diagnostics")

RUN_RE = re.compile(r"data_hr(?P<hr>\d+)_nx(?P<lr>\d+)_(?P<idx>\d+)")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _discover_datasets(root: str) -> list[dict]:
    """Return list of dicts with keys: path, hr_nx, lr_nx, idx."""
    out = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        m = RUN_RE.match(name)
        if m:
            out.append(
                dict(
                    path=os.path.join(root, name),
                    hr_nx=int(m.group("hr")),
                    lr_nx=int(m.group("lr")),
                    idx=int(m.group("idx")),
                    name=name,
                )
            )
    return out


def _load_trajectory(dataset_path: str, traj_idx: int = 0) -> np.ndarray:
    """Load q array (nsteps, nz, ny, nx) from zarr."""
    zarr_path = os.path.join(dataset_path, "trajectories.zarr")
    root = zarr.open_group(zarr_path, mode="r")
    trajs = root["trajectories"]
    keys = sorted(trajs.keys())
    if traj_idx >= len(keys):
        traj_idx = 0
    arr = trajs[keys[traj_idx]][:]          # (nsteps, nz, ny, nx)
    return np.asarray(arr, dtype=np.float64)


def _load_metadata(dataset_path: str) -> dict:
    meta_path = os.path.join(dataset_path, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}


def _coarsen_spectral(q: np.ndarray, target_n: int) -> np.ndarray:
    """Galerkin-truncate q (..., ny, nx) to target_n x target_n, then IFFT back
    to physical space at the *original* resolution so fields can be compared."""
    ny, nx = q.shape[-2], q.shape[-1]
    nk = target_n // 2
    qh = np.fft.rfftn(q, axes=(-2, -1))
    # Build truncated spectrum (zero-pad outside target wavenumbers)
    qh_trunc = np.zeros_like(qh)
    qh_trunc[..., :nk, :nk + 1] = qh[..., :nk, :nk + 1]
    qh_trunc[..., -nk:, :nk + 1] = qh[..., -nk:, :nk + 1]
    return np.fft.irfftn(qh_trunc, s=(ny, nx), axes=(-2, -1))


def _isotropic_ke_spectrum(q: np.ndarray, Lx: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute isotropic KE spectrum E(k) from vorticity q (..., ny, nx)."""
    q = np.asarray(q, dtype=float)
    ny, nx = q.shape[-2], q.shape[-1]
    dx = Lx / nx
    dy = Lx / ny  # assume square domain

    kx = np.fft.rfftfreq(nx, d=dx) * 2.0 * np.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2.0 * np.pi
    kx2, ky2 = np.meshgrid(kx, ky, indexing="xy")
    k2 = kx2 ** 2 + ky2 ** 2
    # avoid division by zero; DC term → psi=0
    k2_safe = np.where(k2 == 0, 1.0, k2)

    # PV → streamfunction → velocity
    qh = np.fft.rfftn(q, axes=(-2, -1))
    if q.ndim > 2:
        qh = qh.mean(axis=tuple(range(q.ndim - 2)))  # average over leading dims
    psi_h = -qh / k2_safe
    psi_h[0, 0] = 0.0

    uh = 1j * ky2 * psi_h   # u =  ∂ψ/∂y  in spectral
    vh = -1j * kx2 * psi_h  # v = -∂ψ/∂x  in spectral
    ke_spec = 0.5 * (np.abs(uh) ** 2 + np.abs(vh) ** 2)

    kmag = np.sqrt(kx2 ** 2 + ky2 ** 2)
    kmax = int(np.floor(kmag.max()))
    k_bins = np.arange(0, kmax + 1, dtype=float)
    E = np.zeros(len(k_bins))
    for i, kc in enumerate(k_bins):
        mask = (kmag >= kc - 0.5) & (kmag < kc + 0.5)
        E[i] = ke_spec[mask].sum()

    return k_bins, E


# ---------------------------------------------------------------------------
# Figure 1 – HR coarsened data vs doubly-coarsened "naive LR" + difference
# ---------------------------------------------------------------------------

def plot_hr_vs_lr(q: np.ndarray, lr_nx: int, meta: dict, out_path: str, n_frames: int = 4):
    """Show coarsened-HR truth, spectral-truncated naive-LR, and the diff."""
    nsteps = q.shape[0]
    frame_indices = np.linspace(0, nsteps - 1, n_frames, dtype=int)

    # Build doubly-coarsened "naive LR" at half the LR resolution
    half_n = max(lr_nx // 2, 4)
    q_naive = _coarsen_spectral(q, half_n)  # same spatial size, but truncated modes

    q_layer = q[:, 0]          # (nsteps, ny, nx)  – take layer 0
    qn_layer = q_naive[:, 0]

    diff = q_layer - qn_layer

    vmax_q = np.percentile(np.abs(q_layer[frame_indices]), 98)
    vmax_d = np.percentile(np.abs(diff[frame_indices]), 98)

    fig, axes = plt.subplots(3, n_frames, figsize=(3.5 * n_frames, 9))
    fig.suptitle(
        f"Coarsened-HR truth (top) vs naively truncated to {half_n}×{half_n} (mid) and difference (bot)\n"
        f"Dataset: hr{meta.get('parameters', {}).get('hr_nx', '?')} → nx{lr_nx}",
        fontsize=11,
    )

    for col, tidx in enumerate(frame_indices):
        kw_q = dict(cmap="RdBu_r", vmin=-vmax_q, vmax=vmax_q)
        kw_d = dict(cmap="seismic", vmin=-vmax_d, vmax=vmax_d)

        axes[0, col].imshow(q_layer[tidx], origin="lower", **kw_q)
        axes[0, col].set_title(f"t = {tidx}", fontsize=9)
        axes[0, col].set_axis_off()

        axes[1, col].imshow(qn_layer[tidx], origin="lower", **kw_q)
        axes[1, col].set_axis_off()

        im = axes[2, col].imshow(diff[tidx], origin="lower", **kw_d)
        axes[2, col].set_axis_off()

    axes[0, 0].set_ylabel(f"Truth ({lr_nx}×{lr_nx})", fontsize=9)
    axes[1, 0].set_ylabel(f"Naive LR ({half_n}×{half_n})", fontsize=9)
    axes[2, 0].set_ylabel("Difference", fontsize=9)

    # Row colour-bars
    for row, vmax, label in zip(
        [0, 1, 2], [vmax_q, vmax_q, vmax_d], ["q (truth)", "q (naive LR)", "Δq"]
    ):
        sm = plt.cm.ScalarMappable(
            cmap="RdBu_r" if row < 2 else "seismic",
            norm=plt.Normalize(-vmax, vmax),
        )
        sm.set_array([])
        fig.colorbar(sm, ax=axes[row, :], fraction=0.02, pad=0.01, label=label)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2 – Training increments  dq = q[t+1] - q[t]
# ---------------------------------------------------------------------------

def plot_training_increments(q: np.ndarray, lr_nx: int, meta: dict, out_path: str, n_frames: int = 6):
    """Visualise the step-to-step vorticity change — the direct training signal."""
    nsteps = q.shape[0]
    frame_indices = np.linspace(0, nsteps - 2, n_frames, dtype=int)

    dq = np.diff(q[:, 0], axis=0)  # (nsteps-1, ny, nx)
    vmax = np.percentile(np.abs(dq[frame_indices]), 99)

    ncols = n_frames
    fig, axes = plt.subplots(2, ncols, figsize=(3 * ncols, 6))
    fig.suptitle(
        f"Training increments  dq = q[t+1] − q[t]   (nx={lr_nx})\n"
        "Top: instantaneous q — Bottom: increment dq",
        fontsize=11,
    )

    vmax_q = np.percentile(np.abs(q[:, 0][frame_indices]), 98)
    for col, tidx in enumerate(frame_indices):
        axes[0, col].imshow(q[tidx, 0], origin="lower", cmap="RdBu_r",
                            vmin=-vmax_q, vmax=vmax_q)
        axes[0, col].set_title(f"t={tidx}", fontsize=8)
        axes[0, col].set_axis_off()

        axes[1, col].imshow(dq[tidx], origin="lower", cmap="seismic",
                            vmin=-vmax, vmax=vmax)
        axes[1, col].set_axis_off()

    for row, (vmax_r, label, cmap) in enumerate(
        [(vmax_q, "q", "RdBu_r"), (vmax, "dq", "seismic")]
    ):
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-vmax_r, vmax_r))
        sm.set_array([])
        fig.colorbar(sm, ax=axes[row, :], fraction=0.02, pad=0.01, label=label)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 3 – KE spectra across all available datasets
# ---------------------------------------------------------------------------

def plot_ke_spectra(datasets: list[dict], traj_idx: int, out_path: str):
    """Overlay isotropic KE spectra for all discovered datasets."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title("Isotropic KE spectra — all datasets", fontsize=12)

    cmap = plt.cm.viridis
    colours = cmap(np.linspace(0, 0.85, len(datasets)))

    for ds, colour in zip(datasets, colours):
        try:
            q = _load_trajectory(ds["path"], traj_idx)
        except Exception as e:
            print(f"  [skip KE spectrum] {ds['name']}: {e}")
            continue

        meta = _load_metadata(ds["path"])
        Lx = meta.get("parameters", {}).get("Lx", 2 * np.pi)

        # Average over several frames for a stable estimate
        n_avg = min(20, q.shape[0])
        sample = q[-n_avg:, 0]       # (n_avg, ny, nx)
        k_all, E_all = [], []
        for frame in sample:
            k, E = _isotropic_ke_spectrum(frame, Lx)
            k_all.append(k)
            E_all.append(E)
        k = k_all[0]
        E_mean = np.mean(E_all, axis=0)

        label = f"hr{ds['hr_nx']}→nx{ds['lr_nx']} (run {ds['idx']:02d})"
        ax.loglog(k[1:], E_mean[1:], label=label, color=colour, lw=1.8)

    # reference k^-3 slope
    k_ref = np.array([2, 30], dtype=float)
    E_ref = k_ref ** -3
    # scale to sit in the middle of the plot
    yl = ax.get_ylim()
    if yl[0] > 0:
        scale = 10 ** (0.5 * (np.log10(yl[0]) + np.log10(yl[1]))) / E_ref.mean()
        ax.loglog(k_ref, E_ref * scale, "k--", lw=1.2, label="k⁻³")

    ax.set_xlabel("Wavenumber k")
    ax.set_ylabel("E(k)")
    ax.legend(fontsize=8, framealpha=0.7)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 4 – Temporal variance / RMS
# ---------------------------------------------------------------------------

def plot_temporal_variance(q: np.ndarray, lr_nx: int, meta: dict, out_path: str):
    """Spatial RMS of q vs time — sanity-check that the trajectory is well-behaved."""
    rms = np.sqrt(np.mean(q[:, 0] ** 2, axis=(-2, -1)))   # (nsteps,)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"Temporal statistics — nx={lr_nx}", fontsize=11)

    axes[0].plot(rms, lw=1.2, color="steelblue")
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("Spatial RMS(q)")
    axes[0].set_title("Spatial RMS over time")
    axes[0].grid(True, alpha=0.3)

    # Histogram of q values
    axes[1].hist(q[:, 0].ravel(), bins=80, density=True, color="steelblue", alpha=0.75)
    axes[1].set_xlabel("q")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Distribution of vorticity values")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 5 – Spectral energy transfer (increment spectrum)
# ---------------------------------------------------------------------------

def plot_increment_spectrum(q: np.ndarray, lr_nx: int, meta: dict, out_path: str):
    """Compare KE spectrum of q vs dq to show which scales are actively forced."""
    Lx = meta.get("parameters", {}).get("Lx", 2 * np.pi)

    n_avg = min(50, q.shape[0] - 1)
    sample_q  = q[-n_avg - 1 : -1, 0]
    sample_dq = np.diff(q[-n_avg - 1:, 0], axis=0)

    k_q_list, E_q_list, E_dq_list = [], [], []
    for i in range(n_avg):
        k, Eq  = _isotropic_ke_spectrum(sample_q[i],  Lx)
        _, Edq = _isotropic_ke_spectrum(sample_dq[i], Lx)
        k_q_list.append(k)
        E_q_list.append(Eq)
        E_dq_list.append(Edq)

    k     = k_q_list[0]
    E_q   = np.mean(E_q_list,  axis=0)
    E_dq  = np.mean(E_dq_list, axis=0)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(k[1:], E_q[1:],  label="E(q) — field", color="steelblue", lw=2)
    ax.loglog(k[1:], E_dq[1:], label="E(dq) — increment (training target)", color="tomato", lw=2)
    ax.set_xlabel("Wavenumber k")
    ax.set_ylabel("E(k)")
    ax.set_title(
        f"Field vs increment KE spectrum  (nx={lr_nx}, averaged over {n_avg} steps)\n"
        "The increment spectrum shows which scales carry the subgrid forcing signal."
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default=None,
                   help="Path to a single dataset directory (default: auto-discover all under data/).")
    p.add_argument("--traj_idx", type=int, default=0,
                   help="Which trajectory index to visualise (default: 0).")
    p.add_argument("--out_dir", default=DEFAULT_OUT,
                   help="Directory to save plots (default: outputs/data_diagnostics/).")
    p.add_argument("--n_frames", type=int, default=4,
                   help="Number of snapshot frames to show (default: 4).")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Discover datasets ------------------------------------------------
    if args.data_dir:
        datasets = [
            dict(
                path=os.path.abspath(args.data_dir),
                name=os.path.basename(args.data_dir),
                **{k: int(v) for k, v in RUN_RE.match(os.path.basename(args.data_dir)).groupdict().items()}
                if RUN_RE.match(os.path.basename(args.data_dir))
                else dict(hr_nx=0, lr_nx=0, idx=0),
            )
        ]
        # Flatten the conditional dict construction
        m = RUN_RE.match(os.path.basename(args.data_dir))
        datasets = [dict(
            path=os.path.abspath(args.data_dir),
            name=os.path.basename(args.data_dir),
            hr_nx=int(m.group("hr")) if m else 0,
            lr_nx=int(m.group("lr")) if m else 0,
            idx=int(m.group("idx")) if m else 0,
        )]
    else:
        datasets = _discover_datasets(DATA_ROOT)

    if not datasets:
        sys.exit(f"No datasets found. Checked: {DATA_ROOT}")

    print(f"Found {len(datasets)} dataset(s).")

    # ---- Per-dataset figures -----------------------------------------------
    for ds in datasets:
        tag = ds["name"]
        print(f"\n=== {tag} ===")

        try:
            q = _load_trajectory(ds["path"], args.traj_idx)
        except Exception as e:
            print(f"  [skip] Could not load trajectory: {e}")
            continue

        meta = _load_metadata(ds["path"])
        lr_nx = ds["lr_nx"] or q.shape[-1]

        print(f"  Trajectory shape: {q.shape}  (nsteps, nz, ny, nx)")

        # Fig 1 – HR truth vs naive LR + difference
        plot_hr_vs_lr(
            q, lr_nx, meta,
            out_path=os.path.join(args.out_dir, f"hr_vs_lr_{tag}.png"),
            n_frames=args.n_frames,
        )

        # Fig 2 – Training increments
        plot_training_increments(
            q, lr_nx, meta,
            out_path=os.path.join(args.out_dir, f"training_increments_{tag}.png"),
            n_frames=min(args.n_frames + 2, 6),
        )

        # Fig 4 – Temporal variance
        plot_temporal_variance(
            q, lr_nx, meta,
            out_path=os.path.join(args.out_dir, f"temporal_variance_{tag}.png"),
        )

        # Fig 5 – Increment spectrum
        plot_increment_spectrum(
            q, lr_nx, meta,
            out_path=os.path.join(args.out_dir, f"increment_spectrum_{tag}.png"),
        )

    # ---- Multi-dataset KE spectra overlay ---------------------------------
    print("\n=== KE spectra overlay ===")
    plot_ke_spectra(
        datasets,
        traj_idx=args.traj_idx,
        out_path=os.path.join(args.out_dir, "ke_spectra_all.png"),
    )

    print(f"\nAll plots saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
