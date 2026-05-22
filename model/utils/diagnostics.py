from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import importlib
import model.utils.physics_ops
importlib.reload(model.utils.physics_ops)
from model.utils.physics_ops import (
    invert_pv_to_psi,
    velocity_from_psi,
    isotropic_ke_spectrum,
)

# ============================================================
# Base class
# ============================================================

class Diagnostic:
    name: str
    output: str = "png"

    def run(self, trajs: dict, out_path: str, cadence: int = 10):
        raise NotImplementedError
    

class LossDiagnostic(Diagnostic):
    name = "loss"

    def run(self, trajs, out_path, cadence):
        losses = trajs.get("loss_history", {})

        train = np.asarray(losses.get("train", []))
        test  = np.asarray(losses.get("test", []))
        zero = np.asarray(losses.get("zero", []))

        fig, ax = plt.subplots()

        if train.size:
            ax.plot(np.arange(1, len(train) + 1), train, label="train")
        if test.size:
            ax.plot(np.arange(1, len(test) + 1), test, label="test")
        if zero.size:
            # plot average zero loss
            ax.hlines(zero.mean(), 1, len(test), colors="C2", linestyles="--", label="zero model")
            # one sd range for zero loss
            ax.fill_between(np.arange(1, len(test) + 1), zero.mean() - zero.std(), zero.mean() + zero.std(), color="C2", alpha=0.08)

        ax.set_title("Training / Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True)
        if train.size or test.size:
            ax.legend()

        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ============================================================
# KE Spectrum Animation (time-resolved)
# ============================================================

class KESpectrumAnimationDiagnostic(Diagnostic):
    name = "ke_spectrum_movie"
    output = "gif"

    def run(self, trajs, out_path, cadence):
        grid = trajs.get("grid")
        if grid is None:
            raise KeyError("ke_spectrum_movie requires 'grid' in trajectories")

        # prefer physical predicted frames produced by validation; fallback to 'pred'
        q_truth = trajs.get("truth")
        q_pred = trajs.get("pred_frames")
        zero = trajs.get("zero_frames")
        q_truth = np.asarray(q_truth[10:]) # im trying out skipping first few frames to avoid 0s ?
        q_pred = np.asarray(q_pred[10:])
        zero = np.asarray(zero[10:])

        # compute per-frame spectra helper
        def compute_frame_spectra(q):
            try:
                frames = []
                nt = q.shape[0]
                for t in range(nt):
                    psi_t = invert_pv_to_psi(q[t], grid)
                    u_t, v_t = velocity_from_psi(psi_t, grid) # shape (nz, ny, nx)
                    spec_t = isotropic_ke_spectrum(u_t, v_t, grid)
                    Et = spec_t["E"]
                    frames.append(np.asarray(Et).ravel())
                k = np.asarray(spec_t["k"]).ravel()
                return np.stack(frames, axis=0), k
            except Exception:
                return None, None

        E_truth_frames, k = compute_frame_spectra(q_truth)
        E_pred_frames, _ = compute_frame_spectra(q_pred) 
        E_zero_frames, _ = compute_frame_spectra(zero)

        # averages and stds
        E_truth_avg = E_truth_frames.mean(axis=0)
        E_pred_avg = E_pred_frames.mean(axis=0)
        E_pred_std = E_pred_frames.std(axis=0)
        E_zero_avg = E_zero_frames.mean(axis=0)

        # select frames for animation using cadence
        nt = E_truth_frames.shape[0]
        frame_indices = list(range(0, nt, max(1, cadence)))

        # Build plot: avg lines + shading, instant lines animated on top
        fig, ax = plt.subplots()
        ax.loglog(k[1:], E_truth_avg[1:], label="Truth", color="k")

        try:
            ax.fill_between(k[1:], (E_pred_avg - E_pred_std)[1:], (E_pred_avg + E_pred_std)[1:], color="C1", alpha=0.08)
        except Exception:
            pass

        # instantaneous lines
        Ep0 = E_pred_frames[frame_indices[0]]
        Ez0 = E_zero_frames[frame_indices[0]]
        ln_pred, = ax.loglog(k[1:], Ep0[1:], label="ML", color="C3", linestyle="--")
        ln_zero, = ax.loglog(k[1:], Ez0[1:], label="Zero", color="C2", linestyle="--")

        ax.set_xlabel("k")
        ax.set_ylabel("E(k)")
        ax.set_title(f"KE spectrum (t={frame_indices[0]})")
        ax.grid(True, which="both")
        ax.legend()

        def update(i):
            idx = frame_indices[i]
            if ln_pred is not None:
                Ep = E_pred_frames[idx]
                ln_pred.set_data(k[1:], Ep[1:])
            if ln_zero is not None:
                Ez = E_zero_frames[idx]
                ln_zero.set_data(k[1:], Ez[1:])
            ax.relim()
            ax.autoscale_view()
            ax.set_title(f"KE spectrum (t={idx})")
            return (ln_pred, ln_zero)

        ani = FuncAnimation(fig, update, frames=len(frame_indices), interval=200)
        ani.save(out_path, writer=PillowWriter(fps=5))
        plt.close(fig)



# ============================================================
# MSE
# ============================================================

class MSEDiagnostic(Diagnostic):
    name = "mse"

    def run(self, trajs, out_path, cadence):
        # Prefer full-resolution data if available
        pred = trajs.get("pred_frames")
        truth = trajs.get("truth")

        if pred is None or truth is None:
            raise KeyError("mse diagnostic requires 'pred' and 'truth' in trajectories")

        pred = np.asarray(pred)
        truth = np.asarray(truth)

        # ensure (nt, nz, ny, nx)
        if pred.ndim == 3:
            pred = pred[:, None, ...]
            truth = truth[:, None, ...]

        # Compute MSE averaged over spatial dimensions and layers
        mse = np.mean((pred - truth) ** 2, axis=(-2, -1))  # (nt, nz)
        mse = np.mean(mse, axis=1)                         # (nt,)

        nt = mse.shape[0]
        x = np.arange(nt)

        fig, ax = plt.subplots()
        ax.plot(x, mse, markersize=3, label="MSE")
        ax.set_title("MSE per timestep (domain mean)")
        ax.set_yscale("log")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("MSE")
        if nt > 0:
            ax.set_xlim(0, max(0, nt - 1))
        ax.grid(True)

        # Plot zero-model baseline if provided. Accept scalar or per-timestep array.
        zero_loss = trajs.get("loss_history", {}).get("zero", None)
        if zero_loss is not None:
            zl = np.asarray(zero_loss)
            # Scalar baseline: draw horizontal dashed line
            if zl.ndim == 0 or zl.size == 1:
                val = float(zl.reshape(()))
                if nt > 0:
                    ax.hlines(val, 0, nt - 1, colors="C2", linestyles="--", label="zero model")
                else:
                    ax.axhline(val, color="C2", linestyle="--", label="zero model")
            else:
                # Per-timestep baseline: ensure length matches mse, truncate/pad with NaN if needed
                if zl.shape[0] != mse.shape[0]:
                    if zl.shape[0] > mse.shape[0]:
                        zl = zl[: mse.shape[0]]
                    else:
                        zl = np.pad(zl, (0, mse.shape[0] - zl.shape[0]), constant_values=np.nan)
                ax.plot(x, zl, "--", color="C2", label="zero model")

        if ax.get_legend_handles_labels()[0]:
            ax.legend()

        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ============================================================
# KE Spectrum (time-averaged)
# ============================================================

class KESpectrumDiagnostic(Diagnostic):
    name = "ke_spectrum"

    def run(self, trajs, out_path, cadence):
        # --- get stuff ---
        grid = trajs.get("grid")
        q_truth = trajs.get("truth")
        q_pred  = trajs.get("pred_frames")
        zero = trajs.get("zero_frames")
        q_truth = np.asarray(q_truth)
        q_pred  = np.asarray(q_pred)
        zero = np.asarray(zero)

        # --- helper: compute spectrum from PV (time-averaged) ---
        def compute_avg_spectrum(q):
            psi = invert_pv_to_psi(q, grid)
            u, v = velocity_from_psi(psi, grid)
            spec = isotropic_ke_spectrum(u, v, grid)
            k = np.asarray(spec["k"]).ravel()
            E = np.asarray(spec["E"]).ravel()
            return k, E

        # compute averaged spectra
        k, E_truth_avg = compute_avg_spectrum(q_truth)
        _, E_pred_avg = compute_avg_spectrum(q_pred)
        _, E_zero_avg = compute_avg_spectrum(zero)

        # --- compute per-frame spectra ---
        def compute_frame_spectra(q):
            try:
                frames = []
                for t in range(q.shape[0]):
                    psi_t = invert_pv_to_psi(q[t], grid)
                    u_t, v_t = velocity_from_psi(psi_t, grid)
                    spec_t = isotropic_ke_spectrum(u_t, v_t, grid)
                    Et = np.asarray(spec_t["E"]).ravel()
                    frames.append(Et)
                return np.stack(frames, axis=0)
            except Exception:
                return None

        E_truth_frames = compute_frame_spectra(q_truth)
        E_pred_frames = compute_frame_spectra(q_pred)
        E_zero_frames = compute_frame_spectra(zero)
        E_truth_std = E_truth_frames.std(axis=0)
        E_pred_std = E_pred_frames.std(axis=0)

        # --- plot ---
        fig, ax = plt.subplots()

        ax.loglog(k[1:], E_truth_avg[1:], label="Truth", color="k")
        if E_pred_avg is not None:
            ax.loglog(k[1:], E_pred_avg[1:], label="ML", linestyle="--", color="C1")
        if E_zero_avg is not None:
            ax.loglog(k[1:], E_zero_avg[1:], label="Zero", linestyle="--", color="C2")


        # Shade ±1σ around the mean if available
        try:
            if E_truth_std is not None:
                ax.fill_between(k[1:], (E_truth_avg - E_truth_std)[1:], (E_truth_avg + E_truth_std)[1:], color="k", alpha=0.12)
            if E_pred_std is not None and E_pred_avg is not None:
                ax.fill_between(k[1:], (E_pred_avg - E_pred_std)[1:], (E_pred_avg + E_pred_std)[1:], color="C1", alpha=0.08)
        except Exception:
            pass

        ax.set_xlabel("k")
        ax.set_ylabel("E(k)")
        ax.set_title("Time-averaged KE spectrum")
        ax.grid(True, which="both")
        ax.legend()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ============================================================
# PV Animation
# ============================================================

class VorticityDiagnostic(Diagnostic): # this might need cadence adding to it tbh
    name = "PV"
    output = "gif"

    def run(self, trajs, out_path, cadence):
        if "q" in trajs and trajs["q"] is not None:
            truth = np.asarray(trajs["q"])
        elif "truth" in trajs and trajs["truth"] is not None:
            truth = np.asarray(trajs["truth"])
        else:
            raise KeyError("PV diagnostic requires 'q' or 'truth' in trajectories")
        ml = trajs.get("pred")

        nt, nz = truth.shape[:2]
        cols = 2 if ml is not None else 1

        fig, axes = plt.subplots(nz, cols, squeeze=False,
                                 figsize=(4 * cols, 3 * nz))

        # fixed color scale (important)
        vmin = truth.min()
        vmax = truth.max()

        ims = []

        for layer in range(nz):
            for col in range(cols):
                ax = axes[layer][col]

                src = truth if col == 0 else ml

                im = ax.imshow(
                    src[0, layer],
                    origin="lower",
                    cmap="RdBu_r",
                    vmin=vmin,
                    vmax=vmax,
                    animated=True,
                )

                title = "Truth" if col == 0 else "ML"
                if nz > 1:
                    title += f" (layer {layer})"

                ax.set_title(title)
                ims.append((im, src, layer))

        def update(frame):
            for im, src, layer in ims:
                im.set_data(src[frame, layer])
            return [im for im, _, _ in ims]

        anim = FuncAnimation(fig, update, frames=nt, interval=200)
        anim.save(out_path, writer=PillowWriter(fps=10))
        plt.close(fig)


# ============================================================
# Quad GIF (uses existing helper)
# ============================================================

class QuadGifDiagnostic(Diagnostic):
    name = "quad"
    output = "gif"

    def run(self, trajs, out_path, cadence):
        pred = trajs.get("pred_frames")
        truth = trajs.get("truth")
        pred_np = np.asarray(pred)
        truth_np = np.asarray(truth)
        err = pred_np - truth_np # not used here

        # Get predicted/applied SGS and the target SGS
        sgs_pred = trajs.get("sgs")
        sgs_target = trajs.get("target_sgs")
        sgs_pred_np = np.asarray(sgs_pred)
        sgs_target_np = np.asarray(sgs_target)

        nt = pred_np.shape[0]

        def pad_to_frames(arr):
            if arr is None:
                return np.zeros_like(pred_np)
            if arr.shape[0] == nt:
                return arr
            if arr.shape[0] == nt - 1:
                pad = np.zeros_like(arr[0:1])
                return np.concatenate([pad, arr], axis=0)
            if arr.shape[0] < nt:
                pad = np.zeros((nt - arr.shape[0],) + arr.shape[1:], dtype=arr.dtype)
                return np.concatenate([pad, arr], axis=0)
            return arr[:nt]

        sgs_pred_np = pad_to_frames(sgs_pred_np)
        sgs_target_np = pad_to_frames(sgs_target_np)

        indices = np.arange(0, nt, max(1, int(cadence)))
        if indices.size == 0:
            raise ValueError("No frames selected for quad diagnostic (check cadence)")

        def pick(arr, idx):
            if arr.ndim == 4:
                return arr[idx, 0]
            return arr[idx]

        # robust percentiles
        def pct(a, q):
            try:
                return np.nanpercentile(a, q)
            except Exception:
                return 0.0

        vmin_truth = pct(truth_np, 1)
        vmax_truth = pct(truth_np, 99)
        vmin_sgs_t = pct(sgs_target_np, 1)
        vmax_sgs_t = pct(sgs_target_np, 99)
        vmin_sgs_p = pct(sgs_pred_np, 1)
        vmax_sgs_p = pct(sgs_pred_np, 99)


        # Layout: top row - Truth  | Target SGS 
        # bottom row - ML adjusted | Pred SGS
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        ax_truth = axes[0, 0]
        ax_ml = axes[1, 0]
        ax_target_sgs = axes[0, 1]
        ax_pred_sgs = axes[1, 1]

        im_truth = ax_truth.imshow(pick(truth_np, indices[0]), origin="lower", cmap="RdBu_r", vmin=vmin_truth, vmax=vmax_truth)
        ax_truth.set_title("Truth")
        im_ml = ax_ml.imshow(pick(pred_np, indices[0]), origin="lower", cmap="RdBu_r", vmin=vmin_truth, vmax=vmax_truth)
        ax_ml.set_title("ML adjusted")

        im_target_sgs = ax_target_sgs.imshow(pick(sgs_target_np, indices[0]), origin="lower", cmap="RdBu_r", vmin=vmin_sgs_t, vmax=vmax_sgs_t)
        ax_target_sgs.set_title("Target SGS")

        im_pred_sgs = ax_pred_sgs.imshow(pick(sgs_pred_np, indices[0]), origin="lower", cmap="RdBu_r", vmin=vmin_sgs_p, vmax=vmax_sgs_p)
        ax_pred_sgs.set_title("Predicted SGS")


        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])

        fig.colorbar(im_truth, ax=[ax_truth, ax_ml], shrink=0.6)
        fig.colorbar(im_target_sgs, ax=ax_target_sgs, shrink=0.6)
        fig.colorbar(im_pred_sgs, ax=ax_pred_sgs, shrink=0.6)


        def update(i):
            idx = indices[i]
            im_truth.set_data(pick(truth_np, idx))
            im_ml.set_data(pick(pred_np, idx))
            im_target_sgs.set_data(pick(sgs_target_np, idx))
            im_pred_sgs.set_data(pick(sgs_pred_np, idx))
            fig.suptitle(f"timestep {idx}")
            return im_truth, im_ml, im_target_sgs, im_pred_sgs

        anim = FuncAnimation(fig, update, frames=len(indices), interval=100, blit=False)

        try:
            writer = PillowWriter(fps=10)
            anim.save(out_path, writer=writer)
            plt.close(fig)
        except Exception as e:
            print("Pillow save failed:", e)


# ============================================================
# Zero Comparison Quad GIF (same as above but with zero model SGS as one of the panels)
# ============================================================

class ZeroComparisonDiagnostic(Diagnostic):
    name = "quad"
    output = "gif"

    def run(self, trajs, out_path, cadence):
        pred = trajs.get("pred_frames")
        truth = trajs.get("truth")
        zero = trajs.get("zero_frames")
        pred_np = np.asarray(pred)
        truth_np = np.asarray(truth)
        zero_np = np.asarray(zero) if zero is not None else np.zeros_like(truth_np)

        # Get predicted/applied SGS
        sgs_pred = trajs.get("sgs")
        sgs_pred_np = np.asarray(sgs_pred)

        nt = pred_np.shape[0]

        def pad_to_frames(arr):
            if arr is None:
                return np.zeros_like(pred_np)
            if arr.shape[0] == nt:
                return arr
            if arr.shape[0] == nt - 1:
                pad = np.zeros_like(arr[0:1])
                return np.concatenate([pad, arr], axis=0)
            if arr.shape[0] < nt:
                pad = np.zeros((nt - arr.shape[0],) + arr.shape[1:], dtype=arr.dtype)
                return np.concatenate([pad, arr], axis=0)
            return arr[:nt]

        sgs_pred_np = pad_to_frames(sgs_pred_np)
        zero_np = pad_to_frames(zero_np)

        indices = np.arange(0, nt, max(1, int(cadence)))
        if indices.size == 0:
            raise ValueError("No frames selected for quad diagnostic (check cadence)")

        def pick(arr, idx):
            if arr.ndim == 4:
                return arr[idx, 0]
            return arr[idx]

        # robust percentiles
        def pct(a, q):
            try:
                return np.nanpercentile(a, q)
            except Exception:
                return 0.0

        vmin_truth = pct(truth_np, 1)
        vmax_truth = pct(truth_np, 99)


        # Layout: top row - Truth  | Zero Model
        # bottom row - ML adjusted | Pred SGS
        fig, axes = plt.subplots(1, 3, figsize=(12, 8))
        ax_truth = axes[0]
        ax_ml = axes[1]
        ax_zero = axes[2]

        im_truth = ax_truth.imshow(pick(truth_np, indices[0]), origin="lower", cmap="RdBu_r", vmin=vmin_truth, vmax=vmax_truth)
        ax_truth.set_title("Truth")
        im_ml = ax_ml.imshow(pick(pred_np, indices[0]), origin="lower", cmap="RdBu_r", vmin=vmin_truth, vmax=vmax_truth)
        ax_ml.set_title("ML adjusted")
        im_zero = ax_zero.imshow(pick(zero_np, indices[0]), origin="lower", cmap="RdBu_r", vmin=vmin_truth, vmax=vmax_truth)
        ax_zero.set_title("Without closure")


        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])

        fig.colorbar(im_truth, ax=[ax_truth, ax_ml, ax_zero], shrink=0.6)


        def update(i):
            idx = indices[i]
            im_truth.set_data(pick(truth_np, idx))
            im_ml.set_data(pick(pred_np, idx))
            im_zero.set_data(pick(zero_np, idx))
            fig.suptitle(f"timestep {idx}")
            return im_truth, im_ml, im_zero

        anim = FuncAnimation(fig, update, frames=len(indices), interval=100, blit=False)

        try:
            writer = PillowWriter(fps=10)
            anim.save(out_path, writer=writer)
            plt.close(fig)
        except Exception as e:
            print("Pillow save failed:", e)


# ============================================================
# Energy (timeseries)
# ============================================================

class EnergyDiagnostic(Diagnostic):
    name = "energy"

    def run(self, trajs, out_path):
        if "q" in trajs and trajs["q"] is not None:
            q = np.asarray(trajs["q"])
        elif "truth" in trajs and trajs["truth"] is not None:
            q = np.asarray(trajs["truth"])
        else:
            raise KeyError("KE spectrum diagnostic requires 'q' or 'truth' in trajectories")
        grid = trajs["grid"]

        psi = invert_pv_to_psi(q, grid)
        u, v = velocity_from_psi(psi, grid)

        ke = 0.5 * np.mean(u**2 + v**2, axis=(-2, -1))  # (nt, nz)
        ke = ke.mean(axis=1)

        fig, ax = plt.subplots()
        ax.plot(ke)
        ax.set_title("Kinetic Energy")
        ax.set_xlabel("t")
        ax.set_ylabel("KE")
        ax.grid(True)

        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

# ============================================================
# CFL condition over time
# ============================================================

class CFLDiagnostic(Diagnostic):
    name = "cfl"

    def run(self, trajs, out_path, cadence):
        grid = trajs.get("grid")
        if grid is None:
            raise KeyError("cfl diagnostic requires 'grid' in trajectories")

        q_pred = trajs.get("pred_frames")
        q_pred = np.asarray(q_pred)
        if q_pred.ndim == 3:
            q_pred = q_pred[:, None, ...]

        dt = float(trajs.get("dt", 1.0))
        dx = float(grid.dx)
        dy = float(grid.dy)

        # Subsample with cadence to keep this fast
        nt = q_pred.shape[0]
        frame_indices = list(range(0, nt, max(1, cadence)))

        cfl_vals = []
        for t in frame_indices:
            psi = invert_pv_to_psi(q_pred[t], grid)
            u, v = velocity_from_psi(psi, grid)
            cfl_x = float(np.max(np.abs(u))) * dt / dx
            cfl_y = float(np.max(np.abs(v))) * dt / dy
            cfl_vals.append(max(cfl_x, cfl_y))

        cfl_vals = np.array(cfl_vals)

        fig, ax = plt.subplots()
        ax.plot(frame_indices, cfl_vals, label="CFL (max over domain)")
        ax.axhline(1.0, color='r', linestyle='--', label='CFL = 1 (unstable above)')
        ax.set_title("CFL condition over time")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("CFL number")
        ax.legend()
        ax.grid(True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

_REGISTRY = {
    "loss": LossDiagnostic,
    "mse": MSEDiagnostic,
    "ke_spectrum": KESpectrumDiagnostic,
    "PV": VorticityDiagnostic,
    "quad": QuadGifDiagnostic,
    'zero': ZeroComparisonDiagnostic,
    "energy": EnergyDiagnostic,
    "ke_spectrum_movie": KESpectrumAnimationDiagnostic,
    'cfl': CFLDiagnostic,
}

def build_diagnostic(name: str) -> Diagnostic:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown diagnostic '{name}'")
    return _REGISTRY[name]()