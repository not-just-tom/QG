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
        n_epochs = losses['n_epochs']

        fig, ax = plt.subplots()

        if train.size:
            ax.plot(np.arange(1, len(train) + 1), train, label="train")
        if test.size:
            ax.plot(np.arange(1, len(test) + 1), test, label="test")
        if zero.size:
            # plot average zero loss per curriculum stage
            x = np.arange(0, len(zero)+1, n_epochs)
            y = [np.mean(zero[i:i+n_epochs]) for i in x]
            ax.step(x, y, label="zero model", linestyle="--", color="C2")

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

        # Auto-determine number of layers from data shape
        # Expected shape: (nt, nz, ny, nx)
        nt = truth.shape[0]
        nz = truth.shape[1] if truth.ndim >= 2 else 1
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
                ims.append((im, src, layer, ax, col))

        def update(frame):
            for im, src, layer, ax, col in ims:
                im.set_data(src[frame, layer])
                title = "Truth" if col == 0 else "ML"
                if nz > 1:
                    title += f" (layer {layer})"
                title += f" - Step {frame}"
                ax.set_title(title)
            return [im for im, _, _, _, _ in ims]

        anim = FuncAnimation(fig, update, frames=nt, interval=200)
        anim.save(out_path, writer=PillowWriter(fps=10))
        plt.close(fig)


# ============================================================
# Comprehensive SGS Diagnostic Quad GIF
# ============================================================

class QuadGifDiagnostic(Diagnostic):
    name = "quad"
    output = "gif"

    def run(self, trajs, out_path, cadence):
        """
        Comprehensive diagnostic showing:
        - Row 1: PV fields (Truth, ML, Zero, Error)  
        - Row 2: SGS forcing fields (Target, Rollout, Teacher-Forced, Difference)
        - Text annotations with statistics
        """
        # Extract data
        pred = trajs.get("pred_frames")
        truth = trajs.get("truth")
        zero = trajs.get("zero_frames")
        grid = trajs.get("grid")
        
        pred_np = np.asarray(pred)
        truth_np = np.asarray(truth)
        zero_np = np.asarray(zero) if zero is not None else None

        # Get SGS data
        sgs_pred = trajs.get("sgs")  # Applied during rollout
        sgs_target = trajs.get("target_sgs")  # Ideal from physics
        sgs_teacher = trajs.get("teacher_forced_sgs")  # Model evaluated at truth states
        
        sgs_pred_np = np.asarray(sgs_pred)
        sgs_target_np = np.asarray(sgs_target)
        sgs_teacher_np = np.asarray(sgs_teacher) if sgs_teacher is not None else None

        nt = pred_np.shape[0]

        # Padding helper
        def pad_to_frames(arr):
            if arr is None:
                return None
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
        sgs_teacher_np = pad_to_frames(sgs_teacher_np)
        zero_np = pad_to_frames(zero_np)

        # Frame indices for animation
        indices = np.arange(0, nt, max(1, int(cadence)))
        if indices.size == 0:
            raise ValueError("No frames selected for quad diagnostic (check cadence)")

        def pick(arr, idx):
            """Extract single frame, handling 3D and 4D arrays"""
            if arr is None:
                return None
            if arr.ndim == 4:
                return arr[idx, 0]
            return arr[idx]

        # Robust percentiles for color scaling
        def pct(a, q):
            try:
                return np.nanpercentile(a, q)
            except Exception:
                return 0.0

        # Color scales
        vmin_pv = pct(truth_np, 1)
        vmax_pv = pct(truth_np, 99)
        
        # Unified SGS scale
        all_sgs = [sgs_target_np, sgs_pred_np]
        if sgs_teacher_np is not None:
            all_sgs.append(sgs_teacher_np)
        all_sgs_concat = np.concatenate([s for s in all_sgs if s is not None], axis=0)
        vmin_sgs = pct(all_sgs_concat, 1)
        vmax_sgs = pct(all_sgs_concat, 99)

        # Error scale
        pv_error = pred_np - truth_np
        vmax_err = np.percentile(np.abs(pv_error), 99)

        # ==== Layout: 2 rows x 4 columns ====
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.3)
        
        # Row 1: PV fields
        ax_truth = fig.add_subplot(gs[0, 0])
        ax_ml = fig.add_subplot(gs[0, 1])
        ax_zero = fig.add_subplot(gs[0, 2])
        ax_pv_err = fig.add_subplot(gs[0, 3])
        
        # Row 2: SGS fields
        ax_target_sgs = fig.add_subplot(gs[1, 0])
        ax_pred_sgs = fig.add_subplot(gs[1, 1])
        ax_teacher_sgs = fig.add_subplot(gs[1, 2])
        ax_sgs_analysis = fig.add_subplot(gs[1, 3])

        # ==== Initialize PV plots (Row 1) ====
        im_truth = ax_truth.imshow(pick(truth_np, indices[0]), origin="lower", 
                                   cmap="RdBu_r", vmin=vmin_pv, vmax=vmax_pv)
        ax_truth.set_title("Truth PV", fontsize=11, fontweight='bold')
        
        im_ml = ax_ml.imshow(pick(pred_np, indices[0]), origin="lower", 
                            cmap="RdBu_r", vmin=vmin_pv, vmax=vmax_pv)
        ax_ml.set_title("ML Rollout PV", fontsize=11, fontweight='bold')
        
        if zero_np is not None:
            im_zero = ax_zero.imshow(pick(zero_np, indices[0]), origin="lower", 
                                    cmap="RdBu_r", vmin=vmin_pv, vmax=vmax_pv)
            ax_zero.set_title("Zero Model PV", fontsize=11, fontweight='bold')
        else:
            ax_zero.text(0.5, 0.5, 'No Zero Model', ha='center', va='center', 
                        transform=ax_zero.transAxes)
            ax_zero.set_title("Zero Model PV", fontsize=11, fontweight='bold')
            im_zero = None
        
        im_pv_err = ax_pv_err.imshow(pick(pv_error, indices[0]), origin="lower", 
                                     cmap="viridis", vmin=-vmax_err, vmax=vmax_err)
        ax_pv_err.set_title("PV Error\n(ML - Truth)", fontsize=11, fontweight='bold')

        # ==== Initialize SGS plots (Row 2) ====
        im_target_sgs = ax_target_sgs.imshow(pick(sgs_target_np, indices[0]), 
                                            origin="lower", cmap="seismic", 
                                            vmin=vmin_sgs, vmax=vmax_sgs)
        ax_target_sgs.set_title("Target SGS\n(Physics @ Truth)", fontsize=11, fontweight='bold')
        
        im_pred_sgs = ax_pred_sgs.imshow(pick(sgs_pred_np, indices[0]), 
                                        origin="lower", cmap="seismic", 
                                        vmin=vmin_sgs, vmax=vmax_sgs)
        ax_pred_sgs.set_title("Rollout SGS\n(Applied in Deployment)", fontsize=11, fontweight='bold')
        
        if sgs_teacher_np is not None:
            im_teacher_sgs = ax_teacher_sgs.imshow(pick(sgs_teacher_np, indices[0]), 
                                                  origin="lower", cmap="seismic", 
                                                  vmin=vmin_sgs, vmax=vmax_sgs)
            ax_teacher_sgs.set_title("Teacher-Forced SGS\n(Model @ Truth)", fontsize=11, fontweight='bold')
        else:
            ax_teacher_sgs.text(0.5, 0.5, 'No Teacher Forcing', ha='center', va='center',
                               transform=ax_teacher_sgs.transAxes)
            ax_teacher_sgs.set_title("Teacher-Forced SGS", fontsize=11, fontweight='bold')
            im_teacher_sgs = None

        # Bottom-right: SGS difference (teacher - rollout) as spatial field
        if sgs_teacher_np is not None:
            sgs_diff = sgs_teacher_np - sgs_pred_np
            vmax_diff = np.percentile(np.abs(sgs_diff), 99)
            im_sgs_diff = ax_sgs_analysis.imshow(pick(sgs_diff, indices[0]), 
                                                origin="lower", cmap="seismic",
                                                vmin=-vmax_diff, vmax=vmax_diff)
            ax_sgs_analysis.set_title("SGS Difference\n(Teacher - Rollout)", fontsize=11, fontweight='bold')
        else:
            sgs_diff = sgs_target_np - sgs_pred_np
            vmax_diff = np.percentile(np.abs(sgs_diff), 99)
            im_sgs_diff = ax_sgs_analysis.imshow(pick(sgs_diff, indices[0]), 
                                                origin="lower", cmap="seismic",
                                                vmin=-vmax_diff, vmax=vmax_diff)
            ax_sgs_analysis.set_title("SGS Error\n(Target - Rollout)", fontsize=11, fontweight='bold')

        # Remove ticks
        for ax in [ax_truth, ax_ml, ax_zero, ax_pv_err, 
                   ax_target_sgs, ax_pred_sgs, ax_teacher_sgs, ax_sgs_analysis]:
            ax.set_xticks([])
            ax.set_yticks([])

        # Colorbars
        fig.colorbar(im_truth, ax=[ax_truth, ax_ml, ax_zero], shrink=0.8, label="PV")
        fig.colorbar(im_pv_err, ax=ax_pv_err, shrink=0.8, label="Error")
        fig.colorbar(im_target_sgs, ax=[ax_target_sgs, ax_pred_sgs, ax_teacher_sgs], 
                    shrink=0.8, label="SGS Forcing")
        fig.colorbar(im_sgs_diff, ax=ax_sgs_analysis, shrink=0.8, label="Δ SGS")

        # ==== Statistics text (will be updated each frame) ====
        stats_text = fig.text(0.5, 0.95, "", ha='center', va='top', fontsize=10,
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        def compute_stats(idx):
            """Compute diagnostic statistics for current frame"""
            # PV errors
            pv_rmse = np.sqrt(np.mean((pick(pred_np, idx) - pick(truth_np, idx))**2))
            pv_max_err = np.max(np.abs(pick(pred_np, idx) - pick(truth_np, idx)))
            
            # SGS errors and correlations
            target = pick(sgs_target_np, idx)
            rollout = pick(sgs_pred_np, idx)
            
            sgs_rmse = np.sqrt(np.mean((rollout - target)**2))
            sgs_rms_target = np.sqrt(np.mean(target**2))
            sgs_rms_pred = np.sqrt(np.mean(rollout**2))
            
            # Correlation
            target_flat = target.ravel()
            rollout_flat = rollout.ravel()
            corr = np.corrcoef(target_flat, rollout_flat)[0, 1]
            
            # Magnitude ratio
            mag_ratio = sgs_rms_pred / sgs_rms_target if sgs_rms_target > 0 else 0
            
            stats = (f"Step {idx} | PV RMSE: {pv_rmse:.3e} | PV Max Error: {pv_max_err:.3e} | "
                    f"SGS RMSE: {sgs_rmse:.3e} | SGS Corr: {corr:.3f} | "
                    f"SGS Mag Ratio: {mag_ratio:.3f}")
            
            # Teacher forcing stats if available
            if sgs_teacher_np is not None:
                teacher = pick(sgs_teacher_np, idx)
                teacher_rmse = np.sqrt(np.mean((teacher - target)**2))
                deployment_gap = np.sqrt(np.mean((teacher - rollout)**2))
                stats += f" | Teacher RMSE: {teacher_rmse:.3e} | Deploy Gap: {deployment_gap:.3e}"
            
            return stats

        def update(i):
            """Update animation frame"""
            idx = indices[i]
            
            # Update PV fields
            im_truth.set_data(pick(truth_np, idx))
            im_ml.set_data(pick(pred_np, idx))
            if im_zero is not None:
                im_zero.set_data(pick(zero_np, idx))
            im_pv_err.set_data(pick(pv_error, idx))
            
            # Update SGS fields
            im_target_sgs.set_data(pick(sgs_target_np, idx))
            im_pred_sgs.set_data(pick(sgs_pred_np, idx))
            if im_teacher_sgs is not None:
                im_teacher_sgs.set_data(pick(sgs_teacher_np, idx))
            
            # Update difference field
            if sgs_teacher_np is not None:
                im_sgs_diff.set_data(pick(sgs_diff, idx))
            else:
                im_sgs_diff.set_data(pick(sgs_diff, idx))
            
            # Update statistics
            stats_text.set_text(compute_stats(idx))
            
            return (im_truth, im_ml, im_pv_err, im_target_sgs, im_pred_sgs, 
                   im_sgs_diff, stats_text)

        anim = FuncAnimation(fig, update, frames=len(indices), interval=150, blit=False)

        try:
            writer = PillowWriter(fps=8)
            anim.save(out_path, writer=writer)
            plt.close(fig)
        except Exception as e:
            print(f"Failed to save quad diagnostic: {e}")
            raise


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

# ============================================================
# Domain-averaged Energy and Enstrophy
# ============================================================

class DomainDiagnostic(Diagnostic):
    name = "domain"

    def run(self, trajs, out_path, cadence):
        grid = trajs.get("grid")
        if grid is None:
            raise KeyError("domain diagnostic requires 'grid' in trajectories")

        # Get truth data
        if "q" in trajs and trajs["q"] is not None:
            q_truth = np.asarray(trajs["q"])
        elif "truth" in trajs and trajs["truth"] is not None:
            q_truth = np.asarray(trajs["truth"])
        else:
            raise KeyError("domain diagnostic requires 'q' or 'truth' in trajectories")

        # Get predicted and zero model data if available
        q_pred = trajs.get("pred_frames")
        q_zero = trajs.get("zero_frames")
        
        if q_pred is not None:
            q_pred = np.asarray(q_pred)
        if q_zero is not None:
            q_zero = np.asarray(q_zero)

        # Compute energy and enstrophy over time for truth
        def compute_timeseries(q):
            """Compute energy and enstrophy for each timestep."""
            energy = []
            enstrophy = []
            
            for t in range(q.shape[0]):
                # Invert PV to streamfunction
                psi_t = invert_pv_to_psi(q[t], grid)
                u_t, v_t = velocity_from_psi(psi_t, grid)
                
                # Domain-averaged kinetic energy
                ke = 0.5 * np.mean(u_t**2 + v_t**2)
                energy.append(ke)
                
                # Domain-averaged enstrophy (using q as vorticity for QG)
                ens = 0.5 * np.mean(q[t]**2)
                enstrophy.append(ens)
            
            return np.array(energy), np.array(enstrophy)

        energy_truth, enstrophy_truth = compute_timeseries(q_truth)
        
        # Compute for predicted and zero models if available
        energy_pred, enstrophy_pred = None, None
        energy_zero, enstrophy_zero = None, None
        
        if q_pred is not None:
            energy_pred, enstrophy_pred = compute_timeseries(q_pred)
        if q_zero is not None:
            energy_zero, enstrophy_zero = compute_timeseries(q_zero)

        # Create side-by-side plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        nt = len(energy_truth)
        time = np.arange(nt)

        # Energy plot
        ax1.plot(time, energy_truth, label="Truth", color="k")
        if energy_pred is not None:
            ax1.plot(time, energy_pred, label="ML", linestyle="--", color="C1")
        if energy_zero is not None:
            ax1.plot(time, energy_zero, label="Zero", linestyle="--", color="C2")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Energy")
        ax1.set_title("Domain-averaged Energy")
        ax1.grid(True)
        ax1.legend()

        # Enstrophy plot
        ax2.plot(time, enstrophy_truth, label="Truth", color="k")
        if enstrophy_pred is not None:
            ax2.plot(time, enstrophy_pred, label="ML", linestyle="--", color="C1")
        if enstrophy_zero is not None:
            ax2.plot(time, enstrophy_zero, label="Zero", linestyle="--", color="C2")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Enstrophy")
        ax2.set_title("Domain-averaged Enstrophy")
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ============================================================
# SGS Spectral Analysis
# ============================================================

class SGSSpectralDiagnostic(Diagnostic):
    name = "sgs_spectrum"

    def run(self, trajs, out_path, cadence):
        """
        Analyze spectral properties of SGS forcing to diagnose why KE spectrum fails.
        Shows:
        - Power spectrum of target vs predicted SGS forcing
        - Time-averaged and instantaneous comparisons
        - Identifies scale-dependent errors
        """
        grid = trajs.get("grid")
        if grid is None:
            raise KeyError("sgs_spectrum diagnostic requires 'grid' in trajectories")

        sgs_target = trajs.get("target_sgs")
        sgs_pred = trajs.get("sgs")
        sgs_teacher = trajs.get("teacher_forced_sgs")
        
        if sgs_target is None or sgs_pred is None:
            raise KeyError("sgs_spectrum requires 'target_sgs' and 'sgs' in trajectories")

        sgs_target_np = np.asarray(sgs_target)
        sgs_pred_np = np.asarray(sgs_pred)
        sgs_teacher_np = np.asarray(sgs_teacher) if sgs_teacher is not None else None

        # Ensure same length
        nt = min(sgs_target_np.shape[0], sgs_pred_np.shape[0])
        sgs_target_np = sgs_target_np[:nt]
        sgs_pred_np = sgs_pred_np[:nt]
        if sgs_teacher_np is not None:
            sgs_teacher_np = sgs_teacher_np[:nt]

        def compute_2d_spectrum(field_2d):
            """Compute isotropic power spectrum of a 2D field"""
            if field_2d.ndim == 3:
                field_2d = field_2d[0]  # Take first layer if multi-layer
            
            # FFT
            fft = np.fft.fft2(field_2d)
            power = np.abs(fft) ** 2
            
            # Get wavenumber grid
            ny, nx = field_2d.shape
            kx = np.fft.fftfreq(nx, d=grid.dx)
            ky = np.fft.fftfreq(ny, d=grid.dy)
            kx_grid, ky_grid = np.meshgrid(kx, ky)
            k_mag = np.sqrt(kx_grid**2 + ky_grid**2)
            
            # Bin by radial wavenumber
            k_max = np.sqrt((nx/2)**2 + (ny/2)**2) / max(grid.Lx, grid.Ly)
            k_bins = np.linspace(0, k_max, min(nx, ny) // 2)
            k_centers = (k_bins[:-1] + k_bins[1:]) / 2
            
            spectrum = np.zeros(len(k_centers))
            counts = np.zeros(len(k_centers))
            
            for i in range(len(k_centers)):
                mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
                spectrum[i] = np.sum(power[mask])
                counts[i] = np.sum(mask)
            
            # Normalize
            spectrum = np.where(counts > 0, spectrum / counts, 0)
            
            return k_centers, spectrum

        def compute_time_avg_spectrum(sgs_array):
            """Compute time-averaged spectrum"""
            spectra = []
            for t in range(sgs_array.shape[0]):
                k, spec = compute_2d_spectrum(sgs_array[t])
                spectra.append(spec)
            return k, np.mean(spectra, axis=0), np.std(spectra, axis=0)

        # Compute time-averaged spectra
        k, spec_target_avg, spec_target_std = compute_time_avg_spectrum(sgs_target_np)
        _, spec_pred_avg, spec_pred_std = compute_time_avg_spectrum(sgs_pred_np)
        
        if sgs_teacher_np is not None:
            _, spec_teacher_avg, spec_teacher_std = compute_time_avg_spectrum(sgs_teacher_np)

        # Create comprehensive plot
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # Top row: Spectral comparisons
        ax_spectrum = fig.add_subplot(gs[0, :2])
        ax_ratio = fig.add_subplot(gs[0, 2])
        
        # Bottom row: Error analysis
        ax_error_spec = fig.add_subplot(gs[1, 0])
        ax_correlation = fig.add_subplot(gs[1, 1])
        ax_transfer = fig.add_subplot(gs[1, 2])

        # Main spectrum plot
        ax_spectrum.loglog(k[1:], spec_target_avg[1:], 'k-', linewidth=2, label='Target SGS')
        ax_spectrum.fill_between(k[1:], 
                                 (spec_target_avg - spec_target_std)[1:],
                                 (spec_target_avg + spec_target_std)[1:],
                                 color='k', alpha=0.15)
        
        ax_spectrum.loglog(k[1:], spec_pred_avg[1:], 'C1--', linewidth=2, label='Rollout SGS')
        ax_spectrum.fill_between(k[1:], 
                                 (spec_pred_avg - spec_pred_std)[1:],
                                 (spec_pred_avg + spec_pred_std)[1:],
                                 color='C1', alpha=0.15)
        
        if sgs_teacher_np is not None:
            ax_spectrum.loglog(k[1:], spec_teacher_avg[1:], 'C2:', linewidth=2, label='Teacher SGS')
            ax_spectrum.fill_between(k[1:], 
                                     (spec_teacher_avg - spec_teacher_std)[1:],
                                     (spec_teacher_avg + spec_teacher_std)[1:],
                                     color='C2', alpha=0.15)
        
        ax_spectrum.set_xlabel('Wavenumber k')
        ax_spectrum.set_ylabel('SGS Forcing Power')
        ax_spectrum.set_title('Time-Averaged SGS Forcing Spectrum', fontsize=12, fontweight='bold')
        ax_spectrum.grid(True, which='both', alpha=0.3)
        ax_spectrum.legend()

        # Spectral ratio (how well does prediction match target at each scale?)
        ratio = np.where(spec_target_avg > 1e-20, spec_pred_avg / spec_target_avg, 1.0)
        ax_ratio.semilogx(k[1:], ratio[1:], 'C1-', linewidth=2)
        ax_ratio.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Perfect match')
        ax_ratio.fill_between(k[1:], 0.8, 1.2, color='green', alpha=0.1, label='±20%')
        ax_ratio.set_xlabel('Wavenumber k')
        ax_ratio.set_ylabel('Predicted / Target')
        ax_ratio.set_title('Spectral Amplitude Ratio', fontsize=11, fontweight='bold')
        ax_ratio.grid(True, which='both', alpha=0.3)
        ax_ratio.legend()
        ax_ratio.set_ylim([0, 3])

        # Error spectrum
        error_sgs = sgs_pred_np - sgs_target_np
        _, spec_error_avg, spec_error_std = compute_time_avg_spectrum(error_sgs)
        
        ax_error_spec.loglog(k[1:], spec_target_avg[1:], 'k-', linewidth=2, alpha=0.5, label='Target')
        ax_error_spec.loglog(k[1:], spec_error_avg[1:], 'r-', linewidth=2, label='Error')
        ax_error_spec.set_xlabel('Wavenumber k')
        ax_error_spec.set_ylabel('Power')
        ax_error_spec.set_title('Error Spectrum\n(Target - Predicted)', fontsize=11, fontweight='bold')
        ax_error_spec.grid(True, which='both', alpha=0.3)
        ax_error_spec.legend()

        # Scale-dependent correlation
        def compute_scale_correlation(field1, field2, k_bins_edges):
            """Compute correlation at different scales using filtering"""
            correlations = []
            scales = []
            
            for i in range(len(k_bins_edges) - 1):
                k_low, k_high = k_bins_edges[i], k_bins_edges[i+1]
                
                # Simple low-pass approach: skip for now, use full field correlation
                # This is a placeholder - would need proper bandpass filtering
                corr = np.corrcoef(field1.ravel(), field2.ravel())[0, 1]
                correlations.append(corr)
                scales.append((k_low + k_high) / 2)
            
            return scales, correlations

        # Time-averaged spatial correlation
        corr_time = []
        for t in range(nt):
            target_flat = sgs_target_np[t].ravel()
            pred_flat = sgs_pred_np[t].ravel()
            corr_time.append(np.corrcoef(target_flat, pred_flat)[0, 1])
        
        ax_correlation.plot(np.arange(nt), corr_time, 'C1-', linewidth=1.5)
        ax_correlation.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax_correlation.axhline(1, color='k', linestyle='--', alpha=0.3)
        ax_correlation.set_xlabel('Time Step')
        ax_correlation.set_ylabel('Spatial Correlation')
        ax_correlation.set_title('SGS Correlation\n(Target vs Rollout)', fontsize=11, fontweight='bold')
        ax_correlation.grid(True, alpha=0.3)
        ax_correlation.set_ylim([-1, 1])

        # Energy transfer at different scales (simplified metric)
        # RMS by scale as a proxy for energy transfer
        rms_target = np.sqrt(np.mean(sgs_target_np**2, axis=(1, 2, 3)))
        rms_pred = np.sqrt(np.mean(sgs_pred_np**2, axis=(1, 2, 3)))
        rms_error = np.sqrt(np.mean(error_sgs**2, axis=(1, 2, 3)))
        
        ax_transfer.plot(np.arange(nt), rms_target, 'k-', linewidth=2, label='Target RMS')
        ax_transfer.plot(np.arange(nt), rms_pred, 'C1--', linewidth=2, label='Rollout RMS')
        ax_transfer.plot(np.arange(nt), rms_error, 'r:', linewidth=2, label='Error RMS')
        ax_transfer.set_xlabel('Time Step')
        ax_transfer.set_ylabel('RMS SGS Forcing')
        ax_transfer.set_title('SGS Magnitude Evolution', fontsize=11, fontweight='bold')
        ax_transfer.grid(True, alpha=0.3)
        ax_transfer.legend()

        plt.suptitle('SGS Forcing Spectral Diagnostics', fontsize=14, fontweight='bold')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


_REGISTRY = {
    "loss": LossDiagnostic,
    "ke_spectrum": KESpectrumDiagnostic,
    "PV": VorticityDiagnostic,
    "quad": QuadGifDiagnostic,
    'zero': ZeroComparisonDiagnostic,
    "energy": EnergyDiagnostic,
    "ke_spectrum_movie": KESpectrumAnimationDiagnostic,
    'cfl': CFLDiagnostic,
    "domain": DomainDiagnostic,
    "sgs_spectrum": SGSSpectralDiagnostic,
}

def build_diagnostic(name: str) -> Diagnostic:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown diagnostic '{name}'")
    return _REGISTRY[name]()