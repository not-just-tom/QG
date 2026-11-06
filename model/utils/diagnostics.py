from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for diagnostics
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import importlib
import model.utils.physics_ops
importlib.reload(model.utils.physics_ops)
from model.core.model import QGM
from model.utils.physics_ops import (
    invert_pv_to_psi,
    velocity_from_psi,
    isotropic_ke_spectrum,
    isotropic_scalar_spectrum,
)


def _sanitize_numeric_array(values):
    """Replace non-finite values with zeros so plotting diagnostics stay robust."""
    if values is None:
        return None
    arr = np.asarray(values)
    if arr.dtype.kind in {"U", "S", "O"}:
        return arr
    try:
        return np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        return arr


def _get_traj_array(trajs, *keys, default=None):
    """Return the first available trajectory array from a list of legacy/new keys."""
    for key in keys:
        if key in trajs and trajs.get(key) is not None:
            return trajs.get(key)
    return default


def _prepare_array(values):
    """Normalize assorted trajectory arrays into a form the diagnostics can plot."""
    arr = _sanitize_numeric_array(values)
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return arr.reshape(1, 1, 1, 1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1, 1, 1)
    if arr.ndim == 2:
        return arr[None, ...]
    return arr


def _pick_frame(arr, idx, layer=0):
    """Extract a frame from a prediction/truth array, handling 2D/3D/4D layouts."""
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim == 4:
        return arr[idx, layer]
    if arr.ndim == 3:
        if arr.shape[0] > 1 and idx < arr.shape[0]:
            return arr[idx]
        return arr[layer] if layer < arr.shape[0] else arr[0]
    if arr.ndim == 2:
        return arr
    return arr[idx] if arr.shape[0] > 1 else arr[0]


def _get_reference_model(trajs, grid):
    """Resolve the QG model whose scaling defines the plotted wavenumbers."""
    model = trajs.get("model")
    if model is not None:
        return model

    params = dict(trajs.get("params") or {})
    params.setdefault("nx", grid.nx)
    params.setdefault("ny", grid.ny)
    params.setdefault("nz", getattr(grid, "nz", 1))
    return QGM(params)


def _get_reference_wavenumbers(trajs, grid):
    model = _get_reference_model(trajs, grid)
    return model.beta, model.kmin, model.kmax, 1.0 / model.Ld, model.nz


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
        losses = trajs.get("loss_history") or {}
        if not isinstance(losses, dict):
            losses = {}

        train = np.asarray(losses.get("train", []), dtype=float)
        test = np.asarray(losses.get("test", []), dtype=float)
        zero = np.asarray(losses.get("zero", []), dtype=float)
        n_epochs = losses.get("n_epochs", 1)
        if n_epochs is None or n_epochs <= 0:
            n_epochs = 1

        fig, ax = plt.subplots()

        if train.size:
            ax.plot(np.arange(1, len(train) + 1), train, label="train")
        if test.size:
            ax.plot(np.arange(1, len(test) + 1), test, label="test")
        if zero.size:
            # plot average zero loss per curriculum stage
            x = np.arange(1, len(zero)+1)
            y = np.repeat([np.mean(zero[i:i+n_epochs]) for i in range(0, len(zero), n_epochs)], n_epochs)
            ax.step(x, y, label="zero model", linestyle="--", color="C2")

        ax.set_title("Train / Test Loss")
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
        q_truth = _sanitize_numeric_array(trajs.get("truth"))
        q_pred = _sanitize_numeric_array(trajs.get("pred"))
        zero = _sanitize_numeric_array(trajs.get("zero"))
        q_truth = q_truth[10:] # im trying out skipping first few frames to avoid 0s ?
        zero = zero[10:]
        if q_pred is not None:
            q_pred = q_pred[10:]

        beta, k_min, k_max, k_deformation, nz = _get_reference_wavenumbers(trajs, grid)

        def rhines_wavenumber(frame):
            psi = invert_pv_to_psi(_sanitize_numeric_array(frame), grid)
            u, v = velocity_from_psi(psi, grid)
            U_rms = np.sqrt(np.mean(u**2 + v**2))
            return np.sqrt(beta / U_rms)

        # compute per-frame spectra helper
        def compute_frame_spectra(q):
            try:
                frames = []
                nt = q.shape[0]
                for t in range(nt):
                    q_t = _sanitize_numeric_array(q[t])
                    psi_t = invert_pv_to_psi(q_t, grid)
                    u_t, v_t = velocity_from_psi(psi_t, grid) # shape (nz, ny, nx)
                    spec_t = isotropic_ke_spectrum(u_t, v_t, grid)
                    Et = spec_t["E"]
                    frames.append(np.asarray(Et).ravel())
                k = np.asarray(spec_t["k"]).ravel()
                return np.stack(frames, axis=0), k
            except Exception:
                return None, None

        E_truth_frames, k = compute_frame_spectra(q_truth)
        E_zero, _ = compute_frame_spectra(zero)
        if q_pred is not None:
            E_pred, _ = compute_frame_spectra(q_pred) 

        # averages and stds
        E_truth_avg = E_truth_frames.mean(axis=0)
        if q_pred is not None:
            E_pred_avg = E_pred.mean(axis=0)
            E_pred_std = E_pred.std(axis=0)


        # select frames for animation using cadence
        nt = E_truth_frames.shape[0]
        frame_indices = list(range(0, nt, max(1, cadence)))

        # Build plot: avg lines + shading, instant lines animated on top
        fig, ax = plt.subplots()
        ax.loglog(k[1:], E_truth_avg[1:], label="Truth", color="k")

        try:
            ax.fill_between(k[1:], (E_pred_avg - E_pred_std)[1:], (E_pred_avg + E_pred_std)[1:], color="C1", alpha=0.08)
            Ep0 = E_pred[frame_indices[0]]
            ln_pred, = ax.loglog(k[1:], Ep0[1:], label="ML", color="C3", linestyle="--")
        except Exception:
            pass

        # instantaneous lines
        Ez0 = E_zero[frame_indices[0]]
        ln_zero, = ax.loglog(k[1:], Ez0[1:], label="Zero", color="C2", linestyle="--")
        if nz>1:
            ax.axvline(k_deformation, ymin=0, ymax=1, color='red', linestyle="--")
            ax.text(k_deformation,1e-14,'k_deformation',rotation=270, color='k')
        k_Rhines = rhines_wavenumber(q_truth[frame_indices[0]])
        rhines_line = ax.axvline(k_Rhines, ymin=0, ymax=1, color='red', linestyle="--")
        rhines_label = ax.text(k_Rhines,1e-14,'k_Rhines',rotation=270, color='k')
        ax.axvline(k_min, ymin=0, ymax=1, color='red', linestyle="--")
        ax.text(k_min,1e-14,'k_min',rotation=270, color='k')
        ax.axvline(k_max, ymin=0, ymax=1, color='red', linestyle="--")
        ax.text(k_max,1e-14,'k_max',rotation=270, color='k')
        #ax.axvline(k_nyquist, ymin=0, ymax=1, color='red', linestyle="--")
        #ax.text(k_nyquist,1e-14,'k_nyquist',rotation=270, color='k')
        ax.set_xlabel("k")
        ax.set_ylabel("E(k)")
        ax.set_title(f"KE spectrum (t={frame_indices[0]})")
        ax.grid(True, which="both")
        ax.legend()

        def update(i):
            idx = frame_indices[i]
            k_Rhines = rhines_wavenumber(q_truth[idx])
            rhines_line.set_xdata([k_Rhines, k_Rhines])
            rhines_label.set_position((k_Rhines, 1e-14))
            ax.relim()
            ax.autoscale_view()
            ax.set_title(f"KE spectrum (t={idx})")
            Ez = E_zero[idx]
            ln_zero.set_data(k[1:], Ez[1:])
            if q_pred is not None:
                Ep = E_pred[idx]
                ln_pred.set_data(k[1:], Ep[1:])
                return (ln_pred, ln_zero)
            else: 
                return (None, ln_zero)

        ani = FuncAnimation(fig, update, frames=len(frame_indices), interval=200)
        ani.save(out_path, writer=PillowWriter(fps=5))
        plt.close(fig)


# ============================================================
# KE Spectrum (time-averaged)
# ============================================================



class KESpectrumDiagnostic(Diagnostic):
    name = "ke_spectrum"

    # Hardcoded plotting options
    PLOT_COMPONENTS = True

    def run(self, trajs, out_path, cadence):
        # --- get stuff ---
        grid = trajs.get("grid")
        q_truth = _sanitize_numeric_array(trajs.get("truth"))
        q_pred = _sanitize_numeric_array(trajs.get("pred"))
        zero = _sanitize_numeric_array(trajs.get("zero"))

        beta, k_min, k_max, k_deformation, nz = _get_reference_wavenumbers(trajs, grid)
        k_damping = 0.65 * np.pi / grid.dx

        # --- helper: compute spectra ---
        def compute_avg_spectrum(q):
            if q is None:
                return None, None, None, None

            psi = invert_pv_to_psi(q, grid)
            u, v = velocity_from_psi(psi, grid)

            # Total KE spectrum
            spec_total = isotropic_ke_spectrum(u, v, grid)

            # Individual component spectra
            spec_x = isotropic_ke_spectrum(u, np.zeros_like(v), grid)
            spec_y = isotropic_ke_spectrum(np.zeros_like(u), v, grid)

            k = np.asarray(spec_total["k"]).ravel()
            E_total = np.asarray(spec_total["E"]).ravel()
            E_x = np.asarray(spec_x["E"]).ravel()
            E_y = np.asarray(spec_y["E"]).ravel()

            return k, E_total, E_x, E_y

        # --- compute averaged spectra ---
        k, E_truth_avg, Ex_truth_avg, Ey_truth_avg = compute_avg_spectrum(q_truth)

        _, E_zero_avg, _, _ = compute_avg_spectrum(zero)

        if q_pred is not None:
            _, E_pred_avg, Ex_pred_avg, Ey_pred_avg = compute_avg_spectrum(q_pred)
        else:
            E_pred_avg = Ex_pred_avg = Ey_pred_avg = None

        # --- compute Rhines wavenumber ---
        u, v = velocity_from_psi(invert_pv_to_psi(q_truth, grid), grid)
        U_rms = np.sqrt(np.mean(u**2 + v**2))
        k_Rhines = np.sqrt(beta / U_rms)

        # --- compute per-frame spectra ---
        def compute_frame_spectra(q):
            try:
                total_frames = []
                x_frames = []
                y_frames = []

                for t in range(q.shape[0]):
                    q_t = _sanitize_numeric_array(q[t])
                    psi_t = invert_pv_to_psi(q_t, grid)
                    u_t, v_t = velocity_from_psi(psi_t, grid)

                    total_frames.append(
                        np.asarray(
                            isotropic_ke_spectrum(u_t, v_t, grid)["E"]
                        ).ravel()
                    )
                    x_frames.append(
                        np.asarray(
                            isotropic_ke_spectrum(u_t, np.zeros_like(v_t), grid)["E"]
                        ).ravel()
                    )
                    y_frames.append(
                        np.asarray(
                            isotropic_ke_spectrum(np.zeros_like(u_t), v_t, grid)["E"]
                        ).ravel()
                    )

                return (
                    np.stack(total_frames),
                    np.stack(x_frames),
                    np.stack(y_frames),
                )

            except Exception:
                return None, None, None

        E_truth_frames, Ex_truth_frames, Ey_truth_frames = compute_frame_spectra(q_truth)

        E_truth_std = E_truth_frames.std(axis=0) if E_truth_frames is not None else None
        Ex_truth_std = Ex_truth_frames.std(axis=0) if Ex_truth_frames is not None else None
        Ey_truth_std = Ey_truth_frames.std(axis=0) if Ey_truth_frames is not None else None

        if q_pred is not None:
            E_pred_frames, Ex_pred_frames, Ey_pred_frames = compute_frame_spectra(q_pred)

            E_pred_std = E_pred_frames.std(axis=0) if E_pred_frames is not None else None
            Ex_pred_std = Ex_pred_frames.std(axis=0) if Ex_pred_frames is not None else None
            Ey_pred_std = Ey_pred_frames.std(axis=0) if Ey_pred_frames is not None else None
        else:
            E_pred_std = Ex_pred_std = Ey_pred_std = None

        # --- plotting ---
        fig, ax = plt.subplots(figsize=(9, 6))

        # Total KE: black
        ax.loglog(
            k[1:], E_truth_avg[1:],
            color="k", linestyle="-",
            label="Truth: $K_x+K_y$"
        )

        if E_pred_avg is not None:
            ax.loglog(
                k[1:], E_pred_avg[1:],
                color="k", linestyle="--",
                label="Prediction: $K_x+K_y$"
            )

        # Individual components
        if self.PLOT_COMPONENTS:
            ax.loglog(
                k[1:], Ex_truth_avg[1:],
                color="red", linestyle="-",
                label="Truth: $K_x$"
            )
            ax.loglog(
                k[1:], Ey_truth_avg[1:],
                color="blue", linestyle="-",
                label="Truth: $K_y$"
            )

            if Ex_pred_avg is not None:
                ax.loglog(
                    k[1:], Ex_pred_avg[1:],
                    color="red", linestyle="--",
                    label="Prediction: $K_x$"
                )
                ax.loglog(
                    k[1:], Ey_pred_avg[1:],
                    color="blue", linestyle="--",
                    label="Prediction: $K_y$"
                )

        # Zero baseline: green
        if E_zero_avg is not None:
            ax.loglog(
                k[1:], E_zero_avg[1:],
                color="green", linestyle="-",
                label="Zero"
            )

        # --- uncertainty bands ---
        try:
            if E_truth_std is not None:
                ax.fill_between(
                    k[1:],
                    (E_truth_avg - E_truth_std)[1:],
                    (E_truth_avg + E_truth_std)[1:],
                    color="k", alpha=0.10
                )

            if E_pred_std is not None and E_pred_avg is not None:
                ax.fill_between(
                    k[1:],
                    (E_pred_avg - E_pred_std)[1:],
                    (E_pred_avg + E_pred_std)[1:],
                    color="k", alpha=0.06
                )

            if self.PLOT_COMPONENTS:
                if Ex_truth_std is not None:
                    ax.fill_between(
                        k[1:],
                        (Ex_truth_avg - Ex_truth_std)[1:],
                        (Ex_truth_avg + Ex_truth_std)[1:],
                        color="red", alpha=0.08
                    )

                if Ey_truth_std is not None:
                    ax.fill_between(
                        k[1:],
                        (Ey_truth_avg - Ey_truth_std)[1:],
                        (Ey_truth_avg + Ey_truth_std)[1:],
                        color="blue", alpha=0.08
                    )

                if Ex_pred_std is not None:
                    ax.fill_between(
                        k[1:],
                        (Ex_pred_avg - Ex_pred_std)[1:],
                        (Ex_pred_avg + Ex_pred_std)[1:],
                        color="red", alpha=0.04
                    )

                if Ey_pred_std is not None:
                    ax.fill_between(
                        k[1:],
                        (Ey_pred_avg - Ey_pred_std)[1:],
                        (Ey_pred_avg + Ey_pred_std)[1:],
                        color="blue", alpha=0.04
                    )

        except Exception:
            pass

        # --- reference wavenumbers ---
        if nz > 1:
            ax.axvline(k_deformation, color="grey", linestyle="--")

        ax.axvline(k_Rhines, color="grey", linestyle="--", label="$k_{Rhines}$")
        ax.axvline(k_min, color="grey", linestyle=":")
        ax.axvline(k_max, color="grey", linestyle=":")

        ax.axvline(
            k_damping,
            color="purple",
            linestyle="-.",
            label="Exact-filter damping onset",
        )

        ax.set_xlabel("k")
        ax.set_ylabel("E(k)")
        ax.set_title("Time-averaged KE spectrum")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9)

        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)



# ============================================================
# PV Animation
# ============================================================

class VorticityDiagnostic(Diagnostic): # this might need cadence adding to it tbh
    name = "PV"
    output = "gif"

    def run(self, trajs, out_path, cadence):
        truth = _prepare_array(_get_traj_array(trajs, "truth", "q"))
        if truth is None:
            raise KeyError("PV diagnostic requires 'q' or 'truth' in trajectories")
        ml = _prepare_array(_get_traj_array(trajs, "pred", "pred"))

        # Auto-determine number of layers from data shape
        # Expected shape: (nt, nz, ny, nx)
        nt = truth.shape[0]
        nz = truth.shape[1] if truth.ndim >= 2 else 1
        cols = 2 if ml is not None else 1

        fig, axes = plt.subplots(nz, cols, squeeze=False,
                                 figsize=(4 * cols, 3 * nz))

        # fixed color scale (important)
        vmin = np.nanmin(truth)
        vmax = np.nanmax(truth)

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


class PVSpectrumDiagnostic(Diagnostic):
    name = "pv_spectrum"

    def run(self, trajs, out_path, cadence):

        grid = trajs.get("grid")
        q_truth = _sanitize_numeric_array(trajs.get("truth"))
        q_pred  = _sanitize_numeric_array(trajs.get("pred"))
        q_zero  = _sanitize_numeric_array(trajs.get("zero"))


        def compute_avg_spectrum(q):
            if q is None:
                return None, None

            spec = isotropic_scalar_spectrum(q, grid)

            return (
                np.asarray(spec["k"]).ravel(),
                np.asarray(spec["E"]).ravel(),
            )

        k, E_truth = compute_avg_spectrum(q_truth)
        _, E_zero  = compute_avg_spectrum(q_zero)

        if q_pred is not None:
            _, E_pred = compute_avg_spectrum(q_pred)
        else:
            E_pred = None

        fig, ax = plt.subplots()

        ax.loglog(k[1:], E_truth[1:], "k", label="Truth")

        if E_pred is not None:
            ax.loglog(k[1:], E_pred[1:], color="C1",
                      label="Parameterisation")

        if E_zero is not None:
            ax.loglog(k[1:], E_zero[1:], color="C2",
                      label="Zero")

        ax.set_xlabel("k")
        ax.set_ylabel(r"$Q(k)$")
        ax.set_title("Time-averaged PV spectrum")
        ax.grid(True, which="both")
        ax.legend()

        fig.savefig(out_path,
                    dpi=150,
                    bbox_inches="tight")

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
        pred = _prepare_array(_get_traj_array(trajs, "pred", "pred"))
        truth = _prepare_array(_get_traj_array(trajs, "truth", "q"))
        zero = _prepare_array(_get_traj_array(trajs, "zero", "zero"))
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
            """Extract single frame, handling 2D/3D/4D arrays."""
            return _pick_frame(arr, idx)

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

        # ==== initialise PV plots (Row 1) ====
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

        # ==== initialise SGS plots (Row 2) ====
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
    name = "zero"
    output = "gif"

    def run(self, trajs, out_path, cadence):
        truth = _prepare_array(_get_traj_array(trajs, "truth", "q"))
        zero = _prepare_array(_get_traj_array(trajs, "zero", "zero"))

        truth_np = np.asarray(truth)
        zero_np = np.asarray(zero)
        
        if "pred" in trajs:
            pred = _prepare_array(_get_traj_array(trajs, "pred"))
            pred_np = np.asarray(pred)
        else: 
            pred_np = None

        # Get predicted/applied SGS (if present)
        if "sgs" in trajs:
            sgs_pred = trajs.get("sgs")
            sgs_pred_np = np.asarray(sgs_pred)
        else:
            sgs_pred_np = None

        # Determine number of timesteps from available prediction/truth
        nt = truth_np.shape[0]

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

        # Only prediction-related arrays may need padding
        sgs_pred_np = pad_to_frames(sgs_pred_np)

        indices = np.arange(0, nt, max(1, int(cadence)))
        if indices.size == 0:
            raise ValueError("No frames selected for diagnostic (check cadence)")

        def pick(arr, idx):
            return _pick_frame(arr, idx)

        # Robust colour limits
        def pct(a, q):
            try:
                return np.nanpercentile(a, q)
            except Exception:
                return 0.0

        vmin_truth = pct(truth_np, 1)
        vmax_truth = pct(truth_np, 99)

        # ------------------------------------------------------------------
        # Build panels dynamically
        # ------------------------------------------------------------------
        panels = [
            ("Truth", truth_np),
            ("Without closure", zero_np),
        ]

        if pred_np is not None:
            panels.insert(1, ("Closure adjusted", pred_np))

        nplots = len(panels)

        fig, axes = plt.subplots(1, nplots, figsize=(4 * nplots, 4))

        # Ensure axes is iterable
        if nplots == 1:
            axes = [axes]
        else:
            axes = np.atleast_1d(axes)

        images = []

        for ax, (title, data) in zip(axes, panels):
            im = ax.imshow(
                pick(data, indices[0]),
                origin="lower",
                cmap="RdBu_r",
                vmin=vmin_truth,
                vmax=vmax_truth,
            )
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            images.append(im)

        fig.colorbar(images[0], ax=axes, shrink=0.6)

        def update(i):
            idx = indices[i]

            for im, (_, data) in zip(images, panels):
                im.set_data(pick(data, idx))

            fig.suptitle(f"timestep {idx}")
            return images

        anim = FuncAnimation(
            fig,
            update,
            frames=len(indices),
            interval=100,
            blit=False,
        )

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
        q = _prepare_array(_get_traj_array(trajs, "truth", "q"))
        if q is None:
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


class ZonalDiagnostic(Diagnostic):
    name = "zonal"

    def run(self, trajs, out_path, cadence=10):
        grid = trajs.get("grid")
        if grid is None:
            raise KeyError("zonal requires 'grid' in trajectories")

        def zonal_mean_u(values, key):
            q = _sanitize_numeric_array(values)
            if q is None:
                return None
            q = np.asarray(q)
            if q.ndim == 2:
                q = q[None, None, ...]
            elif q.ndim == 3:
                q = q[:, None, ...]
            if q.ndim != 4:
                raise ValueError(f"{key} must have shape (time, layer, y, x)")
            psi = invert_pv_to_psi(q, grid)
            u, _ = velocity_from_psi(psi, grid)
            return u.mean(axis=-1)

        truth = zonal_mean_u(_get_traj_array(trajs, "truth", "q"), "truth")
        if truth is None:
            raise KeyError("zonal_mean_velocity requires 'truth' or 'q' in trajectories")

        comparisons = [("Truth", truth, "k", "-")]
        for key, label, color, linestyle in (
            ("pred", "Closure adjusted", "C1", "--"),
            ("zero", "Without closure", "C2", ":"),
        ):
            values = zonal_mean_u(trajs.get(key), key)
            if values is not None:
                comparisons.append((label, values, color, linestyle))

        nt, nz, ny = truth.shape
        dt = trajs.get("dt")
        dt = float(dt) if dt is not None else 1.0
        time_edges = np.arange(nt + 1) * dt
        y_edges = np.arange(ny + 1) * grid.dy
        y_centers = (np.arange(ny) + 0.5) * grid.dy

        fig, axes = plt.subplots(
            nz,
            2,
            squeeze=False,
            figsize=(12, 3.5 * nz),
            constrained_layout=True,
        )
        for layer in range(nz):
            ax_hovmoller, ax_profile = axes[layer]
            field = truth[:, layer, :]
            limit = float(np.nanpercentile(np.abs(field), 99))
            if not np.isfinite(limit) or limit == 0:
                limit = 1.0

            image = ax_hovmoller.pcolormesh(
                time_edges,
                y_edges,
                field.T,
                shading="auto",
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
            )
            fig.colorbar(image, ax=ax_hovmoller, label="Zonal-mean u")
            ax_hovmoller.set_title(f"Truth, layer {layer}: zonal-mean velocity")
            ax_hovmoller.set_xlabel("Time (model units)" if dt != 1.0 else "Time step")
            ax_hovmoller.set_ylabel("y")
            ax_hovmoller.grid(False)

            for label, values, color, linestyle in comparisons:
                if layer < values.shape[1]:
                    ax_profile.plot(
                        values[:, layer, :].mean(axis=0),
                        y_centers,
                        color=color,
                        linestyle=linestyle,
                        label=label,
                    )
            ax_profile.set_title(f"Time-mean zonal velocity, layer {layer}")
            ax_profile.set_xlabel("Zonal-mean u")
            ax_profile.set_ylabel("y")
            ax_profile.grid(True, alpha=0.3)
            ax_profile.legend()

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

        q_pred = trajs.get("pred")
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
        q_truth = _prepare_array(_get_traj_array(trajs, "truth", "q"))
        if q_truth is None:
            raise KeyError("domain diagnostic requires 'q' or 'truth' in trajectories")

        # Get predicted and zero model data if available
        q_pred = _prepare_array(_get_traj_array(trajs, "pred", "pred"))
        q_zero = _prepare_array(_get_traj_array(trajs, "zero", "zero"))

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
        sgs_rollout = trajs.get("sgs")
        sgs_teacher = trajs.get("teacher_forced_sgs")
        sgs_pred = sgs_teacher if sgs_teacher is not None else sgs_rollout
        
        if sgs_target is None or sgs_pred is None:
            raise KeyError(
                "sgs_spectrum requires 'target_sgs' and either "
                "'teacher_forced_sgs' or 'sgs' in trajectories"
            )

        sgs_target_np = np.asarray(sgs_target)
        sgs_pred_np = np.asarray(sgs_pred)
        sgs_rollout_np = np.asarray(sgs_rollout) if sgs_rollout is not None else None

        # Ensure same length
        nt = min(sgs_target_np.shape[0], sgs_pred_np.shape[0])
        sgs_target_np = sgs_target_np[:nt]
        sgs_pred_np = sgs_pred_np[:nt]
        if sgs_rollout_np is not None:
            sgs_rollout_np = sgs_rollout_np[:nt]

        def compute_2d_spectrum(field_2d):
            """Compute isotropic power, averaging over all head/layer axes."""
            field_2d = np.asarray(field_2d, dtype=float)
            fft = np.fft.fft2(field_2d, axes=(-2, -1))
            power = np.abs(fft) ** 2
            if power.ndim > 2:
                power = power.mean(axis=tuple(range(power.ndim - 2)))

            ny, nx = field_2d.shape[-2:]
            kx = 2 * np.pi * np.fft.fftfreq(nx, d=grid.dx)
            ky = 2 * np.pi * np.fft.fftfreq(ny, d=grid.dy)
            kx_grid, ky_grid = np.meshgrid(kx, ky)
            k_mag = np.sqrt(kx_grid**2 + ky_grid**2)

            dk = min(2 * np.pi / grid.Lx, 2 * np.pi / grid.Ly)
            k_bins = np.arange(0, k_mag.max() + dk, dk)
            if k_bins.size < 2:
                k_bins = np.array([0.0, dk])
            k_centers = (k_bins[:-1] + k_bins[1:]) / 2
            spectrum = np.zeros(len(k_centers))
            counts = np.zeros(len(k_centers))
            for i in range(len(k_centers)):
                mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
                spectrum[i] = np.sum(power[mask])
                counts[i] = np.sum(mask)
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
        
        if sgs_rollout_np is not None:
            _, spec_rollout_avg, spec_rollout_std = compute_time_avg_spectrum(sgs_rollout_np)

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
        ax_spectrum.loglog(k, spec_target_avg, 'k-', linewidth=2, label='Interval SGS target')
        ax_spectrum.fill_between(k,
                                 spec_target_avg - spec_target_std,
                                 spec_target_avg + spec_target_std,
                                 color='k', alpha=0.15)
        
        ax_spectrum.loglog(k, spec_pred_avg, 'C2:', linewidth=2, label='Teacher-forced closure')
        ax_spectrum.fill_between(k,
                                 spec_pred_avg - spec_pred_std,
                                 spec_pred_avg + spec_pred_std,
                                 color='C2', alpha=0.15)
        
        if sgs_rollout_np is not None:
            ax_spectrum.loglog(k, spec_rollout_avg, 'C1--', linewidth=2, label='Rollout SGS')
            ax_spectrum.fill_between(k,
                                     spec_rollout_avg - spec_rollout_std,
                                     spec_rollout_avg + spec_rollout_std,
                                     color='C1', alpha=0.15)
        
        ax_spectrum.set_xlabel('Wavenumber k')
        ax_spectrum.set_ylabel('SGS Forcing Power')
        ax_spectrum.set_title('Time-Averaged SGS Forcing Spectrum', fontsize=12, fontweight='bold')
        ax_spectrum.grid(True, which='both', alpha=0.3)
        ax_spectrum.legend()

        # Spectral ratio (how well does prediction match target at each scale?)
        ratio = np.where(spec_target_avg > 1e-20, spec_pred_avg / spec_target_avg, 1.0)
        ax_ratio.semilogx(k, ratio, 'C2-', linewidth=2)
        ax_ratio.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Perfect match')
        ax_ratio.fill_between(k[1:], 0.8, 1.2, color='green', alpha=0.1, label='±20%')
        ax_ratio.set_xlabel('Wavenumber k')
        ax_ratio.set_ylabel('Teacher-forced / Target')
        ax_ratio.set_title('Spectral Amplitude Ratio', fontsize=11, fontweight='bold')
        ax_ratio.grid(True, which='both', alpha=0.3)
        ax_ratio.legend()
        ax_ratio.set_ylim([0, 3])

        # Error spectrum
        error_sgs = sgs_pred_np - sgs_target_np
        _, spec_error_avg, spec_error_std = compute_time_avg_spectrum(error_sgs)
        
        ax_error_spec.loglog(k, spec_target_avg, 'k-', linewidth=2, alpha=0.5, label='Target')
        ax_error_spec.loglog(k, spec_error_avg, 'r-', linewidth=2, label='Error')
        ax_error_spec.set_xlabel('Wavenumber k')
        ax_error_spec.set_ylabel('Power')
        ax_error_spec.set_title('Error Spectrum\n(Teacher-forced - Target)', fontsize=11, fontweight='bold')
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
        ax_correlation.set_title('SGS Correlation\n(Target vs Teacher-forced)', fontsize=11, fontweight='bold')
        ax_correlation.grid(True, alpha=0.3)
        ax_correlation.set_ylim([-1, 1])

        # Energy transfer at different scales (simplified metric)
        # RMS by scale as a proxy for energy transfer
        rms_target = np.sqrt(np.mean(sgs_target_np**2, axis=(1, 2, 3)))
        rms_pred = np.sqrt(np.mean(sgs_pred_np**2, axis=(1, 2, 3)))
        rms_error = np.sqrt(np.mean(error_sgs**2, axis=(1, 2, 3)))
        
        ax_transfer.plot(np.arange(nt), rms_target, 'k-', linewidth=2, label='Target RMS')
        ax_transfer.plot(np.arange(nt), rms_pred, 'C2:', linewidth=2, label='Teacher-forced RMS')
        if sgs_rollout_np is not None:
            rms_rollout = np.sqrt(np.mean(sgs_rollout_np**2, axis=(1, 2, 3)))
            ax_transfer.plot(np.arange(nt), rms_rollout, 'C1--', linewidth=2, label='Rollout RMS')
        ax_transfer.plot(np.arange(nt), rms_error, 'r:', linewidth=2, label='Error RMS')
        ax_transfer.set_xlabel('Time Step')
        ax_transfer.set_ylabel('RMS SGS Forcing')
        ax_transfer.set_title('SGS Magnitude Evolution', fontsize=11, fontweight='bold')
        ax_transfer.grid(True, alpha=0.3)
        ax_transfer.legend()

        plt.suptitle('SGS Forcing Spectral Diagnostics', fontsize=14, fontweight='bold')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

# ============================================================
# ========= Model Comparison Diagnostics =====================
# ============================================================ 

class MultiModelComparisonDiagnostic(Diagnostic):
    """Compare KE spectra across multiple models."""
    name = "ke_comparison"
    output = "png"

    def run(self, trajs, out_path, cadence):
        grid = trajs.get("grid")
        model_names = trajs.get("model_names", [])
        
        if not model_names:
            print("No models to compare")
            return
        
        q_truth = trajs.get("truth")
        if q_truth is None:
            print("No truth trajectory provided")
            return
        
        q_truth = np.asarray(q_truth)
        
        # Compute truth spectrum
        def compute_avg_spectrum(q):
            psi = invert_pv_to_psi(q, grid)
            u, v = velocity_from_psi(psi, grid)
            spec = isotropic_ke_spectrum(u, v, grid)
            k = np.asarray(spec["k"]).ravel()
            E = np.asarray(spec["E"]).ravel()
            return k, E
        
        k, E_truth_avg = compute_avg_spectrum(q_truth)
        k_damping = 0.65 * np.pi / grid.dx
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Left plot: KE spectra
        ax.loglog(k[1:], E_truth_avg[1:], label="Truth", color="black", linewidth=2)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
        
        for i, model_name in enumerate(model_names):
            pred_key = f"pred_{model_name}"
            q_pred = trajs.get(pred_key)
            
            if q_pred is not None:
                q_pred = np.asarray(q_pred)
                _, E_pred_avg = compute_avg_spectrum(q_pred)
                
                ax.loglog(k[1:], E_pred_avg[1:], label=model_name, 
                         color=colors[i], linestyle='--', alpha=0.8)

        ax.axvline(
            k_damping,
            color="purple",
            linestyle=":",
            label="Exact-filter damping onset",
        )
        
        ax.set_xlabel("Wavenumber k")
        ax.set_ylabel("E(k)")
        ax.set_title("KE Spectrum Comparison")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        
        # Right plot: Relative errors
        # ax = axes[1]
        #
        # for i, model_name in enumerate(model_names):
        #     pred_key = f"pred_{model_name}"
        #     q_pred = trajs.get(pred_key)
        #
        #     if q_pred is not None:
        #         q_pred = np.asarray(q_pred)
        #         _, E_pred_avg = compute_avg_spectrum(q_pred)
        #
        #         # Compute relative error
        #         rel_error = np.abs(E_pred_avg - E_truth_avg) / (E_truth_avg + 1e-10)
        #
        #         ax.loglog(k[1:], rel_error[1:], label=model_name,
        #                   color=colors[i], alpha=0.8)
        #
        # ax.set_xlabel("Wavenumber k")
        # ax.set_ylabel("Relative Error")
        # ax.set_title("Relative Error in KE Spectrum")
        # ax.grid(True, which="both", alpha=0.3)
        # ax.legend()
        
        plt.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


class MultiModelLossDiagnostic(Diagnostic):
    """Compare training loss across multiple models."""
    name = "multi_loss"
    output = "png"

    def run(self, trajs, out_path, cadence):
        model_names = trajs.get("model_names", [])
        
        if not model_names:
            print("No models to compare")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
        
        # Left plot: Training loss
        ax = axes[0]
        for i, model_name in enumerate(model_names):
            loss_key = f"loss_history_{model_name}"
            losses = trajs.get(loss_key, {})
            
            if losses:
                train = np.asarray(losses.get("train", []))
                if train.size:
                    ax.plot(np.arange(1, len(train) + 1), train, 
                           label=model_name, color=colors[i], alpha=0.8)
        
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training Loss")
        ax.set_title("Training Loss Comparison")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_yscale('log')
        
        # Right plot: Test loss
        ax = axes[1]
        for i, model_name in enumerate(model_names):
            loss_key = f"loss_history_{model_name}"
            losses = trajs.get(loss_key, {})
            
            if losses:
                test = np.asarray(losses.get("test", []))
                if test.size:
                    ax.plot(np.arange(1, len(test) + 1), test, 
                           label=model_name, color=colors[i], alpha=0.8)
        
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test Loss")
        ax.set_title("Test Loss Comparison")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_yscale('log')
        
        plt.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


class MultiModelLossRolloutDiagnostic(Diagnostic):
    """Compare per-step validation loss using each model's training objective."""
    name = "loss_rollout"
    output = "png"

    def run(self, trajs, out_path, cadence):
        model_names = trajs.get("model_names", [])
        fig, ax = plt.subplots(figsize=(9, 5))
        plotted = False
        if model_names:
            results = trajs.get("model_results", {})
            for model_name in model_names:
                result = results.get(model_name)
                if result is None:
                    raise ValueError(
                        f"Missing comparison results for model {model_name!r}"
                    )
                loss_by_step = np.asarray(
                    result.get("validation_loss_rollout")
                )
                if loss_by_step.ndim != 1 or loss_by_step.size == 0:
                    raise ValueError(
                        f"Model {model_name!r} has no per-step validation loss"
                    )
                loss_name = result.get("loss_name", "configured loss")
                model_line, = ax.plot(
                    np.arange(1, loss_by_step.size + 1),
                    loss_by_step,
                    label=f"{model_name} ({loss_name})",
                )
                zero_loss_by_step = result.get("zero_validation_loss_rollout")
                if zero_loss_by_step is not None:
                    zero_loss_by_step = np.asarray(zero_loss_by_step)
                    if (
                        zero_loss_by_step.ndim != 1
                        or zero_loss_by_step.size != loss_by_step.size
                    ):
                        raise ValueError(
                            f"Model {model_name!r} has invalid zero-model "
                            "per-step validation loss"
                        )
                    ax.plot(
                        np.arange(1, zero_loss_by_step.size + 1),
                        zero_loss_by_step,
                        color=model_line.get_color(),
                        linestyle="--",
                        label=f"zero ({model_name}; {loss_name})",
                    )
                plotted = True
        else:
            loss_by_step = trajs.get("loss_rollout")
            if loss_by_step is None:
                print("No per-step validation loss available")
                plt.close(fig)
                return
            loss_by_step = np.asarray(loss_by_step)
            if loss_by_step.ndim != 1 or loss_by_step.size == 0:
                raise ValueError("Per-step validation loss must be a non-empty 1D array")
            loss_name = trajs.get("loss_name", "configured loss")
            ax.plot(
                np.arange(1, loss_by_step.size + 1),
                loss_by_step,
                label=str(loss_name),
            )
            zero_loss_by_step = trajs.get("zero_loss_rollout")
            if zero_loss_by_step is not None:
                zero_loss_by_step = np.asarray(zero_loss_by_step)
                if (
                    zero_loss_by_step.ndim != 1
                    or zero_loss_by_step.size != loss_by_step.size
                ):
                    raise ValueError(
                        "Zero-model per-step validation loss must match "
                        "the model loss shape"
                    )
                ax.plot(
                    np.arange(1, zero_loss_by_step.size + 1),
                    zero_loss_by_step,
                    label=f"zero ({loss_name})",
                    linestyle="--",
                    color="C2",
                )
            plotted = True

        if not plotted:
            plt.close(fig)
            print("No model rollout predictions available")
            return

        ax.set_xlabel("Validation rollout step")
        ax.set_ylabel("Configured training loss")
        ax.set_title("Per-step validation rollout loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


class ParetoValidationDiagnostic(Diagnostic):
    name = "pareto_validation"
    output = "png"

    def run(self, trajs, out_path, cadence):

        results = trajs.get("model_results", {})

        if not results:
            print("No model results available")
            return

        fig, ax = plt.subplots(figsize=(8, 6))

        xs = []
        ys = []
        labels = []
        zero_baseline_rollout = None
        training_horizon_steps = None

        for model_name, result in results.items():
            model_mae = np.asarray(result.get("validation_mae_rollout"))
            zero_mae_rollout = np.asarray(
                result.get("zero_validation_mae_rollout")
            )
            result_horizon_steps = int(result["training_horizon_steps"])
            if result_horizon_steps <= 0:
                raise ValueError(
                    f"Model {model_name!r} has an invalid training horizon"
                )
            if training_horizon_steps is None:
                training_horizon_steps = result_horizon_steps
            elif result_horizon_steps != training_horizon_steps:
                raise ValueError(
                    "Models must use the same maximum training horizon"
                )
            if model_mae.ndim != 1 or model_mae.size == 0:
                raise ValueError(
                    f"Model {model_name!r} has no per-step validation MAE"
                )
            if zero_mae_rollout.ndim != 1 or zero_mae_rollout.size == 0:
                raise ValueError(
                    f"Model {model_name!r} has no per-step zero-model MAE baseline"
                )
            if not np.all(np.isfinite(model_mae)) or not np.all(
                np.isfinite(zero_mae_rollout)
            ):
                raise ValueError(
                    f"Model {model_name!r} has non-finite validation MAE values"
                )
            if zero_mae_rollout.size < training_horizon_steps:
                raise ValueError(
                    f"Model {model_name!r} has only {zero_mae_rollout.size} "
                    "zero-model validation steps; the training horizon is "
                    f"{training_horizon_steps}"
                )

            zero_baseline = float(
                np.max(zero_mae_rollout[:training_horizon_steps])
            )
            if zero_baseline_rollout is None:
                zero_baseline_rollout = zero_mae_rollout
            elif (
                zero_mae_rollout.shape != zero_baseline_rollout.shape
                or not np.allclose(zero_mae_rollout, zero_baseline_rollout)
            ):
                raise ValueError(
                    "Models must use the same zero-model validation baseline"
                )
            survival_steps = int(
                np.argmax(model_mae > zero_baseline)
                if np.any(model_mae > zero_baseline)
                else model_mae.size
            )
            xs.append(float(np.mean(model_mae)))
            ys.append(survival_steps)
            labels.append(model_name)

        xs = np.asarray(xs)
        ys = np.asarray(ys)

        ax.scatter(xs, ys, s=80, alpha=0.8)

        for x, y, label in zip(xs, ys, labels):
            ax.annotate(
                label,
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        pareto_indices = [
            i for i in range(len(xs))
            if not any(
                xs[j] <= xs[i]
                and ys[j] >= ys[i]
                and (xs[j] < xs[i] or ys[j] > ys[i])
                for j in range(len(xs))
                if j != i
            )
        ]
        pareto_indices.sort(key=lambda i: xs[i])

        ax.plot(
            xs[pareto_indices],
            ys[pareto_indices],
            "--",
            linewidth=2,
            label="Pareto Front",
        )

        ax.set_xlabel("Mean Validation MAE")
        ax.set_ylabel("Rollout Survival Time (steps)")

        ax.set_title(
            "Pareto Comparison: Error vs Survival Relative to Zero Model"
        )

        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

        plt.tight_layout()
        fig.savefig(
            out_path,
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)


class ParetoRolloutAccuracyDiagnostic(Diagnostic):
    """Compare rollout latency against accuracy relative to zero and HR references."""
    name = "pareto_rollout_accuracy"
    output = "png"

    def run(self, trajs, out_path, cadence):
        results = trajs.get("model_results", {})
        if not results:
            print("No model results available")
            return

        points = []
        zero_result = next(
            (
                result
                for result in results.values()
                if result.get("model_type") == "zero"
            ),
            None,
        )
        zero_error_reference = float(
            zero_result["validation_mae"]
            if zero_result is not None
            else next(iter(results.values()))["zero_validation_mae"]
        )
        if zero_result is not None and not np.isclose(
            zero_error_reference, float(zero_result["zero_validation_mae"])
        ):
            raise ValueError(
                "The explicit zero model does not match the physics-only zero rollout"
            )
        high_resolution_times = np.asarray(
            [
                result["high_resolution_rollout_seconds"]
                for result in results.values()
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(high_resolution_times)) or np.any(
            high_resolution_times <= 0
        ):
            raise ValueError("Invalid high-resolution rollout timing baseline")
        high_resolution_seconds = float(np.mean(high_resolution_times))
        for model_name, result in results.items():
            rollout_seconds = float(result["rollout_prediction_seconds"])
            accuracy_error = float(result["validation_mae"])
            zero_error = float(result["zero_validation_mae"])
            if (
                not np.isfinite(rollout_seconds)
                or rollout_seconds <= 0
                or not np.isfinite(accuracy_error)
                or accuracy_error < 0
                or not np.isfinite(zero_error)
                or zero_error < 0
            ):
                raise ValueError(
                    f"Model {model_name!r} has invalid rollout time or validation MAE"
                )
            if not np.isclose(zero_error, zero_error_reference):
                raise ValueError(
                    "Models must use the same zero-model accuracy baseline"
                )
            if zero_error_reference > 0:
                accuracy_score = 1.0 - accuracy_error / zero_error_reference
            else:
                accuracy_score = 1.0 if accuracy_error == 0 else 0.0
            points.append((rollout_seconds, accuracy_score, model_name))
        points.append(
            (high_resolution_seconds, 1.0, "High-resolution truth reference")
        )

        points.sort(key=lambda point: point[0])
        fig, ax = plt.subplots(figsize=(8, 6))
        times = np.asarray([point[0] for point in points])
        accuracy_scores = np.asarray([point[1] for point in points])
        high_resolution_index = next(
            i for i, point in enumerate(points)
            if point[2] == "High-resolution truth reference"
        )
        regular_indices = [
            i for i in range(len(points)) if i != high_resolution_index
        ]
        ax.scatter(
            times[regular_indices],
            accuracy_scores[regular_indices],
            s=80,
            alpha=0.8,
        )
        ax.scatter(
            [times[high_resolution_index]],
            [accuracy_scores[high_resolution_index]],
            marker="*",
            s=180,
            color="black",
            label="High-resolution truth reference",
            zorder=3,
        )

        for rollout_seconds, accuracy_score, model_name in points:
            if model_name == "High-resolution truth reference":
                continue
            ax.annotate(
                model_name,
                (rollout_seconds, accuracy_score),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        pareto_indices = [
            i for i in range(len(points))
            if not any(
                times[j] <= times[i]
                and accuracy_scores[j] >= accuracy_scores[i]
                and (
                    times[j] < times[i]
                    or accuracy_scores[j] > accuracy_scores[i]
                )
                for j in range(len(points))
                if j != i
            )
        ]
        pareto_indices.sort(key=lambda i: times[i])

        ax.plot(
            times[pareto_indices],
            accuracy_scores[pareto_indices],
            "--",
            linewidth=2,
            label="Pareto Front",
        )
        ax.set_xscale("log")
        ax.set_xlabel("Mean rollout prediction time (seconds / trajectory)")
        ax.set_ylabel("Accuracy improvement over zero-model MAE")
        ax.set_title("Pareto Comparison: Rollout Latency vs Relative Accuracy")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


class MultiModelPVRolloutDiagnostic(Diagnostic):
    """Animate truth and all model PV rollouts on synchronized frames."""
    name = "pv_rollout"
    output = "gif"

    def run(self, trajs, out_path, cadence):
        truth = _prepare_array(_sanitize_numeric_array(trajs.get("truth")))
        model_names = trajs.get("model_names", [])
        predictions = {}
        if truth is not None:
            for model_name in model_names:
                prediction = _prepare_array(
                    _sanitize_numeric_array(trajs.get(f"pred_{model_name}"))
                )
                if prediction is not None:
                    predictions[model_name] = prediction

        if truth is None or not predictions:
            print("No comparable model PV rollouts available")
            return

        arrays = [truth, *predictions.values()]
        if any(array.ndim != 4 for array in arrays):
            raise ValueError("PV rollouts must have shape (time, layer, y, x)")
        n_steps = min(array.shape[0] for array in arrays)
        n_layers = min(array.shape[1] for array in arrays)
        if n_steps == 0 or n_layers == 0:
            print("No PV rollout frames available")
            return

        truth = truth[:n_steps, :n_layers]
        predictions = {
            name: prediction[:n_steps, :n_layers]
            for name, prediction in predictions.items()
        }
        scale = float(np.nanpercentile(np.abs(truth), 99.5))
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0

        panels = []
        for layer in range(n_layers):
            panels.append((f"Truth, layer {layer}", truth[:, layer]))
            panels.extend(
                (f"{name}, layer {layer}", prediction[:, layer])
                for name, prediction in predictions.items()
            )

        ncols = min(4, len(panels))
        nrows = int(np.ceil(len(panels) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4 * ncols, 3.5 * nrows),
            squeeze=False,
            constrained_layout=True,
        )
        axes_flat = axes.flat
        images = []
        used_axes = []
        for ax, (label, trajectory) in zip(axes_flat, panels):
            image = ax.imshow(
                trajectory[0],
                origin="lower",
                cmap="RdBu_r",
                vmin=-scale,
                vmax=scale,
                animated=True,
            )
            ax.set_title(label)
            ax.set_xticks([])
            ax.set_yticks([])
            images.append(image)
            used_axes.append(ax)

        for ax in list(axes_flat)[len(panels):]:
            ax.set_visible(False)
        fig.colorbar(images[0], ax=used_axes, shrink=0.8, label="PV")

        frame_indices = np.arange(0, n_steps, max(1, int(cadence)))
        if frame_indices[-1] != n_steps - 1:
            frame_indices = np.append(frame_indices, n_steps - 1)
        title = fig.suptitle("")

        def update(frame_index):
            frame = frame_indices[frame_index]
            for image, (_, trajectory) in zip(images, panels):
                image.set_data(trajectory[frame])
            title.set_text(f"PV rollout comparison, step {frame}")
            return (*images, title)

        animation = FuncAnimation(
            fig,
            update,
            frames=len(frame_indices),
            interval=180,
            blit=False,
        )
        animation.save(out_path, writer=PillowWriter(fps=5))
        plt.close(fig)


class MultiModelPVFinalSnapshotDiagnostic(Diagnostic):
    """Render the final synchronized truth and model PV rollout frames."""
    name = "pv_final_snapshot"
    output = "png"

    def run(self, trajs, out_path, cadence):
        truth = _prepare_array(_sanitize_numeric_array(trajs.get("truth")))
        model_names = trajs.get("model_names", [])
        predictions = {}
        if truth is not None:
            for model_name in model_names:
                prediction = _prepare_array(
                    _sanitize_numeric_array(trajs.get(f"pred_{model_name}"))
                )
                if prediction is not None:
                    predictions[model_name] = prediction

        if truth is None or not predictions:
            print("No comparable model PV rollouts available")
            return

        arrays = [truth, *predictions.values()]
        if any(array.ndim != 4 for array in arrays):
            raise ValueError("PV rollouts must have shape (time, layer, y, x)")
        n_steps = min(array.shape[0] for array in arrays)
        n_layers = min(array.shape[1] for array in arrays)
        if n_steps == 0 or n_layers == 0:
            print("No PV rollout frames available")
            return

        final_step = n_steps - 1
        truth = truth[:n_steps, :n_layers]
        predictions = {
            name: prediction[:n_steps, :n_layers]
            for name, prediction in predictions.items()
        }
        scale = float(np.nanpercentile(np.abs(truth), 99.5))
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0

        panels = []
        for layer in range(n_layers):
            panels.append((f"Truth, layer {layer}", truth[final_step, layer]))
            panels.extend(
                (f"{name}, layer {layer}", prediction[final_step, layer])
                for name, prediction in predictions.items()
            )

        ncols = min(4, len(panels))
        nrows = int(np.ceil(len(panels) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4 * ncols, 3.5 * nrows),
            squeeze=False,
        )
        axes_flat = axes.flat
        images = []
        used_axes = []
        for ax, (label, snapshot) in zip(axes_flat, panels):
            image = ax.imshow(
                snapshot,
                origin="lower",
                cmap="RdBu_r",
                vmin=-scale,
                vmax=scale,
            )
            ax.set_title(label)
            ax.set_xticks([])
            ax.set_yticks([])
            images.append(image)
            used_axes.append(ax)

        for ax in list(axes_flat)[len(panels):]:
            ax.set_visible(False)
        fig.colorbar(images[0], ax=used_axes, shrink=0.8, label="PV")
        fig.suptitle(f"Final PV rollout comparison, step {final_step}")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


class SGSDiagnostic(Diagnostic):
    """Animate closure-head increments against interval-mean inferred SGS targets."""
    name = "sgs"
    output = "gif"

    def run(self, trajs, out_path, cadence):
        prediction = np.asarray(trajs.get("teacher_forced_sgs"), dtype=float)
        target = np.asarray(trajs.get("target_sgs"), dtype=float)
        rollout = trajs.get("sgs")
        rollout = np.asarray(rollout, dtype=float) if rollout is not None else prediction
        if prediction.shape != target.shape:
            raise ValueError(
                "teacher_forced_sgs and target_sgs must have matching shapes; "
                f"got {prediction.shape} and {target.shape}"
            )
        if rollout.shape != target.shape:
            raise ValueError(
                "sgs and target_sgs must have matching shapes; "
                f"got {rollout.shape} and {target.shape}"
            )
        if prediction.ndim != 4:
            raise ValueError(
                "SGS contributions must have shape (time, heads, y, x); "
                f"got {prediction.shape}"
            )
        if prediction.shape[0] == 0:
            raise ValueError("No SGS contribution frames are available")

        n_heads = prediction.shape[1]
        frame_indices = np.arange(0, prediction.shape[0], max(1, int(cadence)))
        if frame_indices[-1] != prediction.shape[0] - 1:
            frame_indices = np.append(frame_indices, prediction.shape[0] - 1)
        fig, axes = plt.subplots(
            n_heads,
            4,
            figsize=(16, max(4, 3.8 * n_heads)),
            squeeze=False,
        )
        head_statistics = []
        all_fields = np.concatenate((target.ravel(), prediction.ravel(), rollout.ravel()))
        field_scale = max(float(np.nanpercentile(np.abs(all_fields), 99)), 1e-12)
        all_differences = prediction - target
        difference_scale = max(
            float(np.nanpercentile(np.abs(all_differences), 99)), 1e-12
        )
        images = []

        for head in range(n_heads):
            predicted_head = prediction[:, head]
            target_head = target[:, head]
            valid = np.isfinite(predicted_head) & np.isfinite(target_head)
            predicted_values = predicted_head[valid]
            target_values = target_head[valid]
            correlation = (
                float(np.corrcoef(predicted_values, target_values)[0, 1])
                if predicted_values.size > 1
                and np.std(predicted_values) > 0
                and np.std(target_values) > 0
                else float("nan")
            )
            rmse = (
                float(np.sqrt(np.mean((predicted_values - target_values) ** 2)))
                if predicted_values.size
                else float("nan")
            )
            head_statistics.append((predicted_values, target_values, correlation, rmse))

            target_image = axes[head, 0].imshow(
                target[0, head], origin="lower", cmap="RdBu_r",
                vmin=-field_scale, vmax=field_scale,
            )
            axes[head, 0].set_title(f"Head {head}: interval-mean SGS target")
            fig.colorbar(target_image, ax=axes[head, 0], fraction=0.046)
            images.append(target_image)

            prediction_image = axes[head, 1].imshow(
                prediction[0, head], origin="lower", cmap="RdBu_r",
                vmin=-field_scale, vmax=field_scale,
            )
            axes[head, 1].set_title(f"Head {head}: closure head addition")
            fig.colorbar(prediction_image, ax=axes[head, 1], fraction=0.046)
            images.append(prediction_image)

            rollout_image = axes[head, 2].imshow(
                rollout[0, head], origin="lower", cmap="RdBu_r",
                vmin=-field_scale, vmax=field_scale,
            )
            axes[head, 2].set_title(f"Head {head}: rollout-applied addition")
            fig.colorbar(rollout_image, ax=axes[head, 2], fraction=0.046)
            images.append(rollout_image)

            error_image = axes[head, 3].imshow(
                all_differences[0, head], origin="lower", cmap="RdBu_r",
                vmin=-difference_scale, vmax=difference_scale,
            )
            axes[head, 3].set_title(
                f"Head {head}: error (r={correlation:.3f}, RMSE={rmse:.2e})"
            )
            fig.colorbar(error_image, ax=axes[head, 3], fraction=0.046)
            images.append(error_image)

        all_prediction = np.concatenate(
            [item[0] for item in head_statistics if item[0].size]
        ) if any(item[0].size for item in head_statistics) else np.array([])
        all_target = np.concatenate(
            [item[1] for item in head_statistics if item[1].size]
        ) if any(item[1].size for item in head_statistics) else np.array([])
        if all_prediction.size:
            overall_corr = (
                float(np.corrcoef(all_prediction, all_target)[0, 1])
                if all_prediction.size > 1
                and np.std(all_prediction) > 0
                and np.std(all_target) > 0
                else float("nan")
            )
        else:
            overall_corr = float("nan")
        print(f"Teacher-forced SGS correlation across heads: {overall_corr:.6f}")
        for head, (_, _, correlation, rmse) in enumerate(head_statistics):
            print(f"  Head {head}: Pearson r={correlation:.6f}")
            print(f"  Head {head}: RMSE={rmse:.6e}")

        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])

        title = fig.suptitle("")

        def update(frame_index):
            frame = frame_indices[frame_index]
            for head in range(n_heads):
                images[head * 4].set_data(target[frame, head])
                images[head * 4 + 1].set_data(prediction[frame, head])
                images[head * 4 + 2].set_data(rollout[frame, head])
                images[head * 4 + 3].set_data(all_differences[frame, head])
            title.set_text(f"Closure SGS additions vs interval target (frame {frame})")
            return (*images, title)

        fig.tight_layout()
        animation = FuncAnimation(
            fig, update, frames=len(frame_indices), interval=180, blit=False
        )
        animation.save(out_path, writer=PillowWriter(fps=5))
        plt.close(fig)

_REGISTRY = {
    "loss": LossDiagnostic,
    "ke_spectrum": KESpectrumDiagnostic,
    "ke_spectrum_movie": KESpectrumAnimationDiagnostic,
    'vorticity': VorticityDiagnostic,
    'pv_spectrum': PVSpectrumDiagnostic,
    'quad': QuadGifDiagnostic,
    'zero': ZeroComparisonDiagnostic,
    'energy': EnergyDiagnostic,
    'zonal': ZonalDiagnostic,
    'cfl': CFLDiagnostic,
    "domain": DomainDiagnostic,
    "sgs_spectrum": SGSSpectralDiagnostic,
    "sgs": SGSDiagnostic,
    "ke_comparison": MultiModelComparisonDiagnostic,
    "multi_loss": MultiModelLossDiagnostic,
    "loss_rollout": MultiModelLossRolloutDiagnostic,
    "pv_rollout": MultiModelPVRolloutDiagnostic,
    "pv_final_snapshot": MultiModelPVFinalSnapshotDiagnostic,
    "pareto_validation": ParetoValidationDiagnostic,
    "pareto_rollout_accuracy": ParetoRolloutAccuracyDiagnostic,
}

def build_diagnostic(name: str) -> Diagnostic:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown diagnostic '{name}'")
    return _REGISTRY[name]()