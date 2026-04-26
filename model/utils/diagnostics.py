from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
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

    def run(self, trajs: dict, out_path: str):
        raise NotImplementedError
    

class LossDiagnostic(Diagnostic):
    name = "loss"

    def run(self, trajs, out_path):
        losses = trajs.get("loss_history", {})

        train = np.asarray(losses.get("train", []))
        test  = np.asarray(losses.get("test", []))

        fig, ax = plt.subplots()

        if train.size:
            ax.plot(np.arange(1, len(train) + 1), train, label="train")
        if test.size:
            ax.plot(np.arange(1, len(test) + 1), test, label="test")

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

    def run(self, trajs, out_path):
        from matplotlib.animation import FuncAnimation, PillowWriter

        # Read inputs: truth and optional prediction
        if "truth" in trajs and trajs["truth"] is not None:
            q_truth = np.asarray(trajs["truth"])
        elif "q" in trajs and trajs["q"] is not None:
            q_truth = np.asarray(trajs["q"])
        else:
            raise KeyError("ke_spectrum_movie requires 'truth' or 'q' in trajectories")
        q_pred = trajs.get("pred")
        q_pred = None if q_pred is None else np.asarray(q_pred)
        grid = trajs.get("grid")
        if grid is None:
            raise KeyError("ke_spectrum_movie requires 'grid' in trajectories")

        # cadence for subsampling frames
        cadence = 10

        # Ensure we have time axis and layer axis (nt, nz, ny, nx)
        if q_truth.ndim == 3:
            q_truth = q_truth[:, None, ...]
        if q_pred is not None and q_pred.ndim == 3:
            q_pred = q_pred[:, None, ...]

        nt = q_truth.shape[0]
        frames = list(range(0, nt, cadence))
        if len(frames) == 0:
            raise ValueError("No frames selected for KE spectrum movie (check cadence)")

        # Precompute first selected frame to build axes
        psi0 = invert_pv_to_psi(q_truth[frames[0]], grid)
        u0, v0 = velocity_from_psi(psi0, grid)
        spec0 = isotropic_ke_spectrum(u0, v0, grid)
        k = np.asarray(spec0["k"]).ravel()

        # initial spectra
        E_truth0 = spec0["E"]
        if E_truth0.ndim > 1:
            E_truth0 = E_truth0.mean(axis=0)
        E_truth0 = np.asarray(E_truth0).ravel()

        if q_pred is not None:
            psi0p = invert_pv_to_psi(q_pred[frames[0]], grid)
            up0, vp0 = velocity_from_psi(psi0p, grid)
            spec0p = isotropic_ke_spectrum(up0, vp0, grid)
            E_pred0 = spec0p["E"]
            if E_pred0.ndim > 1:
                E_pred0 = E_pred0.mean(axis=0)
            E_pred0 = np.asarray(E_pred0).ravel()
        else:
            E_pred0 = None

        fig, ax = plt.subplots()
        ln_truth, = ax.loglog(k[1:], E_truth0[1:], label="Truth", color="k")
        if E_pred0 is not None:
            ln_pred, = ax.loglog(k[1:], E_pred0[1:], label="ML", linestyle="--", color="C1")
        else:
            ln_pred = None

        ax.set_xlabel("k")
        ax.set_ylabel("E(k)")
        ax.set_title(f"KE spectrum (t={frames[0]})")
        ax.grid(True)
        if ln_pred is not None:
            ax.legend()

        def update(frame_idx):
            frame = frames[frame_idx]
            psi = invert_pv_to_psi(q_truth[frame], grid)
            u, v = velocity_from_psi(psi, grid)
            spec = isotropic_ke_spectrum(u, v, grid)
            E = spec["E"]
            if E.ndim > 1:
                E = E.mean(axis=0)
            E = np.asarray(E).ravel()
            ln_truth.set_data(k[1:], E[1:])

            if q_pred is not None and ln_pred is not None:
                psi_p = invert_pv_to_psi(q_pred[frame], grid)
                up, vp = velocity_from_psi(psi_p, grid)
                spec_p = isotropic_ke_spectrum(up, vp, grid)
                Ep = spec_p["E"]
                if Ep.ndim > 1:
                    Ep = Ep.mean(axis=0)
                Ep = np.asarray(Ep).ravel()
                ln_pred.set_data(k[1:], Ep[1:])

            ax.relim()
            ax.autoscale_view()
            ax.set_title(f"KE spectrum (t={frame})")
            if ln_pred is not None:
                return (ln_truth, ln_pred)
            return (ln_truth,)

        ani = FuncAnimation(fig, update, frames=len(frames), interval=200)
        ani.save(out_path, writer=PillowWriter(fps=5))
        plt.close(fig)



# ============================================================
# MSE
# ============================================================

class MSEDiagnostic(Diagnostic):
    name = "mse"

    def run(self, trajs, out_path):
        # Prefer full-resolution data if available
        pred = trajs.get("pred_full", trajs.get("pred"))
        truth = trajs.get("truth_full", trajs.get("truth"))

        pred = np.asarray(pred)
        truth = np.asarray(truth)

        # ensure (nt, nz, ny, nx)
        if pred.ndim == 3:
            pred = pred[:, None, ...]
            truth = truth[:, None, ...]

        mse = np.mean((pred - truth) ** 2, axis=(-2, -1))  # (nt, nz)
        mse = np.mean(mse, axis=1)                         # (nt,)

        fig, ax = plt.subplots()
        ax.plot(mse, "-o", markersize=3)
        ax.set_title("MSE per timestep")
        ax.set_xlabel("t")
        ax.set_ylabel("MSE")
        ax.grid(True)

        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ============================================================
# KE Spectrum (time-averaged)
# ============================================================

class KESpectrumDiagnostic(Diagnostic):
    name = "ke_spectrum"

    def run(self, trajs, out_path):
        grid = trajs["grid"]

        # --- get PV fields ---
        # Prefer full-resolution inputs when available for spectral diagnostics
        q_truth = trajs.get("truth_full", trajs.get("truth"))
        q_pred  = trajs.get("pred_full", trajs.get("pred"))
        q_truth = np.asarray(q_truth)
        q_pred  = None if q_pred is None else np.asarray(q_pred)

        # --- helper: compute spectrum from PV ---
        def compute_spectrum(q):
            psi = invert_pv_to_psi(q, grid)
            u, v = velocity_from_psi(psi, grid)
            spec = isotropic_ke_spectrum(u, v, grid)
            return spec["k"], spec["E"].mean(axis=0).squeeze()  # time-average

        k, E_truth = compute_spectrum(q_truth)

        if q_pred is not None:
            _, E_pred = compute_spectrum(q_pred)
        else:
            E_pred = None

        # --- plot ---
        fig, ax = plt.subplots()

        ax.loglog(k[1:], E_truth[1:], label="Truth", color="k")

        if E_pred is not None:
            ax.loglog(k[1:], E_pred[1:], label="ML", linestyle="--")

        ax.set_xlabel("k")
        ax.set_ylabel("E(k)")
        ax.set_title("Time-averaged KE spectrum")
        ax.grid(True, which="both")

        if E_pred is not None:
            ax.legend()

        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ============================================================
# PV Animation
# ============================================================

class VorticityDiagnostic(Diagnostic):
    name = "PV"
    output = "gif"

    def run(self, trajs, out_path):
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

    def run(self, trajs, out_path):
        pred = trajs.get("pred")
        truth = trajs.get("truth")
        sgs_pred = trajs.get("sgs")
        pred_np = np.asarray(pred)
        truth_np = np.asarray(truth)
        err = pred_np - truth_np

        indices = np.arange(0, pred_np.shape[0], 10)
        # determine color limits per panel
        vmin_truth = np.percentile(truth_np, 1)
        vmax_truth = np.percentile(truth_np, 99)
        vmin_err = np.percentile(err, 1)
        vmax_err = np.percentile(err, 99)
        vmin_sgs = np.percentile(sgs_pred, 1)
        vmax_sgs = np.percentile(sgs_pred, 99)

        fig, axes = plt.subplots(2,2,figsize=(8,8))
        ax_truth = axes[0,0]
        ax_ml = axes[0,1]
        ax_err = axes[1,0]
        ax_sgs = axes[1,1]

        im_truth = ax_truth.imshow(truth_np[0,0], origin='lower', cmap='RdBu_r', vmin=vmin_truth, vmax=vmax_truth)
        ax_truth.set_title('Truth')
        im_ml = ax_ml.imshow(pred_np[0,0], origin='lower', cmap='RdBu_r', vmin=vmin_truth, vmax=vmax_truth)
        ax_ml.set_title('ML adjusted')
        im_err = ax_err.imshow(err[0,0], origin='lower', cmap='RdBu_r', vmin=vmin_err, vmax=vmax_err)
        ax_err.set_title('Error')
        im_sgs = ax_sgs.imshow(sgs_pred[0,0], origin='lower', cmap='RdBu_r', vmin=vmin_sgs, vmax=vmax_sgs)
        ax_sgs.set_title('SGS')

        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])

        fig.colorbar(im_truth, ax=[ax_truth, ax_ml], shrink=0.6)
        fig.colorbar(im_err, ax=ax_err, shrink=0.6)
        fig.colorbar(im_sgs, ax=ax_sgs, shrink=0.6)

        def update(i):
            idx = indices[i]
            im_truth.set_data(truth_np[idx,0])
            im_ml.set_data(pred_np[idx,0])
            im_err.set_data(err[idx,0])
            if idx < sgs_pred.shape[0]:
                im_sgs.set_data(sgs_pred[idx,0])
            fig.suptitle(f'timestep {idx}')
            return im_truth, im_ml, im_err, im_sgs

        anim = FuncAnimation(fig, update, frames=len(indices), interval=100, blit=False)

        try:
            from matplotlib.animation import PillowWriter
            writer = PillowWriter(fps=10)
            anim.save(out_path, writer=writer)
            print('Saved gif to', out_path)
        except Exception as e:
            print('Pillow save failed:', e)


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

_REGISTRY = {
    "loss": LossDiagnostic,
    "mse": MSEDiagnostic,
    "ke_spectrum": KESpectrumDiagnostic,
    "PV": VorticityDiagnostic,
    "quad": QuadGifDiagnostic,
    "energy": EnergyDiagnostic,
    "ke_spectrum_movie": KESpectrumAnimationDiagnostic,
}

def build_diagnostic(name: str) -> Diagnostic:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown diagnostic '{name}'")
    return _REGISTRY[name]()