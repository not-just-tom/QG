"""Diagnostics helpers pulled out of solver for clarity and reuse.

Functions are JAX-friendly where appropriate and accept numpy/jax arrays.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
from jax.numpy.fft import rfftn
from collections import defaultdict
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from model.core.grid import Grid
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import model.core.states as states

def build_diagnostic(name: str):
    mapping = {
        "PV": VorticityDiagnostic,
        "zonal": ZonalMeanVelocity,
        "ke_spectrum": KESpectrumDiagnostic,
        "energy": EnergyDiagnostic,
        "enstrophy": EnstrophyDiagnostic,
    }
    cls = mapping.get(name)
    return cls() if cls is not None else None

class Diagnostic(ABC):
    name: str

    @abstractmethod
    def requires(self) -> set[str]:
        """Fields required from solver (e.g. psi, zeta)."""

    @abstractmethod
    def retrieve(self, state, grid=None):
        """Pull the value from TempStates"""

    # ----------------- Plotting interface ---------------------
    def init_plot(self, ax, sample, grid=None):
        """Create initial matplotlib artists for this diagnostic.

        Returns a dict of artist handles and any auxiliary state needed
        by `update_plot`.
        """
        raise NotImplementedError()
    
    def update_plot(self, artists: dict, sample):
        """Update the given artists with new `sample` data (one frame)."""
        raise NotImplementedError()

class Recorder:
    def __init__(self, cfg, stepped_model):
        self.stepped_model = stepped_model
        self.grid = Grid(cfg.params.Lx, cfg.params.nx)
        diag_cfg = cfg.plotting
        self.cadence = int(getattr(diag_cfg, "cadence", 100))

        animate = set(getattr(diag_cfg, "animate", []))
        final = set(getattr(diag_cfg, "final", []))

        animate.add('PV')
        self.animate_names = animate
        self.final_names = final 

        self.diagnostics = {}
        for name in self.animate_names | self.final_names:
            d = build_diagnostic(name)
            if d is None:
                raise ValueError(f"Unknown diagnostic: {name}")
            self.diagnostics[name] = d

        self.animate_buffers = defaultdict(list)
        self.final_buffers   = {}

    def sample(self, state):
        for name, d in self.diagnostics.items():
            # sanity check
            for field in d.requires():
                self._get_field(state, field)
            value = d.retrieve(state, grid=self.grid)
            value = jax.device_get(value)

            if name in self.animate_names:
                self.animate_buffers[name].append(value)

            if name in self.final_names:
                self.final_buffers[name] = value

    def sample_full_state(self, model, stepper_state):
        """Sample from a stepped model state by expanding to full state on host.

        Args:
            model: model instance implementing `get_full_state(state)`
            stepper_state: the StepperState or object containing `.state` (inner State)
        """
        full_state = model.get_full_state(stepper_state.state)
        for name, d in self.diagnostics.items():
            # sanity check
            for field in d.requires():
                self._get_field(full_state, field)
            value = d.retrieve(full_state, grid=self.grid)
            value = jax.device_get(value)

            if name in self.animate_names:
                self.animate_buffers[name].append(value)

            if name in self.final_names:
                self.final_buffers[name] = value

    def finalize_from_spectral(self, q_traj):
        """Convert a batch of spectral PV snapshots into physical-space frames

        q_traj: spectral array-like with shape (n_frames, ...) 
        """
        if self.grid is None:
            raise ValueError("Recorder.grid must be set to finalize spectral data")

        phys = jnp.fft.irfftn(q_traj, s=(self.grid.ny, self.grid.nx), axes=(-2, -1))
        phys = jax.device_get(phys)
        

        try:
            # q_traj expected shape (n_frames, ...). Compute per-frame max.
            magnitudes = jnp.max(jnp.abs(q_traj).reshape((q_traj.shape[0], -1)), axis=1)
            nonzero_mask = jnp.asarray(magnitudes > 0)
            phys_kept = phys[nonzero_mask]
            phys_kept = jax.device_get(phys_kept)
        except Exception:
            # Fall back to keeping all frames if indexing fails for some reason
            phys_kept = jax.device_get(phys)

        # Overwrite previous animate buffer for PV with the processed frames
        self.animate_buffers["PV"] = list(phys_kept)

    def animate(
        self,
        cfg,
        q_traj,
        outname: str = "../diagnostics.gif",
        plots: list | None = None,
        cadence: int | None = None,
    ):
        """Create a minimal animation: compute diagnostics per selected
        spectral frame, store into `animate_buffers`, then animate by
        delegating plotting to each Diagnostic implementation.
        Also produce final static plots for energy and enstrophy (if present).
        """
        cadence = int(cadence or getattr(cfg.diagnostics, "cadence", self.cadence))

        q_traj = jnp.asarray(q_traj)
        n_frames = q_traj.shape[0]
        indices = np.arange(0, n_frames, cadence)

        # selected spectral frames and FFT axes
        selected = q_traj[indices]
        spec_axes = (-2, -1)

        # Build batched FullState and perform a single batched inversion call.
        nsel = selected.shape[0]

        spec_shape = tuple(self.grid.spectral_state_shape)
        real_shape = tuple(self.grid.real_state_shape)

        ph_zeros = jnp.zeros((nsel,) + spec_shape, dtype=selected.dtype)
        u_zeros = jnp.zeros((nsel,) + real_shape, dtype=selected.dtype)
        v_zeros = jnp.zeros((nsel,) + real_shape, dtype=selected.dtype)
        dqhdt_zeros = jnp.zeros((nsel,) + spec_shape, dtype=selected.dtype)

        inner = states.State(qh=selected, _q_shape=(self.grid.ny, self.grid.nx))

        full = states.FullState(
            state=inner,
            ph=ph_zeros,
            u=u_zeros,
            v=v_zeros,
            dqhdt=dqhdt_zeros,
        )

        inv_fn = jax.jit(self.stepped_model.model._invert)
        full_inv = inv_fn(full)

        ph_spec = full_inv.ph
        psi_phys = jnp.fft.irfftn(ph_spec, s=(self.grid.ny, self.grid.nx), axes=spec_axes)
        q_phys = jnp.fft.irfftn(selected, s=(self.grid.ny, self.grid.nx), axes=spec_axes)

        from types import SimpleNamespace

        # ensure buffers exist for animations and final diagnostics
        for name in (self.animate_names | self.final_names):
            self.animate_buffers.setdefault(name, [])

        # compute diagnostics per selected (cadence) frame
        for i in range(psi_phys.shape[0]):
            psi_f = psi_phys[i]
            q_f = q_phys[i]

            # compute u,v from psi: u = dψ/dy, v = -dψ/dx
            u = jnp.gradient(psi_f, self.grid.dy, axis=-2)
            v = -jnp.gradient(psi_f, self.grid.dx, axis=-1)

            zeta = q_f
            state = SimpleNamespace(q=q_f, psi=psi_f, u=u, v=v, zeta=zeta)

            for name, d in self.diagnostics.items():
                try:
                    val = d.retrieve(state, grid=self.grid)
                except Exception:
                    continue
                val = jax.device_get(val)
                # record into animate buffers when requested for animation
                # or when it's listed as a 'final' diagnostic (we want a
                # time-series for final diagnostics so we can plot them)
                if name in self.animate_names or name in self.final_names:
                    self.animate_buffers[name].append(val)
                # update final buffer with last-seen value
                self.final_buffers[name] = val

        # minimal plotting: stack diagnostics vertically and defer to each Diagnostic
        plots = plots or list(self.animate_names)
        plots = [p for p in plots if p in self.animate_buffers]
        if not plots:
            raise ValueError("No plots available to animate")

        n_plots = len(plots)
        first = self.animate_buffers[plots[0]]
        n_anim_frames = len(first)

        fig, axes = plt.subplots(n_plots, 1, figsize=(6, 3 * n_plots))
        if n_plots == 1:
            axes = [axes]

        artists = []
        for ax, name in zip(axes, plots):
            diag = self.diagnostics[name]
            sample0 = np.asarray(self.animate_buffers[name][0])
            arts = diag.init_plot(ax, sample0, grid=self.grid)
            artists.append((diag, arts))

        def _update(i):
            out = []
            for diag, arts in artists:
                sample = np.asarray(self.animate_buffers[diag.name][i])
                diag.update_plot(arts, sample)
                if "artists" in arts:
                    out.extend(arts["artists"])
            return out

        anim = FuncAnimation(fig, _update, frames=n_anim_frames, blit=False)
        try:
            writer = PillowWriter(fps=10)
            anim.save(outname, writer=writer)
        finally:
            plt.close(fig)

        # animation done; final plots are handled by `plot_final`
        return

    def plot_final(self, outname: str = "diagnostics_final.gif"):
        """Save final time-series plots for selected diagnostics (energy, enstrophy).

        `outname` is used as a base; PNGs are written as `<base>_energy.png` etc.
        """
        # Combine energy and enstrophy on a single final figure if available
        names = [n for n in ("energy", "enstrophy") if n in self.animate_buffers and len(self.animate_buffers[n]) > 0]
        if not names:
            return

        series = {n: np.asarray(self.animate_buffers[n]) for n in names}

        # Align lengths (use minimum length if mismatch)
        lengths = [s.shape[0] for s in series.values()]
        nframe = min(lengths)
        x = np.arange(nframe)

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = {"energy": "C0", "enstrophy": "C1"}
        for name, s in series.items():
            arr = np.asarray(s[:nframe])
            # flatten in case scalar per frame
            arr = arr.reshape(-1)
            ax.plot(x, arr, label=name, color=colors.get(name, None))

        ax.set_xlabel("frame")
        ax.set_title("Energy and Enstrophy (final)")
        ax.legend()

        out_png = outname.replace('.gif', '_energy_enstrophy.png')
        fig.savefig(out_png, bbox_inches='tight')
        plt.close(fig)

    def _get_field(self, full_state, field):
        # first try TempStates
        if hasattr(full_state, field):
            return getattr(full_state, field)

        # then try inner State
        if hasattr(full_state.state, field):
            return getattr(full_state.state, field)

        raise AttributeError(f"Field '{field}' not found in TempStates or State")


class VorticityDiagnostic(Diagnostic):
    name = "PV"

    def requires(self):
        return {"q"}  # vorticity

    def retrieve(self, state, grid=None):
        return state.q

    def init_plot(self, ax, sample, grid=None):
        # sample: (nz, ny, nx) or (ny, nx)
        if sample.ndim == 3:
            frame = sample[0]
        else:
            frame = sample
        im = ax.imshow(np.asarray(frame), origin="lower", extent=(0, getattr(grid, "Lx", 1), 0, getattr(grid, "Ly", 1)), cmap="RdBu_r")
        ax.set_title("Potential Vorticity")
        cb = ax.figure.colorbar(im, ax=ax)
        return {"im": im, "cbar": cb, "artists": [im]}

    def update_plot(self, artists: dict, sample):
        if sample.ndim == 3:
            frame = sample[0]
        else:
            frame = sample
        artists["im"].set_data(np.asarray(frame))


class ZonalMeanVelocity(Diagnostic):
    name = "zonal"

    def requires(self):
        return {"u"}

    def retrieve(self, state, grid=None):
        # state.u expected shape (nz, ny, nx)
        u = state.u
        # mean over zonal direction
        um = jnp.mean(u, axis=1)
        return um

    def init_plot(self, ax, sample, grid=None):
        # sample: (nz, ny) or (ny,)
        nx = sample.shape[-1]
        grid_y = np.linspace(0, getattr(grid, "Ly", 1), nx)
        if sample.ndim == 2:
            line, = ax.plot(grid_y, np.asarray(np.real(sample[0])), color="k")
        else:
            line, = ax.plot(grid_y, np.asarray(np.real(sample)), color="k")
        ax.set_title("Zonal Mean Velocity")
        ax.set_xlabel("y")
        ax.set_ylabel("u")
        return {"line": line, "artists": [line], "grid_y": grid_y}

    def update_plot(self, artists: dict, sample):
        # use grid_y saved during init_plot to keep axes consistent
        if "grid_y" in artists:
            grid_y = artists["grid_y"]
        else:
            # fallback: infer from sample length
            if sample.ndim == 2:
                nx = np.asarray(sample[0]).shape[0]
            else:
                nx = np.asarray(sample).shape[0]
            grid_y = np.linspace(0, 1, nx)

        if sample.ndim == 2:
            ydata = np.asarray(np.real(sample[0]))
        else:
            ydata = np.asarray(np.real(sample))

        artists["line"].set_data(grid_y, ydata)

# the rest need working on vvv

class KESpectrumDiagnostic(Diagnostic):
    name = "ke_spectrum"

    def requires(self):
        return {"psi"}

    def compute(self, fields, grid):
        psi = fields["psi"]
        psih = rfftn(psi, axes=(-2, -1), norm="ortho")

        uh = -1j * grid.KY * psih
        vh =  1j * grid.KX * psih

        KE2D = 0.5 * (jnp.abs(uh)**2 + jnp.abs(vh)**2)

        return KE2D
    
    def reduce(self, KE2D, grid):
        kmag = np.asarray(grid.Kmag).astype(int)
        KE = np.asarray(KE2D)

        kmax = kmag.max()
        E_k = np.bincount(kmag.ravel(), weights=KE.ravel(), minlength=kmax+1)
        return E_k

    def retrieve(self, state, grid=None):
        # Expect `state` to expose `psi` in physical space
        fields = {"psi": state.psi}
        KE2D = self.compute(fields, grid)
        E_k = self.reduce(KE2D, grid)
        return jnp.asarray(E_k)

    def init_plot(self, ax, sample, grid=None):
        spec = np.asarray(sample)
        k = np.arange(spec.shape[0])
        line, = ax.loglog(k + 1e-8, spec)
        ax.set_title("Kinetic Energy Spectrum")
        ax.set_xlabel("k")
        ax.set_ylabel("E(k)")
        return {"line": line, "artists": [line]}

    def update_plot(self, artists: dict, sample):
        spec = np.asarray(sample)
        k = np.arange(spec.shape[0])
        artists["line"].set_data(k, spec)


class EnergyDiagnostic(Diagnostic):
    name = "energy"

    def requires(self):
        return {"psi"}

    def compute(self, fields, grid):
        psi = fields["psi"]
        return compute_ke(psi, grid)

    def reduce(self, value, grid=None):
        return value

    def retrieve(self, state, grid=None):
        fields = {"psi": state.psi}
        val = self.compute(fields, grid)
        return jnp.asarray(val)

    def init_plot(self, ax, sample, grid=None):
        arr = np.asarray(sample)
        if arr.ndim == 0:
            arr = np.array([arr])
        x = np.arange(arr.shape[0])
        line, = ax.plot(x, arr)
        ax.set_title("Energy")
        ax.set_xlabel("frame")
        ax.set_ylabel("energy")
        return {"line": line, "artists": [line]}

    def update_plot(self, artists: dict, sample):
        line = artists["line"]
        xd = np.asarray(line.get_xdata())
        yd = np.asarray(line.get_ydata())
        xd = np.append(xd, xd.size)
        yd = np.append(yd, np.asarray(sample))
        line.set_data(xd, yd)


class EnstrophyDiagnostic(Diagnostic):
    name = "enstrophy"

    def requires(self):
        return {"zeta"}

    def compute(self, fields, grid):
        zeta = fields["zeta"]
        return compute_enstrophy(zeta, grid)

    def reduce(self, value, grid=None):
        return value

    def retrieve(self, state, grid=None):
        fields = {"zeta": state.zeta}
        val = self.compute(fields, grid)
        return jnp.asarray(val)

    def init_plot(self, ax, sample, grid=None):
        arr = np.asarray(sample)
        if arr.ndim == 0:
            arr = np.array([arr])
        x = np.arange(arr.shape[0])
        line, = ax.plot(x, arr)
        ax.set_title("Enstrophy")
        ax.set_xlabel("frame")
        ax.set_ylabel("enstrophy")
        return {"line": line, "artists": [line]}

    def update_plot(self, artists: dict, sample):
        line = artists["line"]
        xd = np.asarray(line.get_xdata())
        yd = np.asarray(line.get_ydata())
        xd = np.append(xd, xd.size)
        yd = np.append(yd, np.asarray(sample))
        line.set_data(xd, yd)


def compute_ke(psi: jnp.ndarray, grid) -> float:
    """Compute total kinetic energy: 0.5 * mean(u^2 + v^2) * dx * dy

    Arguments:
        psi: streamfunction in physical space
        grid: `Grid` instance providing dx, dy
    Returns:
        KE as a float
    """
    # u = dψ/dy, v = -dψ/dx (finite-diff approx consistent with other code)
    u = jnp.gradient(psi, grid.dy, axis=-2)
    v = -jnp.gradient(psi, grid.dx, axis=-1)
    ke = 0.5 * jnp.mean(u ** 2 + v ** 2) * grid.dx * grid.dy
    return float(ke)


def compute_enstrophy(zeta: jnp.ndarray, grid) -> float:
    enst = 0.5 * jnp.mean(zeta ** 2) * grid.dx * grid.dy
    return float(enst)


def forcing_stats(forcing_h, grid) -> dict:
    """Return quick statistics for forcing (spectral and physical)."""
    f_spec_max = float(jnp.max(jnp.abs(forcing_h)))
    f_spec_mean = float(jnp.mean(jnp.abs(forcing_h)))
    # physical
    f_phys = jnp.fft.irfftn(forcing_h, s=(grid.ny, grid.nx))
    f_phys_max = float(jnp.max(jnp.abs(f_phys)))
    f_phys_mean = float(jnp.mean(jnp.abs(f_phys)))
    return {
        "f_spec_max": f_spec_max,
        "f_spec_mean": f_spec_mean,
        "f_phys_max": f_phys_max,
        "f_phys_mean": f_phys_mean,
    }


def check_nans(arr) -> int:
    """Return number of NaNs in array (int)."""
    return int(jnp.isnan(arr).sum())
