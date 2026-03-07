"""Diagnostics helpers pulled out of solver for clarity and reuse.

Functions are JAX-friendly where appropriate and accept numpy/jax arrays.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
from jax.numpy.fft import rfftn
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from model.core.grid import Grid
from matplotlib.animation import FuncAnimation, PillowWriter
import model.core.states as states
import os
from model.utils.config import Config
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
CONFIG_DEFAULT_PATH = os.path.join(BASE_DIR, "config", "default.yaml")
cfg = Config.load_config(CONFIG_DEFAULT_PATH)


class ReconstructedState:
    """
    Lightweight offline stand-in for a live model state.

    Attributes mirror what diagnostics expect: `qh`, `q`, `psi`, `u`, `v`, `zeta`.
    """

    def __init__(self, qh, q, psi, u, v, zeta=None):
        self.qh = qh
        self.q = q
        self.psi = psi
        self.u = u
        self.v = v
        self.zeta = q if zeta is None else zeta

def build_diagnostic(name: str, *, nz: int):
    """
    Factory for diagnostics.

    Args:
        name: diagnostic name (string from config)
        nz: number of vertical layers

    Returns:
        Diagnostic instance

    Raises:
        ValueError if diagnostic name is unknown
    """
    registry = {
        "PV": VorticityDiagnostic,
        "zonal": ZonalMeanVelocityDiagnostic,
        "ke_spectrum": KESpectrumDiagnostic,
        "energy": EnergyDiagnostic,
        "enstrophy": EnstrophyDiagnostic,
        "cfl": CFLDiagnostic,
        'drift': DriftDiagnostic,
    }

    cls = registry.get(name)
    if cls is None:
        raise ValueError(f"Unknown diagnostic '{name}'")

    return cls(nz=nz)

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
    """
    Recorder orchestrates
      - buffering diagnostic outputs
      - animation layout
      - final plot layout
    """

    def __init__(self, cfg, model):
        self.model = model
        self.grid = Grid(cfg.params.Lx, cfg.params.nx)
        self.nz = int(cfg.params.nz)

        diag_cfg = cfg.plotting
        self.cadence = int(getattr(diag_cfg, "cadence", 1))
        self.animate_names = set(diag_cfg.animate)
        self.final_names = set(diag_cfg.final)

        # instantiate diagnostics
        self.diagnostics = {}
        for name in set(self.animate_names) | set(self.final_names):
            diag = build_diagnostic(name, nz=self.nz)
            if diag is None:
                raise ValueError(f"Unknown diagnostic '{name}'")
            self.diagnostics[name] = diag

        # buffers[name] = list of snapshot values
        self.buffers = {name: [] for name in set(self.animate_names) | set(self.final_names)}

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------

    def sample(self, state):
        """
        Sample diagnostics from a FULL physical-space state.
        """
        for name, diag in self.diagnostics.items():
            for field in diag.requires():
                self._get_field(state, field)

            val = diag.retrieve(state, grid=self.grid)
            self.buffers[name].append(jax.device_get(val))

    # ------------------------------------------------------------------
    # reconstruction (offline)
    # ------------------------------------------------------------------

    def reconstruct_states(self, q_traj):
        """
        Batch-reconstruct a list/array of spectral PV frames into
        a list of `ReconstructedState` objects using the model's
        inversion machinery. This avoids redoing the expensive
        invert step for each frame when a batch is available.
        """
        selected = jnp.asarray(q_traj)
        # assume selected shape (nt, nz, ny, nk)
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
        #print('diagnostic: ', full.qh.shape)

        inv_fn = jax.jit(self.model.model._invert)
        full_inv = inv_fn(full)

        ph_spec = full_inv.ph
        psi_phys = jnp.fft.irfftn(ph_spec, s=(self.grid.ny, self.grid.nx), axes=(-2, -1))
        q_phys = jnp.fft.irfftn(selected, s=(self.grid.ny, self.grid.nx), axes=(-2, -1))

        states_out = []
        for i in range(psi_phys.shape[0]):
            psi_f = psi_phys[i]
            q_f = q_phys[i]

            u = jnp.gradient(psi_f, self.grid.dy, axis=-2)
            v = -jnp.gradient(psi_f, self.grid.dx, axis=-1)

            zeta = q_f
            states_out.append(ReconstructedState(qh=selected[i], q=q_f, psi=psi_f, u=u, v=v, zeta=zeta))

        return states_out

    def reconstruct_state(self, qh):
        """
        Reconstruct a single spectral frame into a `ReconstructedState`.
        Uses the same machinery as `reconstruct_states` but for one frame.
        """
        qh = jnp.asarray(qh)
        spec_shape = tuple(self.grid.spectral_state_shape)
        real_shape = tuple(self.grid.real_state_shape)

        ph_zeros = jnp.zeros(spec_shape, dtype=qh.dtype)
        u_zeros = jnp.zeros(real_shape, dtype=qh.dtype)
        v_zeros = jnp.zeros(real_shape, dtype=qh.dtype)
        dqhdt_zeros = jnp.zeros(spec_shape, dtype=qh.dtype)

        inner = states.State(qh=qh, _q_shape=(self.grid.ny, self.grid.nx))

        full = states.FullState(
            state=inner,
            ph=ph_zeros,
            u=u_zeros,
            v=v_zeros,
            dqhdt=dqhdt_zeros,
        )

        inv_fn = jax.jit(self.model.model._invert)
        full_inv = inv_fn(full)

        ph_spec = full_inv.ph
        psi_phys = jnp.fft.irfftn(ph_spec, s=(self.grid.ny, self.grid.nx), axes=(-2, -1))
        q_phys = jnp.fft.irfftn(qh, s=(self.grid.ny, self.grid.nx), axes=(-2, -1))

        u = jnp.gradient(psi_phys, self.grid.dy, axis=-2)
        v = -jnp.gradient(psi_phys, self.grid.dx, axis=-1)
        zeta = q_phys

        return ReconstructedState(qh=qh, q=q_phys, psi=psi_phys, u=u, v=v, zeta=zeta)

    # ------------------------------------------------------------------
    # animation
    # ------------------------------------------------------------------

    def animate(self, cfg, outbase, q_traj=None, outname: str = "animations.gif", fps: int = 10):
        """
        Animate diagnostics listed in cfg.diagnostics.animate.
        """
        # If a spectral trajectory is provided, reconstruct pseudo-states
        # (batched) and populate diagnostic buffers from those reconstructed states.
        if q_traj is not None:
            # assume q_traj already contains only the cadence frames (run.py slices)
            recon_states = self.reconstruct_states(q_traj)

            # ensure buffers exist
            for name in (self.animate_names | self.final_names):
                self.buffers.setdefault(name, [])

            # compute diagnostics per reconstructed (cadence) frame
            for state in recon_states:
                for name, d in self.diagnostics.items():
                    try:
                        val = d.retrieve(state, grid=self.grid)
                    except Exception:
                        continue
                    val = jax.device_get(val)
                    self.buffers.setdefault(name, []).append(val)

                # record CFL if requested (use cfg.plotting.dt if present)
                try:
                    dt = float(getattr(cfg.plotting, "dt", None))
                except Exception:
                    dt = None

                if dt is not None and "cfl" in self.diagnostics:
                    umax = float(jnp.max(jnp.abs(state.u)))
                    vmax = float(jnp.max(jnp.abs(state.v)))
                    cfl_val = dt * (umax / getattr(self.grid, "dx", 1.0) + vmax / getattr(self.grid, "dy", 1.0))
                    self.buffers.setdefault("cfl", []).append(cfl_val)

        plots = [
            name for name in self.animate_names
            if name in self.diagnostics
            and getattr(self.diagnostics[name], "visualize", "frame") == "frame"
        ]

        if not plots:
            return

        # determine layout
        axes_per_diag = {
            name: self.diagnostics[name].n_axes()
            for name in plots
        }

        total_axes = sum(axes_per_diag.values())

        fig, axes = plt.subplots(
            total_axes, 1,
            figsize=(6, 3 * total_axes),
            squeeze=False,
        )

        axes = axes[:, 0]  # flatten
        artists = []
        ax_idx = 0

        # initialize plots
        # ensure buffers were populated
        empty = [name for name in plots if len(self.buffers.get(name, [])) == 0]
        if empty:
            raise ValueError(
                f"No buffered samples for diagnostics: {empty}. "
                "Pass a cadence-sliced `q_traj` to `animate` or call `sample()` before animating."
            )

        for name in plots:
            diag = self.diagnostics[name]
            sample0 = self.buffers[name][0]

            for layer in range(diag.n_axes()):
                ax = axes[ax_idx]
                arts = diag.init_plot(
                    ax, sample0, grid=self.grid, layer=layer
                )
                artists.append((name, diag, arts))
                ax_idx += 1

        n_frames = min(len(self.buffers[name]) for name in plots)

        def _update(i):
            out = []
            for name, diag, arts in artists:
                sample = self.buffers[name][i]
                diag.update_plot(arts, sample)
                out.extend(arts.get("artists", []))
            return out

        anim = FuncAnimation(fig, _update, frames=n_frames, blit=False)

        try:
            anim.save(f'{outbase}/{outname}', writer=PillowWriter(fps=fps))
        finally:
            plt.close(fig)

    # ------------------------------------------------------------------
    # final plots
    # ------------------------------------------------------------------

    def plot_final(self, outbase: str):
        """
        Produce final plots for diagnostics listed in cfg.diagnostics.final.
        """
        for name in self.final_names:
            diag = self.diagnostics.get(name)
            if diag is None:
                continue

            samples = self.buffers[name]
            if not samples:
                continue

            # convert buffered samples to host (NumPy) once to avoid repeated device->host sync
            try:
                samples_host = [jax.device_get(s) for s in samples]
            except Exception:
                samples_host = [np.asarray(s) for s in samples]

            reduced = diag.reduce(samples_host, grid=self.grid)

            n_axes = diag.n_axes()
            fig, axes = plt.subplots(
                n_axes, 1,
                figsize=(6, 3 * n_axes),
                squeeze=False,
            )
            axes = axes[:, 0]

            for layer in range(n_axes):
                diag.plot_final(
                    axes[layer],
                    reduced,
                    grid=self.grid,
                    layer=layer,
                )

            fig.tight_layout()
            fig.savefig(f"{outbase}/{name}.png", bbox_inches="tight")
            plt.close(fig)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _get_field(self, state, field):
        if hasattr(state, field):
            return getattr(state, field)
        if hasattr(state.state, field):
            return getattr(state.state, field)
        raise AttributeError(f"Field '{field}' not found in state")



class VorticityDiagnostic(Diagnostic):
    """
    Potential vorticity / vorticity diagnostic.

    Produces a (nz, ny, nx) field in physical space.
    Handles both single-layer and multi-layer cases uniformly.
    """

    name = "PV"

    # semantic metadata (used by Recorder)
    kind = "field"                 # "field" | "scalar" | "spectrum"
    temporal = "instant"           # "instant" | "timeseries"
    cmap = "RdBu_r"

    def __init__(self, nz: int):
        self.nz = int(nz)

    # ------------------------------------------------------------------
    # data interface
    # ------------------------------------------------------------------

    def requires(self) -> set[str]:
        return {"q"}   # physical-space vorticity

    def retrieve(self, state, grid=None):
        """
        Returns:
            q with shape (nz, ny, nx)
        """
        q = state.q

        # enforce (nz, ny, nx) shape
        if q.ndim == 2:
            q = q[None, ...]   # (1, ny, nx)

        if q.shape[0] != self.nz:
            raise ValueError(
                f"{self.name}: expected nz={self.nz}, got {q.shape[0]}"
            )

        return q

    # ------------------------------------------------------------------
    # plotting interface
    # ------------------------------------------------------------------

    def n_axes(self) -> int:
        """
        Number of matplotlib axes required for animation.

        Recorder uses this to allocate subplots.
        """
        return self.nz

    def init_plot(self, ax, sample, grid=None, layer: int = 0):
        """
        Initialize artists for ONE layer on ONE axis.

        Args:
            ax: matplotlib axis
            sample: (nz, ny, nx) array
            layer: which layer this axis represents
        """
        field = sample[layer]

        im = ax.imshow(
            field,
            origin="lower",
            extent=(0, grid.Lx, 0, grid.Ly),
            cmap=self.cmap,
        )

        ax.set_title(f"PV (layer {layer})" if self.nz > 1 else "PV")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        cbar = ax.figure.colorbar(im, ax=ax)

        return {
            "im": im,
            "cbar": cbar,
            "layer": layer,
            "artists": [im],
        }

    def update_plot(self, artists: dict, sample):
        """
        Update artists for ONE frame.

        Args:
            artists: dict returned by init_plot
            sample: (nz, ny, nx)
        """
        layer = artists["layer"]
        artists["im"].set_data(sample[layer])


class ZonalMeanVelocityDiagnostic(Diagnostic):
    name = "zonal"
    def __init__(self, nz: int):
        self.nz = int(nz)

    def requires(self):
        return {"u"}

    def retrieve(self, state, grid=None):
        # state.u expected shape (nz, ny, nx)
        u = state.u
        # mean over zonal direction
        um = jnp.mean(u, axis=1)

        if um.ndim == 1:
            um = um[None, ...]

        if um.shape[0] != self.nz:
            raise ValueError(f"{self.name}: expected nz={self.nz}, got {um.shape[0]}")

        return um

    def n_axes(self) -> int:
        return self.nz

    def init_plot(self, ax, sample, grid=None, layer: int = 0):
        # sample: (nz, ny)
        ny = sample.shape[-1]
        grid_y = np.linspace(0, getattr(grid, "Ly", 1), ny)
        y0 = np.asarray(np.real(sample[layer]))
        (line,) = ax.plot(grid_y, y0, color="k")
        ax.set_title(f"Zonal Mean Velocity (layer {layer})" if self.nz > 1 else "Zonal Mean Velocity")
        ax.set_xlabel("y")
        ax.set_ylabel("u")
        return {"line": line, "artists": [line], "grid_y": grid_y, "layer": layer}

    def update_plot(self, artists: dict, sample):
        layer = artists.get("layer", 0)
        grid_y = artists.get("grid_y")
        if grid_y is None:
            ny = sample.shape[-1]
            grid_y = np.linspace(0, 1, ny)

        ydata = np.asarray(np.real(sample[layer]))
        artists["line"].set_data(grid_y, ydata)

    def reduce(self, samples, grid=None):
        # samples: list of (nz, ny)
        arr = jnp.stack(samples, axis=0)  # (nt, nz, ny)
        mean = jnp.mean(arr, axis=0)     # (nz, ny)
        return mean

    def plot_final(self, ax, reduced, grid=None, layer: int = 0):
        # reduced: (nz, ny)
        ny = reduced.shape[-1]
        grid_y = np.linspace(0, getattr(grid, "Ly", 1), ny)
        ax.plot(grid_y, np.asarray(reduced[layer]), color="k")
        ax.set_title(f"Zonal Mean Velocity (mean, layer {layer})" if self.nz > 1 else "Zonal Mean Velocity (mean)")
        ax.set_xlabel("y")
        ax.set_ylabel("u")


class CFLDiagnostic(Diagnostic):
    name = "cfl"
    kind = "scalar"
    temporal = "timeseries"
    title = "CFL estimate"
    ylabel = "CFL"

    def requires(self):
        return {"u", "v"}

    def retrieve(self, state, grid=None):
        # compute per-layer CFL if u,v are layered
        u = state.u
        v = state.v

        # grid may carry dt; fallback to grid.dt if present else 1.0
        dt = getattr(grid, "dt", None)
        if dt is None:
            dt = 1.0

        umax = jnp.max(jnp.abs(u), axis=(-2, -1))
        vmax = jnp.max(jnp.abs(v), axis=(-2, -1))

        cfl = dt * (umax / getattr(grid, "dx", 1.0) + vmax / getattr(grid, "dy", 1.0))

        # ensure shape (nz,) for multi-layer
        if cfl.ndim == 0:
            cfl = cfl[None]

        return cfl

    def __init__(self, nz: int):
        self.nz = int(nz)

    def n_axes(self) -> int:
        return self.nz

    def init_plot(self, ax, sample, grid=None, layer: int = 0):
        # sample: (nz,) or scalar
        y0 = np.asarray(sample[layer])
        (line,) = ax.plot([0], [y0], color="C3")
        ax.set_title(self.title)
        ax.set_xlabel("frame")
        ax.set_ylabel(self.ylabel)
        return {"line": line, "artists": [line], "layer": layer}

    def update_plot(self, artists: dict, sample):
        layer = artists.get("layer", 0)
        y = float(np.asarray(sample[layer]))
        line = artists["line"]
        xd = np.asarray(line.get_xdata())
        yd = np.asarray(line.get_ydata())
        xd = np.append(xd, xd.size)
        yd = np.append(yd, y)
        line.set_data(xd, yd)

    def reduce(self, samples, grid=None):
        # samples: list of (nz,) -> (nt, nz) -> return (nz, nt)
        arr = jnp.stack(samples, axis=0)
        return arr.T

    def plot_final(self, ax, reduced, grid=None, layer: int = 0):
        data = np.asarray(reduced[layer])
        x = np.arange(data.shape[0])
        ax.plot(x, data, color="C3")
        ax.set_title(self.title)
        ax.set_xlabel("frame")
        ax.set_ylabel(self.ylabel)

class KESpectrumDiagnostic(Diagnostic):
    """
    Kinetic Energy spectrum diagnostic.

    Produces an isotropic KE spectrum E(k) for each vertical layer.

    - animate: instantaneous spectrum per frame
    - final: time-averaged spectrum
    """

    name = "ke_spectrum"
    kind = "spectrum"
    temporal = "instant"
    xlabel = "k"
    ylabel = "E(k)"

    def __init__(self, nz: int):
        self.nz = int(nz)
        self.beta = getattr(cfg.params, "beta", 10.0)
        self.Ly = getattr(cfg.params, "Ly", cfg.params.Lx)
        self.njets = getattr(cfg.plotting, "n_jets", 6)
    # ------------------------------------------------------------------
    # data interface
    # ------------------------------------------------------------------

    def requires(self) -> set[str]:
        return {"u", "v"}

    def retrieve(self, state, grid=None):
        """
        Returns:
            E : array of shape (nz, nk)
            k : array of shape (nk,)
        """
        u = state.u
        v = state.v

        # enforce (nz, ny, nx)
        if u.ndim == 2:
            u = u[None, ...]
            v = v[None, ...]

        nz, ny, nx = u.shape

        # Fourier transforms
        uh = rfftn(u, axes=(-2, -1))
        vh = rfftn(v, axes=(-2, -1))

        # spectral KE density
        ke_spec = 0.5 * (jnp.abs(uh) ** 2 + jnp.abs(vh) ** 2)

        # wavenumbers
        kx = jnp.fft.rfftfreq(nx, d=getattr(grid, "dx", 1.0))
        ky = jnp.fft.fftfreq(ny, d=getattr(grid, "dy", 1.0))
        kx2, ky2 = jnp.meshgrid(kx, ky, indexing="xy")
        kmag = jnp.sqrt(kx2**2 + ky2**2)

        # isotropic binning
        kmax = int(jnp.max(kmag))
        nbins = kmax + 1

        E = []
        for layer in range(nz):
            spec = ke_spec[layer]
            Ek = jnp.zeros(nbins)

            for i in range(nbins):
                mask = (kmag >= i) & (kmag < i + 1)
                Ek = Ek.at[i].set(jnp.sum(spec[mask]))

            E.append(Ek)

        E = jnp.stack(E, axis=0)  # (nz, nk)
        k = jnp.arange(nbins)

        return {"E": E, "k": k}

    # ------------------------------------------------------------------
    # plotting interface
    # ------------------------------------------------------------------

    def n_axes(self) -> int:
        return self.nz

    def init_plot(self, ax, sample, grid=None, layer: int = 0):
        """
        Initialize KE spectrum plot for one layer.
        """
        E = np.asarray(sample["E"][layer])
        k = np.asarray(sample["k"])

        (line,) = ax.loglog(k[1:], E[1:], color="k")

        ax.set_title(
            f"KE Spectrum (layer {layer})" if self.nz > 1 else "KE Spectrum"
        )
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.grid(True, which="both", ls=":")

        return {
            "line": line,
            "layer": layer,
            "artists": [line],
        }

    def update_plot(self, artists: dict, sample):
        layer = artists["layer"]
        line = artists["line"]

        E = np.asarray(sample["E"][layer])
        k = np.asarray(sample["k"])

        line.set_data(k[1:], E[1:])

    # ------------------------------------------------------------------
    # reduction for final plots
    # ------------------------------------------------------------------

    def reduce(self, samples, grid=None):
        """
        Time-average KE spectrum.

        samples: list of dicts with keys {"E", "k"}
        """
        E_stack = jnp.stack([s["E"] for s in samples], axis=0)
        E_mean = jnp.mean(E_stack, axis=0)
        k = samples[0]["k"]

        return {"E": E_mean, "k": k}

    def plot_final(self, ax, reduced, grid=None, layer: int = 0):
        # Make the final KE spectrum plot slightly bigger for readability
        fig = ax.get_figure()
        w, h = fig.get_size_inches()
        fig.set_size_inches(max(w * 1.25, 8), h * 1.25)

        E = np.asarray(reduced["E"][layer])
        k = np.asarray(reduced["k"])

        # add Rhines wavenumber and slope reference lines
        U_target = self.beta * (self.Ly / (jnp.pi * self.njets))**2
        L_rh = (2*U_target / self.beta) ** 0.5
        k_rh = 1 / L_rh
        kR = 2 * jnp.pi * self.njets / self.Ly
        ax.axvline(x=k_rh, color="C4", ls="--", label="Rhines Scale Wavenumber")
        ax.axvline(x=kR, color="C1", ls="--", label="initial jet wavenumber (post-scaling)")

        _k = np.asarray(k[1:])
        _ref_clipped = np.clip(_k, 1, kR)
        ax.loglog(_ref_clipped, (10 * _ref_clipped **(-5.0)), color="C3", ls="--", label="$k^{-5}$")

        _k = np.asarray(k[1:])
        _ref_clipped = np.clip(_k, kR, 10)
        ax.loglog(_ref_clipped, (1e-3 * _ref_clipped **(-5/3))+1e-3, color="C2", ls="--", label="$k^{-5/3}$")
        #ax.loglog(k[1:], (1e-3 * k[1:] **(-3.0))+1e-2, color="C4", ls="--", label="$k^{-3}$")

        ax.loglog(k[1:], E[1:], color="k")
        ax.set_title(
            f"Mean KE Spectrum (layer {layer})" if self.nz > 1 else "Mean KE Spectrum"
        )
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.grid(True, which="both", ls=":")
        ax.legend()




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
        psi = state.psi

        if psi.ndim == 2:
            val = compute_ke(psi, grid)
            arr = jnp.asarray([val])
        else:
            # per-layer energies
            vals = [compute_ke(psi[layer], grid) for layer in range(psi.shape[0])]
            arr = jnp.asarray(vals)

        return arr

    def init_plot(self, ax, sample, grid=None):
        # sample: (nz,) per-frame energies
        nz = sample.shape[0]
        (line,) = ax.plot([], [])
        ax.set_title("Energy")
        ax.set_xlabel("frame")
        ax.set_ylabel("energy")
        return {"line": line, "artists": [line]}

    def update_plot(self, artists: dict, sample):
        # sample: (nz,) - this artist corresponds to a single layer
        layer = artists.get("layer", 0)
        y = float(np.asarray(sample[layer]))
        line = artists["line"]
        xd = np.asarray(line.get_xdata())
        yd = np.asarray(line.get_ydata())
        xd = np.append(xd, xd.size)
        yd = np.append(yd, y)
        line.set_data(xd, yd)

    def __init__(self, nz: int):
        self.nz = int(nz)

    def n_axes(self) -> int:
        return self.nz

    def reduce(self, samples, grid=None):
        # samples: list of (nz,) -> (nt, nz) -> return (nz, nt)
        arr = jnp.stack(samples, axis=0)
        return arr.T

    def plot_final(self, ax, reduced, grid=None, layer: int = 0):
        data = np.asarray(reduced[layer])
        x = np.arange(data.shape[0])
        ax.plot(x, data, color="C0")
        ax.set_title("Energy (final)")
        ax.set_xlabel("frame")
        ax.set_ylabel("energy")


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
        zeta = state.zeta

        if zeta.ndim == 2:
            val = compute_enstrophy(zeta, grid)
            arr = jnp.asarray([val])
        else:
            vals = [compute_enstrophy(zeta[layer], grid) for layer in range(zeta.shape[0])]
            arr = jnp.asarray(vals)

        return arr

    def init_plot(self, ax, sample, grid=None):
        arr = np.asarray(sample)
        if arr.ndim == 0:
            arr = np.array([arr])
        (line,) = ax.plot([], [])
        ax.set_title("Enstrophy")
        ax.set_xlabel("frame")
        ax.set_ylabel("enstrophy")
        return {"line": line, "artists": [line]}

    def update_plot(self, artists: dict, sample):
        layer = artists.get("layer", 0)
        y = float(np.asarray(sample[layer]))
        line = artists["line"]
        xd = np.asarray(line.get_xdata())
        yd = np.asarray(line.get_ydata())
        xd = np.append(xd, xd.size)
        yd = np.append(yd, y)
        line.set_data(xd, yd)

    def __init__(self, nz: int):
        self.nz = int(nz)

    def n_axes(self) -> int:
        return self.nz

    def reduce(self, samples, grid=None):
        arr = jnp.stack(samples, axis=0)
        return arr.T

    def plot_final(self, ax, reduced, grid=None, layer: int = 0):
        data = np.asarray(reduced[layer])
        x = np.arange(data.shape[0])
        ax.plot(x, data, color="C1")
        ax.set_title("Enstrophy (final)")
        ax.set_xlabel("frame")
        ax.set_ylabel("enstrophy")


class DriftDiagnostic(Diagnostic):
    name = "drift"
    def __init__(self, nz: int):
        self.nz = int(nz)

    def requires(self):
        return {"q"}

    def retrieve(self, state, grid=None):
        q = state.q # expected shape (nz, ny, nx)
        qm = jnp.mean(q, axis=-1)

        if qm.ndim == 1:
            qm = qm[None, ...]

        if qm.shape[0] != self.nz:
            raise ValueError(f"{self.name}: expected nz={self.nz}, got {qm.shape[0]}")

        return qm

    def n_axes(self) -> int:
        return self.nz

    def reduce(self, samples, grid=None):
        # samples: list of (nz, ny)
        arr = jnp.stack(samples, axis=0)   # (nt, nz, ny)
        return jax.device_get(arr)

    def plot_final(self, ax, reduced, grid=None, layer: int = 0):
        # reduced: numpy array (nt, nz, ny)
        if not isinstance(reduced, np.ndarray):
            try:
                reduced = jax.device_get(reduced)
            except Exception:
                reduced = np.asarray(reduced)

        # select layer: image shape (nt, ny)
        im = reduced[:, layer, :].T # (ny, nt)
        nt = im.shape[1]
        ly = float(getattr(grid, "Ly", im.shape[0]))
        extent = (0, nt, 0, ly)

        im = ax.imshow(im, extent=extent, aspect="auto", origin="lower", cmap="RdBu_r")
        ax.set_xlabel("frame")
        ax.set_ylabel("y")
        ax.figure.colorbar(im, ax=ax)


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
