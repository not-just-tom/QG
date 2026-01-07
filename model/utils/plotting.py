import os
import zarr

PLOT_REGISTRY = {}

def register_plot(name):
    def wrapper(func):
        PLOT_REGISTRY[name] = func
        return func
    return wrapper


@register_plot("energy")
def plot_energy(zarr_root, outdir):
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    E = zarr_root["energy"][:]

    plt.figure()
    plt.plot(E)
    plt.xlabel("Sample")
    plt.ylabel("Energy")
    plt.title("Total Energy")
    plt.savefig(f"{outdir}/energy.png")
    plt.close()


@register_plot("vorticity_snapshot")
def plot_vorticity_snapshot(root, outdir, frame=-1):
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    zeta = root["zeta"][frame]
    vlim = max(abs(zeta.min()), abs(zeta.max()))

    plt.figure()
    plt.imshow(zeta, origin="lower", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    plt.colorbar(label="ζ")
    plt.title(f"Vorticity (frame {frame})")
    plt.savefig(f"{outdir}/vorticity_{frame}.png")
    plt.close()


@register_plot("vorticity_gif")
def plot_vorticity_gif(root, outdir):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    os.makedirs(outdir, exist_ok=True)
    zeta = root["zeta"][:]

    fig, ax = plt.subplots()
    im = ax.imshow(zeta[0], cmap="RdBu_r", origin="lower")

    def update(i):
        im.set_data(zeta[i])
        ax.set_title(f"Frame {i}")
        return [im]

    anim = FuncAnimation(fig, update, frames=len(zeta))
    try:
        from matplotlib.animation import PillowWriter
        anim.save(f"{outdir}/vorticity.gif", writer=PillowWriter(fps=10))
    except Exception:
        # Fallback: save PNG frames
        for i in range(len(zeta)):
            fig, ax = plt.subplots()
            ax.imshow(zeta[i], cmap="RdBu_r", origin="lower")
            ax.set_title(f"Frame {i}")
            fig.savefig(f"{outdir}/vorticity_{i}.png")
            plt.close(fig)
    plt.close(fig)


def run_plots(zarr_path, plots, outdir):
    root = zarr.open(zarr_path, mode="r")

    for p in plots:
        if p not in PLOT_REGISTRY:
            raise ValueError(f"Unknown plot: {p}")
        PLOT_REGISTRY[p](root, outdir) 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_model(
    model,
    recorder,
    nsteps=2000,
    frame_interval=10,
    outname="test.gif",
    plots: list | None = None,
):
    """
    Build an animation from data already stored in `recorder`.

    This function does NOT advance the model; it reads the recorded time series
    (recorder.animate_buffers and recorder.final_buffers) and constructs an
    animation for the requested `plots`.
    """

    plots = plots or sorted(list(recorder.animate_names))
    if not plots:
        raise ValueError("No plots requested for animation (set diagnostics.animate in config or pass plots=)")

    # Gather series data
    series = {}
    max_frames = 0
    for name in plots:
        if name == "zeta":
            arr = recorder.animate_buffers.get("zeta", [])
            s = np.asarray(arr)
            # Ensure shape (nframes, ny, nx) if possible
        elif name in ("energy", "enstrophy"):
            s = np.asarray(recorder.final_buffers.get(name, []))
        elif name == "ke_spectrum":
            s = recorder.final_buffers.get("ke_spectrum", [])
            s = list(s)
        else:
            raise ValueError(f"Unsupported animate plot: {name}")

        length = len(s)
        max_frames = max(max_frames, length)
        series[name] = s

    if max_frames == 0:
        raise ValueError("No recorded data available for requested plots")

    n_panels = len(plots)
    nrows, ncols = 1, n_panels
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))

    # ensure axs is list-like
    if n_panels == 1:
        axs = [axs]
    else:
        axs = list(np.ravel(axs))

    artists = {}

    # precompute reduced KE spectra if present
    ke_series = []
    if "ke_spectrum" in plots:
        raw = series.get("ke_spectrum", [])
        diag = getattr(recorder, "_diag_map", {}).get("ke_spectrum")
        for v in raw:
            if diag is not None and hasattr(diag, "reduce"):
                ke_series.append(diag.reduce(v, model.grid))
            else:
                ke_series.append(np.asarray(v).sum(axis=0))

    # initialize panels with first-available frames
    for ax, name in zip(axs, plots):
        if name == "zeta":
            arr = recorder.animate_buffers.get("zeta", [])
            # ensure shape (nframes, ny, nx)
            if len(arr) > 0:
                zarr = np.stack([a.squeeze() for a in arr])
            else:
                z0 = np.array(model.fields["zeta"])
                zarr = np.expand_dims(z0, axis=0)
            artists["zeta"] = ax.imshow(zarr[0], origin="lower", cmap="RdBu_r")
            fig.colorbar(artists["zeta"], ax=ax)
            ax.set_title("Vorticity")
            series["zeta"] = zarr
        elif name in ("energy", "enstrophy"):
            vals = np.asarray(series.get(name, []))
            line, = ax.plot([], [], lw=2)
            artists[name] = line
            ax.set_xlabel("Step")
            ax.set_ylabel(name)
            ax.set_title(name.capitalize())
            # set provisional limits; will autoscale during update
            ax.set_xlim(0, max_frames * (frame_interval or 1))
            ax.set_ylim(0, max(1.0, vals.max() if vals.size else 1.0))
            series[name] = vals
        elif name == "ke_spectrum":
            if len(ke_series) == 0:
                artists["ke_spectrum"], = ax.plot([], [], lw=2)
            else:
                k = np.arange(len(ke_series[0]))
                artists["ke_spectrum"], = ax.plot(k, ke_series[0], lw=2)
                ax.set_xlim(0, len(k))
                ax.set_ylim(max(1e-12, ke_series[0].min()), max(ke_series[0].max() * 1.1, 1e-12))
            ax.set_xlabel("k")
            ax.set_ylabel("E(k)")
            ax.set_title("KE Spectrum")
            ax.set_yscale("log")
            series["ke_spectrum"] = ke_series

    # x values for time-series plots
    xs = np.arange(max_frames) * (frame_interval or 1)

    def update(i):
        for name in plots:
            if name == "zeta":
                zarr = series["zeta"]
                idx = min(i, zarr.shape[0] - 1)
                artists["zeta"].set_data(zarr[idx])
            elif name in ("energy", "enstrophy"):
                vals = series[name]
                if vals.size == 0:
                    continue
                ys = vals[: min(i + 1, vals.size) ]
                xi = xs[: len(ys) ]
                artists[name].set_data(xi, ys)
                ax = axs[plots.index(name)]
                ax.relim()
                ax.autoscale_view()
            elif name == "ke_spectrum":
                ks = series.get("ke_spectrum", [])
                if len(ks) == 0:
                    continue
                ek = ks[min(i, len(ks) - 1)]
                k = np.arange(len(ek))
                artists["ke_spectrum"].set_data(k, ek)
                ax = axs[plots.index("ke_spectrum")]
                ax.set_xlim(0, len(k))
                ax.set_ylim(max(1e-12, ek.min()), max(ek.max() * 1.1, 1e-12))

        # set a generic title with sample index
        for ax in axs:
            ax.set_title(f"Step {i * (frame_interval or 1)}")

        return list(artists.values())

    anim = FuncAnimation(fig, update, frames=max_frames, blit=False)

    anim.save(outname, fps=10)
    plt.close(fig)
    print(f"Saved animation to {outname}")


# ===== not needed but im keeping until i check all the diagnositics ===== #
import jax.numpy as jnp
@staticmethod
def compute_ke_spectrum(psi, grid):
    """
    Compute 1D isotropic KE spectrum E(k) using uh, vh
    """
    # Compute velocity in Fourier space
    psih = 0#rfftn(psi, axes=(-2,-1), norm='ortho')
    uh = -1j * grid.KY * psih
    vh =  1j * grid.KX * psih

    # KE density in spectral space
    KE2D = 0.5 * (jnp.abs(uh)**2 + jnp.abs(vh)**2)

    # bin into isotropic shells
    kmax = int(jnp.max(grid.Kmag))
    kbins = jnp.arange(kmax+1)

    # flatten arrays
    kmag_flat = grid.Kmag.flatten().astype(int)
    KE_flat = KE2D.flatten()

    # accumulate into bins
    E_k = jnp.zeros(kmax+1, dtype=float).at[kbins].add(
        jnp.bincount(kmag_flat, weights=KE_flat, length=kmax+1)
    )

    return np.array(E_k), np.array(kbins)

def make_animation(
        self, 
        nsteps=5000, 
        frame_interval=100,
        outname="../outputs/qg_gpu.gif",
        stats=None        # flags for which plots, currently ['zonal', 'energy', 'enstrophy', ke_spec]
    ):

    # Normalize stats
    if stats is None:
        stats = []
    valid_stats = ["zonal", "energy", 'enstrophy', 'kespec']
    stats = [s for s in stats if s in valid_stats]

    # setup figs dynamically 
    n_panels = 1 + len(stats)
    panel_indices = {"vort": 0}

    if "zonal" in stats:
        panel_indices["zonal"] = len(panel_indices)
    if "energy" in stats:
        panel_indices["energy"] = len(panel_indices)
    if "enstrophy" in stats:
        panel_indices["enstrophy"] = len(panel_indices)
    if "kespec" in stats:
        panel_indices["kespec"] = len(panel_indices)

    # Compute subplot grid
    if n_panels == 1:
        nrows, ncols = 1, 1
    else:
        nrows = 2
        ncols = int(np.ceil(n_panels / 2))

    fig, axs = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 5 * nrows),
        constrained_layout=False
    )

    axs = np.ravel(axs)

    # Always ensure axs is a list-like
    if n_panels == 1:
        axs = [axs]

    ax_vort = axs[panel_indices["vort"]]
    y = self.grid.y
    dx = self.grid.dx
    dy = self.grid.dy

    umean_list = []
    energy_list = []
    enstrophy_list = []
    ke_list = []
    time_list_energy = []
    time_list_enstro = []
    time_list_ke = []

    # vorticity (always)
    zeta = np.array(self.fields["zeta"])
    vmin, vmax = float(zeta.min()), float(zeta.max())

    im = ax_vort.imshow(
        zeta, origin="lower",
        cmap="RdBu_r",
        extent=(-self.grid.Lx/2, self.grid.Lx/2,
                -self.grid.Ly/2, self.grid.Ly/2),
        animated=True #vmin=vmin, vmax=vmax
    )
    ax_vort.set_title("Vorticity")
    ax_vort.set_xlabel("x")
    ax_vort.set_ylabel("y")

    # zonal mean velocity
    if "zonal" in stats:
        ax_umean = axs[panel_indices["zonal"]]
        line_umean, = ax_umean.plot(np.zeros_like(y), y)
        ax_umean.set_title("Zonal-mean U")
        ax_umean.set_xlabel("Ū(y)")
        ax_umean.set_ylabel("y")
        ax_umean.grid(True)
    else:
        line_umean = None

    # energy
    if "energy" in stats:
        ax_energy = axs[panel_indices["energy"]]
        line_energy, = ax_energy.plot([], [])
        ax_energy.set_title("Energy vs Time")
        ax_energy.set_xlabel("Step")
        ax_energy.set_ylabel("Energy")
        ax_energy.grid(True)
    else:
        line_energy = None

    # enstrophy
    if "enstrophy" in stats:
        ax_enstrophy = axs[panel_indices["enstrophy"]]
        line_enstrophy, = ax_enstrophy.plot([], [])
        ax_enstrophy.set_title("Enstrophy vs Time")
        ax_enstrophy.set_xlabel("Step")
        ax_enstrophy.set_ylabel("Enstrophy")
        ax_enstrophy.grid(True)
    else:
        line_enstrophy = None

    # KE spectrum
    if "kespec" in stats:
        ax_spec = axs[panel_indices["kespec"]] if "kespec" in panel_indices else axs[-1]
        line_spec, = ax_spec.loglog([], [])
        ax_spec.set_title("KE Spectrum")
        ax_spec.set_xlabel("k")
        ax_spec.set_ylabel("E(k)")
        ax_spec.grid(True, which='both')
    else:
        line_spec = None

    # update func
    def update(frame):
        nonlocal umean_list, energy_list, enstrophy_list, ke_list, time_list_energy, time_list_enstro, time_list_ke

        if frame == 0:
            zeta = np.array(self.fields["zeta"])
            psi = np.array(self.fields["psi"])
        else:
            self.steps(frame_interval)
            zeta = np.array(self.fields["zeta"])
            psi = np.array(self.fields["psi"])

        # vorticity update
        im.set_array(zeta)
        ax_vort.set_title(f"Vorticity (step {frame * frame_interval})")

        # Compute velocities if needed
        if "zonal" in stats or "energy" in stats or "enstrophy" in stats:
            u = -(np.roll(psi, -1, 0) - np.roll(psi, 1, 0)) / (2 * dy)
            v = (np.roll(psi, -1, 1) - np.roll(psi, 1, 1)) / (2 * dx)
        
        # zonal mean velocity
        if "zonal" in stats:
            ubar = u.mean(axis=1)
            line_umean.set_xdata(ubar)
            ax_umean.relim()
            ax_umean.autoscale_view()
            umean_list.append(ubar)

        # energy
        if "energy" in stats:
            KE = 0.5 * np.mean(u**2 + v**2)*dx*dy
            energy_list.append(KE)
            time_list_energy.append(frame * frame_interval)

            line_energy.set_xdata(time_list_energy)
            line_energy.set_ydata(energy_list)
            ax_energy.relim()
            ax_energy.autoscale_view()

        # enstrophy
        if "enstrophy" in stats:
            enstro = 0.5 * np.mean(zeta**2)*dx*dy
            enstrophy_list.append(enstro)
            time_list_enstro.append(frame * frame_interval)

            line_enstrophy.set_xdata(time_list_enstro)
            line_enstrophy.set_ydata(enstrophy_list)
            ax_enstrophy.relim()
            ax_enstrophy.autoscale_view()

        if "kespec" in stats:
#            E_k, kbins = Solver.compute_ke_spectrum(psi, self.grid)
#            ke_list.append(E_k)
 #           time_list_ke.append(kbins)
  #          line_spec.set_ydata(E_k[1:])
   #         line_spec.set_xdata(kbins[1:])
            ax_spec.relim()
            ax_spec.autoscale_view()

        return [x for x in [im, line_umean, line_energy, line_enstrophy, line_spec] if x is not None]


    frames = nsteps // frame_interval + 1
    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(outname, fps=10)
    plt.close(fig)
    print(f"Saved animation to {outname}")

    # Compute subplot for final plots 
    if n_panels <4:
        nrows, ncols = 1, n_panels
    else:
        nrows = 2
        ncols = int(np.ceil(n_panels / 2))

    fig2, axs2 = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 5 * nrows),
        constrained_layout=True
    )

    axs2 = np.ravel(axs2)

    # --- final: time-averaged zonal velocity ---
    if "zonal" in stats:
        ax2_umean = axs2[panel_indices["zonal"]]
        umean_timeavg = np.mean(np.array(umean_list), axis=0)
        ax2_umean.plot(umean_timeavg, y)
        ax2_umean.set_title("Final Time-averaged Zonal Velocity")
        ax2_umean.set_ylabel("y")
        ax2_umean.set_xlabel("Ū(y)")
        ax2_umean.grid(True)

    # --- final: energy vs time ---
    if "energy" in stats:
        ax2_energy = axs2[panel_indices["energy"]]
        ax2_energy.plot(time_list_energy, energy_list)
        ax2_energy.set_title("Energy vs Time (Final)")
        ax2_energy.set_xlabel("Step")
        ax2_energy.set_ylabel("Energy")
        ax2_energy.grid(True)

    # --- final: enstrophy vs time ---
    if "enstrophy" in stats:
        ax2_enstro = axs2[panel_indices["enstrophy"]]
        ax2_enstro.plot(time_list_enstro, enstrophy_list)
        ax2_enstro.set_title("Enstrophy vs Time (Final)")
        ax2_enstro.set_xlabel("Step")
        ax2_enstro.set_ylabel("Enstrophy")
        ax2_enstro.grid(True)

    if "kespec" in stats:
        ke_stack = np.vstack(ke_list)  
        E_k_timeavg = ke_stack.mean(axis=0)
        k_vals = np.arange(E_k_timeavg.size)

        ax2_kespec = axs2[panel_indices["kespec"]]
        ax2_kespec.loglog(k_vals[1:], E_k_timeavg[1:])
        ax2_kespec.set_title("Time-Averaged Energy Spectra")
        ax2_kespec.set_xlabel("k")
        ax2_kespec.set_ylabel("E(k)")
        ax2_kespec.grid(True)

    # Save all final panels together
    fig2.savefig("../outputs/final_summary.png")
    plt.close(fig2)