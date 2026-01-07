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
    Dynamic animation driver using diagnostics registered in the Recorder.

    Supports plots: 'zeta' (vorticity image), 'energy' (time series),
    'enstrophy' (time series), and 'ke_spectrum' (radial KE spectrum).
    """

    plots = plots or sorted(list(recorder.animate_names))
    if not plots:
        raise ValueError("No plots requested for animation (set diagnostics.animate in config or pass plots=)")

    n_panels = len(plots)
    nrows, ncols = 1, n_panels
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))

    # ensure axs is list-like
    if n_panels == 1:
        axs = [axs]
    else:
        axs = list(np.ravel(axs))

    artists = {}
    timeseries = {name: [] for name in plots if name in ("energy", "enstrophy")}

    # initialize panels
    for ax, name in zip(axs, plots):
        if name == "zeta":
            zeta0 = recorder.animate_buffers.get("zeta")
            if zeta0 is None:
                zeta0 = np.array(model.fields["zeta"])
            im = ax.imshow(zeta0, origin="lower", cmap="RdBu_r")
            artists["zeta"] = im
            fig.colorbar(im, ax=ax)
            ax.set_title("Vorticity")
        elif name in ("energy", "enstrophy"):
            line, = ax.plot([], [], lw=2)
            artists[name] = line
            ax.set_xlim(0, nsteps)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Step")
            ax.set_ylabel(name)
            ax.set_title(name.capitalize())
        elif name == "ke_spectrum":
            line, = ax.plot([], [], lw=2)
            artists["ke_spectrum"] = line
            ax.set_xlabel("k")
            ax.set_ylabel("E(k)")
            ax.set_title("KE Spectrum")
            ax.set_yscale("log")
            ax.set_ylim(1e-12, 1e-6)
        else:
            raise ValueError(f"Unsupported animate plot: {name}")

    def update(frame):
        # advance the model
        model.steps(frame_interval)
        # sample diagnostics AFTER stepping
        recorder.sample(model)

        # update each artist
        for name in plots:
            if name == "zeta":
                z = recorder.animate_buffers.get("zeta")
                if z is None:
                    continue
                artists["zeta"].set_data(z)
            elif name in ("energy", "enstrophy"):
                val = recorder.animate_buffers.get(name)
                if val is None:
                    continue
                val = float(val)
                timeseries[name].append(val)
                xs = np.arange(len(timeseries[name])) * frame_interval
                artists[name].set_data(xs, timeseries[name])
                ax = axs[plots.index(name)]
                ax.relim()
                ax.autoscale_view()
            elif name == "ke_spectrum":
                val = recorder.animate_buffers.get("ke_spectrum")
                if val is None:
                    continue
                # reduce to 1D radial spectrum if possible
                diag = getattr(recorder, "_diag_map", {}).get("ke_spectrum")
                if diag is not None and hasattr(diag, "reduce"):
                    E_k = diag.reduce(val, model.grid)
                else:
                    E_k = np.asarray(val).sum(axis=0)
                k = np.arange(len(E_k))
                artists["ke_spectrum"].set_data(k, E_k)
                ax = axs[plots.index("ke_spectrum")]
                ax.set_xlim(0, len(E_k))
                ax.set_ylim(max(1e-12, E_k.min()), max(E_k.max() * 1.1, 1e-12))
        for ax in axs:
            ax.set_title(f"Step {model.n}")

        return list(artists.values())

    frames = max(1, nsteps // frame_interval)
    anim = FuncAnimation(fig, update, frames=frames, blit=False)

    anim.save(outname, fps=10)
    plt.close(fig)
    print(f"Saved animation to {outname}")
