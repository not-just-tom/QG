import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
def animate(recorder, grid, cadence=100, outname="test.gif", plots: list | None = None):
    """
    Build an animation from data already stored in `recorder`.

    Uses Recorder's animate_buffers and final_buffers, aligned with its diagnostics.
    """

    # Only include plots that actually exist in recorder diagnostics
    available_plots = list(recorder.diagnostics.keys())
    plots = plots or available_plots
    plots = [p for p in plots if p in available_plots]
    if not plots:
        raise ValueError("No valid diagnostics available for animation")

    # Gather series data
    series = {}
    max_frames = 0
    for name in plots:
        if name in recorder.animate_buffers:
            arr = recorder.animate_buffers.get(name, [])
            if len(arr) > 0:
                series[name] = np.stack([np.array(a) for a in arr])
                max_frames = max(max_frames, series[name].shape[0])
            else:
                series[name] = np.empty((0,))
        elif name in recorder.final_buffers:
            val = recorder.final_buffers.get(name)
            # wrap scalar into array for plotting consistency
            series[name] = np.array([val]) if np.isscalar(val) else np.array(val)
            max_frames = max(max_frames, series[name].shape[0])
        else:
            series[name] = np.empty((0,))

    if max_frames == 0:
        raise ValueError("No recorded data available for requested plots")

    n_panels = len(plots)
    fig, axs = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    axs = [axs] if n_panels == 1 else list(np.ravel(axs))

    artists = {}
    xs = np.arange(max_frames) * cadence

    # initialize plots
    for ax, name in zip(axs, plots):
        data = series[name]
        if name == "ke_spectrum":
            # first frame
            ek = data[0] if len(data) > 0 else np.zeros(1)
            k = np.arange(len(ek))
            artists[name], = ax.plot(k, ek, lw=2)
            ax.set_xlabel("k")
            ax.set_ylabel("E(k)")
            ax.set_title("KE Spectrum")
            ax.set_yscale("log")
        elif data.ndim == 3:  # 2D field over time
            # Use grid coordinates for extent
            if grid is not None:
                extent = [-grid.Lx/2, grid.Lx/2, -grid.Ly/2, grid.Ly/2]
                artists[name] = ax.imshow(data[0], origin="lower", cmap="RdBu_r", extent=extent)
            else:
                artists[name] = ax.imshow(data[0], origin="lower", cmap="RdBu_r")
            fig.colorbar(artists[name], ax=ax)
            ax.set_title(name)
            if grid is not None:
                ax.set_xlabel("x")
                ax.set_ylabel("y")
        else:  # time series scalar
            line, = ax.plot([], [], lw=2)
            artists[name] = line
            ax.set_xlabel("Step")
            ax.set_ylabel(name)
            ax.set_title(name)
            ax.set_xlim(0, max_frames * cadence)
            ax.set_ylim(0, max(1.0, data.max() if data.size else 1.0))

    def update(i):
        for name in plots:
            data = series[name]
            if name == "ke_spectrum":
                ek = data[min(i, len(data) - 1)]
                k = np.arange(len(ek))
                artists[name].set_data(k, ek)
                ax = axs[plots.index(name)]
                ax.set_ylim(max(1e-12, ek.min()), max(ek.max() * 1.1, 1e-12))
            elif data.ndim == 3:  # 2D field
                idx = min(i, data.shape[0] - 1)
                artists[name].set_data(data[idx])
            else:  # time series scalar
                ys = data[: min(i + 1, len(data))]
                xi = xs[: len(ys)]
                artists[name].set_data(xi, ys)
                ax = axs[plots.index(name)]
                ax.relim()
                ax.autoscale_view()

            ax = axs[plots.index(name)]
            ax.set_title(f"{name} (Step {i * cadence})")

        return list(artists.values())

    anim = FuncAnimation(fig, update, frames=max_frames, blit=False)
    anim.save(outname, fps=10)
    plt.close(fig)
    print(f"Saved animation to {outname}")


