import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import cmocean as cmo
import jax.numpy as jnp
import matplotlib.gridspec as gridspec





def animate(recorder, grid, cadence=100, outname="test.gif", plots: list | None = None):
    """
    Build an animation from data already stored in `recorder`.
    
    Automatically detects single-layer vs two-layer model and uses appropriate plotting.
    Uses Recorder's animate_buffers and final_buffers, aligned with its diagnostics.
    """
    # Check if we have two-layer data
    available_plots = list(recorder.diagnostics.keys())
    plots = plots or available_plots
    plots = [p for p in plots if p in available_plots]
    
    if not plots:
        raise ValueError("No valid diagnostics available for animation")
    
    """
    Build an animation for two-layer model data from `recorder`.
    
    For 2D fields with two layers, creates side-by-side plots for each layer.
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
            series[name] = np.array([val]) if np.isscalar(val) else np.array(val)
            max_frames = max(max_frames, series[name].shape[0])
        else:
            series[name] = np.empty((0,))

    if max_frames == 0:
        raise ValueError("No recorded data available for requested plots")

    # Determine subplot layout - 1 column per layer for multi-layer fields
    n_base_plots = len(plots)
    subplot_cols = []
    for name in plots:
        data = series[name]
        if data.ndim == 4:  # (time, nz, ny, nx) - multi-layer field
            nz = data.shape[1]
            subplot_cols.extend([1] * nz)
        else:
            subplot_cols.append(1)
    
    n_total_cols = sum(subplot_cols)
    fig, axs_flat = plt.subplots(1, n_total_cols, figsize=(5 * n_total_cols, 5))
    axs_flat = [axs_flat] if n_total_cols == 1 else list(np.ravel(axs_flat))

    artists = {}
    xs = np.arange(max_frames) * cadence
    ax_idx = 0

    # Initialize plots
    for name in plots:
        data = series[name]
        if name == "ke_spectrum":
            ax = axs_flat[ax_idx]
            ek = data[0] if len(data) > 0 else np.zeros(1)
            k = np.arange(len(ek))
            artists[name], = ax.plot(k, ek, lw=2)
            ax.set_xlabel("k")
            ax.set_ylabel("E(k)")
            ax.set_title("KE Spectrum")
            ax.set_yscale("log")
            ax_idx += 1
        elif data.ndim == 4:  # Multi-layer 2D field (time, nz, ny, nx)
            extent = [-grid.Lx/2, grid.Lx/2, -grid.Ly/2, grid.Ly/2] if grid else None
            nz = data.shape[1]
            for layer in range(nz):
                ax_layer = axs_flat[ax_idx]
                im = ax_layer.imshow(data[0, layer], origin="lower", cmap="RdBu_r", extent=extent)
                fig.colorbar(im, ax=ax_layer)
                ax_layer.set_title(f"{name} - Layer {layer+1}")
                if grid:
                    ax_layer.set_xlabel("x")
                    ax_layer.set_ylabel("y")
                artists[f"{name}_layer{layer}"] = im
                ax_idx += 1
        elif data.ndim == 3:  # Single-layer 2D field (time, ny, nx)
            ax = axs_flat[ax_idx]
            extent = [-grid.Lx/2, grid.Lx/2, -grid.Ly/2, grid.Ly/2] if grid else None
            im = ax.imshow(data[0], origin="lower", cmap="RdBu_r", extent=extent)
            fig.colorbar(im, ax=ax)
            ax.set_title(name)
            if grid:
                ax.set_xlabel("x")
                ax.set_ylabel("y")
            artists[name] = im
            ax_idx += 1
        elif name == "zonal":
            # data shape: (time, nz, ny) or (time, ny)
            data = series[name]
            y = np.arange(0.5, grid.ny, 1.0) / grid.ny * grid.Ly if grid else np.arange(data.shape[-1])
            if data.ndim == 3:
                nz = data.shape[1]
                for layer in range(nz):
                    ax_layer = axs_flat[ax_idx]
                    line, = ax_layer.plot(y, data[0, layer], lw=2)
                    ax_layer.set_xlabel("y")
                    ax_layer.set_ylabel("zonal")
                    ax_layer.set_title(f"{name} - Layer {layer+1}")
                    artists[f"{name}_layer{layer}"] = line
                    ax_idx += 1
            else:
                ax = axs_flat[ax_idx]
                line, = ax.plot(y, data[0], lw=2)
                ax.set_xlabel("y")
                ax.set_ylabel("zonal")
                ax.set_title(name)
                artists[name] = line
                ax_idx += 1
        else:  # Time series scalar
            ax = axs_flat[ax_idx]
            line, = ax.plot([], [], lw=2)
            artists[name] = line
            ax.set_xlabel("Step")
            ax.set_ylabel(name)
            ax.set_title(name)
            ax.set_xlim(0, max_frames * cadence)
            ax.set_ylim(0, max(1.0, data.max() if data.size else 1.0))
            ax_idx += 1

    def update(i):
        ax_idx = 0
        for name in plots:
            data = series[name]
            if name == "ke_spectrum":
                ek = data[min(i, len(data) - 1)]
                k = np.arange(len(ek))
                artists[name].set_data(k, ek)
                ax = axs_flat[ax_idx]
                ax.set_ylim(max(1e-12, ek.min()), max(ek.max() * 1.1, 1e-12))
                ax.set_title(f"{name} (Step {i * cadence})")
                ax_idx += 1
            elif data.ndim == 4:  # Multi-layer field
                idx = min(i, data.shape[0] - 1)
                nz = data.shape[1]
                for layer in range(nz):
                    artists[f"{name}_layer{layer}"].set_array(data[idx, layer])
                    axs_flat[ax_idx].set_title(f"{name} - Layer {layer+1} (Step {i * cadence})")
                    ax_idx += 1
            elif name == "zonal":
                data = series[name]
                idx = min(i, data.shape[0] - 1)
                if data.ndim == 3:
                    nz = data.shape[1]
                    for layer in range(nz):
                        artists[f"{name}_layer{layer}"].set_ydata(data[idx, layer])
                        axs_flat[ax_idx].set_title(f"{name} - Layer {layer+1} (Step {i * cadence})")
                        ax_idx += 1
                else:
                    artists[name].set_ydata(data[idx])
                    axs_flat[ax_idx].set_title(f"{name} (Step {i * cadence})")
                    ax_idx += 1
            elif data.ndim == 3:  # Single-layer field
                idx = min(i, data.shape[0] - 1)
                artists[name].set_array(data[idx])
                axs_flat[ax_idx].set_title(f"{name} (Step {i * cadence})")
                ax_idx += 1
            else:  # Time series
                ys = data[: min(i + 1, len(data))]
                xi = xs[: len(ys)]
                artists[name].set_data(xi, ys)
                ax = axs_flat[ax_idx]
                ax.relim()
                ax.autoscale_view()
                ax.set_title(f"{name} (Step {i * cadence})")
                ax_idx += 1

        return list(artists.values())

    anim = FuncAnimation(fig, update, frames=max_frames, blit=False)
    anim.save(outname, fps=10)
    plt.close(fig)
    print(f"Saved animation to {outname}")
    



def make_triple_gif(hr_q, lr_q, sgs_q, out_file="q_triple.gif", cadence=100):
    """
    Create a GIF with 3 panels side by side:
    - hr_q: high-res field, shape (nt, ny, nx)
    - lr_q: low-res field, shape (nt, ny, nx)
    - sgs_q: SGS forcing, shape (nt, ny, nx)
    """
    nt = hr_q.shape[0]

    # Determine global vmin/vmax for HR and LR
    hr_vmax = jnp.max(jnp.abs(hr_q))
    lr_vmax = jnp.max(jnp.abs(lr_q))
    sgs_vmax = jnp.max(jnp.abs(sgs_q))

    fig = plt.figure(figsize=(12, 4), constrained_layout=True)
    gs = gridspec.GridSpec(1, 3, figure=fig)

    ax_hr = fig.add_subplot(gs[0])
    ax_lr = fig.add_subplot(gs[1])
    ax_sgs = fig.add_subplot(gs[2])

    im_hr = ax_hr.imshow(hr_q[0])#, cmap=cmo.balance)#, vmin=-hr_vmax, vmax=hr_vmax)
    ax_hr.set_title("High-Res n=256")
    ax_hr.axis("off")

    im_lr = ax_lr.imshow(lr_q[0])#, cmap=cmo.balance, vmin=-lr_vmax, vmax=lr_vmax)
    ax_lr.set_title("Low-Res n=32")
    ax_lr.axis("off")

    im_sgs = ax_sgs.imshow(sgs_q[0])#, cmap=cmo.curl, vmin=-sgs_vmax, vmax=sgs_vmax)
    ax_sgs.set_title("SGS Forcing")
    ax_sgs.axis("off")

    def update(frame):
        im_hr.set_array(hr_q[frame])
        im_lr.set_array(lr_q[frame])
        im_sgs.set_array(sgs_q[frame])
        ax_hr.set_title(f"High-Res, n=256 step={frame*cadence}")
        ax_lr.set_title(f"Low-, n=64 step={frame*cadence}")
        ax_sgs.set_title(f"SGS step={frame*cadence}")
        return [im_hr, im_lr, im_sgs]

    ani = animation.FuncAnimation(
        fig, update, frames=nt, blit=False
    )
    ani.save(out_file, fps=10)
    plt.close(fig)
    print(f"Saved GIF to {out_file}")


def step_model_n_steps(n, stepped_model, initial_state):
    """
    Step the model n times from an initial state.
    
    Args:
        n: Number of steps to take
        stepped_model: SteppedModel instance
        initial_state: Initial state to start from
        
    Returns:
        Final state after n steps
    """
    state = initial_state
    for _ in range(n):
        state = stepped_model.step_model(state)
    return state

def plot_state(state, grid, title="Potential Vorticity", outname="state.png"):
    """
    Plot a state's potential vorticity field.
    
    Args:
        state: State object with q field
        grid: Grid object with nx, ny, Lx, Ly properties
        title: Plot title
        outname: Output filename for saving
    """
    # Create coordinate arrays from grid properties
    x_extent = [0, grid.Lx]
    y_extent = [0, grid.Ly]
    q = np.array(state.q)
    
    # Determine if single or multi-layer
    if q.ndim == 3 and q.shape[0] > 1:
        n_layers = q.shape[0]
        # Plot each layer
        fig, axes = plt.subplots(1, n_layers, figsize=(6*n_layers, 5))
        if n_layers == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            im = ax.imshow(q[i], extent=[*x_extent, *y_extent], 
                          origin='lower', cmap='RdBu_r', aspect='auto')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'{title} - Layer {i+1}')
            plt.colorbar(im, ax=ax, label='q')
    else:
        # Single field or single layer
        if q.ndim == 3:
            q = q[0]  # Take first layer if shape is (1, ny, nx)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(q, extent=[*x_extent, *y_extent], 
                      origin='lower', cmap='RdBu_r', aspect='auto')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='q')
    
    plt.tight_layout()
    plt.savefig(outname, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {outname}")
    print(f"Shape: {q.shape}, min: {np.min(q):.3e}, max: {np.max(q):.3e}")
    plt.show()
    
    return fig



def plot_initial_condition(n_steps, model, state):
    """
    Plot the initial condition (or after n_steps) from the configuration.
    """

    sm = model
    # Step forward if requested
    if n_steps > 0:
        print(f"Stepping model {n_steps} steps...")
        state = step_model_n_steps(n_steps, sm, state)
        title = f"Potential Vorticity after {n_steps} steps"
        outname = f"state_after_{n_steps}_steps.png"
    else:
        title = "Initial Potential Vorticity"
        outname = "initial_condition.png"
    
    # Get full state and plot
    full_state = sm.get_full_state(state)
    plot_state(full_state, model.get_grid(), title=title, outname=outname)