import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import os
import re
import json
from types import SimpleNamespace
import importlib
import model.utils.diagnostics
importlib.reload(model.utils.diagnostics)
from model.utils.diagnostics import build_diagnostic


RUN_RE = re.compile(r"output_nx(?P<hr>\d+)_(?P<idx>\d{2})")

def metadata_matches(requested: dict, stored: dict) -> bool:
    return canonicalize(requested) == canonicalize(stored)

def canonicalize(params: dict) -> dict:
    def round_floats(x):
        if isinstance(x, float):
            return round(x, 12)
        if isinstance(x, dict):
            return {k: round_floats(v) for k, v in sorted(x.items())}
        if isinstance(x, list):
            return [round_floats(v) for v in x]
        return x

    return round_floats(params)

def find_output_dir(base_dir, params, model_type):
    """
    Function for finding the output dir so I can keep 
    some seblance of organisation in the outputs
    """
    lr_nx = params["nx"]
    hr_nx = params['hr_nx']
    prefix = f"{model_type}_{hr_nx}to{lr_nx}_"
    candidates = []

    # just in case lmao 
    os.makedirs(base_dir, exist_ok=True)

    for name in os.listdir(base_dir):
        m = RUN_RE.fullmatch(name)
        if m is None:
            continue
        if int(m["hr"]) != lr_nx:
            continue

        run_dir = os.path.join(base_dir, name)
        meta_path = os.path.join(run_dir, "metadata.json")
        if not os.path.exists(meta_path):
            continue

        try:
            with open(meta_path) as f:
                stored_meta = json.load(f)
        except Exception:
            continue

        # Exact metadata match
        if metadata_matches(params, stored_meta.get("parameters", {})):
            return run_dir, True

        try:
            candidates.append(int(m["idx"]))
        except Exception:
            continue

    # No match found
    next_idx = max(candidates, default=0) + 1
    run_name = f"{prefix}{next_idx:02d}"
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    meta_path = os.path.join(run_dir, "metadata.json")
    metadata = {"parameters": params}
    try:
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)
    except Exception:
        # If writing fails, still return the path so caller can attempt to write
        pass

    return run_dir, False

class Plotter:
    def __init__(self, cfg, trajectories=None, out_dir=None):
        self.cfg = cfg
        self.trajs = dict(trajectories or {})
        self.out_dir = out_dir or getattr(cfg.filepaths, "out_dir", ".")

        self.plot_list = list(getattr(cfg.plotting, "plot", []) or ["mse", "quad"])

        if "grid" not in self.trajs:
            self.trajs["grid"] = self._make_grid()

    def plot(self):
        os.makedirs(self.out_dir, exist_ok=True)

        for name in self.plot_list:
            try:
                diag = build_diagnostic(name)
                out_path = os.path.join(self.out_dir, f"{name}.{diag.output}")
                diag.run(self.trajs, out_path)
                print(f"Saved {out_path}")
            except Exception as e:
                print(f"{name} failed: {e}")

    def _make_grid(self):
        Lx = float(getattr(self.cfg.params, "Lx", 1.0))
        Ly = float(getattr(self.cfg.params, "Ly", Lx))
        nx = int(getattr(self.cfg.params, "nx", 64))
        ny = nx

        dx = Lx / nx
        dy = Ly / ny

        return SimpleNamespace(Lx=Lx, Ly=Ly, dx=dx, dy=dy, nx=nx, ny=ny)


    
# =====================================================================
# This was legacy code and idk if it works or not but it stays for now
# =====================================================================
def gif_that(q_state, out_file='plotting.gif', cadence=100):
    'just a simple plotting'
    q_state = q_state[::cadence] 
    nt = q_state.shape[0]

    # Determine global vmin/vmax for HR and LR
    fig = plt.figure(figsize=(4, 4), constrained_layout=True)
    gs = gridspec.GridSpec(1, 1, figure=fig)

    ax = fig.add_subplot(gs[0])

    im = ax .imshow(q_state[0])#, cmap=cmo.balance)#, vmin=-hr_vmax, vmax=hr_vmax)
    ax.set_title("State")
    ax.axis("off")

    def update(frame):
        im.set_array(q_state[frame])
        ax.set_title(f"PV state, step={frame*cadence}")
        return im

    ani = animation.FuncAnimation(
        fig, update, frames=nt, blit=False
    )
    ani.save(out_file, fps=10)


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