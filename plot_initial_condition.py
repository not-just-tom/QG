"""
Script to plot the initial condition from run.py configuration
"""
import importlib 
import model.core.model
import model.core.states
import model.core.grid
import model.core.steppers
import model.utils.plotting
importlib.reload(model.core.states)
importlib.reload(model.core.model)
importlib.reload(model.core.grid)
importlib.reload(model.core.steppers)
importlib.reload(model.utils.plotting)
from model.utils.config import Config
from model.core.steppers import SteppedModel, build_stepper
from model.core.model import create_model
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os
import jax

jax.config.update("jax_enable_x64", True)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
CONFIG_DEFAULT_PATH = os.path.join(BASE_DIR, "QG", "config", "default.yaml")

# Load config
with open(CONFIG_DEFAULT_PATH) as f:
    cfg_dict = yaml.safe_load(f)

params = dict(cfg_dict["params"])
cfg = Config.load_config(CONFIG_DEFAULT_PATH)
dt = cfg.plotting.dt

# Create model and initialize
n_layers = params.pop('n_layers', 1)
model = create_model(params, n_layers=n_layers)
stepper = build_stepper(cfg.plotting.stepper, dt)
sm = SteppedModel(model=model, stepper=stepper)

# Get initial state
state = sm.initialise(params['seed'])
full_state = sm.get_full_state(state)

# Get grid for plotting
grid = model.get_grid()
x = np.array(grid.x)
y = np.array(grid.y)

# Get potential vorticity field
q = np.array(full_state.q)

# Determine if single or multi-layer
if q.ndim == 3:
    n_layers = q.shape[0]
    # Plot each layer
    fig, axes = plt.subplots(1, n_layers, figsize=(6*n_layers, 5))
    if n_layers == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        im = ax.imshow(q[i], extent=[x.min(), x.max(), y.min(), y.max()], 
                      origin='lower', cmap='RdBu_r', aspect='auto')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Initial PV - Layer {i+1}')
        plt.colorbar(im, ax=ax, label='q')
else:
    # Single field
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(q, extent=[x.min(), x.max(), y.min(), y.max()], 
                  origin='lower', cmap='RdBu_r', aspect='auto')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Initial Potential Vorticity')
    plt.colorbar(im, ax=ax, label='q')

plt.tight_layout()
plt.savefig('initial_condition.png', dpi=150, bbox_inches='tight')
print(f"Initial condition plot saved to initial_condition.png")
print(f"Shape: {q.shape}, min: {q.min():.3e}, max: {q.max():.3e}")
plt.show()
