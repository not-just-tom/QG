"""Small demo training loop for spectral correction model.

This provides a convenience function `train_spectral_demo` that synthesizes a
"truth" spectral correction and fits the spectral parameters to it using SGD.
It is intentionally simple and meant as a starting point for more advanced
training (e.g., differentiable time integration, multi-step losses, etc.).
"""
import os
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
CONFIG_DEFAULT_PATH = os.path.join(BASE_DIR,"config", "default.yaml")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import importlib 
import model.core.steppers
importlib.reload(model.core.steppers)
import model.core.model
import model.ML.utils.coarsen
importlib.reload(model.core.model)
importlib.reload(model.ML.utils.coarsen)
from model.core.model import create_model
from model.utils.diagnostics import Recorder
from model.ML.utils.coarsen import Coarsener
import logging
import pathlib
import yaml
import jax
import functools
import optax
import equinox as eqx
import cmocean.cm as cmo
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from model.utils.config import Config
from model.utils.logging import configure_logging
from model.core.steppers import SteppedModel, build_stepper
from model.ML.forced_model import ForcedModel
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from model.utils.plotting import make_triple_gif
from model.ML.utils.utils import parameterization

# === just loading in params as dict === #
with open(CONFIG_DEFAULT_PATH) as f:
    cfg_dict = yaml.safe_load(f)

params = dict(cfg_dict["params"])

def run():

    #========
    cfg = Config.load_config(CONFIG_DEFAULT_PATH)
    out_dir = pathlib.Path(cfg.filepaths.out_dir)
    if out_dir.is_file():
        raise ValueError(f"Path must be a directory, not a file: {cfg.filepaths.out_dir}")
    out_dir.mkdir(exist_ok=True)
    configure_logging(level=cfg.filepaths.log_level, out_file="../logs/run.log") #return to this to put numbers on it 
    logger = logging.getLogger("main")

    # load config values
    dt = cfg.plotting.dt
    nsteps = cfg.plotting.nsteps
    cadence = cfg.plotting.cadence
    cfg_stepper = cfg.plotting.stepper
    outname = "../outputs/training_test.gif"

    # load ML bits 

    def param_to_single(param):
        if eqx.is_inexact_array(param):
            if param.dtype == jnp.dtype(jnp.float64):
                return param.astype(jnp.float32)
            elif param.dtype == jnp.dtype(jnp.complex128):
                return param.astype(jnp.complex64)
        return param


    def module_to_single(module):
        return jax.tree.map(param_to_single, module)


    class NNParam(eqx.Module):
        ops: eqx.nn.Sequential

        def __init__(self, key):
            key1, key2 = jax.random.split(key, 2)
            self.ops = eqx.nn.Sequential(
                [
                    eqx.nn.Conv2d(
                        in_channels=2,
                        out_channels=5,
                        kernel_size=3,
                        padding="SAME",
                        key=key1,
                        padding_mode="CIRCULAR",
                    ),
                    eqx.nn.Conv2d(
                        in_channels=5,
                        out_channels=2,
                        kernel_size=3,
                        padding="SAME",
                        key=key2,
                        padding_mode="CIRCULAR",
                    ),
                ]
            )

        def __call__(self, x, *, key=None):
            return self.ops(x, key=key)
        
    @jax.tree_util.register_pytree_node_class
    class Operator1(Coarsener):
        @property
        def spectral_filter(self):
            return self.lr_model.filtr

        def coarsen_state(self, state):
            lr_state = self.lr_model.initialise(42)
            nk = lr_state.qh.shape[-2] // 2
            trunc = jnp.concatenate(
                [
                    state.qh[:, :nk, :nk + 1],
                    state.qh[:, -nk:, :nk + 1],
                ],
                axis=-2,
            )
            spectral_filter = self.spectral_filter
            if trunc.ndim == 3 and spectral_filter.ndim == 2:
                spectral_filter = jnp.expand_dims(spectral_filter, 0)
            filtered = trunc * spectral_filter / self.ratio**2
            return lr_state.update(qh=filtered)
    
    net = module_to_single(NNParam(key=jax.random.key(123)))
        
    learning_rate = cfg.ml.learning_rate
    optim = optax.adam(learning_rate)
    optim_state = optim.init(eqx.filter(net, eqx.is_array))
    

    # Instantiate the model from configs
    n_layers = params.pop('n_layers', 1)  # Extract n_layers, default to 1
    model = create_model(params, n_layers=n_layers)
    stepper = build_stepper(cfg_stepper, dt)
    stepped_model = SteppedModel(model=model, stepper=stepper)

    recorder = Recorder(cfg, grid=model.get_grid())
    init_state = stepped_model.initialise(params['seed'])
    coarsener = Operator1(stepped_model.model, 32)

    @functools.partial(jax.jit, static_argnames=["nsteps"])
    def generate_train_data(seed, nsteps):

        def step(carry, _x):
            next_state = stepped_model.step_model(carry)
            small_state =coarsener.coarsen_state(carry.state)
            return next_state, small_state.q

        _final_big_state, target_q = jax.lax.scan(
            step, stepped_model.initialise(seed), None, length=nsteps
        )
        return target_q

    target_q = generate_train_data(123, nsteps=100)

   # === jax.jit functionality === #
    def roll_out_with_net(init_q, net, nsteps):

        @parameterization
        def net_parameterization(state, param_aux, model):
            assert param_aux is None
            q = state.q
            # Scale states to improve stability
            # This 1e-6 is for illustration only
            q_in = (q / 1e-6).astype(jnp.float32)
            q_param = net(q.astype(jnp.float32))
            return 1e-6 * q_param.astype(q.dtype), None

        # Extrace the small model from the coarsener
        # Then wrap it in the network parameterization and stepper
        # Make sure to match time steps
        small_model = SteppedModel(
            model=ForcedModel(
                model=coarsener.lr_model,
                param_func=net_parameterization,
            ),
            stepper=stepper,
        )
        # Package our state
        # First, package it for the base model
        base_state = small_model.model.model.initialise(params['seed']
        ).update(q=init_q)
        # Next, wrap it for the parameterization and stepper
        init_state = small_model.initialize_stepper_state(
            small_model.model.initialise_param_state(base_state)
        )

        def step(carry, _x):
            next_state = small_model.step_model(carry)
            # NOTE: Be careful! We output the *old* state for the trajectory
            # Otherwise the initial step would be skipped
            return next_state, carry.state.model_state.q

        # Roll out the state
        _final_step, traj = jax.lax.scan(
            step, init_state, None, length=nsteps
        )
        return traj
    
    def compute_traj_errors(target_q, net):
        rolled_out = roll_out_with_net(
            init_q=target_q[0],
            net=net,
            nsteps=target_q.shape[0],
        )
        err = rolled_out - target_q
        return err 

    @eqx.filter_jit
    def train_batch(batch, net, optim_state):

        def loss_fn(net, batch):
            err = jax.vmap(functools.partial(compute_traj_errors, net=net))(batch)
            mse = jnp.mean(err**2)
            return mse
        
        # Compute loss value and gradients
        loss, grads = eqx.filter_value_and_grad(loss_fn)(net, batch)

        # Gradient sanity checks (finite + bounded)
        GRAD_MIN_ABS = 1e-30
        GRAD_MAX_ABS = 1e6

        def _check_grads(grads):
            leaves = jax.tree_util.tree_leaves(grads)
            for g in leaves:
                if g is None:
                    continue
                g_np = np.asarray(g)
                if not np.all(np.isfinite(g_np)):
                    raise ValueError("Non-finite gradient detected")
                abs_g = np.abs(g_np)
                if np.any(abs_g > GRAD_MAX_ABS):
                    raise ValueError("Gradient too large detected")
                if np.any((abs_g > 0) & (abs_g < GRAD_MIN_ABS)):
                    raise ValueError("Gradient too small detected")

        jax.debug.callback(_check_grads, grads)

        print("grads:", grads)
        print("optim_state:", optim_state)
        # Update the network weights
        updates, new_optim_state = optim.update(grads, optim_state, net)
        new_net = eqx.apply_updates(net, updates)
        # Return the loss, updated net, updated optimizer 
        return loss, new_net, new_optim_state

    BATCH_SIZE = 8
    BATCH_STEPS = 10


    np_rng = np.random.default_rng(seed=456)
    losses = []
    for batch_i in range(50):
        # Rudimentary shuffling in lieu of real data loader
        batch = np.stack(
            [
                target_q[start:start+BATCH_STEPS]
                for start in np_rng.integers(
                    0, target_q.shape[0] - BATCH_STEPS, size=BATCH_SIZE
                )
            ]
        )

        loss, net, optim_state = train_batch(batch, net, optim_state)
        losses.append(loss)
        if (batch_i + 1) % 5 == 0:
            print(f"Step {batch_i + 1:02}: loss={loss.item():.4E}")

    plt.plot(np.arange(len(losses)) + 1, losses)
    plt.xlabel("Step")
    plt.ylabel("Step Loss")
    plt.grid(True)


if __name__ == "__main__":
    run()