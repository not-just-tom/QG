import sys
import os
# Ensure the workspace root is in the path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_workspace_root = os.path.dirname(_script_dir)
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)

import functools
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import importlib
import yaml

_module_names = [
    'model.core.grid',
    'model.core.states',
    'model.core.kernel',
    'model.core.model',
    'model.core.steppers',
    'model.ML.utils.utils',
    'model.ML.utils.coarsen',
    'model.ML.forced_model',
    'model.utils.config',
]

for _mod_name in _module_names:
    if _mod_name in sys.modules:
        importlib.reload(sys.modules[_mod_name])
    else:
        __import__(_mod_name)

from model.core.model import QGM
from model.core.steppers import SteppedModel, AB3Stepper
from model.ML.forced_model import ForcedModel
from model.ML.utils.utils import parameterization
from model.ML.utils.coarsen import Coarsener
from model.utils.config import Config

jax.config.update("jax_enable_x64", True)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
CONFIG_DEFAULT_PATH = os.path.join(BASE_DIR, "config", "default.yaml")

@jax.tree_util.register_pytree_node_class
class Operator1(Coarsener):
    @property
    def spectral_filter(self):
        return self.lr_model._dealias


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
                    in_channels=1,
                    out_channels=5,
                    kernel_size=3,
                    padding="SAME",
                    key=key1,
                    padding_mode="CIRCULAR",
                ),
                eqx.nn.Lambda(jax.nn.relu),
                eqx.nn.Conv2d(
                    in_channels=5,
                    out_channels=1,
                    kernel_size=3,
                    padding="SAME",
                    key=key2,
                    padding_mode="CIRCULAR",
                ),
            ]
        )

    def __call__(self, x, *, key=None):
        return self.ops(x, key=key)
    

# === just loading in params as dict === #
with open(CONFIG_DEFAULT_PATH) as f:
    cfg_dict = yaml.safe_load(f)

params = dict(cfg_dict["params"])

# === other important variables for training === #
cfg = Config.load_config(CONFIG_DEFAULT_PATH)

dt = cfg.plotting.dt
learning_rate = cfg.ml.learning_rate
batch_size = cfg.ml.batch_size
batch_steps = cfg.ml.batch_steps


hr_model = SteppedModel(
    model=QGM(params),
    stepper=AB3Stepper(dt=dt),  # build stepper is definitely defunct now. remove at convenience. 
)

coarse_op = Operator1(hr_model.model, 32)

# Ensure that all module weights are float32
net = module_to_single(NNParam(key=jax.random.key(123)))

optim = optax.adam(learning_rate)
optim_state = optim.init(eqx.filter(net, eqx.is_array))
key = jax.random.PRNGKey(42)
@functools.partial(jax.jit, static_argnames=["num_steps"])
def generate_train_data(key, num_steps):

    def step(carry, _x):
        next_state = hr_model.step_model(carry)
        lr_state = coarse_op.coarsen_state(carry.state)
        return next_state, lr_state.q
    
    _final_big_state, target_q = jax.lax.scan(
        step, hr_model.initialise(key), None, length=num_steps
    )
    return target_q

target_q = generate_train_data(key, num_steps=100)

def roll_out_with_net(init_q, net, num_steps):
    
    @parameterization
    def net_parameterization(state, param_aux, model):
        assert param_aux is None
        q = state.q
        q_param = net(q.astype(jnp.float32))
        return q_param.astype(q.dtype), None

    # Extrace the small model from the coarsener
    # Then wrap it in the network parameterization and stepper
    # Make sure to match time steps
    # CRITICAL: Use the small dt for coarsened model - it needs fine timesteps to remain stable
    lr_model = SteppedModel(
        model=ForcedModel(
            model=coarse_op.lr_model,
            param_func=net_parameterization,
        ),
        stepper=AB3Stepper(dt=dt),
    )
    # Package our state
    # Convert init_q from physical to spectral space (State only stores qh!)
    init_qh = jnp.fft.rfftn(init_q, axes=(-2, -1), norm='ortho')
    base_state = lr_model.model.model.initialise(jax.random.PRNGKey(0)).update(qh=init_qh)
    # Next, wrap it for the parameterization and stepper
    init_state = lr_model.initialize_stepper_state(
        lr_model.model.initialise_param_state(base_state)
    )

    def step(carry, _x):
        next_state = lr_model.step_model(carry)
        return next_state, carry.state.model_state.q

    # Roll out the state
    _final_step, traj = jax.lax.scan(
        step, init_state, None, length=num_steps
    )
    return traj

def compute_traj_errors(target_q, net):
    rolled_out = roll_out_with_net(
        init_q=target_q[0],
        net=net,
        num_steps=target_q.shape[0],
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

    
    # Update the network weights
    updates, new_optim_state = optim.update(grads, optim_state, net)
    new_net = eqx.apply_updates(net, updates)
    # Return the loss, updated net, updated optimizer state
    return loss, new_net, new_optim_state

np_rng = np.random.default_rng(seed=456)
losses = []
for batch_i in range(100):
    # Rudimentary shuffling in lieu of real data loader
    batch = np.stack(
        [
            target_q[start:start+batch_steps]
            for start in np_rng.integers(
                0, target_q.shape[0] - batch_steps, size=batch_size
            )
        ]
    )
    loss, net, optim_state = train_batch(batch, net, optim_state)
    losses.append(loss)
    if (batch_i + 1) % 10 == 0:
        print(f"Step {batch_i + 1:02}: loss={loss.item():.4E}")

plt.plot(np.arange(len(losses)) + 1, losses)
plt.xlabel("Step")
plt.ylabel("Step Loss")
plt.grid(True)