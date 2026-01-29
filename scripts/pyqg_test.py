import functools
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import matplotlib.pyplot as plt
from model.core.model import create_model
from model.core.steppers import SteppedModel, build_stepper
from model.ML.forced_model import ForcedModel
from model.ML.utils.utils import parameterization
import inspect
import abc

def model_to_args(model):
    return {
        arg: getattr(model, arg) for arg in inspect.signature(type(model)).parameters
    }

def coarsen_model(big_model, small_nx):
    if big_model.nx != big_model.ny:
        raise ValueError("coarsening tested only for square shapes")
    if small_nx >= big_model.nx:
        raise ValueError(
            f"coarsening output is not strictly smaller (got {big_model.nx} to {small_nx})"
        )
    if small_nx % 2 != 0:
        raise ValueError(f"coarsening output should be even-valued, requested {small_nx}")
    model_args = model_to_args(big_model)
    model_args["nx"] = small_nx
    model_args["ny"] = small_nx
    return type(big_model)(**model_args)


class SpectralCoarsener(abc.ABC):
    def __init__(self, big_model, small_nx):
        self.big_model = big_model
        self.small_nx = small_nx

    @property
    def small_model(self):
        return coarsen_model(self.big_model, self.small_nx)

    @property
    def ratio(self):
        return self.big_model.nx / self.small_nx

    def coarsen_state(self, state):
        if (
            jax.eval_shape(lambda s: s.q, state).shape
            != (self.big_model.nz, self.big_model.ny, self.big_model.nx)
        ):
            raise ValueError(f"incorrect input size {state.qh.shape}")
        out_state = self.small_model.create_initial_state(
            jax.random.key(0)
        )
        nk = out_state.qh.shape[-2] // 2
        trunc = jnp.concatenate(
            [
                state.qh[:, :nk, :nk + 1],
                state.qh[:, -nk:, :nk + 1],
            ],
            axis=-2,
        )
        filtered = trunc * self.spectral_filter / self.ratio**2
        return out_state.update(qh=filtered)

    def compute_q_total_forcing(self, state):
        coarsened_deriv = self.coarsen_state(self.big_model.get_updates(state))
        small_deriv = self.small_model.get_updates(self.coarsen_state(state))
        return coarsened_deriv.q - small_deriv.q

    @property
    @abc.abstractmethod
    def spectral_filter(self):
        pass

    def tree_flatten(self):
        return [self.big_model], self.small_nx

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(big_model=children[0], small_nx=aux_data)


@jax.tree_util.register_pytree_node_class
class Operator1(SpectralCoarsener):
    @property
    def spectral_filter(self):
        return self.small_model.filtr


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
                eqx.nn.Lambda(jax.nn.relu),
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

DT = 36.0
DT_GENERATE = 0.0001  # Very small dt for beta=10.0 regime (Rossby wave timescale ~ 0.016)
LEARNING_RATE = 5e-4

params = {
    "nx": 128,
    "ny": 128,
    "L": 2 * jnp.pi,
    "W": 2 * jnp.pi,
    "rek": 5.787e-7,
    "filterfac": 23.6,
    "beta": 10.0,
    "rd": 15000.0,
    "delta": 0.25,
    "H1": 500.0,
    "U1": 0.025,
    "U2": 0.0,
}

big_model = SteppedModel(
    model=create_model(params, n_layers=2),
    stepper=build_stepper("AB3Stepper", dt=DT_GENERATE),  # Use small dt for stable generation
)

coarse_op = Operator1(big_model.model, 32)

# Ensure that all module weights are float32
net = module_to_single(NNParam(key=jax.random.key(123)))

optim = optax.adam(LEARNING_RATE)
optim_state = optim.init(eqx.filter(net, eqx.is_array))

@functools.partial(jax.jit, static_argnames=["num_steps"])
def generate_train_data(seed, num_steps):

    def step(carry, _x):
        next_state = big_model.step_model(carry)
        small_state = coarse_op.coarsen_state(carry.state)
        return next_state, small_state.q

    # With dt=0.0005, we need 7,200,000 steps to reach 1 hour
    # To match the rollout timescale in training (100 steps * dt=3600 = 360,000 seconds)
    # We need 360,000 / 0.0005 = 720,000,000 steps which is too many
    # Instead: 100 steps at dt=0.0005 = 0.05 seconds (physical time)
    # This gives us 100 coarsened snapshots to learn from
    _final_big_state, target_q = jax.lax.scan(
        step, big_model.initialise(seed), None, length=num_steps
    )
    return target_q

target_q = generate_train_data(123, num_steps=100)
print(f"Generated target_q shape: {target_q.shape}")
print(f"target_q finite check: {jnp.all(jnp.isfinite(target_q))}")
print(f"target_q range: [{jnp.min(target_q):.4e}, {jnp.max(target_q):.4e}]")
print(f"target_q mean: {jnp.mean(target_q):.4e}, std: {jnp.std(target_q):.4e}")

def roll_out_with_net(init_q, net, num_steps):
    
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
    # CRITICAL: Use the small dt for coarsened model - it needs fine timesteps to remain stable
    small_model = SteppedModel(
        model=ForcedModel(
            model=coarse_op.small_model,
            param_func=net_parameterization,
        ),
        stepper=build_stepper("AB3Stepper", dt=DT_GENERATE),
    )
    # Package our state
    # First, package it for the base model
    base_state = small_model.model.model.initialise(0).update(q=init_q)
    # Next, wrap it for the parameterization and stepper
    init_state = small_model.initialize_stepper_state(
        small_model.model.initialise_param_state(base_state)
    )

    def step(carry, _x):
        next_state = small_model.step_model(carry)
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

BATCH_SIZE = 8
BATCH_STEPS = 10
assert BATCH_STEPS >= 2

np_rng = np.random.default_rng(seed=456)
losses = []
for batch_i in range(100):
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
    if (batch_i + 1) % 10 == 0:
        print(f"Step {batch_i + 1:02}: loss={loss.item():.4E}")

plt.plot(np.arange(len(losses)) + 1, losses)
plt.xlabel("Step")
plt.ylabel("Step Loss")
plt.grid(True)