from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

from model.ML.architectures.build_model import (
    ClosureAdapter,
    build_closure,
    init_latent_state,
)
from model.ML.architectures.memory import TemporalMemory, TemporalMemoryState


def make_config(model_type, memory_enabled):
    architecture_configs = {
        "cnn": {
            "nlayers": 2,
            "kernel_size": 3,
            "width": 4,
            "activation": "gelu",
        },
        "fno": {
            "width": 4,
            "xmodes": 2,
            "ymodes": 2,
            "projection_width": 4,
            "depth": 1,
        },
        "gru": {
            "hidden_channels": 2,
            "kernel_size": 3,
            "activation": "gelu",
        },
        "rno": {
            "width": 4,
            "xmodes": 2,
            "ymodes": 2,
            "projection_width": 4,
            "depth": 1,
            "hidden_channels": 2,
            "activation": "gelu",
        },
    }
    return OmegaConf.create(
        {
            "params": {"nz": 1, "seed": 1},
            "ml": {
                "model_type": model_type,
                "memory": {"enabled": memory_enabled, "history_length": 2},
            },
            "architectures": {
                model_type: architecture_configs[model_type],
            },
        }
    )


def call_through_adapter(closure, q, latent_state):
    params, static = eqx.partition(closure, eqx.is_array)
    adapter = ClosureAdapter(
        closure_static=static,
        trajs=jnp.stack((q, q)),
        closure_scale=0.2,
        dt=1.0,
    )
    return adapter.predict(
        state=SimpleNamespace(q=q),
        closure_params=params,
        latent_state=latent_state,
        model=None,
    )


def test_fno_without_memory_is_called_statelessly():
    q = jnp.ones((1, 8, 8), dtype=jnp.float32)
    closure = build_closure(make_config("fno", memory_enabled=False))

    prediction, next_state = call_through_adapter(closure, q, None)

    assert prediction.shape == q.shape
    assert next_state is None


def test_memory_wraps_stateless_and_recurrent_closures():
    q = jnp.arange(64, dtype=jnp.float32).reshape((1, 8, 8))
    normalized_q = (q - jnp.mean(q)) / (jnp.std(q) + 1e-6)
    for model_type in ("cnn", "fno", "gru", "rno"):
        closure = build_closure(make_config(model_type, memory_enabled=True))
        assert isinstance(closure, TemporalMemory)

        init_state = init_latent_state(closure)
        latent_state = init_state(SimpleNamespace(q=q), None, None)
        assert latent_state.history.shape == (2, 8, 8)
        assert not bool(latent_state.initialized)

        prediction, next_state = call_through_adapter(closure, q, latent_state)

        assert prediction.shape == q.shape
        assert isinstance(next_state, TemporalMemoryState)
        assert next_state.history.shape == latent_state.history.shape
        assert bool(next_state.initialized)
        assert jnp.allclose(
            next_state.history,
            jnp.concatenate((normalized_q, normalized_q), axis=0),
        )

        def step(state, _):
            output, next_state = closure(q, state)
            return next_state, output

        _, predictions = jax.lax.scan(step, latent_state, None, length=2)
        assert predictions.shape == (2,) + q.shape
