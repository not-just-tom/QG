import equinox as eqx
import jax
import jax.numpy as jnp


class TemporalMemoryState(eqx.Module):
    history: jnp.ndarray
    initialized: jnp.ndarray
    inner_state: object


class TemporalMemory(eqx.Module):
    """Adds a rolling field history to any channel-compatible ML closure."""

    ml = True
    stateful = True

    closure: eqx.Module
    history_encoder: eqx.nn.Conv2d
    field_channels: int = eqx.field(static=True)
    history_length: int = eqx.field(static=True)

    def __init__(
        self,
        closure,
        field_channels,
        history_length,
        key,
    ):
        if isinstance(history_length, bool) or not isinstance(history_length, int):
            raise ValueError("memory history_length must be an integer")
        if history_length < 1:
            raise ValueError("memory history_length must be at least 1")

        self.closure = closure
        self.field_channels = int(field_channels)
        self.history_length = int(history_length)
        self.history_encoder = eqx.nn.Conv2d(
            in_channels=(self.history_length + 1) * self.field_channels,
            out_channels=self.field_channels,
            kernel_size=1,
            key=key,
        )

    def __call__(self, q, aux):
        if q.shape[0] != self.field_channels:
            raise ValueError("Memory input channels do not match field_channels")
        if aux.history.shape != (
            self.history_length * self.field_channels,
            *q.shape[-2:],
        ):
            raise ValueError("Memory history shape does not match the current field")
        q = q.astype(jnp.float32)
        history = aux.history.astype(jnp.float32)
        history = jax.lax.cond(
            aux.initialized,
            lambda stored: stored,
            lambda _: jnp.concatenate([q] * self.history_length, axis=0),
            history,
        )
        encoded_input = self.history_encoder(jnp.concatenate((history, q), axis=0))

        if getattr(self.closure, "stateful", False):
            prediction, next_inner_state = self.closure(
                encoded_input,
                aux.inner_state,
            )
        else:
            prediction = self.closure(encoded_input)
            next_inner_state = aux.inner_state

        next_history = jnp.concatenate(
            (history[self.field_channels :], q),
            axis=0,
        )
        return prediction, TemporalMemoryState(
            history=next_history,
            initialized=jnp.asarray(True),
            inner_state=next_inner_state,
        )
