import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp

from model.ML.architectures.fno import FNO as FourierOperator
from model.ML.architectures.gru import GRUHiddenState


class RNO(eqx.Module):
    """Recurrent Neural Operator.

    The FNO predicts a spatial dynamics update. A GRU-style recurrent
    mechanism then updates a hidden state and produces the final closure.
    """

    ml = True
    stateful = True

    operator: FourierOperator
    embed: eqx.nn.Conv2d
    reset_gate: eqx.nn.Conv2d
    update_gate: eqx.nn.Conv2d
    candidate: eqx.nn.Conv2d
    proj: eqx.nn.Conv2d
    hidden_channels: int
    activation: str

    def __init__(
        self,
        key=jax.random.PRNGKey(0),
        width=32,
        xmodes=16,
        ymodes=16,
        projection_width=128,
        depth=4,
        hidden_channels=32,
        activation="gelu",
        cfg=None,
        **kwargs,
    ):
        if cfg is None:
            raise ValueError("cfg is required to construct RNO")

        in_channels = int(cfg.params.nz)
        out_channels = in_channels

        if not isinstance(activation, str):
            raise ValueError(f"Unsupported activation: {activation}")

        activation = activation.lower()
        if activation not in {"tanh", "gelu", "relu", "elu", "leaky_relu"}:
            raise ValueError(f"Unsupported activation: {activation}")

        self.hidden_channels = int(hidden_channels)
        self.activation = activation

        keys = jax.random.split(key, 6)

        # Spatial neural operator producing the instantaneous dynamics update.
        self.operator = FourierOperator(
            width=width,
            xmodes=xmodes,
            ymodes=ymodes,
            projection_width=projection_width,
            depth=depth,
            activation=activation,
            key=keys[0],
            cfg=cfg,
        )

        # Combine the current state and the FNO update.
        self.embed = eqx.nn.Conv2d(
            in_channels=2 * in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            key=keys[1],
        )

        self.reset_gate = eqx.nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            padding_mode="CIRCULAR",
            key=keys[2],
        )

        self.update_gate = eqx.nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            padding_mode="CIRCULAR",
            key=keys[3],
        )

        self.candidate = eqx.nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            padding_mode="CIRCULAR",
            key=keys[4],
        )

        self.proj = eqx.nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            key=keys[5],
        )

    def _activate(self, x):
        if self.activation == "tanh":
            return jnp.tanh(x)
        if self.activation == "gelu":
            return jax.nn.gelu(x)
        if self.activation == "relu":
            return jax.nn.relu(x)
        if self.activation == "elu":
            return jax.nn.elu(x)
        return jax.nn.leaky_relu(x)

    def __call__(self, qh, aux):
        """Compute one recurrent neural-operator dynamics update.

        Parameters
        ----------
        qh:
            Current physical-space state, shaped ``(nz, ny, nx)``.
        aux:
            GRUHiddenState containing the recurrent hidden state.
        """
        qh = qh.astype(jnp.float32)

        # FNO spatial operator: approximates the instantaneous dynamics.
        operator_update = self.operator(qh)

        # Encode current state and spatial update together.
        x = self.embed(jnp.concatenate([qh, operator_update], axis=0))
        h = aux.hidden_state

        reset = jax.nn.sigmoid(self.reset_gate(x))
        update = jax.nn.sigmoid(self.update_gate(x))

        candidate = self.candidate(reset * h + x)
        h_tilde = self._activate(candidate)

        # GRU recurrent update.
        h_next = (1.0 - update) * h + update * h_tilde

        # Return the learned dynamics update, not the integrated state.
        dq = self.proj(h_next)

        return dq, GRUHiddenState(hidden_state=h_next)

    @property
    def model_type(self):
        return "rno"

    @property
    def hchannels(self):
        return self.hidden_channels