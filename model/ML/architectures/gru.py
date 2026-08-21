import jax
import jax.numpy as jnp
import equinox as eqx
import dataclasses
import model.utils.pytree as Pytree

@Pytree.register_pytree_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class GRUHiddenState:
    """Hidden param_aux for the hidden states in the GRU to be passed through
    the Forced Model.
    """
    hidden_state: jnp.ndarray


class GRUClosure(eqx.Module):
    """GRU-style closure head with a hidden state.
    """

    ml = True
    stateful = True
    embed: eqx.nn.Conv2d
    reset_gate: eqx.nn.Conv2d
    update_gate: eqx.nn.Conv2d
    candidate: eqx.nn.Conv2d
    proj: eqx.nn.Conv2d
    hidden_channels: int
    # kernel_size: int # fix: dont know why it doesnt like this
    activation: str

    def __init__(
        self,
        key=jax.random.PRNGKey(0),
        hidden_channels=32,
        kernel_size=3,
        activation=None,
        cfg=None,
        **kwargs,
    ):
        in_channels = cfg.params.nz
        out_channels = in_channels
        padding = kernel_size // 2
        keys = jax.random.split(key, 5)
        self.hidden_channels = hidden_channels

        if not isinstance(activation, str):
            raise ValueError(f"Unsupported activation: {activation}")
        activation = activation.lower()
        if activation not in {"tanh", "gelu", "relu", "elu", "leaky_relu"}:
            raise ValueError(f"Unsupported activation: {activation}")

        self.embed = eqx.nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            padding=0,
            key=keys[0],
        )
        self.reset_gate = eqx.nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            key=keys[1],
            padding_mode="CIRCULAR",
        )
        self.update_gate = eqx.nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            key=keys[2],
            padding_mode="CIRCULAR",
        )
        self.candidate = eqx.nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            key=keys[3],
            padding_mode="CIRCULAR",
        )
        self.proj = eqx.nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            key=keys[4],
        )
        self.activation = activation

    def __call__(self, qh, aux):
        """Return dq and the next hidden state."""
        x = self.embed(qh)
        h = aux.hidden_state
        r = jax.nn.sigmoid(self.reset_gate(x))
        z = jax.nn.sigmoid(self.update_gate(x))
        candidate = self.candidate(r * h + x)
        if self.activation == "tanh":
            h_tilde = jnp.tanh(candidate)
        elif self.activation == "gelu":
            h_tilde = jax.nn.gelu(candidate)
        elif self.activation == "relu":
            h_tilde = jax.nn.relu(candidate)
        elif self.activation == "elu":
            h_tilde = jax.nn.elu(candidate)
        else:
            h_tilde = jax.nn.leaky_relu(candidate)
        h_next = (1.0 - z) * h + z * h_tilde

        dq = self.proj(h_next)
        return dq, GRUHiddenState(hidden_state=h_next)

    @property
    def model_type(self):
        return 'gru'

    @property
    def hchannels(self):
        return self.hidden_channels