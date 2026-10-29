import jax
import jax.numpy as jnp
import equinox as eqx
import dataclasses
import model.utils.pytree as Pytree

_ACTIVATIONS = {
    "tanh": jnp.tanh,
    "gelu": jax.nn.gelu,
    "relu": jax.nn.relu,
    "elu": jax.nn.elu,
    "leaky_relu": jax.nn.leaky_relu,
}


def _check_activation(activation):
    if not isinstance(activation, str):
        raise ValueError(f"Unsupported activation: {activation}")
    activation = activation.lower()
    if activation not in _ACTIVATIONS:
        raise ValueError(f"Unsupported activation: {activation}")
    return activation


@Pytree.register_pytree_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class GANState:
    """PRNG key carried through the Forced Model so the generator can draw
    fresh latent noise at every call while staying functionally pure.
    """
    key: jnp.ndarray


class Generator(eqx.Module):
    """Convolutional generator: (qh, latent noise) -> dq."""

    embed: eqx.nn.Conv2d
    hidden: eqx.nn.Conv2d
    proj: eqx.nn.Conv2d
    noise_channels: int
    activation: str

    def __init__(self, key, in_channels, hidden_channels, noise_channels,
                 kernel_size, activation):
        padding = kernel_size // 2
        k0, k1, k2 = jax.random.split(key, 3)
        self.noise_channels = noise_channels
        self.activation = activation
        # qh and noise are concatenated along the channel axis
        self.embed = eqx.nn.Conv2d(
            in_channels + noise_channels, hidden_channels, kernel_size,
            padding=padding, padding_mode="CIRCULAR", key=k0,
        )
        self.hidden = eqx.nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size,
            padding=padding, padding_mode="CIRCULAR", key=k1,
        )
        self.proj = eqx.nn.Conv2d(
            hidden_channels, in_channels, 1, padding=0, key=k2,
        )

    def __call__(self, qh, noise):
        act = _ACTIVATIONS[self.activation]
        x = jnp.concatenate([qh, noise], axis=0)
        x = act(self.embed(x))
        x = act(self.hidden(x))
        return self.proj(x)


class Discriminator(eqx.Module):
    """PatchGAN-style critic: (qh, dq) -> scalar logit (mean over patches)."""

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    out: eqx.nn.Conv2d
    activation: str

    def __init__(self, key, in_channels, hidden_channels, kernel_size,
                 activation):
        padding = kernel_size // 2
        k0, k1, k2 = jax.random.split(key, 3)
        self.activation = activation
        # conditioned on qh, so input is [qh, dq]
        self.conv1 = eqx.nn.Conv2d(
            2 * in_channels, hidden_channels, kernel_size,
            padding=padding, padding_mode="CIRCULAR", key=k0,
        )
        self.conv2 = eqx.nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size,
            padding=padding, padding_mode="CIRCULAR", key=k1,
        )
        self.out = eqx.nn.Conv2d(
            hidden_channels, 1, 1, padding=0, key=k2,
        )

    def __call__(self, qh, dq):
        act = jax.nn.leaky_relu if self.activation == "leaky_relu" else _ACTIVATIONS[self.activation]
        x = jnp.concatenate([qh, dq], axis=0)
        x = act(self.conv1(x))
        x = act(self.conv2(x))
        return jnp.mean(self.out(x))  # scalar logit


class GANClosure(eqx.Module):
    """Conditional GAN closure. The generator is the closure used at
    inference; the discriminator is only used during training.
    """

    ml = True
    stateful = True
    generator: Generator
    discriminator: Discriminator
    hidden_channels: int
    noise_channels: int
    activation: str

    def __init__(
        self,
        key=jax.random.PRNGKey(0),
        hidden_channels=32,
        noise_channels=4,
        kernel_size=3,
        activation=None,
        cfg=None,
        **kwargs,
    ):
        in_channels = cfg.params.nz
        activation = _check_activation(activation)
        gkey, dkey = jax.random.split(key)

        self.hidden_channels = hidden_channels
        self.noise_channels = noise_channels
        self.activation = activation
        self.generator = Generator(
            gkey, in_channels, hidden_channels, noise_channels,
            kernel_size, activation,
        )
        self.discriminator = Discriminator(
            dkey, in_channels, hidden_channels, kernel_size, activation,
        )

    def __call__(self, qh, aux):
        """Return a sampled dq and the next PRNG state."""
        key, subkey = jax.random.split(aux.key)
        noise = jax.random.normal(
            subkey, (self.noise_channels,) + qh.shape[-2:], dtype=qh.dtype
        )
        dq = self.generator(qh, noise)
        return dq, GANState(key=key)

    @property
    def model_type(self):
        return 'gan'

    @property
    def hchannels(self):
        return self.hidden_channels


# ---------------------------------------------------------------------------
# Training losses (non-saturating GAN + optional L1 regression to the target)
# ---------------------------------------------------------------------------

def discriminator_loss(disc, qh, dq_true, dq_fake):
    real = disc(qh, dq_true)
    fake = disc(qh, jax.lax.stop_gradient(dq_fake))
    return jax.nn.softplus(-real) + jax.nn.softplus(fake)


def generator_loss(disc, qh, dq_true, dq_fake, l1_weight=100.0):
    adv = jax.nn.softplus(-disc(qh, dq_fake))
    l1 = jnp.mean(jnp.abs(dq_fake - dq_true))
    return adv + l1_weight * l1