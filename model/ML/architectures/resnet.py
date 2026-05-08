import jax
import jax.numpy as jnp
import equinox as eqx
from typing import List, Callable, Any


class ResNet(eqx.Module):
    """Configurable ResNet closure. Defaults based on Maddison (2026).
    See https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024MS004883 for details."""

    # Equinox requires attributes to be declared as dataclass fields
    convs: List[Any]
    projs: List[Any]
    alphas: jnp.ndarray
    activation: Any
    alpha_input: jnp.ndarray
    alpha_output: jnp.ndarray

    def __init__(
        self,
        key=jax.random.PRNGKey(0),
        nlayers=7,
        in_channels=1,
        out_channels=1,
        kernel_size=5,
        width=64,
        activation="elu",
        cfg=None,
        **kwargs,
    ):
        if nlayers < 1:
            raise ValueError("nlayers must be >= 1")

        padding = kernel_size // 2
        # split keys for convs and projections
        keys = jax.random.split(key, nlayers + (nlayers + 1))
        conv_keys = keys[:nlayers]
        proj_keys = keys[nlayers:]
        beta = cfg.params.beta
        Lx = cfg.params.Lx
        Ly = cfg.params.Lx
        tau0=1 #this is wrong and is a test for the normalization factors to be computed correctly based on the parameters of the system. The value of tau0 should not affect the training dynamics as it is absorbed into the normalization factors, but it should be set to a physically meaningful value for interpretability. In the future, we may want to make this a configurable parameter as well.
        rho0=1 #same comment as tau0

        # Compute normalization factors: S_theta(q(h?)) = alpha_out * F(alpha_in * q(h?))
        ai = 1.0 / (abs(float(beta)) * float(Lx))
        ao = abs(float(tau0)) * float(jnp.pi) / (float(rho0) * float(Ly) * float(Lx))


        # activation function wrapper
        if isinstance(activation, str) and activation.lower() == "tanh":
            act = eqx.nn.Lambda(jnp.tanh)
        elif isinstance(activation, str) and activation.lower() == "gelu":
            act = eqx.nn.Lambda(jax.nn.gelu)
        elif isinstance(activation, str) and activation.lower() == "relu":
            act = eqx.nn.Lambda(jax.nn.relu)
        elif isinstance(activation, str) and activation.lower() == "elu":
            act = eqx.nn.Lambda(jax.nn.elu)
        elif isinstance(activation, str) and activation.lower() == "leaky_relu":
            act = eqx.nn.Lambda(jax.nn.leaky_relu)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        convs = []
        for i in range(nlayers):
            in_ch = in_channels if i == 0 else width
            out_ch = width
            convs.append(
                eqx.nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    padding=padding,
                    key=conv_keys[i],
                    padding_mode='CIRCULAR',
                )
            )

        # projection convs: one for input, one for each layer output
        n_skips = nlayers + 1
        projs = []
        for i in range(n_skips):
            proj_in = in_channels if i == 0 else width
            projs.append(
                eqx.nn.Conv2d(
                    in_channels=proj_in,
                    out_channels=out_channels,
                    kernel_size=1,
                    padding=0,
                    key=proj_keys[i],
                )
            )

        # trainable scalar multipliers for each skip connection
        # Biases are intentionally omitted: a constant additive term per step
        # accumulates as a linear drift in q over the rollout and destabilises.
        alphas = 1e-2*jnp.ones((n_skips,))

        self.convs = convs
        self.projs = projs
        self.alphas = alphas
        self.activation = act

        # store as jax arrays for safe broadcasting in jitted code
        self.alpha_input = jnp.array(ai, dtype=jnp.float32)
        self.alpha_output = jnp.array(ao, dtype=jnp.float32)

    def __call__(self, qh):
        # qh expected shape: (batch, channels, H, W) or (channels, H, W)
        x_in = qh

        # Apply input normalization factor before feeding into the NN
        x_in_scaled = self.alpha_input * x_in

        # collect outputs after each conv+activation
        outs = []
        x = x_in_scaled
        for conv in self.convs:
            x = conv(x)
            x = self.activation(x)
            outs.append(x)


        total = 0
        # Use the scaled input for the input projection as well so that
        # F_theta sees the normalized input consistently across all skips.
        for i, feature in enumerate([x_in_scaled] + outs):
            proj = self.projs[i](feature)
            total = total + self.alphas[i] * proj

        # Apply output normalization factor
        return self.alpha_output * total