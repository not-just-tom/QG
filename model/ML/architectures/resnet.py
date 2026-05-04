import jax
import jax.numpy as jnp
import equinox as eqx


class ResNet(eqx.Module):
    """Configurable ResNet closure. Defaults based on Maddison (2026).
    See https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024MS004883 for details."""

    def __init__(
        self,
        key=jax.random.PRNGKey(0),
        nlayers=7,
        in_channels=1,
        out_channels=1,
        kernel_size=5,
        width=64,
        activation="elu",
        **kwargs,
    ):
        if nlayers < 1:
            raise ValueError("nlayers must be >= 1")

        padding = kernel_size // 2
        # split keys for convs and projections
        keys = jax.random.split(key, nlayers + (nlayers + 1))
        conv_keys = keys[:nlayers]
        proj_keys = keys[nlayers:]

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
                    padding_mode='zeros',
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

        # trainable scalar multipliers and scalar biases for each skip
        alphas = jnp.ones((n_skips,))
        biases = jnp.zeros((n_skips,))

        self.convs = convs
        self.projs = projs
        self.alphas = alphas
        self.biases = biases
        self.activation = act

    def __call__(self, qh):
        # qh expected shape: (batch, channels, H, W) or (channels, H, W)
        x_in = qh

        # collect outputs after each conv+activation
        outs = []
        x = x_in
        for conv in self.convs:
            x = conv(x)
            x = self.activation(x)
            outs.append(x)


        total = 0
        for i, feature in enumerate([x_in] + outs):
            proj = self.projs[i](feature)
            # broadcast scalar alpha and bias across channels/spatial dims
            total = total + self.alphas[i] * proj + self.biases[i]

        return total