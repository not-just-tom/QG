import jax
import jax.numpy as jnp
import equinox as eqx


class SpatialBatchNorm(eqx.Module):
    """Channel-wise normalization over the spatial domain of one field."""

    weight: jax.Array
    bias: jax.Array
    eps: float

    def __init__(self, channels, eps=1e-5):
        self.weight = jnp.ones((channels,))
        self.bias = jnp.zeros((channels,))
        self.eps = eps

    def __call__(self, x):
        mean = jnp.mean(x, axis=(-2, -1), keepdims=True)
        variance = jnp.mean((x - mean) ** 2, axis=(-2, -1), keepdims=True)
        x = (x - mean) * jax.lax.rsqrt(variance + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class CNN(eqx.Module):
    """Eight-layer fully convolutional closure."""

    layers: list
    constrain_output_mean: bool
    ml = True

    def __init__(
        self,
        key=jax.random.PRNGKey(0),
        nlayers=8,
        kernel_size=3,
        activation="relu",
        constrain_output_mean=True,
        cfg=None,
        **kwargs,
    ):
        if nlayers != 8:
            raise ValueError("nlayers must be 8")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        in_channels = cfg.params.nz if cfg is not None else 1
        out_channels = in_channels
        widths = (128, 64, 32, 32, 32, 32, 32, out_channels)
        keys = jax.random.split(key, len(widths))
        layers = []

        if not isinstance(activation, str) or activation.lower() != "relu":
            raise ValueError("CNN activation must be 'relu'")
        act = eqx.nn.Lambda(jax.nn.relu)

        for i, out_ch in enumerate(widths):
            in_ch = in_channels if i == 0 else widths[i - 1]
            layer_kernel_size = 5 if i < 2 else kernel_size
            layers.append(
                eqx.nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=layer_kernel_size,
                    padding=layer_kernel_size // 2,
                    key=keys[i],
                    padding_mode="CIRCULAR",
                )
            )
            if i < len(widths) - 1:
                layers.append(SpatialBatchNorm(out_ch))
                layers.append(act)

        self.layers = layers
        self.constrain_output_mean = constrain_output_mean

    def __call__(self, qh):
        x = qh
        for layer in self.layers:
            x = layer(x)
        if self.constrain_output_mean:
            x = x - jnp.mean(x, axis=(-2, -1), keepdims=True)
        return x

    @property
    def model_type(self):
        return "cnn"
