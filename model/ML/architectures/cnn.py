import jax
import jax.numpy as jnp
import equinox as eqx


class CNN(eqx.Module):
    """Configurable CNN closure."""
    layers: list

    def __init__(
        self,
        key=jax.random.PRNGKey(0),
        nlayers=3,
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        width=64,
        padding=None,
        activation="relu",
        **kwargs,
    ):
        if nlayers < 2:
            raise ValueError("nlayers must be >= 2")
        # force same-padding so the network preserves spatial dimensions
        padding = kernel_size // 2

        keys = jax.random.split(key, nlayers)
        layers = []

        if isinstance(activation, str) and activation.lower() == "tanh":
            act = eqx.nn.Lambda(jnp.tanh)
        elif isinstance(activation, str) and activation.lower() == "gelu":
            act = eqx.nn.Lambda(jax.nn.gelu)
        else:
            act = eqx.nn.Lambda(jax.nn.relu)

        for i in range(nlayers):
            in_ch = in_channels if i == 0 else width
            out_ch = out_channels if i == (nlayers - 1) else width
            layers.append(
                eqx.nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    padding=padding,
                    key=keys[i],
                    padding_mode='CIRCULAR',
                )
            )
            if i < (nlayers - 1):
                layers.append(act)

        self.layers = layers

    def __call__(self, qh):
        x = qh # add batch dim
        for layer in self.layers:
            x = layer(x)
        return x # remove batch dim