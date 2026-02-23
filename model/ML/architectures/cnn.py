import jax
import jax.numpy as jnp
import equinox as eqx


class CNN(eqx.Module):
    """In the works"""
    layers: list

    def __init__(self, seed, nlayers, in_channels, out_channels, kernel_size, width, padding): # maybe making this a params dict later
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, nlayers+1)
        for i in range(nlayers):
            if i == 0:
                in_ch = in_channels
            else:
                in_ch = width
            if i == nlayers-1:
                out_ch = out_channels
            else:
                out_ch = width
            layer = eqx.nn.Conv2d(in_ch, out_ch, kernel_size, padding, key=keys[i], padding_mode="CIRCULAR")
            relu = eqx.nn.Lambda(jax.nn.relu)
            self.layers.append(layer, relu)
        self.layers.pop() # remove final relu

    def __call__(self, qh):
        x = qh # add batch dim
        for layer in self.layers:
            x = layer(x)
        return x # remove batch dim