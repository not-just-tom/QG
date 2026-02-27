import jax
import jax.numpy as jnp
import equinox as eqx


class ConvBlock(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d

    def __init__(self, in_ch, out_ch, key):
        k1, k2 = jax.random.split(key, 2)
        self.conv1 = eqx.nn.Conv2d(
            in_ch, out_ch, kernel_size=3, padding=1,
            padding_mode="CIRCULAR", key=k1
        )
        self.conv2 = eqx.nn.Conv2d(
            out_ch, out_ch, kernel_size=3, padding=1,
            padding_mode="CIRCULAR", key=k2
        )

    def __call__(self, x):
        x = jax.nn.relu(self.conv1(x))
        x = jax.nn.relu(self.conv2(x))
        return x


class DownBlock(eqx.Module):
    conv: ConvBlock
    pool: eqx.nn.MaxPool2d

    def __init__(self, in_ch, out_ch, key):
        k1, k2 = jax.random.split(key, 2)
        self.conv = ConvBlock(in_ch, out_ch, k1)
        self.pool = eqx.nn.MaxPool2d(kernel_size=2, stride=2)

    def __call__(self, x):
        h = self.conv(x)
        x = self.pool(h)
        return x, h   # return skip connection


class UpBlock(eqx.Module):
    up: eqx.nn.ConvTranspose2d
    conv: ConvBlock

    def __init__(self, in_ch, out_ch, key):
        k1, k2 = jax.random.split(key, 2)
        self.up = eqx.nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=2, stride=2, key=k1
        )
        self.conv = ConvBlock(in_ch, out_ch, k2)

    def __call__(self, x, skip):
        x = self.up(x)
        x = jnp.concatenate([x, skip], axis=0)  # channel concat
        x = self.conv(x)
        return x


class UNet(eqx.Module):
    downs: list
    ups: list
    bottleneck: ConvBlock
    final: eqx.nn.Conv2d

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=32,
        depth=4,
        key=jax.random.PRNGKey(0),
    ):
        keys = jax.random.split(key, 2 * depth + 2)

        # Encoder
        downs = []
        ch = in_channels
        for i in range(depth):
            out_ch = base_channels * 2**i
            downs.append(DownBlock(ch, out_ch, keys[i]))
            ch = out_ch
        self.downs = downs

        # Bottleneck
        self.bottleneck = ConvBlock(ch, ch * 2, keys[depth])
        ch = ch * 2

        # Decoder
        ups = []
        for i in reversed(range(depth)):
            out_ch = base_channels * 2**i
            ups.append(UpBlock(ch, out_ch, keys[depth + 1 + i]))
            ch = out_ch
        self.ups = ups

        # Final projection
        self.final = eqx.nn.Conv2d(
            ch, out_channels, kernel_size=1, key=keys[-1]
        )

    def __call__(self, x):
        skips = []

        # Down path
        for down in self.downs:
            x, h = down(x)
            skips.append(h)

        # Bottleneck
        x = self.bottleneck(x)

        # Up path
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)

        return self.final(x)