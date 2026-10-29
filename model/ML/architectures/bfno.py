import jax
import jax.numpy as jnp
import equinox as eqx

from model.ML.architectures.fno import SpectralConv2d


class FNOBlock(eqx.Module):
    spectral: SpectralConv2d
    local: eqx.nn.Conv2d
    activation: eqx.nn.Lambda

    def __init__(self, width, modes1, modes2, key):
        spectral_key, local_key = jax.random.split(key)
        self.spectral = SpectralConv2d(
            width,
            width,
            xmodes=modes2,
            ymodes=modes1,
            key=spectral_key,
        )
        self.local = eqx.nn.Conv2d(
            width,
            width,
            kernel_size=1,
            key=local_key,
            padding_mode="CIRCULAR",
        )
        self.activation = eqx.nn.Lambda(jax.nn.gelu)

    def __call__(self, x):
        return self.activation(self.spectral(x) + self.local(x))


class BFNO(eqx.Module):
    """Barebones 2D Fourier neural operator for a single QG closure step."""

    ml = True

    field_channels: int = eqx.field(static=True)
    fc0: eqx.nn.Conv2d
    fno_blocks: tuple
    fc1: eqx.nn.Conv2d
    fc2: eqx.nn.Conv2d

    def __init__(
        self,
        cfg=None,
        in_channels=None,
        out_channels=None,
        width=32,
        modes1=12,
        modes2=12,
        layers=4,
        projection_width=128,
        key=jax.random.PRNGKey(0),
        **_kwargs,
    ):
        if cfg is not None:
            field_channels = int(cfg.params.nz)
        elif in_channels is not None:
            field_channels = int(in_channels)
        elif out_channels is not None:
            field_channels = int(out_channels)
        else:
            field_channels = 1

        if in_channels is not None and int(in_channels) != field_channels:
            raise ValueError(
                f"BFNO input channels must match cfg.params.nz "
                f"({field_channels}), got {in_channels}"
            )
        if out_channels is not None and int(out_channels) != field_channels:
            raise ValueError(
                f"BFNO output channels must match cfg.params.nz "
                f"({field_channels}), got {out_channels}"
            )
        if min(field_channels, width, layers, projection_width) <= 0:
            raise ValueError(
                "field_channels, width, layers, and projection_width must be positive"
            )
        if min(modes1, modes2) <= 0:
            raise ValueError("modes1 and modes2 must be positive")

        self.field_channels = field_channels
        keys = jax.random.split(key, layers + 3)
        self.fc0 = eqx.nn.Conv2d(
            field_channels,
            width,
            kernel_size=1,
            key=keys[0],
            padding_mode="CIRCULAR",
        )
        self.fno_blocks = tuple(
            FNOBlock(width, modes1, modes2, keys[i + 1])
            for i in range(layers)
        )
        self.fc1 = eqx.nn.Conv2d(
            width,
            projection_width,
            kernel_size=1,
            key=keys[-2],
            padding_mode="CIRCULAR",
        )
        self.fc2 = eqx.nn.Conv2d(
            projection_width,
            field_channels,
            kernel_size=1,
            key=keys[-1],
            padding_mode="CIRCULAR",
        )

    def __call__(self, q):
        if q.ndim != 3 or q.shape[0] != self.field_channels:
            raise ValueError(
                f"BFNO expects ({self.field_channels}, H, W) input, got {q.shape}"
            )

        x = q.astype(jnp.float32)
        x = self.fc0(x)
        for block in self.fno_blocks:
            x = block(x)
        x = jax.nn.gelu(self.fc1(x))
        return self.fc2(x).astype(q.dtype)

    @property
    def model_type(self):
        return "bfno"
