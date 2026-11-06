
import jax
import jax.numpy as jnp
import equinox as eqx


class SpectralConv2d(eqx.Module):
    """2D Fourier convolution with learned low-frequency complex weights."""

    weights1: jnp.ndarray
    weights2: jnp.ndarray
    in_channels: int
    out_channels: int
    xmodes: int
    ymodes: int

    def __init__(
        self,
        in_channels,
        out_channels,
        xmodes,
        ymodes,
        key,
        spectral_init_power=0.5,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.xmodes = xmodes
        self.ymodes = ymodes

        k1, k2 = jax.random.split(key)

        # Complex variance is controlled explicitly.
        scale = spectral_init_power / jnp.sqrt(
            in_channels * out_channels
        )

        def initialise(k):
            real, imag = jax.random.split(k)
            return scale * (
                jax.random.normal(
                    real, (in_channels, out_channels, ymodes, xmodes)
                )
                + 1j * jax.random.normal(
                    imag, (in_channels, out_channels, ymodes, xmodes)
                )
            )

        self.weights1 = initialise(k1)
        self.weights2 = initialise(k2)

    def __call__(self, x):
        unbatched = x.ndim == 3
        if unbatched:
            x = x[None]

        _, _, height, width = x.shape

        x_ft = jnp.fft.rfft2(x, axes=(-2, -1))
        out_ft = jnp.zeros(
            (
                x.shape[0],
                self.out_channels,
                height,
                width // 2 + 1,
            ),
            dtype=x_ft.dtype,
        )

        ym = min(self.ymodes, height// 2)
        xm = min(self.xmodes, width // 2 + 1)

        if ym > 0 and xm > 0:
            out_ft = out_ft.at[:, :, :ym, :xm].set(
                jnp.einsum(
                    "bixy,ioxy->boxy",
                    x_ft[:, :, :ym, :xm],
                    self.weights1[:, :, :ym, :xm],
                )
            )

            out_ft = out_ft.at[:, :, -ym:, :xm].set(
                jnp.einsum(
                    "bixy,ioxy->boxy",
                    x_ft[:, :, -ym:, :xm],
                    self.weights2[:, :, :ym, :xm],
                )
            )

        # Enforce Hermitian symmetry on rFFT boundary columns.
        boundary_columns = [0]
        if width % 2 == 0:
            boundary_columns.append(width // 2)

        y_partner = (-jnp.arange(height)) % height
        boundary = out_ft[..., boundary_columns]
        paired = jnp.take(boundary, y_partner, axis=-2)

        out_ft = out_ft.at[..., boundary_columns].set(
            0.5 * (boundary + jnp.conj(paired))
        )

        output = jnp.fft.irfft2(
            out_ft,
            s=(height, width),
            axes=(-2, -1),
        )

        return output[0] if unbatched else output


class FNOBlock(eqx.Module):
    """Residual spectral + local 3x3 FNO block."""

    spectral: SpectralConv2d
    local: eqx.nn.Conv2d
    spectral_scale: jnp.ndarray
    local_scale: jnp.ndarray
    residual_scale: jnp.ndarray
    norm: object
    activation: object

    def __init__(
        self,
        width,
        xmodes,
        ymodes,
        key,
        spectral_init_power=0.5,
        residual_init=0.1,
        use_layer_norm=False,
        activation="gelu",
    ):
        ks = jax.random.split(key, 3)

        self.spectral = SpectralConv2d(
            width,
            width,
            xmodes,
            ymodes,
            ks[0],
            spectral_init_power=spectral_init_power,
        )

        # CHANGED: kernel_size=3, padding=1, padding_mode="CIRCULAR"
        # This keeps (Height, Width) perfectly identical to the input.
        self.local = eqx.nn.Conv2d(
            width,
            width,
            kernel_size=3,
            padding=1, 
            padding_mode="CIRCULAR",
            key=ks[1],
        )

        self.spectral_scale = jnp.asarray(1.0)
        self.local_scale = jnp.asarray(1.0)
        self.residual_scale = jnp.asarray(residual_init)

        self.norm = (
            eqx.nn.LayerNorm(width)
            if use_layer_norm
            else eqx.nn.Identity()
        )

        activations = {
            "gelu": jax.nn.gelu,
            "relu": jax.nn.relu,
            "tanh": jnp.tanh,
            "silu": jax.nn.silu,
        }
        self.activation = activations[activation]

    def __call__(self, x):
        spectral = self.spectral(x)
        local = self.local(x)

        # Now shapes match exactly, broadcasting works seamlessly
        update = (
            self.spectral_scale * spectral
            + self.local_scale * local
        )

        output = x + self.residual_scale * update

        if not isinstance(self.norm, eqx.nn.Identity):
            output = jax.vmap(
                jax.vmap(self.norm, in_axes=1, out_axes=1),
                in_axes=1,
                out_axes=1,
            )(output)

        return self.activation(output)



class OriginalFNO2d(eqx.Module):
    """FNO mapping a temporal field window to a closure increment."""

    input_proj: eqx.nn.Conv2d
    blocks: tuple
    proj1: eqx.nn.Conv2d
    proj2: eqx.nn.Conv2d
    field_channels: int

    def __init__(
        self,
        field_channels,
        xmodes=12,
        ymodes=12,
        width=20,
        projection_width=128,
        depth=4,
        spectral_init_power=0.5,
        residual_init=0.1,
        use_layer_norm=False,
        activation="gelu",
        key=jax.random.PRNGKey(0),
    ):
        self.field_channels = int(field_channels)

        if depth < 1:
            raise ValueError("depth must be at least 1")

        keys = jax.random.split(key, depth + 3)

        self.input_proj = eqx.nn.Conv2d(
            self.field_channels,
            width,
            kernel_size=1,
            key=keys[0],
            padding_mode="CIRCULAR",
        )

        self.blocks = tuple(
            FNOBlock(
                width=width,
                xmodes=xmodes,
                ymodes=ymodes,
                key=keys[i + 1],
                spectral_init_power=spectral_init_power,
                residual_init=residual_init,
                use_layer_norm=use_layer_norm,
                activation=activation,
            )
            for i in range(depth)
        )

        self.proj1 = eqx.nn.Conv2d(
            width,
            projection_width,
            kernel_size=1,
            key=keys[-2],
            padding_mode="CIRCULAR",
        )

        self.proj2 = eqx.nn.Conv2d(
            projection_width,
            self.field_channels,
            kernel_size=1,
            key=keys[-1],
            padding_mode="CIRCULAR",
        )

    def __call__(self, q):
        x = self.input_proj(q)

        for block in self.blocks:
            x = block(x)

        x = jax.nn.gelu(self.proj1(x))
        return self.proj2(x)


class FNO(eqx.Module):
    """Stateless FNO closure; temporal memory is configured separately."""

    fno: OriginalFNO2d
    field_channels: int

    def __init__(
        self,
        cfg,
        key=jax.random.PRNGKey(0),
        xmodes=12,
        ymodes=12,
        width=20,
        projection_width=128,
        depth=4,
        spectral_init_power=0.5,
        residual_init=0.1,
        use_layer_norm=False,
        activation="gelu",
        **kwargs,
    ):
        self.field_channels = int(cfg.params.nz)

        self.fno = OriginalFNO2d(
            field_channels=self.field_channels,
            xmodes=xmodes,
            ymodes=ymodes,
            width=width,
            projection_width=projection_width,
            depth=depth,
            spectral_init_power=spectral_init_power,
            residual_init=residual_init,
            use_layer_norm=use_layer_norm,
            activation=activation,
            key=key,
        )

    def __call__(self, q):
        return self.fno(q.astype(jnp.float32))

    @property
    def model_type(self):
        return "fno"
