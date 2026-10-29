import jax
import jax.numpy as jnp
import equinox as eqx
import warnings

class SpectralConv2d(eqx.Module):
    """
    2D Fourier layer that multiplies learned complex weights on low-frequency modes.
    """
    in_channels: int
    out_channels: int
    xmodes: int  # modes in x-direction
    ymodes: int  # modes in y-direction
    weights1: jnp.ndarray  # complex weights for positive y frequencies
    weights2: jnp.ndarray  # complex weights for negative y frequencies
    gate: jnp.ndarray  # learnable scalar multiplying the whole spectral branch
    kref: float = eqx.field(static=True)
    kpower: float = eqx.field(static=True)

    def __init__(
        self,
        in_channels,
        out_channels,
        xmodes,
        ymodes,
        key=None,
        spectral_init_power=0.5,
        gate_init=1.0,
        kref=4.0,
        kpower=0.0,
    ):
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.xmodes = int(xmodes)
        self.ymodes = int(ymodes)
        if self.xmodes <= 0 or self.ymodes <= 0:
            raise ValueError("xmodes and ymodes must be positive")
        # Learnable scalar branch gate (init < 1 lets the net start as its local
        # path and bring the spectral branch in only where it helps).
        self.gate = jnp.asarray(gate_init, dtype=jnp.float32)
        # Optional fixed isotropic gain 1 + (|k|/kref)^kpower on the output
        # coefficients (kpower=0 disables). Preconditions k^p-type operators.
        self.kref = float(kref)
        self.kpower = float(kpower)
        
        if key is None:
            key = jax.random.PRNGKey(0)
        
        k1, k2, k3, k4 = jax.random.split(key, 4)
        # A complex weight has two independently initialized components. This
        # base scale gives E[|w|^2] = 1 / in_channels before mode reweighting.
        scale = 1.0 / jnp.sqrt(2.0 * in_channels)
        positive_y_modes = jnp.arange(ymodes, dtype=jnp.float32)[:, None]
        negative_y_modes = jnp.arange(ymodes, 0, -1, dtype=jnp.float32)[:, None]
        x_modes = jnp.arange(xmodes, dtype=jnp.float32)[None, :]
        positive_scale = (
            1.0 + jnp.sqrt(positive_y_modes**2 + x_modes**2)
        ) ** spectral_init_power
        negative_scale = (
            1.0 + jnp.sqrt(negative_y_modes**2 + x_modes**2)
        ) ** spectral_init_power
        positive_scale = positive_scale / jnp.sqrt(jnp.mean(positive_scale**2))
        negative_scale = negative_scale / jnp.sqrt(jnp.mean(negative_scale**2))
        
        # initialise two sets of complex weights
        weight_scale1 = scale * positive_scale[None, None, :, :]
        weight_scale2 = scale * negative_scale[None, None, :, :]
        real1 = jax.random.normal(
            k1, (in_channels, out_channels, ymodes, xmodes), dtype=jnp.float32
        ) * weight_scale1
        imag1 = jax.random.normal(
            k2, (in_channels, out_channels, ymodes, xmodes), dtype=jnp.float32
        ) * weight_scale1
        self.weights1 = real1 + 1j * imag1
        
        real2 = jax.random.normal(
            k3, (in_channels, out_channels, ymodes, xmodes), dtype=jnp.float32
        ) * weight_scale2
        imag2 = jax.random.normal(
            k4, (in_channels, out_channels, ymodes, xmodes), dtype=jnp.float32
        ) * weight_scale2
        self.weights2 = real2 + 1j * imag2

    def compl_mul2d(self, input, weights):
        """Multiply Fourier modes with channel-wise complex weights."""
        return jnp.einsum("bixy,ioxy->boxy", input, weights)

    def _wavenumber_gain(self, H, W):
        """Real, |k|-only gain: keeps Hermitian symmetry and translation equivariance."""
        if self.kpower == 0.0:
            return jnp.ones((H, W // 2 + 1), dtype=jnp.float32)
        ky = jnp.fft.fftfreq(H) * H
        kx = jnp.fft.rfftfreq(W) * W
        kmag = jnp.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
        return 1.0 + (kmag / self.kref) ** self.kpower

    def _spectral_coefficients(self, x):
        if x.ndim not in (3, 4):
            raise ValueError(
                "SpectralConv2d expects shape (C, H, W) or (B, C, H, W)"
            )
        unbatched = x.ndim == 3
        if unbatched:
            x = x[None, ...]

        B, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {C}"
            )

        # Compute Fourier coefficients using real FFT (more efficient for real inputs)
        x_ft = jnp.fft.rfft2(x, axes=(-2, -1))

        # Output tensor in Fourier domain
        out_ft = jnp.zeros(
            (B, self.out_channels, H, W // 2 + 1), dtype=x_ft.dtype
        )

        # Multiply relevant Fourier modes
        # The final rFFT axis is x; the full FFT axis is y.
        if self.ymodes > H // 2 or self.xmodes > W // 2 + 1:
            warnings.warn(
                f"Configured FNO modes (ymodes={self.ymodes}, "
                f"xmodes={self.xmodes}) exceed the available grid modes "
                f"(ymodes={H // 2}, xmodes={W // 2 + 1}); excess modes are unused.",
                RuntimeWarning,
                stacklevel=2,
            )
        y_modes = min(self.ymodes, H // 2)
        x_modes = min(self.xmodes, W // 2 + 1)

        if y_modes and x_modes:
            out_ft = out_ft.at[:, :, :y_modes, :x_modes].set(
                self.compl_mul2d(
                    x_ft[:, :, :y_modes, :x_modes],
                    self.weights1[:, :, :y_modes, :x_modes],
                )
            )

            # The y axis is uncompressed, so retain its negative frequencies too.
            out_ft = out_ft.at[:, :, -y_modes:, :x_modes].set(
                self.compl_mul2d(
                    x_ft[:, :, -y_modes:, :x_modes],
                    self.weights2[:, :, :y_modes, :x_modes],
                )
            )

            # The x=0 and x-Nyquist columns are self-conjugate in an rFFT.
            # Constrain those columns so irfft2 does not silently project away
            # independently learned positive/negative-y coefficients.
            boundary_x_modes = [0]
            if W % 2 == 0 and x_modes > W // 2:
                boundary_x_modes.append(W // 2)

            for x_mode in boundary_x_modes:
                out_ft = out_ft.at[:, :, :, x_mode].set(0.0)

                zero_mode = self.compl_mul2d(
                    x_ft[:, :, :1, x_mode : x_mode + 1],
                    jnp.real(self.weights1[:, :, :1, x_mode : x_mode + 1]),
                )
                out_ft = out_ft.at[:, :, 0, x_mode].set(zero_mode[:, :, 0, 0])

                paired_modes = min(y_modes - 1, (H - 1) // 2)
                for mode in range(1, paired_modes + 1):
                    positive_mode = self.compl_mul2d(
                        x_ft[:, :, mode : mode + 1, x_mode : x_mode + 1],
                        self.weights1[:, :, mode : mode + 1, x_mode : x_mode + 1],
                    )
                    positive_mode = positive_mode[:, :, 0, 0]
                    out_ft = out_ft.at[:, :, mode, x_mode].set(positive_mode)
                    out_ft = out_ft.at[:, :, -mode, x_mode].set(
                        jnp.conj(positive_mode)
                    )

                if y_modes < H // 2 or H % 2:
                    # The negative cutoff mode has no positive counterpart in
                    # the usual slice; add its conjugate partner explicitly.
                    negative_mode = H - y_modes
                    positive_mode = y_modes
                    negative_value = self.compl_mul2d(
                        x_ft[
                            :,
                            :,
                            negative_mode : negative_mode + 1,
                            x_mode : x_mode + 1,
                        ],
                        self.weights2[:, :, :1, x_mode : x_mode + 1],
                    )[:, :, 0, 0]
                    out_ft = out_ft.at[:, :, negative_mode, x_mode].set(
                        negative_value
                    )
                    out_ft = out_ft.at[:, :, positive_mode, x_mode].set(
                        jnp.conj(negative_value)
                    )
                elif H % 2 == 0 and y_modes == H // 2:
                    # The y-Nyquist coefficient is self-conjugate as well.
                    nyquist_mode = H // 2
                    nyquist_value = self.compl_mul2d(
                        x_ft[
                            :,
                            :,
                            nyquist_mode : nyquist_mode + 1,
                            x_mode : x_mode + 1,
                        ],
                        jnp.real(self.weights2[:, :, :1, x_mode : x_mode + 1]),
                    )[:, :, 0, 0]
                    out_ft = out_ft.at[:, :, nyquist_mode, x_mode].set(
                        nyquist_value
                    )
        
        out_ft = out_ft * self._wavenumber_gain(H, W)[None, None, :, :]
        return out_ft[0] if unbatched else out_ft

    def __call__(self, x):
        if x.ndim not in (3, 4):
            raise ValueError(
                "SpectralConv2d expects shape (C, H, W) or (B, C, H, W)"
            )
        height, width = x.shape[-2:]
        x_ft = self._spectral_coefficients(x)
        return self.gate * jnp.fft.irfft2(
            x_ft, s=(height, width), axes=(-2, -1)
        )


class MemoryState(eqx.Module):
    """Rolling history of the nine snapshots preceding the current field.

    history shape: (9 * C, H, W)
    The current field supplied to the model completes the ten-step window.
    """
    history: jnp.ndarray
    initialized: jnp.ndarray


class OriginalFNO2d(eqx.Module):
    """Creator-style 2D FNO, adapted to channels-first Equinox tensors.

    Input:  (10*C, H, W), containing ten field snapshots.
    Output: (C, H, W), normalized closure increment.
    """
    input_proj: eqx.nn.Conv2d
    spec_layers: tuple
    w_layers: tuple
    proj1: eqx.nn.Conv2d
    proj2: eqx.nn.Conv2d
    activation: object
    field_channels: int
    depth: int = eqx.field(static=True)

    def __init__(
        self,
        field_channels,
        xmodes=12,
        ymodes=12,
        width=20,
        projection_width=128,
        depth=4,
        activation="gelu",
        spectral_init_power=0.5,
        key=jax.random.PRNGKey(0),
    ):
        self.field_channels = int(field_channels)
        self.depth = int(depth)
        if self.depth <= 0:
            raise ValueError("depth must be positive")

        activation = str(activation).lower()
        activation_functions = {
            "gelu": jax.nn.gelu,
            "relu": jax.nn.relu,
            "elu": jax.nn.elu,
            "tanh": jnp.tanh,
            "leaky_relu": jax.nn.leaky_relu,
        }
        if activation not in activation_functions:
            raise ValueError(
                f"Unsupported activation {activation!r}; expected one of "
                f"{sorted(activation_functions)}"
            )
        keys = jax.random.split(key, 2 * self.depth + 3)

        self.input_proj = eqx.nn.Conv2d(
            in_channels=10 * self.field_channels,
            out_channels=width,
            kernel_size=1,
            key=keys[0],
            padding_mode="CIRCULAR",
        )

        self.spec_layers = tuple(
            SpectralConv2d(
                width, width, xmodes, ymodes,
                key=keys[1 + i],
                gate_init=1.0,
                kpower=0.0,
                spectral_init_power=spectral_init_power,
            )
            for i in range(self.depth)
        )

        # The original architecture uses pointwise linear branches.
        self.w_layers = tuple(
            eqx.nn.Conv2d(
                width, width, kernel_size=1,
                key=keys[1 + self.depth + i],
                padding_mode="CIRCULAR",
            )
            for i in range(self.depth)
        )

        self.proj1 = eqx.nn.Conv2d(
            width, projection_width, kernel_size=1,
            key=keys[-2],
            padding_mode="CIRCULAR",
        )
        self.proj2 = eqx.nn.Conv2d(
            projection_width, self.field_channels, kernel_size=1,
            key=keys[-1],
            padding_mode="CIRCULAR",
        )
        self.activation = eqx.nn.Lambda(activation_functions[activation])

    def __call__(self, history_and_current):
        x = self.input_proj(history_and_current)

        for i, (spec, local) in enumerate(
            zip(self.spec_layers, self.w_layers)
        ):
            x = spec(x) + local(x)
            if i < self.depth - 1:
                x = self.activation(x)

        x = self.activation(self.proj1(x))
        return self.proj2(x)


class FNO(eqx.Module):
    """FNO with a rolling ten-snapshot input window."""

    fno: OriginalFNO2d
    field_channels: int

    def __init__(
        self,
        cfg,
        key=jax.random.PRNGKey(0), #fix: this
        xmodes=12,
        ymodes=12,
        width=20,
        projection_width=128,
        depth=4,
        activation="gelu",
        spectral_init_power=0.5,
        **kwargs,
    ):
        channels = int(cfg.params.nz)
        self.field_channels = channels
        self.fno = OriginalFNO2d(
            field_channels=channels,
            xmodes=xmodes,
            ymodes=ymodes,
            width=width,
            projection_width=projection_width,
            depth=depth,
            activation=activation,
            spectral_init_power=spectral_init_power,
            key=key,
        )

    def init_state(self, q):
        """Start with nine copies of the initial field."""
        q = q.astype(jnp.float32)
        history = jnp.concatenate([q] * 9, axis=0)
        return MemoryState(history=history, initialized=jnp.asarray(True))

    def __call__(self, q, aux):
        q = q.astype(jnp.float32)
        C = self.field_channels

        # Shapes are static under JAX tracing.
        if (
            not isinstance(aux, MemoryState)
            or aux.history.shape[0] != 9 * C
            or aux.history.shape[-2:] != q.shape[-2:]
        ):
            history = jnp.concatenate([q] * 9, axis=0)
        else:
            history = jax.lax.cond(
                aux.initialized,
                lambda stored_history: stored_history,
                lambda _: jnp.concatenate([q] * 9, axis=0),
                aux.history.astype(jnp.float32),
            )

        # Ten snapshots: nine historical fields + current field.
        window = jnp.concatenate((history, q), axis=0)
        closure_increment = self.fno(window)

        # Shift history and append the current field.
        next_history = jnp.concatenate(
            (history[C:], q), axis=0
        )

        return closure_increment, MemoryState(
            history=next_history,
            initialized=jnp.asarray(True),
        )

    @property
    def model_type(self):
        return "fno"

    @property
    def stateful(self):
        return True
