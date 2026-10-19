import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Sequence, Optional

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
        # x expected shape: (C, H, W)
        x = x[None, ...]
        
        B, C, H, W = x.shape
        
        # Compute Fourier coefficients using real FFT (more efficient for real inputs)
        x_ft = jnp.fft.rfft2(x, axes=(-2, -1))
        
        # Output tensor in Fourier domain
        out_ft = jnp.zeros((B, self.out_channels, H, W // 2 + 1), dtype=jnp.complex64)
        
        # Multiply relevant Fourier modes
        # The final rFFT axis is x; the full FFT axis is y.
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
        
        return out_ft[0] * self._wavenumber_gain(H, W)[None, :, :]

    def __call__(self, x):
        height, width = x.shape[-2:]
        x_ft = self._spectral_coefficients(x)
        return self.gate * jnp.fft.irfft2(
            x_ft, s=(height, width), axes=(-2, -1)
        )


class FNO(eqx.Module):
    """
    Fourier Neural Operator parameterisation
    
    Architecture:
    - Lift input features -> `width` via a 1x1 conv
    - Optionally augment PV inputs with streamfunction, Laplacian, and velocity
      channels derived using the configured QG inversion.
    - Repeat `n_layers` residual blocks of parallel spectral and local
      convolutions.
    - Project back to `out_channels` and add a linear local-stencil input skip
      (circular `skip_kernel` x `skip_kernel` conv, no bias).
    - Resolution-invariant via spectral operations
    
    Args:
        width: Hidden channel dimension (default: 32)
        xmodes: Number of Fourier modes in x-direction (default: 16)
        ymodes: Number of Fourier modes in y-direction (default: 16)
        depth: Number of Fourier layers (default: 4)
        activation: activation function 
        key: JAX PRNG key
        cfg: Configuration object
        physics_features: Add QG-inverted streamfunction, Laplacian, and velocity channels.
        spectral_init_power: Radial power used to scale spectral-weight initialization.
        skip_kernel: Odd kernel size of the linear input->output skip (1 = old pointwise skip).
        spectral_gate_init: Initial value of each spectral layer's learnable scalar gate.
        spectral_gain_power: kpower of the fixed gain 1 + (|k|/spectral_kref)^p (0 = off).
        spectral_kref: Reference wavenumber (integer mode units) for that gain.
    
    Input/Output:
        - Accepts (C,H,W)
        - Returns same shape as input
    """
    input_proj: eqx.nn.Conv2d
    spec_layers: Sequence[SpectralConv2d]
    w_layers: Sequence[eqx.nn.Conv2d]
    proj1: eqx.nn.Conv2d
    proj2: eqx.nn.Conv2d
    skip_proj: eqx.nn.Conv2d
    activation: Optional[str]
    n_layers: int
    physics_features: bool
    Lx: float
    Ly: float
    Ld: float
    delta: float

    def __init__(
        self,
        width: int = 32,
        xmodes: int = 16,
        ymodes: int = 16,
        projection_width: int = 128,
        depth: int = 4,
        activation: Optional[str] = 'gelu',
        key=jax.random.PRNGKey(0),
        cfg=None,
        physics_features: bool = True,
        spectral_init_power: float = 0.5,
        skip_kernel: int = 5,
        spectral_gate_init: float = 1.0,
        spectral_gain_power: float = 0.0,
        spectral_kref: float = 4.0,
        **kwargs,
    ):
        
        in_channels = cfg.params.nz
        out_channels = in_channels 
        if in_channels not in (1, 2):
            raise ValueError("FNO physics features support only nz=1 or nz=2")
        self.physics_features = bool(physics_features)
        raw_lx = float(cfg.params.Lx)
        raw_ly = float(getattr(cfg.params, "Ly", raw_lx) or raw_lx)
        length_scale = (
            raw_ly if getattr(cfg.params, "nondimensional", True) else 1.0
        )
        self.Lx = raw_lx / length_scale
        self.Ly = raw_ly / length_scale
        self.Ld = float(cfg.params.Ld) / length_scale
        self.delta = float(getattr(cfg.params, "delta", 1.0))
        self.n_layers = depth

        # set up activation
        if isinstance(activation, str) and activation.lower() == "tanh":
            self.activation = eqx.nn.Lambda(jnp.tanh)
        elif isinstance(activation, str) and activation.lower() == "gelu":
            self.activation = eqx.nn.Lambda(jax.nn.gelu)
        elif isinstance(activation, str) and activation.lower() == "relu":
            self.activation = eqx.nn.Lambda(jax.nn.relu)
        elif isinstance(activation, str) and activation.lower() == "elu":
            self.activation = eqx.nn.Lambda(jax.nn.elu)
        elif isinstance(activation, str) and activation.lower() == "leaky_relu":
            self.activation = eqx.nn.Lambda(jax.nn.leaky_relu)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Split keys for all layers
        keys = jax.random.split(key, 2 * depth + 4)
        k0 = keys[0]
        
        # Lifting layer: in_channels -> width
        self.input_proj = eqx.nn.Conv2d(
            in_channels * (5 if self.physics_features else 1),
            width,
            kernel_size=1,
            key=k0,
            padding_mode="CIRCULAR"
        )

        # Fourier layers
        spec_layers = []
        w_layers = []
        for i in range(depth):
            ks = keys[1 + i]
            kw = keys[1 + depth + i]
            spec_layers.append(
                SpectralConv2d(
                    width,
                    width,
                    xmodes,
                    ymodes,
                    key=ks,
                    spectral_init_power=spectral_init_power,
                    gate_init=spectral_gate_init,
                    kref=spectral_kref,
                    kpower=spectral_gain_power,
                )
            )
            w_layers.append(eqx.nn.Conv2d(
                width, width, kernel_size=3, padding=1, key=kw,
                padding_mode="CIRCULAR"
            ))

        self.spec_layers = spec_layers
        self.w_layers = w_layers

        # Projection layers: width -> projection_width -> out_channels
        k_proj1 = keys[-2]
        k_proj2 = keys[-1]
        self.proj1 = eqx.nn.Conv2d(
            width, projection_width, kernel_size=1, key=k_proj1,
            padding_mode="CIRCULAR"
        )
        self.proj2 = eqx.nn.Conv2d(
            projection_width, out_channels, kernel_size=1, key=k_proj2,
            padding_mode="CIRCULAR"
        )
        if skip_kernel < 1 or skip_kernel % 2 != 1:
            raise ValueError(f"skip_kernel must be a positive odd integer, got {skip_kernel}")
        # Linear local-stencil path q -> closure. No bias: a constant offset would
        # add a spatially uniform tendency that drifts the domain-mean PV.
        self.skip_proj = eqx.nn.Conv2d(
            in_channels, out_channels, kernel_size=skip_kernel,
            padding=skip_kernel // 2, use_bias=False, key=keys[-3],
            padding_mode="CIRCULAR"
        )

    def prepare_inputs(self, q):
        """Return PV plus QG-derived physical-space feature channels."""
        if not self.physics_features:
            return q.astype(jnp.float32)

        height, width = q.shape[-2:]
        kx = 2.0 * jnp.pi * jnp.fft.rfftfreq(
            width, d=self.Lx / width
        )
        ky = 2.0 * jnp.pi * jnp.fft.fftfreq(
            height, d=self.Ly / height
        )
        k2 = ky[:, None] ** 2 + kx[None, :] ** 2
        qh = jnp.fft.rfft2(q, axes=(-2, -1))

        if q.shape[0] == 1:
            safe_k2 = jnp.where(k2 == 0, 1.0, k2)
            psi_h = -qh / safe_k2[None, :, :]
            psi_h = psi_h.at[..., 0, 0].set(0.0)
        elif q.shape[0] == 2:
            f1 = self.Ld**-2 / (1.0 + self.delta)
            f2 = self.delta * f1
            determinant = k2 * (k2 + f1 + f2)
            safe_determinant = jnp.where(determinant == 0, 1.0, determinant)
            psi_h = jnp.stack(
                (
                    (-(k2 + f2) * qh[0] - f1 * qh[1]) / safe_determinant,
                    (-f2 * qh[0] - (k2 + f1) * qh[1]) / safe_determinant,
                ),
                axis=0,
            )
            psi_h = jnp.where(determinant[None, :, :] == 0, 0.0, psi_h)
        else:
            raise ValueError(f"Expected 1 or 2 PV layers, got {q.shape[0]}")

        laplacian_qh = -k2[None, :, :] * qh
        u_h = -1j * ky[None, :, None] * psi_h
        v_h = 1j * kx[None, None, :] * psi_h
        features = tuple(
            jnp.fft.irfft2(value, s=(height, width), axes=(-2, -1))
            for value in (psi_h, laplacian_qh, u_h, v_h)
        )
        return jnp.concatenate((q, *features), axis=0).astype(jnp.float32)

    def __call__(self, q):
        """
        Forward pass through FNO.
        
        Args:
            q: Input field of shape (C,H,W)
        """
        # Lifting: project to working width
        x = self.input_proj(self.prepare_inputs(q))

        # Combine global spectral and local stencil transformations residually.
        for spec, w in zip(self.spec_layers, self.w_layers):
            x = x + self.activation(spec(x) + w(x))

        # Projection to output space
        x = self.proj1(x)
        x = self.activation(x)
        x = self.proj2(x)
        return x + self.skip_proj(q.astype(jnp.float32))

    @property
    def model_type(self):
        return 'fno'