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
    weights1: jnp.ndarray  # complex weights for lower modes
    weights2: jnp.ndarray  # complex weights for upper modes

    def __init__(self, in_channels, out_channels, xmodes, ymodes, key=None):
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.xmodes = int(xmodes)
        self.ymodes = int(ymodes)
        
        if key is None:
            key = jax.random.PRNGKey(0)
        
        k1, k2, k3, k4 = jax.random.split(key, 4)
        scale = (1.0 / (in_channels * out_channels))
        
        # initialise two sets of complex weights
        real1 = jax.random.normal(k1, (in_channels, out_channels, xmodes, ymodes), dtype=jnp.float32) * scale
        imag1 = jax.random.normal(k2, (in_channels, out_channels, xmodes, ymodes), dtype=jnp.float32) * scale
        self.weights1 = real1 + 1j * imag1
        
        real2 = jax.random.normal(k3, (in_channels, out_channels, xmodes, ymodes), dtype=jnp.float32) * scale
        imag2 = jax.random.normal(k4, (in_channels, out_channels, xmodes, ymodes), dtype=jnp.float32) * scale
        self.weights2 = real2 + 1j * imag2

    def compl_mul2d(self, input, weights):
        """Complex multiplication in Fourier space: (batch, in_ch, x, y) * (in_ch, out_ch, x, y) -> (batch, out_ch, x, y)"""
        return jnp.einsum("bixy,ioxy->boxy", input, weights)

    def __call__(self, x): 
        # x expected shape: (C, H, W)
        x = x[None, ...]
        
        B, C, H, W = x.shape
        
        # Compute Fourier coefficients using real FFT (more efficient for real inputs)
        x_ft = jnp.fft.rfft2(x, axes=(-2, -1))
        
        # Output tensor in Fourier domain
        out_ft = jnp.zeros((B, self.out_channels, H, W // 2 + 1), dtype=jnp.complex64)
        
        # Multiply relevant Fourier modes
        # Lower modes (positive frequencies)
        xmodes = min(self.xmodes, H)
        ymodes = min(self.ymodes, W // 2 + 1)
        
        out_ft = out_ft.at[:, :, :xmodes, :ymodes].set(
            self.compl_mul2d(x_ft[:, :, :xmodes, :ymodes], self.weights1[:, :, :xmodes, :ymodes])
        )
        
        # Upper modes (negative frequencies)
        if xmodes < H:
            out_ft = out_ft.at[:, :, -xmodes:, :ymodes].set(
                self.compl_mul2d(x_ft[:, :, -xmodes:, :ymodes], self.weights2[:, :, :xmodes, :ymodes])
            )
        
        x = jnp.fft.irfft2(out_ft, s=(H, W), axes=(-2, -1))
        return x[0]


class FNO(eqx.Module):
    """
    Fourier Neural Operator parameterisation
    
    Architecture:
    - Lift input channels -> `width` via a 1x1 conv
    - Repeat `n_layers` blocks of (SpectralConv2d + pointwise conv) with GELU activation
    - Project back to `out_channels` via two-layer MLP
    - Resolution-invariant via spectral operations
    
    Args:
        width: Hidden channel dimension (default: 32)
        xmodes: Number of Fourier modes in x-direction (default: 16)
        ymodes: Number of Fourier modes in y-direction (default: 16)
        depth: Number of Fourier layers (default: 4)
        activation: activation function 
        key: JAX PRNG key
        cfg: Configuration object
    
    Input/Output:
        - Accepts (C,H,W)
        - Returns same shape as input
    """
    input_proj: eqx.nn.Conv2d
    spec_layers: Sequence[SpectralConv2d]
    w_layers: Sequence[eqx.nn.Conv2d]
    proj1: eqx.nn.Conv2d
    proj2: eqx.nn.Conv2d
    activation: Optional[str]
    n_layers: int

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
        **kwargs,
    ):
        
        in_channels = cfg.params.nz
        out_channels = in_channels 
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
            in_channels, width, kernel_size=1, key=k0,
            padding_mode="CIRCULAR"
        )

        # Fourier layers
        spec_layers = []
        w_layers = []
        for i in range(depth):
            ks = keys[1 + i]
            kw = keys[1 + depth + i]
            spec_layers.append(SpectralConv2d(width, width, xmodes, ymodes, key=ks))
            w_layers.append(eqx.nn.Conv2d(
                width, width, kernel_size=1, key=kw,
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

    def __call__(self, q):
        """
        Forward pass through FNO.
        
        Args:
            q: Input field of shape (C,H,W)
        """
        # Lifting: project to working width
        x = self.input_proj(q.astype(jnp.float32))

        # Fourier layers with residual connections
        for i, (spec, w) in enumerate(zip(self.spec_layers, self.w_layers)):
            x_spec = spec(x)
            x_local = w(x)
            x = x_spec + x_local
            # Apply activation between layers (not on last layer)
            if i < self.n_layers - 1:
                x = self.activation(x)

        # Projection to output space
        x = self.proj1(x)
        x = self.activation(x)
        x = self.proj2(x)
        return x

    @property
    def model_type(self):
        return 'fno'
