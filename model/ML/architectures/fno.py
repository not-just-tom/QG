import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Sequence, Optional


class SpectralConv2d(eqx.Module):
    """
    2D Fourier layer that multiplies learned complex weights on low-frequency modes.
    
    Implementation based on FourierFlow library approach:
    - Uses rfft2 for real-valued inputs (more efficient)
    - Maintains two weight matrices for different Fourier mode regions
    - Handles both positive and negative frequency components
    """
    in_channels: int
    out_channels: int
    modes1: int  # modes in x-direction
    modes2: int  # modes in y-direction
    weights1: jnp.ndarray  # complex weights for lower modes
    weights2: jnp.ndarray  # complex weights for upper modes

    def __init__(self, in_channels, out_channels, modes1, modes2, key=None):
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        
        if key is None:
            key = jax.random.PRNGKey(0)
        
        k1, k2, k3, k4 = jax.random.split(key, 4)
        scale = (1.0 / (in_channels * out_channels))
        
        # initialise two sets of complex weights
        real1 = jax.random.normal(k1, (in_channels, out_channels, modes1, modes2), dtype=jnp.float32) * scale
        imag1 = jax.random.normal(k2, (in_channels, out_channels, modes1, modes2), dtype=jnp.float32) * scale
        self.weights1 = real1 + 1j * imag1
        
        real2 = jax.random.normal(k3, (in_channels, out_channels, modes1, modes2), dtype=jnp.float32) * scale
        imag2 = jax.random.normal(k4, (in_channels, out_channels, modes1, modes2), dtype=jnp.float32) * scale
        self.weights2 = real2 + 1j * imag2

    def compl_mul2d(self, input, weights):
        """Complex multiplication in Fourier space: (batch, in_ch, x, y) * (in_ch, out_ch, x, y) -> (batch, out_ch, x, y)"""
        return jnp.einsum("bixy,ioxy->boxy", input, weights)

    def __call__(self, x):
        # x expected shape: (C, H, W) or (B, C, H, W)
        added_batch = False
        if x.ndim == 3:
            x = x[None, ...]
            added_batch = True
        
        B, C, H, W = x.shape
        
        # Compute Fourier coefficients using real FFT (more efficient for real inputs)
        x_ft = jnp.fft.rfft2(x, axes=(-2, -1))
        
        # Output tensor in Fourier domain
        out_ft = jnp.zeros((B, self.out_channels, H, W // 2 + 1), dtype=jnp.complex64)
        
        # Multiply relevant Fourier modes
        # Lower modes (positive frequencies)
        modes1 = min(self.modes1, H)
        modes2 = min(self.modes2, W // 2 + 1)
        
        out_ft = out_ft.at[:, :, :modes1, :modes2].set(
            self.compl_mul2d(x_ft[:, :, :modes1, :modes2], self.weights1[:, :, :modes1, :modes2])
        )
        
        # Upper modes (negative frequencies)
        if modes1 < H:
            out_ft = out_ft.at[:, :, -modes1:, :modes2].set(
                self.compl_mul2d(x_ft[:, :, -modes1:, :modes2], self.weights2[:, :, :modes1, :modes2])
            )
        
        # Return to physical space
        x = jnp.fft.irfft2(out_ft, s=(H, W), axes=(-2, -1))
        
        if added_batch:
            return x[0]
        return x


class FNO(eqx.Module):
    """
    Fourier Neural Operator for ocean subgrid parameterization.
    
    Architecture:
    - Lift input channels -> `width` via a 1x1 conv
    - Repeat `n_layers` blocks of (SpectralConv2d + pointwise conv) with GELU activation
    - Project back to `out_channels` via two-layer MLP
    
    Features:
    - Zero-mean normalization for conservation properties (configurable)
    - Circular padding for periodic boundary conditions (configurable)
    - Resolution-invariant via spectral operations
    
    Args:
        width: Hidden channel dimension (default: 32)
        modes_x: Number of Fourier modes in x-direction (default: 16)
        modes_y: Number of Fourier modes in y-direction (default: 16)
        n_layers: Number of Fourier layers (default: 4)
        zero_mean: Apply zero-mean constraint at output (default: True)
        padding: Padding to use ('circular', 'same', or None)
        key: JAX PRNG key
        cfg: Configuration object
    
    Input/Output:
        - Accepts (C,H,W) or (B,C,H,W) formats
        - Returns same shape as input
    """
    input_proj: eqx.nn.Conv2d
    spec_layers: Sequence[SpectralConv2d]
    w_layers: Sequence[eqx.nn.Conv2d]
    proj1: eqx.nn.Conv2d
    proj2: eqx.nn.Conv2d
    zero_mean: bool
    padding_mode: Optional[str]
    n_layers: int

    def __init__(
        self,
        width: int = 32,
        modes_x: int = 16,
        modes_y: int = 16,
        n_layers: int = 4,
        zero_mean: bool = True,
        padding: Optional[str] = 'circular',
        key=jax.random.PRNGKey(0),
        cfg=None,
        **kwargs,
    ):
        # Support legacy parameter names
        modes1 = kwargs.get('modes1', modes_x)
        modes2 = kwargs.get('modes2', modes_y)
        depth = kwargs.get('depth', n_layers)
        
        in_channels = cfg.params.nz if cfg is not None else 1
        out_channels = in_channels
        
        self.zero_mean = zero_mean
        self.padding_mode = padding
        self.n_layers = depth
        
        # Split keys for all layers
        keys = jax.random.split(key, 2 * depth + 4)
        k0 = keys[0]
        
        # Lifting layer: in_channels -> width
        self.input_proj = eqx.nn.Conv2d(
            in_channels, width, kernel_size=1, key=k0,
            padding_mode=padding if padding else 'ZEROS'
        )

        # Fourier layers
        spec_layers = []
        w_layers = []
        for i in range(depth):
            ks = keys[1 + i]
            kw = keys[1 + depth + i]
            spec_layers.append(SpectralConv2d(width, width, modes1, modes2, key=ks))
            w_layers.append(eqx.nn.Conv2d(
                width, width, kernel_size=1, key=kw,
                padding_mode=padding if padding else 'ZEROS'
            ))

        self.spec_layers = spec_layers
        self.w_layers = w_layers

        # Projection layers: width -> 128 -> out_channels
        k_proj1 = keys[-2]
        k_proj2 = keys[-1]
        self.proj1 = eqx.nn.Conv2d(
            width, 128, kernel_size=1, key=k_proj1,
            padding_mode=padding if padding else 'ZEROS'
        )
        self.proj2 = eqx.nn.Conv2d(
            128, out_channels, kernel_size=1, key=k_proj2,
            padding_mode=padding if padding else 'ZEROS'
        )

    def __call__(self, q):
        """
        Forward pass through FNO.
        
        Args:
            q: Input field of shape (C,H,W) or (B,C,H,W)
            
        Returns:
            Output field of same shape as input, with optional zero-mean constraint
        """
        # Accept (C,H,W) or (B,C,H,W)
        readd_batch = False
        x = q
        if x.ndim == 4:
            if x.shape[0] == 1:
                x = x[0]
                readd_batch = True
            else:
                raise ValueError("FNO does not support batched inputs with batch>1 in this codebase")

        if x.ndim != 3:
            raise ValueError("Input must be (C,H,W) or (1,C,H,W)")

        x = x.astype(jnp.float32)

        # Lifting: project to working width
        x = self.input_proj(x)

        # Fourier layers with residual connections
        for i, (spec, w) in enumerate(zip(self.spec_layers, self.w_layers)):
            x_spec = spec(x)
            x_local = w(x)
            x = x_spec + x_local
            # Apply activation between layers (not on last layer)
            if i < self.n_layers - 1:
                x = jax.nn.gelu(x)

        # Projection to output space
        x = self.proj1(x)
        x = jax.nn.gelu(x)
        x = self.proj2(x)

        # Apply zero-mean constraint for conservation properties
        if self.zero_mean:
            # Compute mean over spatial dimensions
            mean_val = jnp.mean(x, axis=(-2, -1), keepdims=True)
            x = x - mean_val

        if readd_batch:
            return x[None, ...]
        return x
