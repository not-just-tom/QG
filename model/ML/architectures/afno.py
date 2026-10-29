import equinox as eqx
import jax
import jax.numpy as jnp


class _Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_features, out_features, key):
        self.weight = 0.02 * jax.random.truncated_normal(
            key, lower=-2.0, upper=2.0, shape=(out_features, in_features)
        )
        self.bias = jnp.zeros((out_features,), dtype=jnp.float32)

    def __call__(self, x):
        return jnp.matmul(x, self.weight.T) + self.bias


class AFNO2D(eqx.Module):
    """Block-diagonal Fourier mixing layer from the AFNO architecture."""

    w1_real: jax.Array
    w1_imag: jax.Array
    b1_real: jax.Array
    b1_imag: jax.Array
    w2_real: jax.Array
    w2_imag: jax.Array
    b2_real: jax.Array
    b2_imag: jax.Array
    hidden_size: int = eqx.field(static=True)
    num_blocks: int = eqx.field(static=True)
    block_size: int = eqx.field(static=True)
    hidden_size_factor: int = eqx.field(static=True)
    sparsity_threshold: float = eqx.field(static=True)
    hard_thresholding_fraction: float = eqx.field(static=True)

    def __init__(
        self,
        hidden_size,
        num_blocks=8,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
        hidden_size_factor=1,
        key=jax.random.PRNGKey(0),
    ):
        if hidden_size <= 0 or num_blocks <= 0 or hidden_size % num_blocks:
            raise ValueError("hidden_size must be positive and divisible by num_blocks")
        if hidden_size_factor <= 0:
            raise ValueError("hidden_size_factor must be positive")
        if sparsity_threshold < 0:
            raise ValueError("sparsity_threshold must be non-negative")
        if not 0.0 < hard_thresholding_fraction <= 1.0:
            raise ValueError("hard_thresholding_fraction must be in (0, 1]")

        self.hidden_size = int(hidden_size)
        self.num_blocks = int(num_blocks)
        self.block_size = self.hidden_size // self.num_blocks
        self.hidden_size_factor = int(hidden_size_factor)
        self.sparsity_threshold = float(sparsity_threshold)
        self.hard_thresholding_fraction = float(hard_thresholding_fraction)

        hidden_width = self.block_size * self.hidden_size_factor
        keys = jax.random.split(key, 8)
        shapes = (
            (self.num_blocks, self.block_size, hidden_width),
            (self.num_blocks, self.block_size, hidden_width),
            (self.num_blocks, hidden_width),
            (self.num_blocks, hidden_width),
            (self.num_blocks, hidden_width, self.block_size),
            (self.num_blocks, hidden_width, self.block_size),
            (self.num_blocks, self.block_size),
            (self.num_blocks, self.block_size),
        )
        params = [
            0.02 * jax.random.normal(k, shape, dtype=jnp.float32)
            for k, shape in zip(keys, shapes)
        ]
        (
            self.w1_real,
            self.w1_imag,
            self.b1_real,
            self.b1_imag,
            self.w2_real,
            self.w2_imag,
            self.b2_real,
            self.b2_imag,
        ) = params

    @staticmethod
    def _softshrink(x, threshold):
        return jnp.sign(x) * jnp.maximum(jnp.abs(x) - threshold, 0.0)

    def __call__(self, x):
        if x.ndim != 3 or x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"AFNO2D expects (H, W, {self.hidden_size}) input, got {x.shape}"
            )

        residual = x
        x = x.astype(jnp.float32)
        height, width, channels = x.shape
        x_ft = jnp.fft.rfft2(x, axes=(0, 1), norm="ortho")
        x_ft = x_ft.reshape(
            height, width // 2 + 1, self.num_blocks, self.block_size
        )

        hidden_width = self.block_size * self.hidden_size_factor
        o1_real = jnp.zeros(
            (height, width // 2 + 1, self.num_blocks, hidden_width),
            dtype=jnp.float32,
        )
        o1_imag = jnp.zeros_like(o1_real)
        o2_real = jnp.zeros_like(x_ft.real)
        o2_imag = jnp.zeros_like(x_ft.real)

        total_modes = height // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)
        row_slice = slice(total_modes - kept_modes, total_modes + kept_modes)
        col_slice = slice(0, kept_modes)
        selected = x_ft[row_slice, col_slice]

        first_real = (
            jnp.einsum("...bi,bio->...bo", selected.real, self.w1_real)
            - jnp.einsum("...bi,bio->...bo", selected.imag, self.w1_imag)
            + self.b1_real
        )
        first_imag = (
            jnp.einsum("...bi,bio->...bo", selected.imag, self.w1_real)
            + jnp.einsum("...bi,bio->...bo", selected.real, self.w1_imag)
            + self.b1_imag
        )
        o1_real = o1_real.at[row_slice, col_slice].set(jax.nn.relu(first_real))
        o1_imag = o1_imag.at[row_slice, col_slice].set(jax.nn.relu(first_imag))

        selected_real = o1_real[row_slice, col_slice]
        selected_imag = o1_imag[row_slice, col_slice]
        second_real = (
            jnp.einsum("...bi,bio->...bo", selected_real, self.w2_real)
            - jnp.einsum("...bi,bio->...bo", selected_imag, self.w2_imag)
            + self.b2_real
        )
        second_imag = (
            jnp.einsum("...bi,bio->...bo", selected_imag, self.w2_real)
            + jnp.einsum("...bi,bio->...bo", selected_real, self.w2_imag)
            + self.b2_imag
        )
        o2_real = o2_real.at[row_slice, col_slice].set(second_real)
        o2_imag = o2_imag.at[row_slice, col_slice].set(second_imag)
        o2_real = self._softshrink(o2_real, self.sparsity_threshold)
        o2_imag = self._softshrink(o2_imag, self.sparsity_threshold)

        out_ft = jax.lax.complex(o2_real, o2_imag).reshape(
            height, width // 2 + 1, channels
        )
        out = jnp.fft.irfft2(out_ft, s=(height, width), axes=(0, 1), norm="ortho")
        return out.astype(residual.dtype) + residual


class AFNOBlock(eqx.Module):
    norm1: eqx.nn.LayerNorm
    filter: AFNO2D
    norm2: eqx.nn.LayerNorm
    fc1: _Linear
    fc2: _Linear
    hidden_size: int = eqx.field(static=True)
    mlp_ratio: float = eqx.field(static=True)
    double_skip: bool = eqx.field(static=True)

    def __init__(
        self,
        hidden_size,
        num_blocks,
        sparsity_threshold,
        hard_thresholding_fraction,
        mlp_ratio,
        double_skip,
        key,
    ):
        key_filter, key_fc1, key_fc2 = jax.random.split(key, 3)
        self.hidden_size = int(hidden_size)
        self.mlp_ratio = float(mlp_ratio)
        self.double_skip = bool(double_skip)
        self.norm1 = eqx.nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.filter = AFNO2D(
            self.hidden_size,
            num_blocks=num_blocks,
            sparsity_threshold=sparsity_threshold,
            hard_thresholding_fraction=hard_thresholding_fraction,
            key=key_filter,
        )
        self.norm2 = eqx.nn.LayerNorm(self.hidden_size, eps=1e-6)
        mlp_hidden_size = int(self.hidden_size * self.mlp_ratio)
        self.fc1 = _Linear(self.hidden_size, mlp_hidden_size, key_fc1)
        self.fc2 = _Linear(mlp_hidden_size, self.hidden_size, key_fc2)

    def _norm(self, layer, x):
        return jax.vmap(jax.vmap(layer))(x)

    def _mlp(self, x):
        x = jax.vmap(jax.vmap(self.fc1))(x)
        x = jax.nn.gelu(x, approximate=False)
        return jax.vmap(jax.vmap(self.fc2))(x)

    def __call__(self, x):
        residual = x
        x = self.filter(self._norm(self.norm1, x))

        if self.double_skip:
            x = x + residual
            residual = x

        x = self._mlp(self._norm(self.norm2, x))
        return x + residual


class AFNO(eqx.Module):
    """AFNO closure for periodic QG fields, using the current field as input.

    The reference model's patch embedding is replaced by a pointwise lift and
    its learned absolute positional embedding is omitted to preserve the QG
    solver grid and periodic translation equivariance.
    """

    ml = True
    stateful = False

    lift: _Linear
    blocks: tuple
    head_norm: eqx.nn.LayerNorm
    head: _Linear
    field_channels: int = eqx.field(static=True)
    width: int = eqx.field(static=True)

    def __init__(
        self,
        cfg,
        key=jax.random.PRNGKey(0),
        width=64,
        depth=4,
        num_blocks=8,
        mlp_ratio=4.0,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
        double_skip=True,
        **kwargs,
    ):
        if cfg is None:
            raise ValueError("cfg is required to construct AFNO")
        if width <= 0 or depth <= 0:
            raise ValueError("width and depth must be positive")
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if width % num_blocks:
            raise ValueError("width must be divisible by num_blocks")

        self.field_channels = int(cfg.params.nz)
        self.width = int(width)
        keys = jax.random.split(key, depth + 2)
        self.lift = _Linear(self.field_channels, self.width, keys[0])
        self.blocks = tuple(
            AFNOBlock(
                self.width,
                num_blocks,
                sparsity_threshold,
                hard_thresholding_fraction,
                mlp_ratio,
                double_skip,
                keys[i + 1],
            )
            for i in range(depth)
        )
        self.head_norm = eqx.nn.LayerNorm(self.width, eps=1e-6)
        self.head = _Linear(self.width, self.field_channels, keys[-1])

    def __call__(self, q):
        if q.ndim != 3 or q.shape[0] != self.field_channels:
            raise ValueError(
                f"AFNO expects ({self.field_channels}, H, W) input, got {q.shape}"
            )
        x = jnp.moveaxis(q.astype(jnp.float32), 0, -1)
        x = jax.vmap(jax.vmap(self.lift))(x)

        for block in self.blocks:
            x = block(x)

        x = jax.vmap(jax.vmap(self.head_norm))(x)
        x = jax.vmap(jax.vmap(self.head))(x)
        return jnp.moveaxis(x, -1, 0).astype(q.dtype)

    @property
    def model_type(self):
        return "afno"
