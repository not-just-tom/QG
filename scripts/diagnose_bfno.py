
"""
diagnose_bfno.py

Tests:
  1. One-example and small-batch overfitting.
  2. Pointwise-only, spectral-only, and combined FNOs.
  3. Residual and non-residual FNO blocks.
  4. Direct input-output skip connection.
  5. Gradient norms by parameter group.
  6. SpectralConv2d response to individual Fourier modes.
  7. Identity, Laplacian, and combined synthetic mappings.

Run:
    python diagnose_bfno.py
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from model.ML.architectures.fno import SpectralConv2d


# ============================================================
# Configuration
# ============================================================

SEED = 42

CHANNELS = 1
HEIGHT = 64
WIDTH_GRID = 64

WIDTH = 32
MODES1 = 12
MODES2 = 12
LAYERS = 4

PROJECTION_WIDTH = 64
LEARNING_RATE = 1e-3

TRAIN_STEPS = 1000
LOG_EVERY = 100

BATCH_SIZE = 32
TINY_BATCH_SIZE = 8

# Set to True to run the more expensive spectral-only tests.
RUN_SPECTRAL_TESTS = True


# ============================================================
# Utilities
# ============================================================

def tree_l2_norm(tree):
    """L2 norm over all array leaves, including complex arrays."""
    leaves = jax.tree_util.tree_leaves(tree)
    leaves = [x for x in leaves if eqx.is_array(x)]

    if not leaves:
        return 0.0

    return jnp.sqrt(
        sum(jnp.sum(jnp.abs(x) ** 2) for x in leaves)
    )


def rms(x):
    return jnp.sqrt(jnp.mean(jnp.square(jnp.abs(x))))


def relative_error(pred, target):
    return rms(pred - target) / (rms(target) + 1e-8)


def print_metrics(name, pred, target):
    print(
        f"{name:24s} "
        f"pred RMS={float(rms(pred)):.4e} "
        f"target RMS={float(rms(target)):.4e} "
        f"relative error={float(relative_error(pred, target)):.4e}"
    )


# ============================================================
# Test FNO architecture
# ============================================================

class TestFNOBlock(eqx.Module):
    spectral: SpectralConv2d
    local: eqx.nn.Conv2d

    spectral_enabled: bool = eqx.field(static=True)
    local_enabled: bool = eqx.field(static=True)
    residual: bool = eqx.field(static=True)

    def __init__(
        self,
        width,
        modes1,
        modes2,
        key,
        spectral_enabled=True,
        local_enabled=True,
        residual=False,
    ):
        k1, k2 = jax.random.split(key)

        self.spectral = SpectralConv2d(
            width,
            width,
            xmodes=modes2,
            ymodes=modes1,
            key=k1,
        )

        self.local = eqx.nn.Conv2d(
            width,
            width,
            kernel_size=1,
            key=k2,
            padding_mode="CIRCULAR",
        )

        self.spectral_enabled = spectral_enabled
        self.local_enabled = local_enabled
        self.residual = residual

    def __call__(self, x):
        y = jnp.zeros_like(x)

        if self.spectral_enabled:
            y = y + self.spectral(x)

        if self.local_enabled:
            y = y + self.local(x)

        y = jax.nn.gelu(y)

        if self.residual:
            return x + y

        return y


class TestBFNO(eqx.Module):
    fc0: eqx.nn.Conv2d
    blocks: tuple
    fc1: eqx.nn.Conv2d
    fc2: eqx.nn.Conv2d

    skip: eqx.nn.Conv2d

    use_skip: bool = eqx.field(static=True)

    def __init__(
        self,
        key,
        spectral_enabled=True,
        local_enabled=True,
        residual=False,
        use_skip=False,
    ):
        keys = jax.random.split(key, LAYERS + 4)

        self.fc0 = eqx.nn.Conv2d(
            CHANNELS,
            WIDTH,
            kernel_size=1,
            key=keys[0],
            padding_mode="CIRCULAR",
        )

        self.blocks = tuple(
            TestFNOBlock(
                WIDTH,
                MODES1,
                MODES2,
                keys[i + 1],
                spectral_enabled=spectral_enabled,
                local_enabled=local_enabled,
                residual=residual,
            )
            for i in range(LAYERS)
        )

        self.fc1 = eqx.nn.Conv2d(
            WIDTH,
            PROJECTION_WIDTH,
            kernel_size=1,
            key=keys[-3],
            padding_mode="CIRCULAR",
        )

        self.fc2 = eqx.nn.Conv2d(
            PROJECTION_WIDTH,
            CHANNELS,
            kernel_size=1,
            key=keys[-2],
            padding_mode="CIRCULAR",
        )

        self.skip = eqx.nn.Conv2d(
            CHANNELS,
            CHANNELS,
            kernel_size=1,
            key=keys[-1],
            padding_mode="CIRCULAR",
        )

        self.use_skip = use_skip

    def __call__(self, q):
        x = self.fc0(q)

        for block in self.blocks:
            x = block(x)

        x = jax.nn.gelu(self.fc1(x))
        y = self.fc2(x)

        if self.use_skip:
            y = y + self.skip(q)

        return y


# ============================================================
# Synthetic data
# ============================================================

def make_data(key, n_samples):
    """
    Generate smooth periodic random fields and their targets.

    Inputs are band-limited random fields.
    Targets:
        identity:      y = q
        laplacian:     y = Laplacian(q)
        combined:      y = 0.5*q + 0.1*Laplacian(q)

    The Laplacian uses periodic boundary conditions.
    """

    key, subkey = jax.random.split(key)

    q = jax.random.normal(
        subkey,
        (n_samples, CHANNELS, HEIGHT, WIDTH_GRID),
    )

    ky = jnp.fft.fftfreq(HEIGHT) * HEIGHT
    kx = jnp.fft.fftfreq(WIDTH_GRID) * WIDTH_GRID

    KX, KY = jnp.meshgrid(kx, ky)

    # Retain only relatively low spatial frequencies.
    mask = (
        (jnp.abs(KX) <= MODES2)
        & (jnp.abs(KY) <= MODES1)
    )

    q_hat = jnp.fft.fft2(q, axes=(-2, -1))
    q_hat = q_hat * mask[None, None, :, :]

    q = jnp.fft.ifft2(q_hat, axes=(-2, -1)).real

    # Spectral Laplacian: -|k|^2 * q_hat.
    lap_hat = -(KX**2 + KY**2) * q_hat
    lap = jnp.fft.ifft2(
        lap_hat,
        axes=(-2, -1),
    ).real

    # Keep target scales reasonable.
    lap = lap / (rms(lap) + 1e-8) * rms(q)

    targets = {
        "identity": q,
        "laplacian": lap,
        "combined": 0.5 * q + 0.1 * lap,
    }

    return q.astype(jnp.float32), {
        name: value.astype(jnp.float32)
        for name, value in targets.items()
    }


# ============================================================
# Training
# ============================================================

def train_model(model, x, y, name, steps=TRAIN_STEPS):
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state, x, y):
        def loss_fn(m):
            pred = jax.vmap(m)(x)
            loss = jnp.mean((pred - y) ** 2)
            return loss

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)

        updates, opt_state = optimizer.update(
            grads,
            opt_state,
            eqx.filter(model, eqx.is_array),
        )

        model = eqx.apply_updates(model, updates)

        return model, opt_state, loss, grads

    print(f"\n{'=' * 65}")
    print(f"Training: {name}")
    print(f"{'=' * 65}")

    initial_pred = jax.vmap(model)(x)
    print_metrics("Initial prediction", initial_pred, y)

    initial_loss = float(jnp.mean((initial_pred - y) ** 2))

    for i in range(steps):
        model, opt_state, loss, grads = step(
            model, opt_state, x, y
        )

        if i % LOG_EVERY == 0 or i == steps - 1:
            pred = jax.vmap(model)(x)

            print(
                f"step={i:5d} "
                f"loss={float(loss):.6e} "
                f"pred_RMS={float(rms(pred)):.4e} "
                f"target_RMS={float(rms(y)):.4e} "
                f"grad_norm={float(tree_l2_norm(grads)):.4e}"
            )

    final_pred = jax.vmap(model)(x)
    final_loss = float(jnp.mean((final_pred - y) ** 2))

    print_metrics("Final prediction", final_pred, y)

    print(
        f"Loss reduction: "
        f"{initial_loss / (final_loss + 1e-12):.2f}x"
    )

    return model


# ============================================================
# Gradient diagnostics
# ============================================================

def inspect_gradients(model, x, y, name):
    print(f"\nGradient diagnostics: {name}")

    def loss_fn(m):
        pred = jax.vmap(m)(x)
        return jnp.mean((pred - y) ** 2)

    grads = eqx.filter_grad(loss_fn)(model)

    print(f"Total gradient norm: {float(tree_l2_norm(grads)):.6e}")
    print(f"fc0 gradient norm:   {float(tree_l2_norm(grads.fc0)):.6e}")
    print(f"fc1 gradient norm:   {float(tree_l2_norm(grads.fc1)):.6e}")
    print(f"fc2 gradient norm:   {float(tree_l2_norm(grads.fc2)):.6e}")
    print(f"skip gradient norm:  {float(tree_l2_norm(grads.skip)):.6e}")

    for i, block in enumerate(grads.blocks):
        print(
            f"block {i}: "
            f"spectral={float(tree_l2_norm(block.spectral)):.6e}, "
            f"local={float(tree_l2_norm(block.local)):.6e}"
        )


# ============================================================
# Spectral operator response test
# ============================================================

def test_spectral_modes(key):
    print(f"\n{'=' * 65}")
    print("SPECTRAL OPERATOR MODE RESPONSE")
    print(f"{'=' * 65}")

    layer = SpectralConv2d(
        WIDTH,
        WIDTH,
        xmodes=MODES2,
        ymodes=MODES1,
        key=key,
    )

    yy = jnp.arange(HEIGHT)[:, None]
    xx = jnp.arange(WIDTH_GRID)[None, :]

    for ky in [1, 2, 4, 8, 12, 16, 20]:
        for kx in [1, 2, 4, 8, 12, 16, 20]:

            field = jnp.sin(
                2 * jnp.pi * (
                    ky * yy / HEIGHT
                    + kx * xx / WIDTH_GRID
                )
            )

            # SpectralConv2d expects (channels, H, W).
            x = jnp.zeros((WIDTH, HEIGHT, WIDTH_GRID))
            x = x.at[0].set(field)

            y = layer(x)

            print(
                f"input mode (ky={ky:2d}, kx={kx:2d}) "
                f"output RMS={float(rms(y)):.4e}"
            )

    # Check gradients through the operator.
    x = jax.random.normal(
        key,
        (WIDTH, HEIGHT, WIDTH_GRID),
    )

    def loss_fn(m):
        return jnp.mean(m(x) ** 2)

    grads = eqx.filter_grad(loss_fn)(layer)

    print(
        "SpectralConv2d gradient norm:",
        float(tree_l2_norm(grads)),
    )


# ============================================================
# Main experiment
# ============================================================

def main():
    key = jax.random.PRNGKey(SEED)
    key, data_key = jax.random.split(key)

    x, targets = make_data(data_key, BATCH_SIZE)

    print("Input shape:", x.shape)

    for name, y in targets.items():
        print(
            f"{name:12s}: "
            f"RMS={float(rms(y)):.5e}, "
            f"shape={y.shape}"
        )

    # --------------------------------------------------------
    # Experiment 1: one-example overfit
    # --------------------------------------------------------

    key, model_key = jax.random.split(key)

    model = TestBFNO(
        model_key,
        spectral_enabled=True,
        local_enabled=True,
        residual=False,
        use_skip=False,
    )

    train_model(
        model,
        x[:1],
        targets["laplacian"][:1],
        "One-example Laplacian overfit",
    )

    # --------------------------------------------------------
    # Experiment 2: tiny-batch overfit
    # --------------------------------------------------------

    key, model_key = jax.random.split(key)

    model = TestBFNO(
        model_key,
        spectral_enabled=True,
        local_enabled=True,
        residual=False,
        use_skip=False,
    )

    train_model(
        model,
        x[:TINY_BATCH_SIZE],
        targets["laplacian"][:TINY_BATCH_SIZE],
        "Tiny-batch Laplacian overfit",
    )

    # --------------------------------------------------------
    # Experiment 3: architectural ablations
    # --------------------------------------------------------

    variants = [
        ("Pointwise only", False, True, False, False),
        ("Spectral only", True, False, False, False),
        ("Combined", True, True, False, False),
        ("Combined + residual", True, True, True, False),
        ("Combined + direct skip", True, True, False, True),
        ("Combined + both skips", True, True, True, True),
    ]

    for (
        name,
        spectral_enabled,
        local_enabled,
        residual,
        use_skip,
    ) in variants:

        key, model_key = jax.random.split(key)

        model = TestBFNO(
            model_key,
            spectral_enabled=spectral_enabled,
            local_enabled=local_enabled,
            residual=residual,
            use_skip=use_skip,
        )

        inspect_gradients(
            model,
            x[:TINY_BATCH_SIZE],
            targets["laplacian"][:TINY_BATCH_SIZE],
            name,
        )

        train_model(
            model,
            x[:TINY_BATCH_SIZE],
            targets["laplacian"][:TINY_BATCH_SIZE],
            name,
        )

    # --------------------------------------------------------
    # Experiment 4: different synthetic targets
    # --------------------------------------------------------

    for target_name, y in targets.items():
        key, model_key = jax.random.split(key)

        model = TestBFNO(
            model_key,
            spectral_enabled=True,
            local_enabled=True,
            residual=True,
            use_skip=True,
        )

        train_model(
            model,
            x[:TINY_BATCH_SIZE],
            y[:TINY_BATCH_SIZE],
            f"Synthetic target: {target_name}",
        )

    # --------------------------------------------------------
    # Experiment 5: spectral operator tests
    # --------------------------------------------------------

    if RUN_SPECTRAL_TESTS:
        key, spectral_key = jax.random.split(key)
        test_spectral_modes(spectral_key)

    print("\nAll diagnostics completed.")


if __name__ == "__main__":
    main()
