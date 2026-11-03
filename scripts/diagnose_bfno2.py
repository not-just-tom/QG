
#!/usr/bin/env python3

import argparse
import importlib
import sys

import jax
import jax.numpy as jnp
import equinox as eqx
import optax


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

def make_bandlimited_data(
    key,
    n=32,
    height=64,
    width=64,
    cutoff=7,
):
    key1, _ = jax.random.split(key)

    q = jax.random.normal(
        key1,
        (n, 1, height, width),
        dtype=jnp.float32,
    )

    ky = jnp.fft.fftfreq(height) * height
    kx = jnp.fft.rfftfreq(width) * width

    KY, KX = jnp.meshgrid(ky, kx, indexing="ij")

    mask = (
        (jnp.abs(KY) <= cutoff)
        & (jnp.abs(KX) <= cutoff)
    )

    qhat = jnp.fft.rfft2(q, axes=(-2, -1))
    qhat = qhat * mask[None, None, :, :]

    q = jnp.fft.irfft2(
        qhat,
        s=(height, width),
        axes=(-2, -1),
    )

    k2 = KY**2 + KX**2

    lap_hat = -k2[None, None, :, :] * qhat

    lap = jnp.fft.irfft2(
        lap_hat,
        s=(height, width),
        axes=(-2, -1),
    )

    q_rms = jnp.sqrt(jnp.mean(q**2))
    lap_rms = jnp.sqrt(jnp.mean(lap**2))

    scale = q_rms / (lap_rms + 1e-12)

    target = scale * lap

    return q, target, KY, KX, scale


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def rms(x):
    return float(jnp.sqrt(jnp.mean(jnp.abs(x)**2)))


def mae(pred, target):
    return float(jnp.mean(jnp.abs(pred - target)))


def rel_l2(pred, target):
    return float(
        jnp.sqrt(jnp.mean(jnp.abs(pred - target)**2))
        / (jnp.sqrt(jnp.mean(jnp.abs(target)**2)) + 1e-12)
    )


def tree_grad_norm(grads):
    leaves = jax.tree_util.tree_leaves(grads)

    squared_norms = [
        jnp.sum(jnp.abs(x)**2)
        for x in leaves
        if x is not None
    ]

    return jnp.sqrt(sum(squared_norms))


# ---------------------------------------------------------------------
# Model loading and construction
# ---------------------------------------------------------------------

def load_spectral_class(module_name, class_name):
    module = importlib.import_module(module_name)

    cls = getattr(module, class_name, None)

    if cls is None:
        raise AttributeError(
            f"{module_name!r} does not expose {class_name!r}"
        )

    return cls


def make_layer(cls, key, modes):
    return cls(
        in_channels=1,
        out_channels=1,
        xmodes=modes,
        ymodes=modes,
        key=key,
        kpower=0.0,
    )


# ---------------------------------------------------------------------
# Parameter modification
# ---------------------------------------------------------------------

def set_parameter(model, name, value):
    return eqx.tree_at(
        lambda m: getattr(m, name),
        model,
        value,
    )


def set_unit_weights(model):
    """Independent model with unit spectral weights."""

    model = set_parameter(
        model,
        "weights1",
        jnp.ones_like(model.weights1),
    )

    model = set_parameter(
        model,
        "weights2",
        jnp.ones_like(model.weights2),
    )

    if hasattr(model, "gate"):
        model = set_parameter(
            model,
            "gate",
            jnp.asarray(1.0, dtype=jnp.float32),
        )

    return model


def set_zero_weights(model):
    """Independent model with zero spectral weights."""

    model = set_parameter(
        model,
        "weights1",
        jnp.zeros_like(model.weights1),
    )

    model = set_parameter(
        model,
        "weights2",
        jnp.zeros_like(model.weights2),
    )

    if hasattr(model, "gate"):
        model = set_parameter(
            model,
            "gate",
            jnp.asarray(1.0, dtype=jnp.float32),
        )

    return model


def set_oracle_laplacian_weights(model, scale):
    """
    Install exact Laplacian multipliers.

    weights1 rows:
        ky = 0, 1, ..., ymodes-1

    weights2 rows:
        ky = -ymodes, ..., -1
    """

    w1 = jnp.zeros_like(model.weights1)
    w2 = jnp.zeros_like(model.weights2)

    ymodes = int(model.ymodes)
    xmodes = int(model.xmodes)

    for iy in range(ymodes):
        ky = iy

        for ix in range(xmodes):
            kx = ix

            value = -scale * (ky**2 + kx**2)

            w1 = w1.at[0, 0, iy, ix].set(value + 0.0j)

    for iy in range(ymodes):
        ky = -(ymodes - iy)

        for ix in range(xmodes):
            kx = ix

            value = -scale * (ky**2 + kx**2)

            w2 = w2.at[0, 0, iy, ix].set(value + 0.0j)

    model = set_parameter(model, "weights1", w1)
    model = set_parameter(model, "weights2", w2)

    if hasattr(model, "gate"):
        model = set_parameter(
            model,
            "gate",
            jnp.asarray(1.0, dtype=jnp.float32),
        )

    return model


# ---------------------------------------------------------------------
# Test 1: mode response
# ---------------------------------------------------------------------

def test_single_modes(model, height=64, width=64):

    print("\n[1] UNIT-WEIGHT SINGLE-MODE RESPONSE")

    unit_model = set_unit_weights(model)

    cases = [
        (1, 1, True),
        (3, 2, True),
        (7, 7, True),
        (8, 1, False),
        (1, 8, False),
        (16, 2, False),
    ]

    for ky, kx, should_pass in cases:

        yy = jnp.arange(height)[:, None]
        xx = jnp.arange(width)[None, :]

        field = jnp.cos(
            2 * jnp.pi * (
                ky * yy / height
                + kx * xx / width
            )
        )

        field = field[None, None, :, :].astype(jnp.float32)

        out = unit_model(field)

        ratio = rms(out) / (rms(field) + 1e-12)

        expected = "PASS" if should_pass else "SUPPRESS"

        print(
            f"  ky={ky:2d}, kx={kx:2d}: "
            f"output/input RMS={ratio:.6f} "
            f"(expected {expected})"
        )


# ---------------------------------------------------------------------
# Test 2: oracle Laplacian
# ---------------------------------------------------------------------

def test_oracle(model, q, target, scale):

    print("\n[2] ORACLE LAPLACIAN (NO TRAINING)")

    oracle_model = set_oracle_laplacian_weights(
        model,
        scale,
    )

    pred = jax.vmap(oracle_model)(q)

    print(f"  target RMS       : {rms(target):.6e}")
    print(f"  prediction RMS   : {rms(pred):.6e}")
    print(f"  target MAE       : {mae(jnp.zeros_like(target), target):.6e}")
    print(f"  oracle MAE       : {mae(pred, target):.6e}")
    print(f"  oracle rel L2    : {rel_l2(pred, target):.6e}")

    return oracle_model


# ---------------------------------------------------------------------
# Test 3: learn Laplacian from zero initialisation
# ---------------------------------------------------------------------

def train_single_spectral(
    cls,
    key,
    q,
    target,
    modes,
    lr=1e-3,
    steps=5000,
):

    print("\n[3] LEARNED SINGLE-SPECTRAL-LAYER LAPLACIAN FIT")

    model = make_layer(cls, key, modes)

    # IMPORTANT: explicitly zero-initialise the training model.
    model = set_zero_weights(model)

    initial_weight_rms = jnp.sqrt(
        (
            jnp.mean(jnp.abs(model.weights1)**2)
            + jnp.mean(jnp.abs(model.weights2)**2)
        ) / 2.0
    )

    print(
        f"  initial spectral weight RMS: "
        f"{float(initial_weight_rms):.6e}"
    )

    # Verify that the initial prediction is zero.
    pred0 = jax.vmap(model)(q)

    print(
        f"  initial prediction RMS: {rms(pred0):.6e}"
    )

    optimizer = optax.adam(lr)

    opt_state = optimizer.init(
        eqx.filter(model, eqx.is_array)
    )

    @eqx.filter_jit
    def train_step(model, opt_state, x, y):

        def loss_fn(m):
            pred = jax.vmap(m)(x)
            return jnp.mean((pred - y)**2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)

        filtered_grads = eqx.filter(grads, eqx.is_array)

        updates, opt_state = optimizer.update(
            filtered_grads,
            opt_state,
            eqx.filter(model, eqx.is_array),
        )

        model = eqx.apply_updates(model, updates)

        grad_norm = tree_grad_norm(filtered_grads)

        return model, opt_state, loss, grad_norm

    for step in range(steps):

        model, opt_state, loss, grad_norm = train_step(
            model,
            opt_state,
            q,
            target,
        )

        if step % 250 == 0 or step == steps - 1:

            pred = jax.vmap(model)(q)

            print(
                f"  step={step:5d} "
                f"MSE={float(loss):.6e} "
                f"MAE={mae(pred, target):.6e} "
                f"relL2={rel_l2(pred, target):.6e} "
                f"predRMS={rms(pred):.6e} "
                f"grad={float(grad_norm):.3e}"
            )

    return model


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--spectral-module",
        default="spectral_conv",
    )

    parser.add_argument(
        "--spectral-class",
        default="SpectralConv2d",
    )

    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cutoff", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1234)

    args = parser.parse_args()

    print("=" * 72)
    print("BFNO SPECTRAL OPERATOR DIAGNOSTIC")
    print("=" * 72)

    print(
        f"grid={args.height}x{args.width}, "
        f"modes={args.modes}, "
        f"bandlimit={args.cutoff}, "
        f"batch={args.batch_size}, "
        f"lr={args.learning_rate:g}, "
        f"steps={args.steps}"
    )

    if args.cutoff >= args.modes:
        raise ValueError(
            "cutoff must be strictly less than modes. "
            "Use cutoff=7 with modes=8."
        )

    try:
        SpectralConv2d = load_spectral_class(
            args.spectral_module,
            args.spectral_class,
        )

    except Exception as exc:
        print(f"\nCould not import SpectralConv2d: {exc}")
        sys.exit(2)

    key = jax.random.PRNGKey(args.seed)

    key, data_key, model_key, train_key = jax.random.split(
        key,
        4,
    )

    q, target, _, _, scale = make_bandlimited_data(
        data_key,
        n=args.batch_size,
        height=args.height,
        width=args.width,
        cutoff=args.cutoff,
    )

    print(f"\nInput RMS : {rms(q):.6e}")
    print(f"Target RMS: {rms(target):.6e}")
    print(f"Laplacian normalisation scale: {float(scale):.6e}")

    # Base model is only a template. Each test constructs its own
    # independent model with the appropriate weights.
    base_model = make_layer(
        SpectralConv2d,
        model_key,
        args.modes,
    )

    # Test 1: independent unit-weight model.
    test_single_modes(
        base_model,
        args.height,
        args.width,
    )

    # Test 2: independent oracle-weight model.
    test_oracle(
        base_model,
        q,
        target,
        scale,
    )

    # Test 3: independent zero-initialised training model.
    train_single_spectral(
        SpectralConv2d,
        train_key,
        q,
        target,
        args.modes,
        lr=args.learning_rate,
        steps=args.steps,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
