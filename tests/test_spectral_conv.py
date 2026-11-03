import jax
import jax.numpy as jnp
import numpy as np
import pytest
import equinox as eqx

from model.ML.architectures.fno import SpectralConv2d


def expected_spectral_output(layer, inputs):
    height, width = inputs.shape[-2:]
    input_spectrum = jnp.fft.rfft2(inputs, axes=(-2, -1))
    expected = jnp.zeros(
        (inputs.shape[0], layer.out_channels, height, width // 2 + 1),
        dtype=input_spectrum.dtype,
    )
    y_modes = min(layer.ymodes, (height - 1) // 2)
    x_modes = min(layer.xmodes, width // 2 + 1)

    if y_modes and x_modes:
        expected = expected.at[:, :, :y_modes, :x_modes].set(
            jnp.einsum(
                "bixy,ioxy->boxy",
                input_spectrum[:, :, :y_modes, :x_modes],
                layer.weights1[:, :, :y_modes, :x_modes],
            )
        )
        expected = expected.at[:, :, -y_modes:, :x_modes].set(
            jnp.einsum(
                "bixy,ioxy->boxy",
                input_spectrum[:, :, -y_modes:, :x_modes],
                layer.weights2[:, :, :y_modes, :x_modes],
            )
        )

    columns = [0]
    if width % 2 == 0:
        columns.append(width // 2)
    partners = (-jnp.arange(height)) % height
    boundary = expected[..., columns]
    expected = expected.at[..., columns].set(
        0.5 * (boundary + jnp.conj(jnp.take(boundary, partners, axis=-2)))
    )
    return expected


@pytest.mark.parametrize("height,width", [(8, 8), (7, 7), (8, 7)])
def test_spectral_output_matches_real_fft_boundary_projection(height, width):
    key, input_key = jax.random.split(jax.random.PRNGKey(height * 100 + width))
    layer = SpectralConv2d(
        in_channels=2,
        out_channels=3,
        xmodes=width // 2 + 1,
        ymodes=(height - 1) // 2,
        key=key,
    )
    inputs = jax.random.normal(input_key, (2, 2, height, width))

    actual = jnp.fft.rfft2(layer(inputs), axes=(-2, -1))
    expected = expected_spectral_output(layer, inputs)

    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=1e-5,
        atol=1e-5,
    )


def test_boundary_projection_preserves_gradients():
    height = width = 8
    layer = SpectralConv2d(
        in_channels=1,
        out_channels=1,
        xmodes=width // 2 + 1,
        ymodes=(height - 1) // 2,
        key=jax.random.PRNGKey(0),
    )
    inputs = jax.random.normal(jax.random.PRNGKey(1), (1, 1, height, width))

    def loss(weights1, weights2):
        current_layer = layer
        current_layer = eqx.tree_at(
            lambda module: module.weights1,
            current_layer,
            weights1,
        )
        current_layer = eqx.tree_at(
            lambda module: module.weights2,
            current_layer,
            weights2,
        )
        return jnp.mean(current_layer(inputs) ** 2)

    weights1_grad, weights2_grad = jax.grad(loss, argnums=(0, 1))(
        layer.weights1,
        layer.weights2,
    )
    assert jnp.all(jnp.isfinite(weights1_grad))
    assert jnp.all(jnp.isfinite(weights2_grad))
    assert jnp.linalg.norm(weights1_grad) > 0
    assert jnp.linalg.norm(weights2_grad) > 0
