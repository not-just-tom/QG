"""Pointwise MLP closure using local physical features."""

import equinox as eqx
import jax
import jax.numpy as jnp


class MLP(eqx.Module):
    """Apply one shared fully connected network at every grid location."""

    layers: list[eqx.nn.Linear]
    nlayers: int
    width: int
    activation: str
    smag_coeff: float
    leith_coeff: float

    requires_physics_features = True

    def __init__(
        self,
        key=jax.random.PRNGKey(0),
        nlayers=3,
        width=20,
        activation="tanh",
        smag_coeff=5.0,
        leith_coeff=5.0,
        cfg=None,
        **kwargs,
    ):
        if nlayers < 2:
            raise ValueError("nlayers must be >= 2 (including the output layer)")
        if width < 1:
            raise ValueError("width must be positive")
        if smag_coeff < 0 or leith_coeff < 0:
            raise ValueError("Smagorinsky and Leith coefficients must be non-negative")

        supported_activations = {"tanh", "gelu", "relu", "elu", "leaky_relu"}
        activation = str(activation).lower()
        if activation not in supported_activations:
            raise ValueError(f"Unsupported activation: {activation}")

        keys = jax.random.split(key, nlayers)
        layers = []
        for i in range(nlayers):
            in_size = 20 if i == 0 else width
            out_size = 1 if i == nlayers - 1 else width
            layers.append(eqx.nn.Linear(in_size, out_size, key=keys[i]))

        self.layers = layers
        self.nlayers = nlayers
        self.width = width
        self.activation = activation
        self.smag_coeff = smag_coeff
        self.leith_coeff = leith_coeff

    def _activate(self, x):
        if self.activation == "tanh":
            return jnp.tanh(x)
        if self.activation == "gelu":
            return jax.nn.gelu(x)
        if self.activation == "relu":
            return jax.nn.relu(x)
        if self.activation == "elu":
            return jax.nn.elu(x)
        return jax.nn.leaky_relu(x)

    def _features(self, state, model):
        """Return (layer, feature, y, x) arrays: vorticity, streamfunction, viscosities."""
        psi_h = model.invert_pv(state)
        psi = model.spectral_to_real(psi_h)
        k2 = jnp.asarray(model.K2, dtype=psi_h.dtype)[None, :, :]
        zeta_h = -k2 * psi_h
        zeta = model.spectral_to_real(zeta_h)

        ikx = jnp.asarray(model.ik, dtype=psi_h.dtype)[None, :, :]
        iky = jnp.asarray(model.il, dtype=psi_h.dtype)[None, :, :]
        u_h = -iky * psi_h
        v_h = ikx * psi_h
        u_x = model.spectral_to_real(ikx * u_h)
        u_y = model.spectral_to_real(iky * u_h)
        v_x = model.spectral_to_real(ikx * v_h)
        v_y = model.spectral_to_real(iky * v_h)
        strain_rate = jnp.sqrt(
            jnp.maximum(
                2 * u_x**2 + 2 * v_y**2 + (u_y + v_x) ** 2,
                0.0,
            )
        )

        zeta_x = model.spectral_to_real(ikx * zeta_h)
        zeta_y = model.spectral_to_real(iky * zeta_h)
        zeta_gradient = jnp.sqrt(zeta_x**2 + zeta_y**2)
        delta = jnp.sqrt(jnp.asarray(model.dx * model.dy, dtype=psi.dtype))
        smag_viscosity = (self.smag_coeff * delta) ** 2 * strain_rate
        leith_viscosity = self.leith_coeff * delta**3 * zeta_gradient

        features = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                features.append(jnp.roll(zeta, shift=(-dy, -dx), axis=(-2, -1)))
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                features.append(jnp.roll(psi, shift=(-dy, -dx), axis=(-2, -1)))
        features.extend((smag_viscosity, leith_viscosity))
        return jnp.stack(features, axis=1)

    def __call__(self, state, model=None):
        if model is None:
            raise ValueError("MLP requires the QG model to construct physical features")

        features = self._features(state, model)
        nz, _, ny, nx = features.shape
        point_features = jnp.moveaxis(features, 1, -1).reshape((-1, 20))

        def apply_point(x):
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i < self.nlayers - 1:
                    x = self._activate(x)
            return x[0]

        prediction = jax.vmap(apply_point)(point_features)
        return prediction.reshape((nz, ny, nx))

    @property
    def model_type(self):
        return "mlp"
