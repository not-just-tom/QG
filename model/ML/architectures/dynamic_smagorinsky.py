"""Germano-Lilly dynamic Smagorinsky closure for the resolved PV field."""

import equinox as eqx
import jax.numpy as jnp


class DynamicSmagorinskyClosure(eqx.Module):
    """Estimate a non-negative Smagorinsky coefficient from a test-filter identity."""

    ml = False
    test_filter_ratio: float
    max_smag_coeff: float

    def __init__(
        self,
        test_filter_ratio=2.0,
        max_smag_coeff=0.23,
        cfg=None,
        **kwargs,
    ):
        if test_filter_ratio <= 1:
            raise ValueError("test_filter_ratio must be greater than 1")
        if max_smag_coeff < 0:
            raise ValueError("max_smag_coeff must be non-negative")
        self.test_filter_ratio = test_filter_ratio
        self.max_smag_coeff = max_smag_coeff

    def _test_filter(self, field, model, filter_mask):
        field_h = model.real_to_spectral(field)
        return model.spectral_to_real(field_h * filter_mask)

    def _test_filter_product(self, left, right, model, filter_mask):
        product_h = model.dealias_spectral(
            model.real_to_spectral(left * right)
        )
        return model.spectral_to_real(product_h * filter_mask)

    def __call__(self, state, *, model=None, dt=None):
        """Return a PV increment using a clipped dynamic viscosity estimate."""
        if dt is None:
            raise ValueError("DynamicSmagorinskyClosure requires the coarse timestep `dt`.")
        if model is None:
            raise ValueError("DynamicSmagorinskyClosure requires the QG model.")

        qh = state.qh
        real_dtype = jnp.real(qh).dtype
        ikx = jnp.asarray(model.ik, dtype=qh.dtype)[None, :, :]
        iky = jnp.asarray(model.il, dtype=qh.dtype)[None, :, :]
        delta = jnp.sqrt(jnp.asarray(model.dx * model.dy, dtype=real_dtype))
        test_delta = self.test_filter_ratio * delta
        test_filter_mask = (
            (jnp.abs(model.KX) <= jnp.pi / (self.test_filter_ratio * model.dx))
            & (jnp.abs(model.KY) <= jnp.pi / (self.test_filter_ratio * model.dy))
        )

        psi_h = model.invert_pv(state)
        u_h = -iky * psi_h
        v_h = ikx * psi_h
        u = model.spectral_to_real(u_h)
        v = model.spectral_to_real(v_h)

        def strain_components(u_h, v_h):
            u_x = model.spectral_to_real(ikx * u_h)
            u_y = model.spectral_to_real(iky * u_h)
            v_x = model.spectral_to_real(ikx * v_h)
            v_y = model.spectral_to_real(iky * v_h)
            s11 = u_x
            s22 = v_y
            s12 = 0.5 * (u_y + v_x)
            magnitude = jnp.sqrt(
                jnp.maximum(
                    2.0 * (s11**2 + s22**2 + 2.0 * s12**2),
                    0.0,
                )
            )
            return s11, s22, s12, magnitude

        s11, s22, s12, strain_magnitude = strain_components(u_h, v_h)

        u_test = self._test_filter(u, model, test_filter_mask)
        v_test = self._test_filter(v, model, test_filter_mask)
        u_test_h = model.real_to_spectral(u_test)
        v_test_h = model.real_to_spectral(v_test)
        s11_test, s22_test, s12_test, strain_magnitude_test = strain_components(
            u_test_h, v_test_h
        )

        l11 = (
            self._test_filter_product(u, u, model, test_filter_mask)
            - u_test**2
        )
        l22 = (
            self._test_filter_product(v, v, model, test_filter_mask)
            - v_test**2
        )
        l12 = (
            self._test_filter_product(u, v, model, test_filter_mask)
            - u_test * v_test
        )

        q11 = delta**2 * strain_magnitude * s11
        q22 = delta**2 * strain_magnitude * s22
        q12 = delta**2 * strain_magnitude * s12
        m11 = 2.0 * (
            self._test_filter(q11, model, test_filter_mask)
            - test_delta**2 * strain_magnitude_test * s11_test
        )
        m22 = 2.0 * (
            self._test_filter(q22, model, test_filter_mask)
            - test_delta**2 * strain_magnitude_test * s22_test
        )
        m12 = 2.0 * (
            self._test_filter(q12, model, test_filter_mask)
            - test_delta**2 * strain_magnitude_test * s12_test
        )

        numerator = jnp.mean(
            l11 * m11 + l22 * m22 + 2.0 * l12 * m12,
            axis=(-2, -1),
        )
        denominator = jnp.mean(
            m11**2 + m22**2 + 2.0 * m12**2,
            axis=(-2, -1),
        )
        eps = jnp.finfo(real_dtype).tiny
        coefficient_squared = jnp.where(
            denominator > eps,
            numerator / jnp.maximum(denominator, eps),
            0.0,
        )
        coefficient_squared = jnp.clip(
            coefficient_squared,
            0.0,
            self.max_smag_coeff**2,
        )
        viscosity = (
            coefficient_squared[:, None, None]
            * delta**2
            * strain_magnitude
        )

        qx = model.spectral_to_real(ikx * qh)
        qy = model.spectral_to_real(iky * qh)
        flux_x_h = model.dealiased_product(viscosity * qx)
        flux_y_h = model.dealiased_product(viscosity * qy)
        tendency = model.spectral_to_real(ikx * flux_x_h + iky * flux_y_h)
        return jnp.asarray(dt, dtype=real_dtype) * tendency

    @property
    def model_type(self):
        return "dynamic_smagorinsky"
