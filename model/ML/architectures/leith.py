"""Leith eddy-viscosity closure for the resolved PV field."""
import equinox as eqx
import jax.numpy as jnp

class LeithClosure(eqx.Module):
    ml = False
    leith_coeff: float

    def __init__(self, leith_coeff, cfg, **kwargs):
        self.leith_coeff = leith_coeff

    def __call__(self, state, *, model=None, dt=None):
        """Return the Leith PV increment for one coarse model timestep.

        The model supplies the QG PV inversion, giving streamfunction in each
        layer.  Relative vorticity is then ``zeta = laplacian(psi)`` and the
        layerwise Leith viscosity is ``C_L Delta**3 |grad(zeta)|``.  The
        closure itself is ``div(nu grad(q))``.  This is valid for both the
        one- and multi-layer PV inversions implemented by the model.
        """
        if dt is None:
            raise ValueError("LeithClosure requires the coarse timestep `dt`.")
        if model is None:
            raise ValueError("LeithClosure requires the QG model for PV inversion.")

        qh = state.qh
        real_dtype = jnp.real(qh).dtype
        ikx = jnp.asarray(model.ik, dtype=qh.dtype)[None, :, :]
        iky = jnp.asarray(model.il, dtype=qh.dtype)[None, :, :]
        delta = jnp.sqrt(jnp.asarray(model.dx * model.dy, dtype=real_dtype))

        # Use the model's existing one- or two-layer PV inversion.  This
        # avoids treating PV as relative vorticity in the baroclinic case.
        psi_h = model.invert_pv(state)
        zeta_h = -jnp.asarray(model.K2, dtype=qh.dtype)[None, :, :] * psi_h
        zeta_x = model.spectral_to_real(ikx * zeta_h)
        zeta_y = model.spectral_to_real(iky * zeta_h)
        zeta_gradient = jnp.sqrt(zeta_x**2 + zeta_y**2)
        viscosity = self.leith_coeff * delta**3 * zeta_gradient

        qx = model.spectral_to_real(ikx * qh)
        qy = model.spectral_to_real(iky * qh)

        # These are nonlinear products.  Match the core advection path by
        # filtering them before taking their spectral divergence.
        flux_x_h = model.dealiased_product(viscosity * qx)
        flux_y_h = model.dealiased_product(viscosity * qy)
        tendency = model.spectral_to_real(ikx * flux_x_h + iky * flux_y_h)
        return jnp.asarray(dt, dtype=real_dtype) * tendency
