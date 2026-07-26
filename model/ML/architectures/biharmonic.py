"""Biharmonic Smagorinsky eddy-viscosity closure for the resolved PV field."""
import equinox as eqx
import jax.numpy as jnp

class BiharmonicClosure(eqx.Module):
    ml = False
    smag_coeff: float  # Often denoted as C_S in literature

    def __init__(self, smag_coeff, cfg, **kwargs):
        self.smag_coeff = smag_coeff

    def __call__(self, state, *, model=None, dt=None):
        """Return the Biharmonic PV increment for one timestep.
        
        Implements variable-coefficient biharmonic Smagorinsky viscosity:
        Tendency = -div( A_4 * grad(laplacian(q)) )
        where A_4 = (C_S * Delta / pi)**4 * |D|
        and |D| is the horizontal deformation rate computed from streamfunction.
        """
        if dt is None:
            raise ValueError("BiharmonicSmagorinskyClosure requires `dt`.")
        if model is None:
            raise ValueError("BiharmonicSmagorinskyClosure requires the QG model.")

        qh = state.qh
        real_dtype = jnp.real(qh).dtype
        
        ikx = jnp.asarray(model.ik, dtype=qh.dtype)[None, :, :]
        iky = jnp.asarray(model.il, dtype=qh.dtype)[None, :, :]
        k2 = jnp.asarray(model.K2, dtype=qh.dtype)[None, :, :]
        delta = jnp.sqrt(jnp.asarray(model.dx * model.dy, dtype=real_dtype))

        # 1. Invert PV to get streamfunction
        psi_h = model.invert_pv(state)
        
        # 2. Compute Horizontal Deformation Strains using derivatives of psi
        # u = -d(psi)/dy  =>  du/dx = -d^2(psi)/dxdy
        # v =  d(psi)/dx  =>  dv/dy =  d^2(psi)/dxdy
        # Tension Strain (DT) = du/dx - dv/dy = -2 * d^2(psi)/dxdy
        psi_xy = model.spectral_to_real(ikx * iky * psi_h)
        tension_strain = -2.0 * psi_xy
        
        # Shearing Strain (DS) = dv/dx + du/dy = d^2(psi)/dx^2 - d^2(psi)/dy^2
        psi_xx = model.spectral_to_real(ikx * ikx * psi_h)
        psi_yy = model.spectral_to_real(iky * iky * psi_h)
        shearing_strain = psi_xx - psi_yy
        
        # Total deformation rate |D|
        deformation_rate = jnp.sqrt(tension_strain**2 + shearing_strain**2)
        
        # 3. Compute Biharmonic Smagorinsky Viscosity (A_4)
        # Scaling factor uses delta**4 as per Griffies & Hallberg / MITgcm standards
        viscosity_4 = ((self.smag_coeff * delta) / jnp.pi)**4 * deformation_rate

        # 4. Apply to hyper-viscous PV gradients
        lap_q_h = -k2 * qh
        lap_q_x = model.spectral_to_real(ikx * lap_q_h)
        lap_q_y = model.spectral_to_real(iky * lap_q_h)

        # 5. De-alias the nonlinear flux products before their divergence.
        flux_x_h = model.dealiased_product(viscosity_4 * lap_q_x)
        flux_y_h = model.dealiased_product(viscosity_4 * lap_q_y)
        
        div_flux_h = ikx * flux_x_h + iky * flux_y_h
        # -div(A_4 grad(laplacian(q))) is fourth-order, not sixth-order.
        tendency = model.spectral_to_real(-div_flux_h)
        return jnp.asarray(dt, dtype=real_dtype) * tendency
