import jax
import jax.numpy as jnp
import re
import logging
from model.core import states

logger = logging.getLogger(__name__)


def build_loss(loss):
    """Build loss function from config string or list of strings.
    
    If loss is a weighted combo, check the elements are valid then combine.
    """
    registry = {
        "mse": MSELoss,
        "mae": MAELoss,
        "ke": KELoss,
        "spectral": SpectralEnergyLoss,
        "multistep": MultiStepStatisticsLoss,
        "maddison": maddison_loss,
    }

    # Combination losses 
    extracted_elements = re.findall(r"[a-zA-Z]+", loss)
    all_valid = all(elem in registry for elem in extracted_elements)
    if all_valid:
        loss_fns = [registry[elem] for elem in extracted_elements]
        logger.info(f"Using combined loss functions: {extracted_elements}")
        
        def combined_loss(residual_q, lr_model):
            """Combine multiple loss functions."""
            total_loss = 0.0
            for loss_fn in loss_fns:
                total_loss = total_loss + loss_fn(residual_q, lr_model)
            return total_loss
        
        return combined_loss
    else:
        raise ValueError(
            f"Unknown loss choice(s) {extracted_elements}, available: {sorted(registry.keys())}"
        )

def KELoss(residual_q, lr_model):
    """Kinetic-energy norm of the velocity error implied by a PV residual.

    The PV residual is inverted to a streamfunction, differentiated to obtain
    velocity errors, then reduced using layer-depth-weighted kinetic energy.
    For batched input, the batch axis is preserved and one loss is returned per
    sample. This is not the squared difference between two scalar KE values.
    """
    if residual_q.ndim not in (4, 5):
        raise ValueError(
            "residual_q must have shape (nsteps, nz, ny, nx) or "
            "(batch, nsteps, nz, ny, nx)"
        )

    nz, ny, nx = residual_q.shape[-3:]
    if nz != lr_model.nz:
        raise ValueError(
            f"residual_q has {nz} layers, but the model has {lr_model.nz}"
        )

    residual_qh = jnp.fft.rfftn(
        residual_q, axes=(-2, -1), norm="ortho"
    )
    fields_per_sample = residual_qh.reshape(
        (-1, nz, ny, nx // 2 + 1)
    )

    def velocity_error(field_qh):
        state = states.State(qh=field_qh, _q_shape=(ny, nx))
        ph = lr_model.invert_pv(state)
        uh = -1j * lr_model.ky[None, :, None] * ph
        vh = 1j * lr_model.kx[None, None, :] * ph
        u = lr_model.spectral_to_real(uh)
        v = lr_model.spectral_to_real(vh)
        return u, v

    u_error, v_error = jax.vmap(velocity_error)(fields_per_sample)
    u_error = u_error.reshape(residual_q.shape)
    v_error = v_error.reshape(residual_q.shape)

    layer_depths = lr_model.Lz
    if layer_depths.ndim == 2:
        layer_depths = layer_depths[0]
    if layer_depths.ndim != 1 or layer_depths.shape[0] < nz:
        raise ValueError(
            f"model layer depths have shape {layer_depths.shape}, expected at "
            f"least {nz} layer values"
        )
    layer_depths = layer_depths[:nz]
    layer_weights = layer_depths / jnp.sum(layer_depths)
    layer_weights = layer_weights.reshape(
        (1,) * (residual_q.ndim - 3) + (nz, 1, 1)
    )
    energy_error = 0.5 * layer_weights * (
        u_error**2 + v_error**2
    )
    energy_error = jnp.sum(energy_error, axis=-3)

    if residual_q.ndim == 5:
        return jnp.mean(energy_error, axis=(1, 2, 3))
    return jnp.mean(energy_error)


def MSELoss(residual_q, lr_model=None):
    """MSE loss: mean squared distance between predicted and target states.

    Args:
        residual_q: Error in q (physical space). Shape: (batch, nsteps, nz, ny, nx) 
                    or (nsteps, nz, ny, nx)
        lr_model: Low-resolution model (not used for L2, but kept for interface consistency)
    
    Returns:
        Per-sample loss if batch dimension present, otherwise scalar.
    """
    axes = tuple(range(1, residual_q.ndim)) if residual_q.ndim > 4 else None
    return jnp.mean(residual_q**2, axis=axes)


def SpectralEnergyLoss(residual_q, lr_model):
    """Spectral energy loss: log-spectral distance of kinetic energy.
    
    LE = integral_k log(E_s(k) / E_q^sτ(k))^2 dk
    
    This loss improves accuracy on fine spatial scales by comparing kinetic energy
    spectra in Fourier space.
    
    Args:
        residual_q: Error in q (physical space). Shape: (batch, nsteps, nz, ny, nx) 
                    or (nsteps, nz, ny, nx)
        lr_model: Low-resolution model with spectral properties
    
    Returns:
        Per-sample loss if batch dimension present, otherwise scalar.
    """
    
    # Convert residual to spectral space
    # rfftn automatically handles multi-dimensional inputs
    residual_qh = jnp.fft.rfftn(residual_q, axes=(-2, -1), norm='ortho')
    
    # Compute kinetic energy spectrum: E(k) = |q|^2
    energy_spec = jnp.abs(residual_qh) ** 2
    
    # Average energy over spatial modes - keep spectrum structure
    # Shape after mean: (batch, nsteps, nz) or (nsteps, nz) depending on input
    energy_mag = jnp.mean(energy_spec, axis=tuple(range(-2, 0)))  # Average over spatial dims
    
    # Avoid log(0) by adding small epsilon
    eps = 1e-10
    energy_mag = jnp.maximum(energy_mag, eps)
    
    # Log-spectral distance: mean of log of energy
    log_energy = jnp.log(energy_mag)
    spectral_loss = jnp.mean(log_energy ** 2)
    
    return spectral_loss

def MultiStepStatisticsLoss(residual_q, lr_model):
    """Multi-step statistics loss: match averaged quantities over unrolled steps.
    
   LMS = ||mean_s(u_s) - mean_s(q(u_s^τ))||
    
    This ensures long-term accuracy by matching the mean flow. Only applied to
    statistically steady simulations.
    
    Args:
        residual_q: Error in q (physical space). Shape: (batch, nsteps, nz, ny, nx) 
                    or (nsteps, nz, ny, nx)
        lr_model: Low-resolution model
    
    Returns:
        Per-sample loss if batch dimension present, otherwise scalar.
    """
    # Average residuals over time steps (unrolled simulation steps)
    if residual_q.ndim == 5:
        # Shape: (batch, nsteps, nz, ny, nx)
        # Average over steps dimension (axis 1)
        residual_q_averaged = jnp.mean(residual_q, axis=1)
    else:
        # Shape: (nsteps, nz, ny, nx)
        # Average over steps dimension (axis 0)
        residual_q_averaged = jnp.mean(residual_q, axis=0)
    
    # MSE of averaged residuals represents error in mean flow
    multistep_loss = jnp.mean(residual_q_averaged ** 2)
    
    return multistep_loss


def maddison_loss(residual_q, lr_model, beta=None, scale_factor=1e4):
    """Compute loss per Maddison (2026) eqn 7: spatial interior weighting, no boundary.
    scale_factor: front multiplier (10^4 in paper, very odd)
    beta: if None, uses lr_model.beta; otherwise uses provided value
    
    Returns per-sample loss if input has batch dimension, otherwise scalar.
    """
    if beta is None:
        beta = float(getattr(lr_model, 'beta', None))
    
    ny, nx = residual_q.shape[-2:]
    interior_mask = jnp.ones((ny, nx))
    dx = lr_model.get_grid().dx
    L = float(lr_model.Lx)
    
    # Normalization
    norm_factor = scale_factor / (beta**2 * L**2 * 4.0 * L**2)
    
    # Squared residual weighted by interior mask and grid spacing dx^2
    weighted_residual_sq = (residual_q ** 2) * interior_mask[None, None, :, :] * (dx**2)
    
    # Mean over all but the first (batch) dimension if present
    # residual_q shape: (batch, nsteps, nz, ny, nx) or (nsteps, nz, ny, nx)
    axes = tuple(range(1, weighted_residual_sq.ndim)) if weighted_residual_sq.ndim > 4 else None
    loss = norm_factor * jnp.mean(weighted_residual_sq, axis=axes)
    return loss      

    
def MAELoss(residual_q, lr_model=None):
    """Mean absolute error loss. Returns per-sample loss if batch dimension present."""
    # Mean over all but the first (batch) dimension if present
    axes = tuple(range(1, residual_q.ndim)) if residual_q.ndim > 4 else None
    return jnp.mean(jnp.abs(residual_q), axis=axes)

