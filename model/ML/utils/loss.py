import jax
import jax.numpy as jnp
import re
import logging
logger = logging.getLogger(__name__)


def build_loss(loss):
    """Build loss function from config string or list of strings.
    
    If loss is a weighted combo, check the elements are valid then combine.
    """
    registry = {
        "mse": MSELoss,
        "mae": MAELoss,
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

