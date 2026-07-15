import jax
import jax.numpy as jnp
import logging
logger = logging.getLogger(__name__)


def build_loss(loss):
    """Build loss function from config string or list of strings.
    
    If loss is a list, returns a combined loss that sums individual losses.
    """
    registry = {
        "mse": MSELoss,
        "mae": MAELoss,
        "spectral": SpectralEnergyLoss,
        "strain": RateOfStrainLoss,
        "multistep": MultiStepStatisticsLoss,
        "maddison": maddison_loss,
    }

    # Handle list of losses (combined loss)
    if isinstance(loss, list):
        loss_fns = [registry.get(l) for l in loss]
        if any(fn is None for fn in loss_fns):
            invalid = [l for l, fn in zip(loss, loss_fns) if fn is None]
            raise ValueError(
                f"Unknown loss choice(s) {invalid}, available: {sorted(registry.keys())}"
            )
        logger.info(f"Using combined loss functions: {loss}")
        
        def combined_loss(residual_q, lr_model):
            """Combine multiple loss functions."""
            total_loss = 0.0
            for loss_fn in loss_fns:
                total_loss = total_loss + loss_fn(residual_q, lr_model)
            return total_loss
        
        return combined_loss
    
    # Handle single loss string
    cls = registry.get(loss)
    if cls is None:
        raise ValueError(
            f"Unknown loss choice '{loss}', available: {sorted(registry.keys())}"
        )
    logger.info(f"Using loss function: {loss}")

    return cls


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
    # Store original shape for batch handling
    is_batched = residual_q.ndim == 5
    
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


def RateOfStrainLoss(residual_q, lr_model):
    """Rate of strain loss: L1 norm of strain rate tensor differences.
    
    LS = sum_ij |S_ij,s - S_ij,s^τ|
    
    where S_ij = 0.5(∂u_i/∂x_j + ∂u_j/∂x_i) is the rate of strain tensor.
    
    This ensures the network output carries information necessary for accurate
    energy transfer computation in the next step.
    
    Args:
        residual_q: Error in q (physical space). Shape: (batch, nsteps, nz, ny, nx) 
                    or (nsteps, nz, ny, nx)
        lr_model: Low-resolution model
    
    Returns:
        Per-sample loss if batch dimension present, otherwise scalar.
    """
    # Convert residual_q to spectral space
    # rfftn will handle the multi-dimensional input correctly
    residual_qh = jnp.fft.rfftn(residual_q, axes=(-2, -1), norm='ortho')
    
    # Get grid spacing for derivative computation
    grid = lr_model.get_grid()
    dy = jnp.asarray(grid.dy, dtype=residual_qh.dtype)
    dx = jnp.asarray(grid.dx, dtype=residual_qh.dtype)
    
    # Get wavenumber grids from model if available, otherwise compute
    if hasattr(lr_model, '_ky') and hasattr(lr_model, '_kx'):
        ky = lr_model._ky
        kx = lr_model._kx
    else:
        # Compute wavenumber grids
        ny = residual_q.shape[-2]
        nx = residual_q.shape[-1]
        kx = jnp.fft.rfftfreq(nx, d=dx / (2 * jnp.pi))
        ky = jnp.fft.fftfreq(ny, d=dy / (2 * jnp.pi))
    
    # Create meshgrid for wavenumbers
    # ky has shape (ny,), kx has shape (nx//2+1,)
    # After meshgrid: KY has shape (ny, nx//2+1), KX has shape (ny, nx//2+1)
    KY, KX = jnp.meshgrid(ky, kx, indexing='ij')
    
    # Compute velocity from streamfunction via inversion
    # For QG: u = -∂ψ/∂y, v = ∂ψ/∂x
    # In spectral: uh = -i*ky*psih, vh = i*kx*psih
    # Broadcast KY and KX to match residual_qh dimensions
    residual_uh = -1j * KY * residual_qh
    residual_vh = 1j * KX * residual_qh
    
    # Transform to physical space
    ny = residual_q.shape[-2]
    nx = residual_q.shape[-1]
    residual_u = jnp.fft.irfftn(residual_uh, axes=(-2, -1), norm='ortho', s=(ny, nx))
    residual_v = jnp.fft.irfftn(residual_vh, axes=(-2, -1), norm='ortho', s=(ny, nx))
    
    # Compute spatial derivatives via finite differences
    # Using central differences on last two dimensions (y, x)
    du_dx = (jnp.roll(residual_u, -1, axis=-1) - jnp.roll(residual_u, 1, axis=-1)) / (2 * dx)
    du_dy = (jnp.roll(residual_u, -1, axis=-2) - jnp.roll(residual_u, 1, axis=-2)) / (2 * dy)
    dv_dx = (jnp.roll(residual_v, -1, axis=-1) - jnp.roll(residual_v, 1, axis=-1)) / (2 * dx)
    dv_dy = (jnp.roll(residual_v, -1, axis=-2) - jnp.roll(residual_v, 1, axis=-2)) / (2 * dy)
    
    # Compute rate of strain tensor: S_ij = 0.5(∂u_i/∂x_j + ∂u_j/∂x_i)
    S_xx = du_dx  # S_xx = ∂u/∂x
    S_yy = dv_dy  # S_yy = ∂v/∂y
    S_xy = 0.5 * (du_dy + dv_dx)  # S_xy = 0.5(∂u/∂y + ∂v/∂x)
    
    # Compute L1 norm of strain rate (sum of absolute values)
    strain_loss = jnp.mean(jnp.abs(S_xx) + jnp.abs(S_yy) + jnp.abs(S_xy))
    
    return strain_loss


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
        beta = float(getattr(lr_model, 'beta', 10.0))
    
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

