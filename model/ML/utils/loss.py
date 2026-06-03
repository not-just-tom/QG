import jax
import jax.numpy as jnp
import logging
logger = logging.getLogger(__name__)


def build_loss(loss):
    registry = {
        "mse": MSELoss,
        "mae": MAELoss,
        "spectral": spectral_loss,
        "maddison": maddison_loss,
    }

    cls = registry.get(loss)
    if cls is None:
        raise ValueError(
            f"Unknown loss choice '{loss}', available: {sorted(registry.keys())}"
        )
    logger.info(f"Using loss function: {loss}")

    return cls

def spectral_loss(pred_qh, target_qh):
    pred_energy = jnp.sum(pred_qh**2, axis=-1)
    target_energy = jnp.sum(target_qh**2, axis=-1)
    energy_diff = pred_energy - target_energy
    return jnp.mean(energy_diff**2)

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
    
def MSELoss(err, lr_model=None):
    """Compute MSE loss. Returns per-sample loss if batch dimension present."""
    # Mean over all but the first (batch) dimension if present
    # err shape: (batch, nsteps, nz, ny, nx) or (nsteps, nz, ny, nx)
    axes = tuple(range(1, err.ndim)) if err.ndim > 4 else None
    return jnp.mean(err**2, axis=axes)

def MAELoss(err, lr_model=None):
    """Compute MAE loss. Returns per-sample loss if batch dimension present."""
    # Mean over all but the first (batch) dimension if present
    axes = tuple(range(1, err.ndim)) if err.ndim > 4 else None
    return jnp.mean(jnp.abs(err), axis=axes)

