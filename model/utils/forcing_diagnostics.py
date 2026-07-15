"""Diagnostics and utilities for stochastic forcing in QG models.

This module provides tools to:
1. Monitor forcing-induced changes in model state
2. Compute appropriate forcing amplitudes based on system energy
3. Detect and diagnose NaN/Inf issues related to forcing
4. Suggest better forcing parameter choices
"""

import logging
import numpy as np
import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


def tune_forcing_parameters(model, state, n_jets, verbose=False):
    """Derive forcing parameters from the requested jet count.

    This keeps the forcing centered on the target jet scale and uses a conservative
    amplitude that scales with the initialized state so the system remains close to the
    requested balance while avoiding large stability errors.
    """
    if n_jets is None or n_jets <= 0:
        return None

    target_wavenumber = (2.0 * jnp.pi * jnp.asarray(n_jets, dtype=model.Kmag.dtype)) / float(model.Ly)
    target_wavenumber = jnp.clip(target_wavenumber, 1.0, jnp.max(model.Kmag))

    U_target = model.beta * (model.Ly / (jnp.pi * n_jets)) ** 2
    U_rms = model.rhines_length(state)[1]
    scaler = U_target / (U_rms + 1e-12)

    forcing_center = target_wavenumber
    forcing_width = jnp.maximum(target_wavenumber / 2.0, 1.0)
    forcing_width = jnp.minimum(forcing_width, 2.0 * target_wavenumber)
    
    # Keep the amplitude small enough to avoid destabilizing the tuned initial state.
    forcing_amplitude = jnp.sqrt(jnp.maximum(scaler, 1e-12) * 1e-4)
    forcing_amplitude = jnp.clip(forcing_amplitude, 1e-8, 1e-3)

    model.forcing_center = forcing_center
    model.forcing_width = forcing_width
    model.forcing_amplitude = forcing_amplitude
    model._update_forcing_mask()

    if verbose:
        jax.debug.print(
            "[forcing-tune] n_jets={n_jets} -> center={center} width={width} amplitude={amplitude} kmin={kmin} kmax={kmax} scaler={scaler}",
            n_jets=n_jets,
            center=forcing_center if forcing_center is not None else 0.0,
            width=forcing_width if forcing_width is not None else 0.0,
            amplitude=forcing_amplitude,
            kmin=getattr(model, "kmin", 0.0),
            kmax=getattr(model, "kmax", 0.0),
            scaler=scaler,
        )

    return {
        "forcing_center": forcing_center,
        "forcing_width": forcing_width,
        "forcing_amplitude": forcing_amplitude,
        "kmin": getattr(model, "kmin", None),
        "kmax": getattr(model, "kmax", None),
        "scaler": scaler,
    }


def compute_initial_energy_diagnostics(model, state):
    """Compute and log diagnostics about the initial state's energy.
    
    Parameters
    ----------
    model : QGM
        The model instance
    state : State
        The initial state
        
    Returns
    -------
    dict
        Dictionary with keys: ke_total, ke_per_layer, u_rms, v_rms, energy_spectrum
    """
    full = model.get_full_state(state)
    u = np.asarray(full.u)
    v = np.asarray(full.v)
    
    # Compute various energy metrics
    ke_per_layer = 0.5 * (np.mean(u ** 2, axis=(-2, -1)) + np.mean(v ** 2, axis=(-2, -1)))
    ke_total = float(np.mean(ke_per_layer))
    u_rms = float(np.sqrt(np.mean(u ** 2)))
    v_rms = float(np.sqrt(np.mean(v ** 2)))
    
    # Get energy spectrum
    energy_spectrum = np.asarray(model.compute_energy_spectrum(state))
    energy_spectrum_mean = float(np.mean(energy_spectrum))
    energy_spectrum_max = float(np.max(energy_spectrum))
    
    diagnostics = {
        'ke_total': ke_total,
        'ke_per_layer': ke_per_layer,
        'u_rms': u_rms,
        'v_rms': v_rms,
        'energy_spectrum_mean': energy_spectrum_mean,
        'energy_spectrum_max': energy_spectrum_max,
    }
    
    logger.info(
        f"Initial state energy diagnostics:\n"
        f"  Total KE: {ke_total:.3e}\n"
        f"  KE per layer: {ke_per_layer}\n"
        f"  U_rms: {u_rms:.3e}, V_rms: {v_rms:.3e}\n"
        f"  Energy spectrum - mean: {energy_spectrum_mean:.3e}, max: {energy_spectrum_max:.3e}"
    )
    
    return diagnostics


def suggest_normalized_forcing_amplitude(model, state, target_energy_change_fraction=0.01, dt=1000):
    """Suggest an appropriate forcing amplitude based on system energy.
    
    This computes a forcing amplitude that would add approximately
    `target_energy_change_fraction` of the current energy per timestep.
    
    Parameters
    ----------
    model : QGM
        The model instance
    state : State
        The current state
    target_energy_change_fraction : float
        Target fraction of energy to add per step (default 0.01 = 1%)
    dt : float
        Timestep size (used for scaling)
        
    Returns
    -------
    suggested_amplitude : float
        Suggested forcing_amplitude value
    """
    ke = float(model.compute_kinetic_energy(state))
    
    # Estimate how much forcing is needed
    # This is a rough heuristic: forcing scales as sqrt(energy)
    if ke > 1e-12:
        # More sophisticated scaling: we want dE/dt ~ target_fraction * E
        # So amplitude ~ sqrt(target_fraction * E / dt)
        suggested = float(np.sqrt(target_energy_change_fraction * ke / (dt + 1e-12)))
    else:
        suggested = 0.0
    
    logger.info(
        f"Forcing amplitude suggestion:\n"
        f"  Current KE: {ke:.3e}\n"
        f"  Target energy change per step: {target_energy_change_fraction * 100:.1f}%\n"
        f"  Timestep dt: {dt}\n"
        f"  Suggested forcing_amplitude: {suggested:.3e}\n"
        f"  Note: adjust based on desired energy balance"
    )
    
    return suggested


def check_forcing_mask_coverage(model):
    """Check and report on the spatial coverage of the forcing mask.
    
    Parameters
    ----------
    model : QGM
        The model instance
    """
    mask = np.asarray(model.forcing_mask)
    
    # Compute mask statistics
    mask_coverage = float(np.mean(mask))
    mask_max = float(np.max(mask))
    mask_min = float(np.min(mask))
    
    # Find the wavenumber range with forcing
    kmag = np.asarray(model.Kmag)
    mask_nonzero = mask > 1e-6
    
    if np.any(mask_nonzero):
        kmag_active = kmag[mask_nonzero]
        k_min_active = float(np.min(kmag_active))
        k_max_active = float(np.max(kmag_active))
        k_mean_active = float(np.mean(kmag_active[mask[mask_nonzero] > 0.5]))
    else:
        k_min_active = k_max_active = k_mean_active = 0.0
    
    logger.info(
        f"Forcing mask coverage:\n"
        f"  Mean coverage: {mask_coverage:.3f} (0-1 scale)\n"
        f"  Max/Min values: {mask_max:.3f} / {mask_min:.3f}\n"
        f"  Active wavenumber range: k ∈ [{k_min_active:.3f}, {k_max_active:.3f}]\n"
        f"  Mean active wavenumber: {k_mean_active:.3f}"
    )
    
    logger.info(
            f"  Annulus parameters: center={model.forcing_center}, width={model.forcing_width}"
        )


def diagnose_nan_issue(model, state, dt=1000, n_steps_to_test=10):
    """Test the model stepping to identify when NaN appears.
    
    This function attempts to step the model and detects where NaN first appears.
    Useful for diagnosing forcing-related instabilities.
    
    Parameters
    ----------
    model : QGM
        The model instance
    state : State
        The initial state
    dt : float
        Timestep size
    n_steps_to_test : int
        Number of steps to test before giving up
        
    Returns
    -------
    diagnostics : dict
        Contains 'nan_appeared_at_step' (int or None), 'final_ke', etc.
    """
    from model.core.steppers import SteppedModel, CNABStepper
    import jax
    
    logger.info(f"Testing for NaN issues over {n_steps_to_test} steps...")
    
    # Create a stepped model
    stepper = CNABStepper(dt=dt)
    stepped_model = SteppedModel(model, stepper)
    
    # Initialize stepper state
    stepper_state = stepped_model.initialise(jax.random.PRNGKey(42))
    
    nan_step = None
    final_ke = None
    
    for step in range(n_steps_to_test):
        try:
            stepper_state = stepped_model.step_model(stepper_state)
            
            # Check for NaN
            qh = np.asarray(stepper_state.state.qh)
            if np.any(np.isnan(qh)) or np.any(np.isinf(qh)):
                nan_step = step
                logger.error(f"NaN detected at step {step}")
                break
            
            # Compute KE for diagnostics
            final_ke = float(model.compute_kinetic_energy(stepper_state.state))
            
            if step % 5 == 0:
                logger.debug(f"  Step {step}: KE = {final_ke:.3e}")
                
        except Exception as e:
            logger.error(f"Exception at step {step}: {e}")
            nan_step = step
            break
    
    diagnostics = {
        'nan_appeared_at_step': nan_step,
        'final_ke': final_ke,
        'steps_without_nan': nan_step if nan_step is not None else n_steps_to_test,
    }
    
    if nan_step is None:
        logger.info(f"No NaN detected in {n_steps_to_test} steps. Final KE: {final_ke:.3e}")
    
    return diagnostics


def print_forcing_summary(model):
    """Print a comprehensive summary of forcing configuration.
    
    Parameters
    ----------
    model : QGM
        The model instance
    """
    logger.info(
        f"\n{'='*70}\n"
        f"FORCING CONFIGURATION SUMMARY\n"
        f"{'='*70}\n"
        f"Forcing Amplitude: {model.forcing_amplitude}\n"
    )
    
    logger.info(
            f"  Center wavenumber: {model.forcing_center}\n"
            f"  Width (σ): {model.forcing_width}\n"
        )
    
    logger.info(
        f"Grid resolution: nx={model.nx}, ny={model.ny}\n"
        f"Domain size: Lx={model.Lx}, Ly={model.Ly}\n"
        f"{'='*70}\n"
    )
    
    check_forcing_mask_coverage(model)
