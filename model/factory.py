"""Factory helpers to build model components from a Config object.

Provides functions to build the `params` dict from a `Config`, choose
stepper, and compute an automatic timestep (simple heuristic).
"""
from __future__ import annotations

from typing import Dict
import math
import jax.numpy as jnp


def build_params_from_config(cfg: Dict) -> Dict:
    params = {}
    params.update(cfg.get("grid", {}))
    params.update(cfg.get("timestep", {}))
    params.update(cfg.get("forcing", {}))
    params.update(cfg.get("dissipation", {}))
    params.setdefault("beta", 10.0)
    # Keep seed handling to caller (PRNG key creation)
    return params


def compute_auto_dt(params: Dict, cfg: Dict) -> float:
    """Compute a simple heuristic dt when `auto_dt` is enabled.

    Heuristic: dt = cfl * min(dx, dy) / (k_f) where k_f is forcing wavenumber
    expressed in physical inverse-length; this is intentionally conservative.
    """
    Lx = params.get("Lx")
    nx = params.get("nx")
    ny = params.get("ny", nx)
    dx = Lx / nx
    dy = params.get("Ly", params.get("Lx")) / ny
    k_f = params.get("k_f", cfg.get("forcing", {}).get("k_f", 8.0))
    cfl = cfg.get("timestep", {}).get("cfl", 0.1)
    # convert domain-centric k_f into inverse-length: (k_f * 2π / Lx)
    k_phys = k_f * 2 * math.pi / Lx
    # Avoid zero division
    k_phys = max(1e-12, k_phys)
    dt = cfl * min(dx, dy) / (k_phys)
    return float(dt)
