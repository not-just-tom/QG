"""ML closure helpers: build an ml_forcing callable to plug into Solver.rhs.

The returned callable must have signature:
    ml_forcing(qh, key, grid, params) -> spectral_forcing (shape like qh)

"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional


def zero_ml_forcing(qh, key, grid, params):
    # return zeros in spectral shape
    return jnp.zeros_like(qh)


def build_ml_forcing(config: Optional[object], grid):
    """Return an ml_forcing callable according to config.

    config can be a dict-like or SimpleNamespace supporting keys:
      - enabled: bool
      - model_path: optional path to a numpy .npz or .npy containing a spectral field

    For now, if enabled and a model_path is provided and loadable as numpy
    array, we return a fixed spectral field function (constant correction).
    Otherwise we return None or the zero forcing (if enabled but no model).
    """
    if config is None:
        return None
    enabled = getattr(config, "enabled", False) if not hasattr(config, "get") else config.get("enabled", False)
    if not enabled:
        return None

    # If a model_path is provided and looks like a numpy file, load it
    model_path = getattr(config, "model_path", None) if not hasattr(config, "get") else config.get("model_path", None)
    if model_path is None:
        # No model yet, return a zero forcing (acts as a placeholder)
        return zero_ml_forcing

    try:
        data = np.load(model_path)
        # If data is an array-like use it as spectral correction
        if isinstance(data, np.ndarray):
            ml_field = jnp.array(data)
        elif isinstance(data, np.lib.npyio.NpzFile):
            # pick the first array
            key = list(data.files)[0]
            ml_field = jnp.array(data[key])
        else:
            return zero_ml_forcing

        @jax.jit
        def ml_forcing(qh, key, grid, params):
            # broadcast/reshape as needed to match qh shape
            if ml_field.shape == qh.shape:
                return ml_field
            # try to slice or pad
            out = jnp.zeros_like(qh)
            miny = min(out.shape[0], ml_field.shape[0])
            minx = min(out.shape[1], ml_field.shape[1])
            out = out.at[:miny, :minx].set(ml_field[:miny, :minx])
            return out

        return ml_forcing
    except Exception:
        # fallback
        return zero_ml_forcing