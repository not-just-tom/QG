"""Diagnostic to measure nonlinear, beta, and residual contributions for QG models.
Run from repo root: python scripts/diagnose_terms.py
"""

import sys
import os
# Ensure the workspace root is in the path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_workspace_root = os.path.dirname(_script_dir)
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)
import yaml
import jax
import jax.numpy as jnp
import numpy as np
import importlib
import model.core.model
importlib.reload(model.core.model)
from model.core.model import create_model
import model.core.states as states

# load params
with open("../config/default.yaml") as f:
    cfg = yaml.safe_load(f)
params = dict(cfg["params"])  # mutable copy

# choose n_layers and construct model
n_layers = params.pop('n_layers', 1)
model = create_model(params, n_layers=n_layers)

# initialise state
state = model.initialise(params.get('seed', 0))
full = model.get_full_state(state)

# compute nonlinear fluxes in real space and their spectral transforms
uq = full.u * full.state.q
vq = full.v * full.state.q

uqh = states._generic_rfftn(uq)
vqh = states._generic_rfftn(vq)

# prepare dealias mask expansion
dealias = getattr(model, "_dealias", None)
if dealias is not None:
    # dealias is (nl,nk); expand to (1,nl,nk) so it multiplies layer-wise when needed
    dmask = jnp.expand_dims(dealias, 0)
else:
    dmask = 1.0

# nonlinear spectral divergence (no dealias)
nonlin_spec = - (jnp.expand_dims(model._ik, 0) * uqh + jnp.expand_dims(model._il, 1) * vqh)
# nonlinear spectral divergence (with dealias applied to nonlinear products)
nonlin_spec_deal = - (
    jnp.expand_dims(model._ik, 0) * (uqh * dmask)
    + jnp.expand_dims(model._il, 1) * (vqh * dmask)
)

# beta / PV-gradient term if available
beta_term = 0
if hasattr(model, "_ikQy") and hasattr(full, "ph"):
    beta_term = - (model._ikQy * full.ph)

# full rhs from kernel
full_rhs = full.dqhdt

# residual = full - (nonlin + beta)
residual = full_rhs - (nonlin_spec + beta_term)
residual_deal = full_rhs - (nonlin_spec_deal + beta_term)

# report norms (L2)
def cnorm(x):
    # convert to real-valued norm
    try:
        return float(jnp.linalg.norm(x).real)
    except Exception:
        return float(np.linalg.norm(np.asarray(x)))

print("Norms (L2) of RHS components:")
print("  Nonlinear (no dealias)   :", cnorm(nonlin_spec))
print("  Nonlinear (with dealias) :", cnorm(nonlin_spec_deal))
print("  Beta term                :", cnorm(beta_term))
print("  Residual (no dealias)    :", cnorm(residual))
print("  Residual (with dealias)  :", cnorm(residual_deal))
print("  Full RHS                 :", cnorm(full_rhs))

print("\nParameter scales:")
for p in ["beta", "rek", "rd", "delta"]:
    val = params.get(p, getattr(model, p, None))
    print(f"  {p} = {val}")

# quick energy check in physical space if available
try:
    u = full.u
    v = full.v
    ke = 0.5 * jnp.sum(u ** 2 + v ** 2)
    print("\nKinetic energy (integral):", float(ke))
except Exception:
    pass
