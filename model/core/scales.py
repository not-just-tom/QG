"""Reference-scale conversions for the quasi-geostrophic model."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RhinesScales:
    """Dimensional reference scales selected by the target jet count."""

    length: float
    velocity: float
    time: float


def apply_rhines_scaling(params: dict) -> tuple[dict, RhinesScales]:
    """Convert dimensional QG parameters to Rhines-scaled parameters."""
    if not params.get("nondimensional", False):
        return dict(params), RhinesScales(1.0, 1.0, 1.0)

    n_jets = params.get("n_jets")
    if n_jets is None or n_jets <= 0:
        raise ValueError("n_jets must be positive when nondimensional=True")

    ly = float(params.get("Ly", params.get("Lx")))
    beta = float(params.get("beta"))
    if ly <= 0 or beta <= 0:
        raise ValueError("Ly and beta must be positive for Rhines scaling")

    length = ly / (math.pi * float(n_jets))
    velocity = beta * length**2
    time = length / velocity

    scaled = dict(params)
    scaled["Lx"] = float(params.get("Lx", ly)) / length
    scaled["Ly"] = ly / length
    scaled["beta"] = 1.0
    scaled["Ld"] = float(params["Ld"]) / length
    scaled["U1"] = float(params.get("U1", 0.0)) / velocity
    scaled["U2"] = float(params.get("U2", 0.0)) / velocity
    scaled["drag"] = float(params.get("drag", 0.0)) * time
    scaled["epsilon"] = float(params.get("epsilon", 0.0)) * length / velocity**3
    if "dt" in params:
        scaled["dt"] = float(params["dt"]) / time
    return scaled, RhinesScales(length, velocity, time)