"""Config loader for QG model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

try:
    import yaml
except Exception as e:  
    yaml = None


DEFAULTS: Dict[str, Any] = {
    "grid": {"nx": 128, "ny": 128, "Lx": 2 * 3.141592653589793},
    "timestep": {"dt": 5e-3, "stepper": "FilteredRK4"},
    "forcing": {"k_f": 8.0, "k_width": 2.0, "epsilon": 1e-3, "seed": 0},
    "dissipation": {"nu": 0.0, "m": 4, "mu": 0.0},
    "diagnostics": {"record_ke": True, "record_enstrophy": True},
    "ml": {"enabled": False, "model_path": None},
}


@dataclass
class Config:
    data: Dict[str, Any] = field(default_factory=lambda: DEFAULTS.copy())

    @classmethod
    def from_file(cls, path: str) -> "Config":
        if yaml is None:
            raise RuntimeError("PyYAML is required to load config files. Install with `pip install pyyaml`.")
        with open(path, "r", encoding="utf-8") as f:
            parsed = yaml.safe_load(f) or {}
        return cls.from_dict(parsed)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        out = DEFAULTS.copy()
        # shallow merge for top-level keys only 
        for k, v in d.items():
            if isinstance(v, dict) and k in out and isinstance(out[k], dict):
                merged = out[k].copy()
                merged.update(v)
                out[k] = merged
            else:
                out[k] = v
        cfg = cls(out)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        # Minimal checks
        g = self.data.get("grid", {})
        t = self.data.get("timestep", {})
        if not (isinstance(g.get("nx"), int) and isinstance(g.get("ny"), int)):
            raise ValueError("grid.nx and grid.ny must be integers")
        if not (isinstance(t.get("dt"), float) or isinstance(t.get("dt"), int)):
            raise ValueError("timestep.dt must be a number")

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return self.data.copy()
