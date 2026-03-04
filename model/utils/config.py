"""Config loader for QG model. I think loads of this is useless so clean it later
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from types import SimpleNamespace
import os
import copy

try:
    import yaml
except Exception as e:  
    yaml = None

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
CONFIG_DEFAULT_PATH = os.path.join(BASE_DIR, "QG", "config", "default.yaml")


def _wrap(d):
    """Recursively convert dicts to objects with attribute access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _wrap(v) for k, v in d.items()})
    return d


DEFAULTS: Dict[str, Any] = {
    "grid": {"nx": 128, "ny": 128, "Lx": 2 * 3.141592653589793},
    "timestep": {"dt": 5e-3, "stepper": "FilteredRK4"},
    "forcing": {"k_f": 8.0, "k_width": 2.0, "epsilon": 1e-3, "seed": 0},
    "dissipation": {"nu": 0.0, "m": 4, "mu": 0.0},
    "diagnostics": {"cadence": 1, "zarr_path": "outputs/data.zarr", "plots": ["energy"]},
    "output": {"outdir": "outputs", "every": 10},
    "logging": {"level": "INFO", "file": None},
    "ml": {"enabled": False, "model_path": None},
    # record the config file used (by default points to the packaged YAML)
    "config": {"path": CONFIG_DEFAULT_PATH},
}


@dataclass
class Config:
    data: Dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULTS))
    _config_path: str = field(default=CONFIG_DEFAULT_PATH, repr=False)

    @classmethod
    def load_config(cls, path: str) -> "Config":
        if yaml is None:
            raise RuntimeError("PyYAML required")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            parsed = yaml.safe_load(f) or {}

        cfg = cls.from_dict(parsed)
        cfg._config_path = os.path.abspath(path)
        return cfg

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        out = copy.deepcopy(DEFAULTS)

        for k, v in d.items():
            if isinstance(v, dict) and k in out:
                out[k].update(v)
            else:
                out[k] = v

        cfg = cls(out)
        cfg.validate()
        return cfg

    @property
    def config_path(self):
        return self._config_path

    def __getattr__(self, name: str):
        if name in self.data:
            return _wrap(self.data[name])
        raise AttributeError(name)

    def validate(self) -> None:
        # Minimal checks
        g = self.data.get("grid", {})
        t = self.data.get("timestep", {})
        if not (isinstance(g.get("nx"), int) and isinstance(g.get("ny"), int)):
            raise ValueError("grid.nx and grid.ny must be integers")
        if not (isinstance(t.get("dt"), float) or isinstance(t.get("dt"), int)):
            raise ValueError("timestep.dt must be a number")
        if not isinstance(t.get("stepper"), str):
            raise ValueError("timestep.stepper must be a string (e.g. 'FilteredRK4')")
        if float(t.get("dt")) <= 0:
            raise ValueError("timestep.dt must be positive")

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return self.data.copy()

    @property
    def grid(self) -> Dict[str, Any]:
        """Return grid parameters."""
        return self.data.get("grid", {}).copy()

    @property
    def time(self):
        """ Dunno for now """
        return _wrap(self.data.get("timestep", {}))
