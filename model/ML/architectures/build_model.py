from abc import ABC, abstractmethod
import importlib
import model.ML.architectures.zero
import model.ML.architectures.cnn
importlib.reload(model.ML.architectures.zero)
importlib.reload(model.ML.architectures.cnn)
from model.ML.architectures.cnn import CNN
from model.ML.architectures.zero import ZeroModel

def _get_arch_params(cfg, arch_name):
    arch_cfg = cfg.get("architectures", {})
    if arch_name in arch_cfg and isinstance(arch_cfg[arch_name], dict):
        return dict(arch_cfg[arch_name])

def build_closure(cfg): 
    _registry = {
        'zero': ZeroModel,
        "CNN": CNN,
        'MLP': None, # add in soon
    }

    arch_name = cfg.ml.model_type
    cls = _registry.get(arch_name)
    if cls is None:
        raise ValueError(f"Unknown ML closure '{arch_name}', Available: {sorted(_registry.keys())}")
    arch_params = _get_arch_params(cfg, arch_name)
    if arch_params is None:
        raise ValueError(f"No parameters found for architecture '{arch_name}' in config.")
    return cls(**arch_params)


class Closure(ABC):
    name: str

    @abstractmethod
    def requires(self) -> set[str]:
        """dunno if i even need this rn"""




        