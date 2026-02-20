from abc import ABC, abstractmethod
import importlib
import model.ML.architectures
importlib.reload(model.ML.architectures)
from model.ML.architectures.cnn import CNN
from model.ML.architectures.zero import ZeroModel

def build_closure(cfg): 
    registry = {
        'zero': ZeroModel,
        "CNN": CNN,
    }

    name = cfg.ml.model
    cls = registry.get(name)
    if cls is None:
        raise ValueError(f"Unknown ML closure '{name}'")

    return cls()


class Closure(ABC):
    name: str

    @abstractmethod
    def requires(self) -> set[str]:
        """dunno if i even need this rn"""




        