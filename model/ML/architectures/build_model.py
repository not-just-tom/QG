import importlib
import model.ML.architectures.zero
import model.ML.architectures.cnn
import model.ML.architectures.unet
importlib.reload(model.ML.architectures.zero)
importlib.reload(model.ML.architectures.cnn)
importlib.reload(model.ML.architectures.unet)
from model.ML.architectures.cnn import CNN
from model.ML.architectures.zero import ZeroModel
from model.ML.architectures.unet import UNet


def _normalize(name):
    return str(name).strip().lower()


def _get_arch_params(cfg, arch_name):
    arch_cfg = cfg.get("architectures", {})
    if not isinstance(arch_cfg, dict):
        return {}

    for key, value in arch_cfg.items():
        if _normalize(key) == _normalize(arch_name) and isinstance(value, dict):
            return dict(value)
    return {}


def _resolve_arch_name(cfg):
    # Prefer model_type, fallback to older model field
    return getattr(cfg.ml, "model_type", getattr(cfg.ml, "model", None))


def build_closure(cfg):
    registry = {
        "zero": ZeroModel,
        "cnn": CNN,
        'unet': UNet,
    }

    arch_name = _resolve_arch_name(cfg)
    cls = registry.get(_normalize(arch_name))
    if cls is None:
        raise ValueError(
            f"Unknown ML closure '{arch_name}', available: {sorted(registry.keys())}"
        )

    arch_params = _get_arch_params(cfg, arch_name)
    return cls(**arch_params)




        