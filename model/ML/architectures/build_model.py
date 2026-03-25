import importlib
import model.ML.architectures.zero
import model.ML.architectures.cnn
import model.ML.architectures.unet
import model.ML.architectures.fno
importlib.reload(model.ML.architectures.zero)
importlib.reload(model.ML.architectures.cnn)
importlib.reload(model.ML.architectures.unet)
importlib.reload(model.ML.architectures.fno)
from model.ML.architectures.cnn import CNN
from model.ML.architectures.zero import ZeroModel
from model.ML.architectures.unet import UNet
from model.ML.architectures.fno import FNO
import equinox as eqx
import numpy as np
import jax
import logging
logger = logging.getLogger(__name__)
from model.ML.utils.utils import module_to_single



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


def build_closure(cfg, loaded_leaves=None):
    registry = {
        "zero": ZeroModel,
        "cnn": CNN,
        'unet': UNet,
        'fno': FNO,
    }

    arch_name = _resolve_arch_name(cfg)
    cls = registry.get(_normalize(arch_name))
    if cls is None:
        raise ValueError(
            f"Unknown ML closure '{arch_name}', available: {sorted(registry.keys())}"
        )

    arch_params = _get_arch_params(cfg, arch_name)
    closure_template = cls(**arch_params)
    if 'loaded_leaves' in locals() and loaded_leaves is not None:
        try:
            template_params, template_static = eqx.partition(closure_template, eqx.is_array)
            tpl_leaves, tpl_treedef = jax.tree_util.tree_flatten(template_params)
            if len(tpl_leaves) != len(loaded_leaves):
                raise ValueError(f"Loaded params length {len(loaded_leaves)} does not match template {len(tpl_leaves)}")

            # cast loaded leaves to template dtypes and build new param pytree
            new_leaves = []
            for tpl, ld in zip(tpl_leaves, loaded_leaves):
                arr = np.asarray(ld)
                # ensure dtype matches template leaf
                try:
                    arr = arr.astype(np.asarray(tpl).dtype)
                except Exception:
                    pass
                new_leaves.append(jax.device_put(arr))

            new_params = jax.tree_util.tree_unflatten(tpl_treedef, new_leaves)
            closure_model = eqx.combine(new_params, template_static)
            logger.info("Reconstructed closure from loaded params")
        except Exception:
            logger.exception("Failed to reconstruct closure from params; falling back to fresh closure")
            closure_model = closure_template
    else:
        closure_model = closure_template

    return module_to_single(closure_model)




        