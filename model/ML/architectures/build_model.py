import importlib
import model.ML.architectures.zero
import model.ML.architectures.cnn
import model.ML.architectures.unet
import model.ML.architectures.fno
import model.ML.architectures.diffusion
import model.ML.architectures.resnet
import model.ML.architectures.mlp
import model.ML.architectures.leith
importlib.reload(model.ML.architectures.zero)
importlib.reload(model.ML.architectures.cnn)
importlib.reload(model.ML.architectures.unet)
importlib.reload(model.ML.architectures.fno)
importlib.reload(model.ML.architectures.diffusion)
importlib.reload(model.ML.architectures.resnet)
importlib.reload(model.ML.architectures.mlp)
importlib.reload(model.ML.architectures.leith)
from model.ML.architectures.cnn import CNN
from model.ML.architectures.zero import ZeroModel
from model.ML.architectures.unet import UNet
from model.ML.architectures.fno import FNO
from model.ML.architectures.diffusion import Diffusion
from model.ML.architectures.resnet import ResNet
from model.ML.architectures.mlp import MLP
from model.ML.architectures.leith import LeithClosure
from model.ML.utils.utils import module_to_single
import equinox as eqx
import numpy as np
import jax
import jax.numpy as jnp
import logging
logger = logging.getLogger(__name__)

def _normalize(name):
    return str(name).strip().lower()

def closure_combiner(
    state,
    closure_params,
    static_closure_obj=None,
    q_mean=None,
    q_std=None,
    dq_mean=None,
    dq_std=None,
):
    """Evaluate closure and return per-step PV increment dQ plus params.
    """
    closure = eqx.combine(closure_params, static_closure_obj)
    q = state.q

    if getattr(closure, 'ml', True):
        # ML closure: normalize input and output if mean/std provided
        if q_mean is None or q_std is None:
            q_in = q
        else:
            q_in = (q - q_mean) / (q_std + 1e-6)

        dq_increment = closure(q_in.astype(jnp.float32)).astype(q.dtype)

        if dq_mean is not None and dq_std is not None:
            dq_increment = (dq_increment * dq_std) + dq_mean
    else:
        dq_increment = closure(state).astype(q.dtype)
    return dq_increment, closure_params

    
def _get_arch_params(cfg, arch_name):
    # Support multiple cfg shapes: dict-like, attribute-style, or OmegaConf
    arch_cfg = {}
    # Prefer mapping-style access if available
    if hasattr(cfg, "get"):
        try:
            arch_cfg = cfg.get("architectures", {})
        except Exception:
            arch_cfg = getattr(cfg, "architectures", {})
    else:
        arch_cfg = getattr(cfg, "architectures", {})

    # If it's an OmegaConf node, convert to plain dict
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(arch_cfg):
            arch_cfg = OmegaConf.to_container(arch_cfg, resolve=True)
    except Exception:
        pass

    if not isinstance(arch_cfg, dict):
        return {}

    # Find matching normalized arch name and return its params
    for key, value in arch_cfg.items():
        if _normalize(key) == _normalize(arch_name) and isinstance(value, dict):
            return dict(value)

    return {}

def build_closure(cfg=None, loaded_leaves=None):
    """Build a closure model, optionally loading from saved parameters.
    
    Args:
        cfg: Configuration object with ml.model_type and architectures section
        loaded_leaves: List of numpy arrays (loaded parameters)
        model_type: Model architecture name (e.g., 'resnet'). Used if cfg is None
        arch_params: Dict of architecture parameters. Used if cfg is None
    
    Returns:
        Closure model (equinox module)
    """
    registry = {
        "zero": ZeroModel,
        "cnn": CNN,
        'unet': UNet,
        'fno': FNO,
        'diffusion': Diffusion,
        'resnet': ResNet,
        'mlp': MLP,
        'leith': LeithClosure,
    }

    arch_name = cfg.ml.model_type
    arch_params_to_use = _get_arch_params(cfg, arch_name)
    
    # Get model class
    cls = registry.get(_normalize(arch_name))
    if cls is None:
        raise ValueError(
            f"Unknown ML closure '{arch_name}', available: {sorted(registry.keys())}"
        )
    
    logger.info("Building closure '%s' with arch params: %s", arch_name, arch_params_to_use)
    closure_template = cls(**arch_params_to_use, cfg=cfg)
    
    # If loaded parameters provided, reconstruct model with them
    if loaded_leaves is not None:
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
        