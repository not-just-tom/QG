import sys
import os
# Ensure the workspace root is in the path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_workspace_root = os.path.dirname(_script_dir)
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)

import functools
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import importlib
import yaml

_module_names = [
    'model.core.grid',
    'model.core.states',
    'model.core.kernel',
    'model.core.model',
    'model.core.steppers',
    'model.ML.utils.utils',
    'model.ML.utils.coarsen',
    'model.ML.forced_model',
    'model.utils.config',
]

for _mod_name in _module_names:
    if _mod_name in sys.modules:
        importlib.reload(sys.modules[_mod_name])
    else:
        __import__(_mod_name)

from model.core.model import QGM
from model.core.steppers import SteppedModel, build_stepper
from model.ML.forced_model import ForcedModel
from model.ML.utils.utils import parameterization
from model.ML.utils.coarsen import Coarsener
from model.utils.config import Config

jax.config.update("jax_enable_x64", True)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
CONFIG_DEFAULT_PATH = os.path.join(BASE_DIR, "config", "default.yaml")