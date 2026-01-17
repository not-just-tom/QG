"""Small demo training loop for spectral correction model.

This provides a convenience function `train_spectral_demo` that synthesizes a
"truth" spectral correction and fits the spectral parameters to it using SGD.
It is intentionally simple and meant as a starting point for more advanced
training (e.g., differentiable time integration, multi-step losses, etc.).
"""
from __future__ import annotations

import time
import jax
import jax.numpy as jnp
import pathlib
import os
import sys
import itertools
import logging
import json 
import platform
import math
import contextlib
import numpy as np
import equinox as eqx
import optax
import re
import importlib
import model.utils
import model.ML
importlib.reload(model.utils)
importlib.reload(model.ML)
from model.utils.logging import configure_logging
from model.ML.utils import build_network, make_json_serializable, load_network_continue


def run(config):
    out_dir = pathlib.Path(config.filepaths.out_dir)
    if out_dir.is_file():
        raise ValueError(f"Path must be a directory, not a file: {config.filepaths.out_dir}")
    out_dir.mkdir(exist_ok=True)
    configure_logging(level=config.filepaths.log_level, out_file="logs/run.log") #return to this to put numbers on it 
    logger = logging.getLogger("main")
    logger.info("Arguments: %s", vars(config))

    # Select seed
    if config.seed is None:
        logger.info("No seed provided")
    else:
        seed = config.params.seed # this needs to be changed
    logger.info("Using seed %d", seed)
    
    