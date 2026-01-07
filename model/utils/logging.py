"""Simple logging configuration for the QG package.

Provides a small helper `get_logger(name)` to obtain a logger pre-configured
for console output and DEBUG/INFO control via the `QG_LOGLEVEL` environment
variable.
"""

from __future__ import annotations

import logging
import os
from typing import Optional


def configure_logging(level: Optional[str] = None):
    level = level or os.environ.get("QG_LOGLEVEL", "INFO")
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger("qg")
    # Add console handler if no handlers present
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        root.addHandler(handler)

    # Optionally log to file if QG_LOGFILE is set
    logfile = os.environ.get("QG_LOGFILE")
    if logfile and not any(isinstance(h, logging.FileHandler) for h in root.handlers):
        fh = logging.FileHandler(logfile)
        fh.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        root.addHandler(fh)

    root.setLevel(numeric_level) 


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(f"qg.{name}")
