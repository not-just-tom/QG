import logging
from pathlib import Path


def configure_logging(level="info", out_file=None):
    num_level = getattr(logging, level.upper(), None)
    if not isinstance(num_level, int):
        raise ValueError("Invalid log level: {}".format(level))
    handlers = [logging.StreamHandler()]
    if out_file:
        p = Path(out_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Open log file in write mode to replace any existing file with same name
        handlers.append(logging.FileHandler(filename=str(p), encoding="utf8", mode="w"))
    logging.basicConfig(
        level=num_level,
        handlers=handlers,
        force=True,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


