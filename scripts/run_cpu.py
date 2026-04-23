#!/usr/bin/env python3
"""Helper to run a single sweep combination by index.

This script selects one combination from the sweep axes in a config
and invokes `run.py --config <tmpfile>` for that single combination.

Designed to be used with SLURM job arrays via `SLURM_ARRAY_TASK_ID`.
"""
import os
import sys
import argparse
import tempfile
import subprocess
import itertools
from omegaconf import OmegaConf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Base config YAML path")
    p.add_argument("--task-index", type=int, default=None, help="Index of sweep combination (overrides SLURM env var)")
    p.add_argument("--outdir-override", default=None, help="Optional outdir override passed into generated config")
    args = p.parse_args()

    task_index = args.task_index
    if task_index is None:
        task_index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

    base_cfg = OmegaConf.load(args.config)
    cfg_plain = OmegaConf.to_container(base_cfg, resolve=True)
    model_type = cfg_plain.get('ml', {}).get('model_type')
    archs = cfg_plain.get('architectures', {})
    arch_cfg = archs.get(model_type, {}) if archs else {}

    sweep_keys = [k for k, v in arch_cfg.items() if isinstance(v, list) and len(v) > 1]
    lists = [arch_cfg[k] for k in sweep_keys]
    if lists:
        combos = list(itertools.product(*lists))
    else:
        combos = [()]

    total = len(combos)
    if task_index < 0 or task_index >= total:
        raise SystemExit(f"task-index {task_index} out of range (0..{total-1})")

    combo = combos[task_index]
    # Build a copy of the config with the chosen combination applied
    cfg_copy = OmegaConf.to_container(base_cfg, resolve=True)
    if 'architectures' not in cfg_copy:
        cfg_copy['architectures'] = {}
    if model_type not in cfg_copy['architectures']:
        cfg_copy['architectures'][model_type] = {}
    for k, val in zip(sweep_keys, combo):
        cfg_copy['architectures'][model_type][k] = val

    # Optional outdir override to keep outputs isolated per task
    if args.outdir_override is not None:
        if 'filepaths' not in cfg_copy:
            cfg_copy['filepaths'] = {}
        cfg_copy['filepaths']['out_dir'] = args.outdir_override

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
    tmp_path = tmp.name
    tmp.close()
    OmegaConf.save(config=OmegaConf.create(cfg_copy), f=tmp_path)

    try:
        cmd = [sys.executable, "run.py", "--config", tmp_path]
        print("Running:", " ".join(cmd))
        subprocess.check_call(cmd)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


if __name__ == "__main__":
    main()
