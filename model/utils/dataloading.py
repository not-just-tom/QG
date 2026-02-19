import os
import json
import re

RUN_RE = re.compile(r"data_hr(?P<hr>\d+)_nx(?P<lr>\d+)_(?P<idx>\d{2})")

def metadata_matches(requested: dict, stored: dict) -> bool:
    return canonicalize(requested) == canonicalize(stored)

def canonicalize(params: dict) -> dict:
    def round_floats(x):
        if isinstance(x, float):
            return round(x, 12)
        if isinstance(x, dict):
            return {k: round_floats(v) for k, v in sorted(x.items())}
        if isinstance(x, list):
            return [round_floats(v) for v in x]
        return x

    return round_floats(params)

def find_existing_run(base_dir, hr_nx, lr_nx, params):
    prefix = f"data_hr{hr_nx}_nx{lr_nx}_"
    candidates = []

    for name in os.listdir(base_dir):
        m = RUN_RE.fullmatch(name)
        if m is None:
            continue
        if int(m["hr"]) != hr_nx or int(m["lr"]) != lr_nx:
            continue

        run_dir = os.path.join(base_dir, name)
        meta_path = os.path.join(run_dir, "metadata.json")
        if not os.path.exists(meta_path):
            continue

        try:
            with open(meta_path) as f:
                stored_meta = json.load(f)
        except Exception:
            continue

        # Exact metadata match → reuse
        if metadata_matches(params, stored_meta["parameters"]):
            return run_dir, True

        candidates.append(int(m["idx"]))

    # No match found → create new
    next_idx = max(candidates, default=0) + 1
    run_name = f"{prefix}{next_idx:02d}"
    run_dir = os.path.join(base_dir, run_name)
    return run_dir, False
