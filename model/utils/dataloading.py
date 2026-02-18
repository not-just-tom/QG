import os
import json

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

def find_existing_run(base_dir, hr_nx, lr_nx, params_hash):
    prefix = f"data_hr{hr_nx}_nx{lr_nx}_"
    for name in os.listdir(base_dir):
        if not name.startswith(prefix):
            continue
        cand_dir = os.path.join(base_dir, name)
        meta_path = os.path.join(cand_dir, "metadata.json")
        if not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path) as f:
                m = json.load(f)
        except Exception:
            continue
        if m.get("params_hash") == params_hash:
            return cand_dir, meta_path, m
    return None, None, None
