#!/usr/bin/env python3
"""Aggregate MSE results across output runs and print top-N performers.

Looks for `metadata.json` and `mse.json` (or `mse.npz`) in each subdirectory of `outputs/`.
"""
import json
import os
import numpy as np
from pathlib import Path


def read_mse(path):
    p = Path(path)
    if p.exists():
        if p.suffix == '.json':
            with open(p) as f:
                data = json.load(f)
            return np.asarray(data.get('mse'))
        elif p.suffix == '.npz':
            with np.load(p) as d:
                return d.get('mse')
    return None


def find_runs(outputs_dir='outputs'):
    runs = []
    base = Path(outputs_dir)
    if not base.exists():
        print(f"Outputs dir '{outputs_dir}' not found.")
        return runs

    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        meta = d / 'metadata.json'
        mse_json = d / 'mse.json'
        mse_npz = d / 'mse.npz'
        if not meta.exists():
            # some runs may nest deeper; attempt to find metadata in first-level subdir
            submeta = None
            for sub in d.iterdir():
                if (sub / 'metadata.json').exists():
                    submeta = sub / 'metadata.json'
                    mse_json = sub / 'mse.json'
                    mse_npz = sub / 'mse.npz'
                    break
            if submeta is None:
                continue
            meta = submeta

        try:
            with open(meta) as f:
                meta_obj = json.load(f)
        except Exception:
            continue

        mse_arr = read_mse(mse_json) if mse_json.exists() else read_mse(mse_npz) if mse_npz.exists() else None
        if mse_arr is None:
            continue

        runs.append({'path': str(d), 'meta': meta_obj, 'mse': np.asarray(mse_arr)})

    return runs


def summarize(runs, top_n=5):
    # compute mean mse across timesteps for ranking
    entries = []
    for r in runs:
        mse = r['mse']
        mean_mse = float(np.nanmean(mse))
        final_mse = float(mse[-1]) if mse.size else float('nan')
        entries.append((mean_mse, final_mse, r))

    entries.sort(key=lambda x: x[0])
    print(f"Found {len(entries)} completed runs. Top {top_n} by mean MSE:")
    for i, (mean_mse, final_mse, r) in enumerate(entries[:top_n], start=1):
        params = r['meta'].get('parameters') or r['meta'].get('model_arch') or {}
        print(f"\n{i}. Path: {r['path']}")
        print(f"   Mean MSE: {mean_mse:.6e}, Final MSE: {final_mse:.6e}")
        print("   Parameters:")
        # pretty-print param dict shallowly
        for k, v in (params.items() if isinstance(params, dict) else []):
            print(f"     - {k}: {v}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--outputs', default='outputs', help='Outputs directory to scan')
    p.add_argument('--top', type=int, default=5, help='Number of top models to print')
    args = p.parse_args()

    runs = find_runs(args.outputs)
    summarize(runs, top_n=args.top)
