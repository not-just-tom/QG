"""CLI runner that builds model structure from YAML config.

This script reads a YAML configuration using `model.config.Config` and
constructs a `QGM` instance using `model.factory` helpers. The config may
select the time-stepper, enable `auto_dt`, select forcing options, and
select diagnostics. Run `python scripts/run_cfg.py --help` for usage.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from model.config import Config
from model.factory import build_params_from_config, compute_auto_dt
from model.model import QGM
from model.diagnostics import compute_ke, compute_enstrophy


def run(cfg_path: str, nsteps: int, outdir: str, record_every: int = 10):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = Config.from_file(cfg_path).to_dict()
    params = build_params_from_config(cfg)

    # handle auto_dt selection
    if cfg.get("timestep", {}).get("auto_dt", False):
        params["dt"] = compute_auto_dt(params, cfg)
        print(f"Auto-selected dt = {params['dt']:.3e}")

    # PRNG key
    params["key"] = jax.random.PRNGKey(cfg.get("forcing", {}).get("seed", 0))

    print(f"Starting run: stepper={params.get('stepper', 'FilteredRK4')}, dt={params.get('dt')}")

    jax.config.update("jax_platform_name", os.environ.get("JAX_PLATFORM_NAME", "cpu"))

    model = QGM(params)
    model.initialize()

    times = []
    ke_vals = []
    enst_vals = []

    t0 = time.time()
    for i in range(1, nsteps + 1):
        model.step()
        if i % record_every == 0 or i == 1:
            psi = model.fields["psi"]
            zeta = model.fields["zeta"]
            ke = compute_ke(psi, model.grid)
            enst = compute_enstrophy(zeta, model.grid)
            times.append(i)
            ke_vals.append(ke)
            enst_vals.append(enst)
            print(f"step {i:5d}: KE={ke:.4e}, Enst={enst:.4e}")

    dt = time.time() - t0
    print(f"Run finished: {nsteps} steps in {dt:.1f}s ({nsteps/dt:.1f} steps/s)")

    plt.figure()
    plt.plot(times, ke_vals, label="KE")
    plt.plot(times, enst_vals, label="Enstrophy")
    plt.xlabel("step")
    plt.ylabel("value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(outdir) / "diagnostics_timeseries.png")
    plt.close()

    z = jnp.array(model.fields["zeta"])
    plt.figure(figsize=(6, 5))
    plt.imshow(z, origin="lower", cmap="RdBu_r")
    plt.colorbar(label="zeta")
    plt.title("Final vorticity")
    plt.tight_layout()
    plt.savefig(Path(outdir) / "zeta_final.png")
    plt.close()

    print(f"Saved outputs to {outdir}")


def main():
    p = argparse.ArgumentParser(description="Run QG from YAML config")
    p.add_argument("--config", default="config/default.yaml", help="Path to YAML config")
    p.add_argument("--nsteps", type=int, default=1000, help="Number of timesteps to run")
    p.add_argument("--outdir", default="outputs/run_cfg", help="Directory to save outputs")
    p.add_argument("--record-every", type=int, default=10, help="Record diagnostics every N steps")
    args = p.parse_args()

    run(args.config, args.nsteps, args.outdir, args.record_every)


if __name__ == "__main__":
    main()
