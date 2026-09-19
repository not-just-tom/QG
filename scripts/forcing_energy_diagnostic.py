"""Check the kinetic-energy rate of stochastic forcing from a zero state."""
import argparse
from pathlib import Path
import sys

# Adds the parent directory (root folder) to the Python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))


import inspect
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from model.core.model import QGM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=BASE_DIR / "config/default.yaml")
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default= BASE_DIR / "outputs/forcing_energy_convergence.png",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    params = dict(OmegaConf.to_container(cfg.params, resolve=True))
    params["nx"] = args.nx or int(params["nx"])
    params["ny"] = params["nx"]
    params["dt"] = float(cfg.plotting.dt)
    model = QGM(params)

    zero_qh = jnp.zeros(
        (model.nz, model.ny, model.nx // 2 + 1), dtype=jnp.complex64
    )
    zero_state = model.set_initial(zero_qh, _q_shape=(model.ny, model.nx))
    keys = jax.random.split(jax.random.PRNGKey(int(params["seed"])), args.samples)

    energies = []
    for key in keys:
        forcing = model.do_stochastic_forcing(zero_state, forcing_key=key)
        forced_state = zero_state.update(qh=zero_state.qh + forcing.qh)
        full_state = model.get_full_state(forced_state)
        layer_energy = 0.5 * jnp.mean(
            full_state.u**2 + full_state.v**2, axis=(-2, -1)
        )
        energies.append(jnp.mean(layer_energy))

    rates = np.cumsum(np.asarray(energies)) / np.arange(1, args.samples + 1)
    rates /= float(model.dt)
    epsilon = float(model.epsilon)

    print(f"epsilon = {epsilon:.6g}")
    print(f"<E(dq)>/dt = {rates[-1]:.6g}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.plot(np.arange(1, args.samples + 1), rates, label=r"running $\langle E(\Delta q)\rangle/dt$")
    plt.axhline(epsilon, color="black", linestyle="--", label=r"$\epsilon$")
    plt.xlabel("Forcing samples")
    plt.ylabel("Energy rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    plt.close()


    print("QGM source:", inspect.getsourcefile(QGM))
    print("JAX:", jax.__version__, "backend:", jax.default_backend())
    print("x64:", jax.config.read("jax_enable_x64"))
    print("nx, ny, dt:", model.nx, model.ny, model.dt)
    print("epsilon:", model.epsilon)
    print("kmin, kmax:", model.kmin, model.kmax)
    print(inspect.getsource(model._invert))


if __name__ == "__main__":
    main()