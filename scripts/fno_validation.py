"""Interrogate an FNO closure against a ResNet closure, one component at a time.

The FNO is loaded the same way as in scripts/diagnose_fno.py: from the run
metadata via build_closure(cfg), with an optional trained checkpoint whose leaf
shapes are validated against the architecture before loading. Data loading and
normalisation also mirror diagnose_fno.py.

Put in scripts/ next to diagnose_fno.py. Examples:

    # everything, real data, trained FNO checkpoint
    python scripts/interrogate_fno.py \
        --metadata outputs/fno/512_to_32/01/metadata.json \
        --data-dir data/data_hr512_nx32_08 \
        --checkpoint-dir saved_closures/fno/512_to_32/01

    # fresh (untrained) FNO built from metadata, real data
    python scripts/interrogate_fno.py --metadata <metadata.json> --data-dir <data_dir> \
        --tests params locality gain truncation fit

    # no metadata / data: default-arch FNO on synthetic jet-like fields
    python scripts/interrogate_fno.py --tests params equivariance locality gain

Tests (pick with --tests, default all):
  params         parameter counts by component, FNO vs ResNet
  equivariance   translation-equivariance of each spectral layer and the full nets
  locality       how local is each net's input->output Jacobian (receptive field)
  gain           linear transfer function per wavenumber shell, zonal (kx=0)
                 vs non-zonal, plus Jacobian spectral norm (rollout amplification)
  truncation     fraction of input / target spectral energy kept for various modes
  weights        (needs --checkpoint-dir) trained-vs-init weight change per |k| shell
  fit            train both nets on synthetic closures (linear, k^4, advective) and
                 compare held-out relative error. Controlled, no rollout involved.
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from omegaconf import OmegaConf

from model.ML.architectures.build_model import build_closure
from model.ML.architectures.resnet import ResNet
from model.ML.utils.dataloading import ZarrDataLoader, checkpointer

ALL_TESTS = ["params", "equivariance", "locality", "gain", "truncation", "weights", "fit"]
DEFAULT_FNO_ARCH = dict(width=32, xmodes=16, ymodes=16, projection_width=64, depth=6, activation="gelu")


# --------------------------------------------------------------------------- utils
def rms(x):
    return float(jnp.sqrt(jnp.mean(jnp.abs(x) ** 2)))


def rel(a, b):
    return float(jnp.linalg.norm(a - b) / (jnp.linalg.norm(b) + 1e-12))


def mk_cfg(nz):
    return SimpleNamespace(params=SimpleNamespace(nz=nz))


def count_params(tree):
    total = 0
    for leaf in jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_array)):
        total += leaf.size * (2 if jnp.iscomplexobj(leaf) else 1)
    return int(total)


def count_by_field(module):
    """Parameter count for every top-level field of an eqx Module (+ total)."""
    out = {}
    for field in dataclasses.fields(module):
        n = count_params(getattr(module, field.name))
        if n:
            out[field.name] = n
    out["total"] = count_params(module)
    return out


def shell_index(h, w):
    ky = np.fft.fftfreq(h) * h
    kx = np.fft.rfftfreq(w) * w
    kmag = np.hypot(ky[:, None], kx[None, :])
    return np.rint(kmag).astype(int), ky, kx


def synthetic_samples(n, nx, nz, n_jets=4, seed=0):
    """Red-spectrum noise plus zonal jets, roughly like a beta-plane PV field."""
    rng = np.random.default_rng(seed)
    shells, ky, kx = shell_index(nx, nx)
    kmag = np.hypot(ky[:, None], kx[None, :])
    amp = np.where(kmag > 0, np.maximum(kmag, 1e-9) ** -1.5, 0.0) * (kmag <= nx // 3)
    yy = np.arange(nx)[:, None] * np.ones((1, nx))
    out = []
    for _ in range(n * nz):
        ph = rng.normal(size=kmag.shape) + 1j * rng.normal(size=kmag.shape)
        f = np.fft.irfft2(amp * ph, s=(nx, nx))
        out.append(f / f.std() + 2.0 * np.sin(2 * np.pi * n_jets * yy / nx))
    return np.asarray(out, np.float32).reshape(n, nz, nx, nx)


# --------------------------------------------------------------------------- loading
def load_samples(args, parameters):
    """Load and normalise samples exactly as diagnose_fno.py does."""
    if args.data_dir is None:
        nx = int(parameters["nx"]) if parameters else args.nx
        nz = int(parameters.get("nz", 1)) if parameters else args.nz
        print("[data] no --data-dir: using synthetic jet-like fields")
        samples = synthetic_samples(args.n_samples, nx, nz, seed=args.seed)
    else:
        data_dir = args.data_dir.resolve()
        data = ZarrDataLoader(str(data_dir))
        trajectory_count = min(args.max_trajectories, data.n_trajectories)
        if trajectory_count == 0:
            raise ValueError(f"No training trajectories found in {data_dir}")
        chunks = []
        for trajectory_index in range(trajectory_count):
            trajectory, _ = data.get_trajectory(trajectory_index)
            indices = np.linspace(0, trajectory.shape[0] - 1, args.frames_per_trajectory, dtype=int)
            chunks.extend(trajectory[indices])
        samples = np.asarray(chunks, dtype=np.float32)
        if samples.ndim != 4:
            raise ValueError(f"Expected sample fields shaped (N, C, H, W), got {samples.shape}")
        if parameters and samples.shape[-2:] != (int(parameters["nx"]), int(parameters["nx"])):
            raise ValueError(
                f"Training data grid {samples.shape[-2:]} does not match metadata nx={parameters['nx']}"
            )

    layer_axis = samples.ndim - 3
    reduce_axes = tuple(axis for axis in range(samples.ndim) if axis != layer_axis)
    q_mean = np.mean(samples, axis=reduce_axes, keepdims=True)
    q_std = np.maximum(np.std(samples, axis=reduce_axes, keepdims=True), 1e-6)
    return ((samples - q_mean) / (q_std + 1e-6)).astype(np.float32)


def build_fno(args):
    """Return (fno, fno_init, parameters, arch).

    With --metadata: build through build_closure(cfg) like diagnose_fno.py, load the
    checkpoint if given (after validating leaf shapes against the template), and build
    a fresh closure for fno_init. Without metadata: construct the FNO class directly
    with default arch (no checkpoint possible).
    """
    if args.metadata is None:
        if args.checkpoint_dir is not None:
            raise SystemExit("--checkpoint-dir needs --metadata")
        from model.ML.architectures.fno import FNO

        fno = FNO(**DEFAULT_FNO_ARCH, key=jax.random.PRNGKey(args.seed), cfg=mk_cfg(args.nz))
        return fno, fno, None, dict(DEFAULT_FNO_ARCH)

    metadata_path = args.metadata.resolve()
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    parameters = metadata["parameters"]
    model_arch = metadata["training"]["model_arch"]
    if metadata.get("model_type") != "fno":
        raise ValueError(f"Expected FNO metadata, got {metadata.get('model_type')!r}")

    cfg = OmegaConf.create(
        {
            "params": parameters,
            "ml": {"model_type": "fno"},
            "architectures": {"fno": model_arch},
        }
    )
    template = build_closure(cfg)  # fresh initialisation
    loaded_leaves = None
    if args.checkpoint_dir is not None:
        loaded_leaves, _, _, _ = checkpointer(
            None, None, str(args.checkpoint_dir.resolve()), save=False
        )
        if loaded_leaves is None:
            raise ValueError(f"No trained closure checkpoint found in {args.checkpoint_dir}")
        template_params, _ = eqx.partition(template, eqx.is_array)
        template_shapes = [leaf.shape for leaf in jax.tree_util.tree_leaves(template_params)]
        loaded_shapes = [np.shape(leaf) for leaf in loaded_leaves]
        if loaded_shapes != template_shapes:
            raise ValueError(
                "Checkpoint parameter shapes do not match the architecture in the run metadata"
            )
    model = build_closure(cfg, loaded_leaves)
    return model, template, parameters, model_arch


def build_resnet(args, nz):
    return ResNet(
        key=jax.random.PRNGKey(args.seed),
        nlayers=args.resnet_nlayers,
        kernel_size=args.resnet_kernel,
        width=args.resnet_width,
        activation="elu",
        cfg=mk_cfg(nz),
    )


# --------------------------------------------------------------------------- tests
def test_params(fno, resnet, **_):
    f, r = count_by_field(fno), count_by_field(resnet)
    return {"fno": f, "resnet": r, "ratio_fno_to_resnet": f["total"] / max(r["total"], 1)}


def test_equivariance(fno, resnet, X, **_):
    x = jnp.asarray(X[0])
    out = {}
    shifts = [(3, 5), (7, 1), (0, 9)]

    def check(fn):
        errs = []
        for dy, dx in shifts:
            a = fn(jnp.roll(x_in, (dy, dx), axis=(-2, -1)))
            b = jnp.roll(fn(x_in), (dy, dx), axis=(-2, -1))
            errs.append(rel(a, b))
        return max(errs)

    x_in = x
    out["fno_full_net_max_rel_err"] = check(fno)
    out["resnet_full_net_max_rel_err"] = check(resnet)
    # each spectral layer on a random multi-channel input
    key = jax.random.PRNGKey(1)
    layer_errs = []
    for layer in fno.spec_layers:
        x_in = jax.random.normal(key, (layer.in_channels,) + x.shape[-2:])
        layer_errs.append(check(layer))
    out["fno_spectral_layer_max_rel_err"] = layer_errs
    out["note"] = "~1e-6 means exactly translation-equivariant; anything larger is a bug."

    # constrained vs plain-irfft2 projection of the independent weights
    layer = fno.spec_layers[0]
    x_in = jax.random.normal(key, (layer.in_channels,) + x.shape[-2:])
    constrained = layer(x_in)
    H, W = x_in.shape[-2:]
    xft = jnp.fft.rfft2(x_in[None], axes=(-2, -1))
    ym, xm = min(layer.ymodes, H // 2), min(layer.xmodes, W // 2 + 1)
    o = jnp.zeros((1, layer.out_channels, H, W // 2 + 1), jnp.complex64)
    o = o.at[:, :, :ym, :xm].set(layer.compl_mul2d(xft[:, :, :ym, :xm], layer.weights1[:, :, :ym, :xm]))
    o = o.at[:, :, -ym:, :xm].set(layer.compl_mul2d(xft[:, :, -ym:, :xm], layer.weights2[:, :, :ym, :xm]))
    if hasattr(layer, "_wavenumber_gain"):  # layers with a fixed |k| gain
        o = o * layer._wavenumber_gain(H, W)[None, None]
    naive = jnp.fft.irfft2(o[0], s=(H, W), axes=(-2, -1))
    out["constrained_vs_naive_irfft_rel_diff"] = rel(constrained, naive)
    return out


def test_locality(fno, resnet, X, **_):
    H, W = X.shape[-2:]
    cy, cx = H // 2, W // 2
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    dist = np.hypot(np.minimum(np.abs(yy - cy), H - np.abs(yy - cy)),
                    np.minimum(np.abs(xx - cx), W - np.abs(xx - cx)))
    res = {}
    for name, net in [("fno", fno), ("resnet", resnet)]:
        mass = np.zeros((H, W))
        for i in range(min(6, len(X))):
            g = jax.grad(lambda z: net(z)[0, cy, cx])(jnp.asarray(X[i]))
            mass += np.asarray(jnp.sum(g ** 2, axis=0))
        mass /= mass.sum() + 1e-30
        order = np.argsort(dist.ravel())
        csum = np.cumsum(mass.ravel()[order])
        d_sorted = dist.ravel()[order]
        res[name] = {
            "radius_50pct_jacobian_mass": float(d_sorted[np.searchsorted(csum, 0.5)]),
            "radius_90pct_jacobian_mass": float(d_sorted[np.searchsorted(csum, 0.9)]),
            "mass_within_3_cells": float(mass[dist <= 3].sum()),
        }
    res["note"] = ("Closures for subgrid terms are physically local. A much wider FNO radius "
                   "means it has to learn locality from data instead of getting it for free.")
    return res


def jac_spectral_norm(net, x, iters=30, seed=0):
    _, vjp = jax.vjp(net, x)
    v = jax.random.normal(jax.random.PRNGKey(seed), x.shape)
    v = v / jnp.linalg.norm(v)
    s = 0.0
    for _ in range(iters):
        _, jv = jax.jvp(net, (x,), (v,))
        (w,) = vjp(jv)
        s = jnp.linalg.norm(w)
        v = w / (s + 1e-30)
    return float(jnp.sqrt(s))


def test_gain(fno, resnet, X, **_):
    H, W = X.shape[-2:]
    shells, ky, kx = shell_index(H, W)
    rng = np.random.default_rng(0)
    modes = {}  # (shell, zonal) -> list[(ky,kx)]
    for iy in range(H):
        for ix in range(W // 2 + 1):
            if iy == 0 and ix == 0:
                continue
            modes.setdefault((int(shells[iy, ix]), ix == 0), []).append((ky[iy], kx[ix]))
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    x0 = jnp.asarray(X[0])
    res = {"shell": [], "fno_zonal": [], "fno_nonzonal": [], "resnet_zonal": [], "resnet_nonzonal": []}
    for sh in range(1, min(H, W) // 2 + 1):
        res["shell"].append(sh)
        for flag, tag in [(True, "zonal"), (False, "nonzonal")]:
            cand = modes.get((sh, flag), [])
            picks = [cand[i] for i in rng.permutation(len(cand))[:4]]
            for name, net in [("fno", fno), ("resnet", resnet)]:
                vals = []
                for (ky_, kx_) in picks:
                    t = np.sqrt(2) * np.cos(2 * np.pi * (kx_ * xx / W + ky_ * yy / H) + rng.uniform(0, 6.28))
                    t = jnp.asarray(np.broadcast_to(t, x0.shape), jnp.float32)
                    _, jv = jax.jvp(net, (x0,), (t,))
                    vals.append(rms(jv))
                res[f"{name}_{tag}"].append(float(np.mean(vals)) if vals else None)
    n_norm = min(3, len(X))
    res["jacobian_spectral_norm"] = {
        "fno": float(np.mean([jac_spectral_norm(fno, jnp.asarray(X[i])) for i in range(n_norm)])),
        "resnet": float(np.mean([jac_spectral_norm(resnet, jnp.asarray(X[i])) for i in range(n_norm)])),
    }
    res["note"] = ("Gain = output rms per unit-rms input perturbation at that |k|. Look for: (a) FNO "
                   "gain at high |k| much larger than ResNet (rollout instability) or ~0 (can't "
                   "act where the closure matters); (b) zonal vs non-zonal mismatch in FNO only.")
    return res


def retained_mask(h, w, xm, ym):
    m = np.zeros((h, w // 2 + 1), bool)
    y, x = min(ym, h // 2), min(xm, w // 2 + 1)
    m[:y, :x] = True
    m[-y:, :x] = True
    return m


def make_targets(X):
    N, C, H, W = X.shape
    ky = np.fft.fftfreq(H) * H
    kx = np.fft.rfftfreq(W) * W
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    K2 = KX ** 2 + KY ** 2
    inv = lambda F: np.fft.irfft2(F, s=(H, W))
    Xh = np.fft.rfft2(X)
    psih = -Xh / np.where(K2 == 0, 1, K2)
    psih[..., 0, 0] = 0
    u, v = inv(-1j * KY * psih), inv(1j * KX * psih)
    adv = -(u * inv(1j * KX * Xh) + v * inv(1j * KY * Xh))
    t = {"linear_laplacian": inv(-K2 * Xh), "hyperdiffusion_k4": inv(-(K2 ** 2) * Xh), "advective_nonlinear": adv}
    return {k: (v_ / (v_.std() + 1e-12)).astype(np.float32) for k, v_ in t.items()}


def test_truncation(X, fno, **_):
    targets = {"input_q": X, **make_targets(X)}
    H, W = X.shape[-2:]
    shells, _, _ = shell_index(H, W)
    model_xm, model_ym = fno.spec_layers[0].xmodes, fno.spec_layers[0].ymodes
    res = {}
    for name, f in targets.items():
        spec = np.mean(np.abs(np.fft.rfft2(f, norm="ortho")) ** 2, axis=(0, 1))
        total = spec.sum() + 1e-30
        entry = {"zonal_kx0_energy_share": float(spec[:, 0].sum() / total)}
        for m in [4, 6, 8, 12, 16]:
            entry[f"modes_{m}x{m}"] = float(spec[retained_mask(H, W, m, m)].sum() / total)
        entry[f"model_modes_y{model_ym}_x{model_xm}"] = float(
            spec[retained_mask(H, W, model_xm, model_ym)].sum() / total
        )
        entry["energy_share_by_shell"] = [float(spec[shells == s].sum() / total) for s in range(0, min(H, W) // 2 + 1)]
        res[name] = entry
    res["note"] = ("If modes_16x16 ~ 1.0 for every row, the FNO's mode truncation does nothing at this "
                   "resolution: it is a full per-wavenumber channel mixer with no locality prior.")
    return res


def test_weights(fno, fno_init, args, **_):
    if args.checkpoint_dir is None:
        return {"skipped": "needs --checkpoint-dir and --metadata"}
    out = []
    for li, (lt, l0) in enumerate(zip(fno.spec_layers, fno_init.spec_layers)):
        ym, xm = lt.ymodes, lt.xmodes
        iy, ix = np.arange(ym)[:, None], np.arange(xm)[None, :]
        k1, k2 = np.rint(np.hypot(iy, ix)).astype(int), np.rint(np.hypot(ym - iy, ix)).astype(int)
        nsh = int(max(k1.max(), k2.max())) + 1
        trained_mag, rel_change = np.zeros(nsh), np.zeros(nsh)
        for s in range(nsh):
            num = den = mag = cnt = 0.0
            for wt, w0, kk in [(lt.weights1, l0.weights1, k1), (lt.weights2, l0.weights2, k2)]:
                m = kk == s
                if not m.any():
                    continue
                wt_, w0_ = np.asarray(wt)[:, :, m], np.asarray(w0)[:, :, m]
                num += np.sum(np.abs(wt_ - w0_) ** 2)
                den += np.sum(np.abs(w0_) ** 2)
                mag += np.sum(np.abs(wt_) ** 2)
                cnt += wt_.size
            rel_change[s] = np.sqrt(num / (den + 1e-30))
            trained_mag[s] = np.sqrt(mag / max(cnt, 1))
        out.append({"layer": li, "rms_weight_by_shell": trained_mag.tolist(),
                    "relative_change_from_init_by_shell": rel_change.tolist()})
    return {"layers": out,
            "note": "Assumes build_closure(cfg) re-creates the original init (same seed). Near-zero "
                    "change at high |k| = those weights never learned anything."}


def fit_one(model, Xtr, Ytr, Xte, Yte, steps, lr, batch, seed):
    params, static = eqx.partition(model, eqx.is_inexact_array)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    state = opt.init(params)

    def loss_fn(p, xb, yb):
        net = eqx.combine(p, static)
        return jnp.mean((jax.vmap(net)(xb) - yb) ** 2)

    @eqx.filter_jit
    def step(p, s, xb, yb):
        l, g = eqx.filter_value_and_grad(loss_fn)(p, xb, yb)
        up, s = opt.update(g, s, p)
        return eqx.apply_updates(p, up), s, l

    rng = np.random.default_rng(seed)
    for _ in range(steps):
        idx = rng.integers(0, len(Xtr), batch)
        params, state, _ = step(params, state, Xtr[idx], Ytr[idx])
    net = eqx.combine(params, static)
    ev = lambda x, y: float(jnp.mean((jax.vmap(net)(x) - y) ** 2) / (jnp.var(y) + 1e-12))
    return {"train_rel_mse": ev(Xtr, Ytr), "test_rel_mse": ev(Xte, Yte)}


def test_fit(fno_init, resnet, X, args, **_):
    targets = make_targets(X)
    n_tr = max(1, int(0.8 * len(X)))
    res = {}
    for name, Y in targets.items():
        Xtr, Xte = jnp.asarray(X[:n_tr]), jnp.asarray(X[n_tr:])
        Ytr, Yte = jnp.asarray(Y[:n_tr]), jnp.asarray(Y[n_tr:])
        res[name] = {
            "fno": fit_one(fno_init, Xtr, Ytr, Xte, Yte, args.fit_steps, args.lr, args.batch, 0),
            "resnet": fit_one(resnet, Xtr, Ytr, Xte, Yte, args.fit_steps, args.lr, args.batch, 0),
        }
    res["note"] = ("Relative MSE (0 = perfect, 1 = predicting the mean). A big train/test gap for the "
                   "FNO relative to the ResNet = parameter-count / inductive-bias problem. If FNO is "
                   "worse even on the linear Laplacian target, the architecture (no linear skip, "
                   "depth of GELUs) is hurting.")
    return res


# --------------------------------------------------------------------------- main
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tests", nargs="+", default=ALL_TESTS, choices=ALL_TESTS)
    p.add_argument("--metadata", type=Path, help="Run metadata JSON (FNO built via build_closure)")
    p.add_argument("--data-dir", type=Path, help="Directory containing the training trajectories.zarr")
    p.add_argument("--checkpoint-dir", type=Path, help="Trained closure directory (needs --metadata)")
    p.add_argument("--max-trajectories", type=int, default=3)
    p.add_argument("--frames-per-trajectory", type=int, default=8)
    p.add_argument("--nz", type=int, default=1, help="synthetic data without metadata only")
    p.add_argument("--nx", type=int, default=32, help="synthetic data without metadata only")
    p.add_argument("--n-samples", type=int, default=24, help="synthetic data only")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resnet-nlayers", type=int, default=7)
    p.add_argument("--resnet-kernel", type=int, default=5)
    p.add_argument("--resnet-width", type=int, default=64)
    p.add_argument("--fit-steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.max_trajectories <= 0 or args.frames_per_trajectory <= 0:
        p.error("Trajectory and frame counts must be positive")
    return args


def main():
    args = parse_args()
    fno, fno_init, parameters, arch = build_fno(args)
    X = load_samples(args, parameters)
    nz = X.shape[1]
    resnet = build_resnet(args, nz)
    print(f"[setup] samples {X.shape}, fno arch {arch}")
    ctx = dict(fno=fno, fno_init=fno_init, resnet=resnet, X=X, args=args)
    fns = {"params": test_params, "equivariance": test_equivariance, "locality": test_locality,
           "gain": test_gain, "truncation": test_truncation, "weights": test_weights, "fit": test_fit}
    report = {
        "metadata": str(args.metadata.resolve()) if args.metadata else None,
        "data_dir": str(args.data_dir.resolve()) if args.data_dir else None,
        "checkpoint_dir": str(args.checkpoint_dir.resolve()) if args.checkpoint_dir else None,
        "samples_shape": list(X.shape),
    }
    for name in args.tests:
        print(f"[run] {name}")
        report[name] = fns[name](**ctx)
    rendered = json.dumps(report, indent=2, default=float)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"\nWrote report to {args.output.resolve()}")


if __name__ == "__main__":
    main()