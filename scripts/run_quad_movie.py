"""Create a quad-panel animation comparing truth, ML-adjusted prediction, error, and SGS.

Saves to outputs/quad_movie.mp4 (ffmpeg) or outputs/quad_movie.gif (Pillow fallback).

Usage:
    python scripts/run_quad_movie.py --nsteps 1000 --cadence 5
"""
import os
import argparse
from omegaconf import OmegaConf
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from model.ML.utils.dataloading import find_existing_run, find_existing_closure, ZarrDataLoader, checkpointer
from model.ML.architectures.build_model import build_closure
from model.ML.utils.coarsen import coarsen
from model.core.model import QGM
from model.core.steppers import SteppedModel, AB3Stepper
from model.ML.train import load_forced_model, roll_out

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'default.yaml')
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'saved_closures')
OUT_DIR = os.path.join(BASE_DIR, 'outputs')
DIAG_DIR = os.path.join(OUT_DIR, 'diagnostics')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DIAG_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--nsteps', type=int, default=1000)
parser.add_argument('--cadence', type=int, default=5)
parser.add_argument('--mitigation', type=str, default='project', choices=['baseline','project','scale','clip'],
                    help='mitigation wrapper to apply to the closure before rollout')
args = parser.parse_args()

cfg = OmegaConf.load(CONFIG_PATH)
params = dict(OmegaConf.to_container(cfg.params, resolve=True))

dt = float(cfg.plotting.dt)
hr_nx = int(params['hr_nx'])
nx = int(params['nx'])
ratio = int(hr_nx // nx)
low_res_dt = dt * ratio

print('Quad movie: dt', dt, 'ratio', ratio, 'low_res_dt', low_res_dt)

# build models
hr_model = SteppedModel(model=QGM({**params, 'nx': hr_nx}), stepper=AB3Stepper(dt=dt))
lr_model = coarsen(hr_model.model, nx)

# Find data run
timing_metadata = {
    'spinup': int(cfg.plotting.spinup),
    'nsteps': int(cfg.plotting.nsteps),
    "dt (original)": float(cfg.plotting.dt),
    'auto_dt': bool(cfg.plotting.auto_dt),
    'final dt': float(dt),
    'batch_steps': int(cfg.ml.batch_steps),
}
run_dir, found = find_existing_run(DATA_DIR, params, timing_metadata)
if not found:
    candidates = [d for d in os.listdir(DATA_DIR) if d.startswith(f"data_hr{hr_nx}_nx{nx}_")]
    if len(candidates) > 0:
        run_dir = os.path.join(DATA_DIR, sorted(candidates)[-1])
        found = True
        print(f"No exact metadata match; falling back to existing run: {run_dir}")
    else:
        raise RuntimeError(f"No matching data run found in {DATA_DIR}.")
print('Using data run:', run_dir)

loader = ZarrDataLoader(run_dir)
traj = loader.get_trajectory(0)
T = traj.shape[0]
print('Loaded trajectory length', T)

# find closure
closure_dir, found = find_existing_closure(MODEL_DIR, params, timing_metadata, cfg.ml.model_type)
if not found:
    candidates = [d for d in os.listdir(MODEL_DIR) if d.startswith(f"{cfg.ml.model_type}_hr")]
    chosen = None
    for d in candidates:
        parts = d.split('_')
        try:
            hr_val = int([p for p in parts if p.startswith('hr')][0][2:])
            nx_val = int([p for p in parts if p.startswith('nx')][0][2:])
        except Exception:
            continue
        if hr_val == hr_nx and nx_val == nx:
            chosen = d
            break
    if chosen is None and len(candidates) > 0:
        chosen = sorted(candidates)[-1]
    if chosen is not None:
        closure_dir = os.path.join(MODEL_DIR, chosen)
        found = True
        print('Falling back to closure dir', closure_dir)
if not found:
    raise RuntimeError('No trained closure found')
print('Using closure dir:', closure_dir)

loaded_leaves, _, ckpt_meta, _ = checkpointer(None, None, closure_dir, save=False)
closure = build_closure(cfg, loaded_leaves)

# small wrapper factory to apply simple mitigations (project is the recommended stabiliser)
def make_wrapped_closure(kind, scale=0.5, clip_val=None, eps=1e-8):
    def wrapped(q):
        out = closure(q)
        if kind == 'baseline':
            return out
        if kind == 'scale':
            return out * scale
        if kind == 'clip':
            if clip_val is None:
                return out
            return jnp.clip(out, -clip_val, clip_val)
        if kind == 'project':
            qh = jnp.fft.rfftn(q, axes=(-2,-1), norm='ortho')
            out_qh = jnp.fft.rfftn(out, axes=(-2,-1), norm='ortho')
            num = jnp.real(jnp.conj(qh) * out_qh)
            den = jnp.abs(qh)**2 + eps
            alpha = num / den
            out_qh_proj = out_qh - alpha * qh
            return jnp.fft.irfftn(out_qh_proj, axes=(-2,-1), norm='ortho', s=out.shape[-2:])
        return out
    return wrapped

# choose effective closure (apply mitigation wrapper if requested)
if args.mitigation == 'baseline':
    effective_closure = closure
elif args.mitigation in ('scale','clip','project'):
    # determine clip value if needed from a small sample (fallback)
    clip_val = None
    try:
        # quick try: compute an approximate clip percentile from a few frames if data available
        clip_val = float(np.load(os.path.join(DIAG_DIR, 'closure_clip_value.npy'))[0])
    except Exception:
        clip_val = None
    effective_closure = make_wrapped_closure(args.mitigation, scale=0.5, clip_val=clip_val)
else:
    effective_closure = closure

# build forced model with effective closure
forced_model, closure_params, closure_static = load_forced_model(lr_model, effective_closure, low_res_dt)

# Closed-loop rollout for nsteps (bounded by available truth length)
nsteps = min(args.nsteps, T-1)
print('Rolling out closed-loop for', nsteps, 'steps')
traj_dqh = roll_out(init_q=traj[0], forced_model=forced_model, nsteps=nsteps, template_state=lr_model.initialise(jax.random.PRNGKey(0)), closure_params=closure_params)

# reconstruct predicted frames
init_qh = jnp.fft.rfftn(traj[0], axes=(-2, -1), norm='ortho')
qh_traj = jnp.concatenate([init_qh[None, ...], init_qh[None, ...] + jnp.cumsum(traj_dqh, axis=0)], axis=0)
real_shape = traj.shape[-2:]
pred_frames = jax.vmap(lambda x: jnp.fft.irfftn(x, axes=(-2, -1), norm='ortho', s=real_shape))(qh_traj)

pred_np = np.asarray(jax.device_get(pred_frames))
truth_np = np.asarray(traj[:pred_np.shape[0]])

# compute sgs by applying closure to predicted frames (except final)
@jax.jit
def _apply_closure(q):
    return effective_closure(q.astype(jnp.float32)).astype(q.dtype)

sgs_pred = np.asarray(jax.device_get(jax.vmap(_apply_closure)(pred_frames[:-1])))

# compute error frames
err = pred_np - truth_np

# animation frames sampling
indices = np.arange(0, pred_np.shape[0], args.cadence)
print('Animating', len(indices), 'frames with cadence', args.cadence)

# determine color limits per panel
vmin_truth = np.percentile(truth_np, 1)
vmax_truth = np.percentile(truth_np, 99)
vmin_err = np.percentile(err, 1)
vmax_err = np.percentile(err, 99)
vmin_sgs = np.percentile(sgs_pred, 1)
vmax_sgs = np.percentile(sgs_pred, 99)

fig, axes = plt.subplots(2,2,figsize=(8,8))
ax_truth = axes[0,0]
ax_ml = axes[0,1]
ax_err = axes[1,0]
ax_sgs = axes[1,1]

im_truth = ax_truth.imshow(truth_np[0,0], origin='lower', cmap='RdBu_r', vmin=vmin_truth, vmax=vmax_truth)
ax_truth.set_title('Truth')
im_ml = ax_ml.imshow(pred_np[0,0], origin='lower', cmap='RdBu_r', vmin=vmin_truth, vmax=vmax_truth)
ax_ml.set_title('ML adjusted')
im_err = ax_err.imshow(err[0,0], origin='lower', cmap='RdBu_r', vmin=vmin_err, vmax=vmax_err)
ax_err.set_title('Error')
im_sgs = ax_sgs.imshow(sgs_pred[0,0], origin='lower', cmap='RdBu_r', vmin=vmin_sgs, vmax=vmax_sgs)
ax_sgs.set_title('SGS')

for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])

fig.colorbar(im_truth, ax=[ax_truth, ax_ml], shrink=0.6)
fig.colorbar(im_err, ax=ax_err, shrink=0.6)
fig.colorbar(im_sgs, ax=ax_sgs, shrink=0.6)

def update(i):
    idx = indices[i]
    im_truth.set_data(truth_np[idx,0])
    im_ml.set_data(pred_np[idx,0])
    im_err.set_data(err[idx,0])
    if idx < sgs_pred.shape[0]:
        im_sgs.set_data(sgs_pred[idx,0])
    fig.suptitle(f'timestep {idx}')
    return im_truth, im_ml, im_err, im_sgs

anim = animation.FuncAnimation(fig, update, frames=len(indices), interval=100, blit=False)

out_mp4 = os.path.join(OUT_DIR, 'quad_movie.mp4')
out_gif = os.path.join(OUT_DIR, 'quad_movie.gif')

saved = False
try:
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='QG'), bitrate=4000)
    anim.save(out_mp4, writer=writer)
    print('Saved movie to', out_mp4)
    saved = True
except Exception as e:
    print('ffmpeg save failed:', e)

if not saved:
    try:
        from matplotlib.animation import PillowWriter
        writer = PillowWriter(fps=10)
        anim.save(out_gif, writer=writer)
        print('Saved gif to', out_gif)
        saved = True
    except Exception as e:
        print('Pillow save failed:', e)

if not saved:
    print('Failed to save animation; consider installing ffmpeg or Pillow support')

print('Done')
