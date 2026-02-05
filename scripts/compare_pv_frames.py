import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def _as_complex(arr: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(arr):
        return arr
    if arr.dtype.fields and "re" in arr.dtype.fields and "im" in arr.dtype.fields:
        return arr["re"] + 1j * arr["im"]
    return arr


def load_julia_pv(jld2_path: Path):
    with h5py.File(jld2_path, "r") as f:
        has_q = "snapshots/q" in f
        has_qh = "snapshots/qh" in f

        if has_q:
            q_group = f["snapshots/q"]
            iterations = sorted(int(k) for k in q_group.keys())
            pv_frames = []
            for i in iterations:
                q = q_group[str(i)][()]
                q = _as_complex(np.asarray(q))
                if np.iscomplexobj(q):
                    q = np.real(q)
                pv_frames.append(q)
        elif has_qh:
            qh_group = f["snapshots/qh"]
            iterations = sorted(int(k) for k in qh_group.keys())
            nx = int(np.asarray(f["grid/nx"][()]))
            ny = int(np.asarray(f["grid/ny"][()]))
            pv_frames = []
            for i in iterations:
                qh = qh_group[str(i)][()]
                qh = _as_complex(np.asarray(qh))
                q = np.fft.irfft2(qh, s=(ny, nx), norm="ortho")
                pv_frames.append(q)
        else:
            raise KeyError("No PV data found. Expected snapshots/q or snapshots/qh in JLD2.")

        t_group = f["snapshots/t"]
        times = np.array([t_group[str(i)][()] for i in iterations], dtype=np.float64)

    return np.stack(pv_frames, axis=0), times


def load_python_pv(npz_path: Path):
    data = np.load(npz_path)
    pv = np.asarray(data["pv"], dtype=np.float64)
    times = np.asarray(data["times"], dtype=np.float64)
    return pv, times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--python",
        dest="python_npz",
        default="../outputs/python_pv_frames.npz",
        help="Path to python PV frames (.npz).",
    )
    parser.add_argument(
        "--jld2",
        dest="jld2_path",
        default="../julia/singlelayerqg_match_python.jld2",
        help="Path to Julia PV snapshots (.jld2).",
    )
    parser.add_argument(
        "--out",
        dest="out_dir",
        default="../outputs",
        help="Output directory for plots.",
    )
    parser.add_argument(
        "--fps",
        dest="fps",
        type=int,
        default=16,
        help="Frames per second for MP4 video.",
    )
    parser.add_argument(
        "--match-initial",
        dest="match_initial",
        action="store_true",
        help="Subtract each model's initial frame so comparisons use the same baseline.",
    )
    args = parser.parse_args()

    python_path = Path(args.python_npz).resolve()
    jld2_path = Path(args.jld2_path).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pv_py, _ = load_python_pv(python_path)
    pv, times = load_julia_pv(jld2_path)
    n = min(len(pv_py), len(pv))
    pv_py = pv_py[:n]
    pv = pv[:n]

    if args.match_initial:
        pv_py0 = pv_py[0].copy()
        pv_jl0 = pv[0].copy()
        pv_py = pv_py - pv_py0
        pv = pv - pv_jl0

    vmax_py = float(np.max(np.abs(pv_py)))
    vmax_jl = float(np.max(np.abs(pv)))
    diff = pv_py[0] - pv[0]
    diff_max = float(np.max(np.abs(diff)))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    im0 = axes[0].imshow(pv_py[0], origin="lower", cmap="viridis", vmin=-vmax_py, vmax=vmax_py)
    axes[0].set_title("Python PV (frame 0)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(pv[0], origin="lower", cmap="viridis", vmin=-vmax_jl, vmax=vmax_jl)
    axes[1].set_title("Julia PV")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(diff, origin="lower", cmap="coolwarm", vmin=-diff_max, vmax=diff_max)
    axes[2].set_title("Difference (Py - JL)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    def update(i):
        pv_py_i = pv_py[i]
        pv_jl_i = pv[i]
        pv_diff_i = pv_py_i - pv_jl_i
        vmax_py_i = float(np.max(np.abs(pv_py_i)))
        vmax_jl_i = float(np.max(np.abs(pv_jl_i)))
        diff_max_i = float(np.max(np.abs(pv_diff_i)))

        im0.set_data(pv_py_i)
        im0.set_clim(-vmax_py_i, vmax_py_i)
        axes[0].set_title(f"Python PV (frame {i})")

        im1.set_data(pv_jl_i)
        im1.set_clim(-vmax_jl_i, vmax_jl_i)

        im2.set_data(pv_diff_i)
        im2.set_clim(-diff_max_i, diff_max_i)
        return im0, im1, im2

    anim = animation.FuncAnimation(fig, update, frames=n, interval=1000 / max(args.fps, 1), blit=False)
    video_out = out_dir / "pv_compare.mp4"
    writer = animation.FFMpegWriter(fps=args.fps)
    try:
        anim.save(video_out, writer=writer)
        print(f"Wrote video to {video_out}")
    except FileNotFoundError:
        gif_out = out_dir / "pv_compare.gif"
        gif_writer = animation.PillowWriter(fps=args.fps)
        anim.save(gif_out, writer=gif_writer)
        print("FFmpeg not found; wrote GIF instead to", gif_out)
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
