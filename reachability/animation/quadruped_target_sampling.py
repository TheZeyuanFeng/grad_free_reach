"""reachability/animation/quadruped_target_sampling.py

Renders a batch of Quadruped.sample draws as a slideshow -- one
MuJoCo frame per sample, no physics stepping, since these are independent
static states, not a rollout -- with a synced scatter plot showing where each
sample's l(x) falls relative to the standing-target boundary. For
qualitatively (and quantitatively -- see the printed summary) checking the
"target" bucket's distribution, without needing a trained checkpoint.

Usage:
    python -m reachability.animation.quadruped_target_sampling
    python -m reachability.animation.quadruped_target_sampling --num_samples 40 --fps 2
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import jax
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

from reachability.dynamics.quadruped import (
    Quadruped, INIT_HEIGHT, STABLE_HEIGHT_TOL, STABLE_DOT_MIN, _find_mjcf,
)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Slideshow a batch of Quadruped.sample draws with an l(x) scatter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--num_samples", type=int, default=30, help="Batch size drawn from dyn.sample().")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=float, default=2.0, help="Slideshow speed (samples/sec, not a physics rate).")
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--camera", default=None, help="Fixed camera name/id; default is a static framing.")
    p.add_argument("--mjcf", default=None)
    p.add_argument("--out", default="/tmp/quadruped_target_sampling.mp4")
    return p


def _find_render_mjcf(override):
    if override is not None:
        return override
    a1_path = _find_mjcf()
    scene_path = os.path.join(os.path.dirname(a1_path), "scene.xml")
    return scene_path if os.path.isfile(scene_path) else a1_path


def _state_to_qpos_qvel(x: np.ndarray, nv: int) -> tuple:
    q, dq, quat_xyzw, base_z = x[0:12], x[12:24], x[24:28], x[28]
    base_lin_vel, base_ang_vel, base_xy = x[29:32], x[32:35], x[35:37]
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    qpos = np.concatenate([base_xy, [base_z], quat_wxyz, q])
    qvel = np.zeros(nv, dtype=np.float64)
    qvel[0:3] = base_lin_vel
    qvel[3:6] = base_ang_vel
    qvel[6:] = dq
    return qpos, qvel


def render_frames(states: np.ndarray, mjcf_path: str, width: int, height: int, camera) -> np.ndarray:
    """One frame per sample -- states are independent, not a trajectory, so
    there's no motion to render within a frame, just a fixed static view of
    each sampled pose (no camera tracking needed: dyn.sample() always
    places base_xy at the origin).
    """
    mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=height, width=width)

    if camera is not None:
        cam = camera
    else:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.0, 0.0, INIT_HEIGHT]
        cam.distance = 1.1
        cam.azimuth = 120.0
        cam.elevation = -20.0

    frames = []
    try:
        for k in range(states.shape[0]):
            qpos, qvel = _state_to_qpos_qvel(states[k], mj_model.nv)
            mj_data.qpos[:] = qpos
            mj_data.qvel[:] = qvel
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(mj_data, camera=cam)
            frames.append(renderer.render().copy())
    finally:
        renderer.close()
    return np.stack(frames, axis=0)


def save_animation(frames, l_vals, base_z, upright, boundary_band, fps, out_path):
    n = frames.shape[0]
    in_target = l_vals < 0.0        # l is negative-is-inside (see Quadruped.l)
    near_boundary = np.abs(l_vals) < boundary_band

    fig = plt.figure(figsize=(10, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], left=0.03, right=0.97,
                           top=0.88, bottom=0.12, wspace=0.28)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[0, 1])

    ax_img.axis("off")
    im = ax_img.imshow(frames[0])

    sc = ax_scatter.scatter(base_z, upright, c=l_vals, cmap="RdYlGn_r", vmin=-1.0, vmax=1.0,
                             s=40, edgecolors="none", alpha=0.85)
    cbar = fig.colorbar(sc, ax=ax_scatter, fraction=0.046, pad=0.04)
    cbar.set_label("l(x)  (< 0 = inside target)", fontsize=8)
    ax_scatter.axvline(INIT_HEIGHT - STABLE_HEIGHT_TOL, color="gray", lw=0.8, ls=":")
    ax_scatter.axvline(INIT_HEIGHT + STABLE_HEIGHT_TOL, color="gray", lw=0.8, ls=":")
    ax_scatter.axhline(STABLE_DOT_MIN, color="gray", lw=0.8, ls=":")
    ax_scatter.set_xlabel("base_z (m)", fontsize=8)
    ax_scatter.set_ylabel("upright", fontsize=8)
    ax_scatter.tick_params(labelsize=7)
    highlight = ax_scatter.scatter([base_z[0]], [upright[0]], s=220, facecolors="none",
                                    edgecolors="black", linewidths=2.0, zorder=5)

    title = fig.suptitle("", fontsize=10, fontweight="bold")

    def update(i):
        im.set_data(frames[i])
        highlight.set_offsets([[base_z[i], upright[i]]])
        status = "IN TARGET" if in_target[i] else ("near boundary" if near_boundary[i] else "outside")
        title.set_text(f"sample draw {i+1}/{n}  ·  l(x)={l_vals[i]:+.3f}  ·  {status}")
        return im, highlight, title

    anim = FuncAnimation(fig, update, frames=n, blit=False)
    try:
        anim.save(out_path, writer=FFMpegWriter(fps=fps, bitrate=4000))
        print(f"Saved animation -> {out_path}")
    except Exception as exc:
        gif_path = os.path.splitext(out_path)[0] + ".gif"
        print(f"MP4 failed ({exc}) — falling back to GIF: {gif_path}")
        anim.save(gif_path, writer=PillowWriter(fps=fps))
        print(f"Saved animation -> {gif_path}")
    plt.close(fig)


def main() -> None:
    args = build_argparser().parse_args()

    dyn = Quadruped(sim_dt=0.005)
    key = jax.random.PRNGKey(args.seed)
    states = dyn.sample(key, args.num_samples)
    states_np = np.asarray(states, dtype=np.float64)

    l_vals = np.asarray(dyn.l(states))
    base_z = states_np[:, 28]
    upright = 1.0 - 2.0 * (states_np[:, 24] ** 2 + states_np[:, 25] ** 2)
    boundary_band = 0.1  # matches BoundaryAwareSampler's default boundary_band

    n_in = int(np.sum(l_vals < 0.0))
    n_near = int(np.sum(np.abs(l_vals) < boundary_band))
    print(f"{args.num_samples} samples: {n_in} inside target (l<0, {100*n_in/args.num_samples:.1f}%), "
          f"{n_near} within |l|<{boundary_band} band ({100*n_near/args.num_samples:.1f}%)")

    mjcf_path = _find_render_mjcf(args.mjcf)
    print(f"Rendering {args.num_samples} frames from {mjcf_path} ...")
    frames = render_frames(states_np, mjcf_path, args.width, args.height, args.camera)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_animation(frames, l_vals, base_z, upright, boundary_band, args.fps, args.out)


if __name__ == "__main__":
    main()
