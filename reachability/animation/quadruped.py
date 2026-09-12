"""animate_quadruped.py — Unitree A1 balance / fall-recovery animation.

Rolls the trained policy out from sampled initial states, picks one
trajectory (default: the least-safe one), and renders the actual MuJoCo body
next to its safety-value, l(x) margin, and base-height traces.

Rendering uses assets/unitree_a1/scene.xml (robot + a floor plane + skybox)
purely as a visual reference — the physics model the policy was trained
against, assets/unitree_a1/a1.xml, has NO ground plane. Quadruped is a
free-fall / self-righting problem (does the robot stay upright and above
IS_FALLEN_HEIGHT within the horizon), not a floor-supported stance, and the
state has no (x, y) — the trunk never translates horizontally, so a static
camera is sufficient.

Usage:
    python -m reachability.animation.quadruped --run_dir runs/quadruped_test
    python -m reachability.animation.quadruped --run_dir runs/quadruped_test \
        --select safest --out /tmp/best.mp4
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

from configs import get_cfg_defaults, PROJECT_NAME
from utils import (
    attach_file_handler,
    find_latest_step,
    load_policy_net,
    load_value_net,
    make_dynamics,
    setup_logging,
)
from eval import rollout_trajectories, _unsafe_mask
from reachability.dynamics.quadruped import IS_FALLEN_HEIGHT, INIT_HEIGHT, _find_mjcf

setup_logging()
log = logging.getLogger(f"{PROJECT_NAME}.{__name__}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Render a trained Quadruped policy rollout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run_dir", required=True,
                   help="Run directory containing config.yaml and checkpoints/.")
    p.add_argument("--ckpt_step", type=int, default=None,
                   help="Checkpoint step to load. Defaults to the highest step found.")
    p.add_argument("--num_states", type=int, default=64,
                   help="Number of initial states rolled out; one is picked to render.")
    p.add_argument("--t_final", type=float, default=None,
                   help="Rollout horizon. Defaults to cfg.GAME.TIME.T.")
    p.add_argument("--dt", type=float, default=None,
                   help="Macro time step. Defaults to cfg.GAME.TIME.DT.")
    p.add_argument("--policy_steps_per_dt", type=int, default=10,
                   help="Policy sub-steps per dt during rollout.")
    p.add_argument("--select", default="least_safe",
                   choices=["least_safe", "safest", "random", "random_unsafe", "index"],
                   help="Which rolled-out trajectory to render.")
    p.add_argument("--select_index", type=int, default=0,
                   help="Used when --select=index.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--camera", default=None,
                   help="Camera name/id for rendering. Defaults to the scene's free camera.")
    p.add_argument("--mjcf", default=None,
                   help="Override MJCF used for rendering (defaults to scene.xml next to a1.xml).")
    p.add_argument("--out", default=None,
                   help="Output .mp4 path. Defaults to <run_dir>/eval/quadruped_<select>_step<N>.mp4")
    return p


# ---------------------------------------------------------------------------
# Trajectory selection
# ---------------------------------------------------------------------------

def choose_index(
    true_score: np.ndarray,
    control_role: str,
    mode: str,
    index: int = 0,
    seed: int = 0,
) -> int:
    """Pick a trajectory index from ``true_score`` (higher = safer iff control_role='max')."""
    N = true_score.shape[0]
    safer_is_higher = (control_role == "max")

    if mode == "safest":
        return int(np.argmax(true_score) if safer_is_higher else np.argmin(true_score))
    if mode == "least_safe":
        return int(np.argmin(true_score) if safer_is_higher else np.argmax(true_score))
    if mode == "random":
        return int(np.random.default_rng(seed).integers(0, N))
    if mode == "random_unsafe":
        unsafe_idx = np.nonzero(np.asarray(_unsafe_mask(true_score, control_role)))[0]
        rng = np.random.default_rng(seed)
        if unsafe_idx.size == 0:
            log.warning("random_unsafe: no unsafe trajectories found; picking randomly.")
            return int(rng.integers(0, N))
        return int(unsafe_idx[rng.integers(0, unsafe_idx.size)])
    if mode == "index":
        if index < 0 or index >= N:
            raise ValueError(f"index {index} out of range for {N} trajectories.")
        return index
    raise ValueError(f"Unknown mode '{mode}'.")


# ---------------------------------------------------------------------------
# State -> MuJoCo qpos/qvel  (mirrors Quadruped._data_from_x)
# ---------------------------------------------------------------------------

def _find_render_mjcf(override: Optional[str] = None) -> str:
    if override is not None:
        return override
    a1_path = _find_mjcf()
    scene_path = os.path.join(os.path.dirname(a1_path), "scene.xml")
    if os.path.isfile(scene_path):
        return scene_path
    log.warning("scene.xml not found next to %s; rendering without a floor/skybox.", a1_path)
    return a1_path


def _state_to_qpos_qvel(x: np.ndarray, nv: int) -> tuple:
    q, dq, quat_xyzw, base_z = x[0:12], x[12:24], x[24:28], x[28]
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    qpos = np.concatenate([[0.0, 0.0, base_z], quat_wxyz, q])
    qvel = np.zeros(nv, dtype=np.float64)
    qvel[6:] = dq
    return qpos, qvel


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_frames(
    traj: np.ndarray,
    mjcf_path: str,
    width: int,
    height: int,
    camera,
) -> np.ndarray:
    """Render one RGB frame per row of ``traj`` (K+1, 29). Returns (K+1, H, W, 3) uint8."""
    mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=height, width=width)

    cam = camera if camera is not None else -1
    frames = []
    try:
        for k in range(traj.shape[0]):
            qpos, qvel = _state_to_qpos_qvel(traj[k], mj_model.nv)
            mj_data.qpos[:] = qpos
            mj_data.qvel[:] = qvel
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(mj_data, camera=cam)
            frames.append(renderer.render().copy())
    finally:
        renderer.close()
    return np.stack(frames, axis=0)


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def save_quadruped_animation(
    frames: np.ndarray,
    sim_t: np.ndarray,
    v_series: np.ndarray,
    l_series: np.ndarray,
    base_z_series: np.ndarray,
    traj_idx: int,
    cost: float,
    step: int,
    fps: int,
    out_path: str,
) -> None:
    K = frames.shape[0] - 1
    fell_at = np.nonzero(l_series > 0.0)[0]
    fell = fell_at.size > 0

    fig = plt.figure(figsize=(10, 5.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1], hspace=0.55, wspace=0.32,
                           left=0.03, right=0.97, top=0.88, bottom=0.10)
    ax_img = fig.add_subplot(gs[:, 0])
    ax_v = fig.add_subplot(gs[0, 1])
    ax_l = fig.add_subplot(gs[1, 1])
    ax_h = fig.add_subplot(gs[2, 1])

    ax_img.axis("off")
    im = ax_img.imshow(frames[0])

    ax_v.plot(sim_t, v_series, color="tab:purple", lw=1.3)
    ax_v.axhline(0.0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax_v.set_ylabel("V(x, t)", fontsize=8)
    ax_v.tick_params(labelsize=7)

    ax_l.plot(sim_t, l_series, color="tab:red", lw=1.3)
    ax_l.axhline(0.0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax_l.set_ylabel("l(x) margin", fontsize=8)
    ax_l.tick_params(labelsize=7)

    ax_h.plot(sim_t, base_z_series, color="tab:blue", lw=1.3)
    ax_h.axhline(IS_FALLEN_HEIGHT, color="red", lw=0.8, ls="--", alpha=0.7, label="fallen")
    ax_h.axhline(INIT_HEIGHT, color="gray", lw=0.8, ls=":", alpha=0.7, label="nominal")
    ax_h.set_ylabel("base_z (m)", fontsize=8)
    ax_h.set_xlabel("time (s)", fontsize=8)
    ax_h.legend(fontsize=6, loc="upper right")
    ax_h.tick_params(labelsize=7)

    for ax in (ax_v, ax_l, ax_h):
        ax.set_xlim(sim_t[0], sim_t[-1])
    vl_v = ax_v.axvline(0.0, color="k", lw=1.2, alpha=0.7)
    vl_l = ax_l.axvline(0.0, color="k", lw=1.2, alpha=0.7)
    vl_h = ax_h.axvline(0.0, color="k", lw=1.2, alpha=0.7)

    status0 = "FELL" if fell else "safe"
    title = fig.suptitle(
        f"Quadruped  ·  step {step}  ·  traj #{traj_idx}  ·  true_score={cost:.3f}"
        f"  ·  t=0.00/{sim_t[-1]:.2f}s  ·  {status0}",
        fontsize=10, fontweight="bold",
    )

    n_frames = max(2, int(round(fps * sim_t[-1])) + 1)
    frame_idx = np.round(np.linspace(0, K, n_frames)).astype(int)

    def update(fi):
        k = frame_idx[fi]
        t = sim_t[k]
        im.set_data(frames[k])
        vl_v.set_xdata([t, t]); vl_l.set_xdata([t, t]); vl_h.set_xdata([t, t])
        status = "FELL" if fell and k >= fell_at[0] else "safe"
        title.set_text(
            f"Quadruped  ·  step {step}  ·  traj #{traj_idx}  ·  true_score={cost:.3f}"
            f"  ·  t={t:.2f}/{sim_t[-1]:.2f}s  ·  {status}"
        )
        return im, vl_v, vl_l, vl_h, title

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)

    try:
        anim.save(out_path, writer=FFMpegWriter(fps=fps, bitrate=4000))
        log.info("Saved animation -> %s", out_path)
    except Exception as exc:
        gif_path = os.path.splitext(out_path)[0] + ".gif"
        log.warning("MP4 failed (%s) — falling back to GIF: %s", exc, gif_path)
        anim.save(gif_path, writer=PillowWriter(fps=fps))
        log.info("Saved animation -> %s", gif_path)

    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_argparser().parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(os.path.join(args.run_dir, "config.yaml"))
    cfg.freeze()
    attach_file_handler(os.path.join(args.run_dir, "animate.log"))

    dyn = make_dynamics(cfg)

    ckpt_dir = os.path.join(args.run_dir, cfg.IO.CKPT_DIRNAME)
    step = args.ckpt_step if args.ckpt_step is not None else find_latest_step(ckpt_dir)

    rngs = nnx.Rngs(args.seed)
    value_net = load_value_net(cfg, dyn, rngs, ckpt_dir, step)
    policy_net = load_policy_net(cfg, dyn, rngs, ckpt_dir, step)

    t_final = args.t_final or float(cfg.GAME.TIME.T)
    dt = args.dt or float(cfg.GAME.TIME.DT)

    key = jax.random.PRNGKey(args.seed)
    x0 = dyn.sample(key, args.num_states)

    log.info("Rolling out %d trajectories (T=%.2f, dt=%.3f) ...", args.num_states, t_final, dt)
    true_score, Xtraj = rollout_trajectories(
        dyn, value_net, policy_net, t_final, dt, cfg, x0, args.policy_steps_per_dt,
    )
    true_score_np = np.asarray(true_score)

    idx = choose_index(true_score_np, cfg.GAME.CONTROL_ROLE, args.select, args.select_index, args.seed)
    cost = float(true_score_np[idx])
    traj = np.asarray(Xtraj[idx], dtype=np.float64)  # (K+1, 29)
    K = traj.shape[0] - 1
    log.info("Selected trajectory #%d (%s)  true_score=%.4f", idx, args.select, cost)

    t_arr = jnp.linspace(t_final, 0.0, K + 1, dtype=dyn.dtype)
    v_series = np.asarray(value_net(jnp.asarray(traj, dtype=dyn.dtype), t_arr)["V"])
    l_series = np.asarray(dyn.l(jnp.asarray(traj, dtype=dyn.dtype)))
    base_z_series = traj[:, 28]
    sim_t = np.linspace(0.0, t_final, K + 1)

    mjcf_path = _find_render_mjcf(args.mjcf)
    log.info("Rendering %d frames from %s ...", K + 1, mjcf_path)
    frames = render_frames(traj, mjcf_path, args.width, args.height, args.camera)

    out_path = args.out or os.path.join(
        args.run_dir, "eval", f"quadruped_{args.select}_step{step}.mp4"
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    save_quadruped_animation(
        frames, sim_t, v_series, l_series, base_z_series,
        idx, cost, step, args.fps, out_path,
    )


if __name__ == "__main__":
    main()
