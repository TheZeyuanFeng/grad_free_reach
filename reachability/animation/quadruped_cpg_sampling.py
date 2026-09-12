"""reachability/animation/quadruped_cpg_sampling.py

Renders a single Quadruped.nominal_policy rollout (MuJoCo video + state
traces) and marks the timesteps at which BoundaryAwareSampler's nominal
bucket would actually snapshot a sample -- for qualitatively checking the
CPG gait and the burn-in / rollout-length sampling window together, without
needing a trained checkpoint or run_dir.

Usage:
    python -m reachability.animation.quadruped_cpg_sampling
    python -m reachability.animation.quadruped_cpg_sampling \
        --num_steps 150 --burn_in 10 --nominal_steps 15 --out /tmp/cpg.mp4
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

from reachability.dynamics.quadruped import (
    Quadruped, INIT_HEIGHT, STABLE_HEIGHT_TOL, STABLE_DOT_MIN, CPG_GAIT_NAMES, _find_mjcf,
)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Render a Quadruped nominal-policy rollout with sample-time markers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--num_steps", type=int, default=120, help="Total macro-steps to roll out and render.")
    p.add_argument("--dt", type=float, default=0.02, help="Macro time step (same units nominal_policy's t uses).")
    p.add_argument("--burn_in", type=int, default=10, help="nominal_burn_in_steps, always applied before sampling.")
    p.add_argument("--nominal_steps", type=int, default=15, help="nominal_steps -- fixed rollout length after burn-in.")
    p.add_argument("--samples_per_rollout", type=int, default=4,
                   help="nominal_samples_per_rollout -- distinct snapshots this one rollout yields (<= nominal_steps).")
    p.add_argument("--gait", default=None, choices=["TROT", "PACE", "BOUND", "WALK"],
                   help="Fix the gait for this render. Default: draw one via nominal_params, like real sampling.")
    p.add_argument("--vary_height", action="store_true",
                   help="Also draw a random stance height via nominal_params (default: fixed default height).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--camera", default=None, help="Fixed camera name/id; overrides chase-cam tracking if set.")
    p.add_argument("--no_track", action="store_true", help="Disable chase-cam tracking (use the scene's free camera).")
    p.add_argument("--cam_distance", type=float, default=1.3)
    p.add_argument("--cam_azimuth", type=float, default=120.0)
    p.add_argument("--cam_elevation", type=float, default=-20.0)
    p.add_argument("--mjcf", default=None)
    p.add_argument("--out", default="/tmp/quadruped_cpg_sampling.mp4")
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


def render_frames(
    traj: np.ndarray, mjcf_path: str, width: int, height: int, camera,
    track: bool = True, cam_distance: float = 1.3, cam_azimuth: float = 120.0,
    cam_elevation: float = -20.0,
) -> np.ndarray:
    """Render one frame per row of ``traj``. With base_xy now a real, moving
    part of the state (see quadruped.py's state-layout docstring), a
    static camera loses the robot as soon as it walks any distance -- track
    re-centers an MjvCamera's lookat on the trunk's (x, y, z) every frame
    (fixed distance/azimuth/elevation, i.e. a chase cam) instead.
    """
    mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=height, width=width)

    if camera is not None:
        cam = camera            # explicit override: fixed named/indexed camera, no tracking
    elif track:
        cam = mujoco.MjvCamera()
        cam.distance = cam_distance
        cam.azimuth = cam_azimuth
        cam.elevation = cam_elevation
    else:
        cam = -1                 # scene's free camera, static

    frames = []
    try:
        for k in range(traj.shape[0]):
            qpos, qvel = _state_to_qpos_qvel(traj[k], mj_model.nv)
            mj_data.qpos[:] = qpos
            mj_data.qvel[:] = qvel
            mujoco.mj_forward(mj_model, mj_data)
            if camera is None and track:
                cam.lookat[:] = qpos[0:3]
            renderer.update_scene(mj_data, camera=cam)
            frames.append(renderer.render().copy())
    finally:
        renderer.close()
    return np.stack(frames, axis=0)


def rollout_cpg(dyn, seed, num_steps, dt, gait, vary_height):
    """Roll out one trajectory. gait=None draws one via dyn.nominal_params,
    same as real nominal-policy sampling would -- rerun with a different
    --seed to see a different gait. Returns (traj, gait_name, stance_z).
    """
    key = jax.random.PRNGKey(seed)
    k_params, k_seed = jax.random.split(key)
    stance_z = None
    if gait is None:
        params = dyn.nominal_params(k_params, 1)
        gait_idx = int(params["gait"][0])
        gait_name = CPG_GAIT_NAMES[gait_idx]
        if vary_height:
            stance_z = float(params["stance_z"][0, 0])
    else:
        gait_name = gait

    x = dyn.nominal_seed(k_seed, 1)
    t_elapsed = jnp.zeros((1,), dyn.dtype)
    xs = [x]
    for _ in range(num_steps):
        u = dyn.nominal_policy(x, t_elapsed, gait=gait_name, stance_z=stance_z)
        x = dyn.step(x, u, None, dt)
        t_elapsed = t_elapsed + dt
        xs.append(x)
    return jnp.concatenate(xs, axis=0), gait_name, stance_z  # (num_steps+1, 37)


def simulate_sample_indices(seed, burn_in, nominal_steps, samples_per_rollout):
    """Mirrors BoundaryAwareSampler._nominal exactly: samples_per_rollout
    DISTINCT snapshot indices (no replacement) picked from the one fixed
    window [burn_in, burn_in + nominal_steps) -- every trajectory runs the
    same fixed length, so diversity comes entirely from which points within
    that shared window get picked, not from varying the rollout length
    itself (an earlier design did that and it biased sampling toward early
    steps -- see sampling.py's _nominal docstring).
    """
    rng = np.random.default_rng(seed + 1)
    idx = burn_in + rng.choice(nominal_steps, size=samples_per_rollout, replace=False)
    return np.sort(idx), nominal_steps


def save_animation(frames, sim_t, base_z, upright, l_vals, sample_idx, burn_in, fps, out_path, gait_name):
    K = frames.shape[0] - 1
    sample_t = sim_t[sample_idx]
    burn_in_t = sim_t[min(burn_in, K)]

    fig = plt.figure(figsize=(10, 5.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1], hspace=0.6, wspace=0.32,
                           left=0.03, right=0.97, top=0.88, bottom=0.10)
    ax_img = fig.add_subplot(gs[:, 0])
    ax_h = fig.add_subplot(gs[0, 1])
    ax_u = fig.add_subplot(gs[1, 1])
    ax_l = fig.add_subplot(gs[2, 1])

    ax_img.axis("off")
    im = ax_img.imshow(frames[0])
    sample_marker = ax_img.text(0.02, 0.96, "", transform=ax_img.transAxes, color="orange",
                                 fontsize=13, fontweight="bold", va="top")

    ax_h.plot(sim_t, base_z, color="tab:blue", lw=1.3)
    ax_h.axhline(INIT_HEIGHT, color="gray", lw=0.8, ls=":", alpha=0.7)
    ax_h.axhspan(INIT_HEIGHT - STABLE_HEIGHT_TOL, INIT_HEIGHT + STABLE_HEIGHT_TOL, color="gray", alpha=0.12)
    ax_h.set_ylabel("base_z (m)", fontsize=8)

    ax_u.plot(sim_t, upright, color="tab:green", lw=1.3)
    ax_u.axhline(STABLE_DOT_MIN, color="red", lw=0.8, ls="--", alpha=0.7)
    ax_u.set_ylabel("upright", fontsize=8)

    ax_l.plot(sim_t, l_vals, color="tab:red", lw=1.3)
    ax_l.axhline(0.0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax_l.set_ylabel("l(x)", fontsize=8)
    ax_l.set_xlabel("time (s)", fontsize=8)

    for ax in (ax_h, ax_u, ax_l):
        ax.axvline(burn_in_t, color="purple", lw=1.0, ls=":", alpha=0.8)
        for st in sample_t:
            ax.axvline(st, color="orange", lw=0.8, alpha=0.35)
        ax.set_xlim(sim_t[0], sim_t[-1])
        ax.tick_params(labelsize=7)

    ax_l.plot([], [], color="purple", ls=":", label="burn-in ends")
    ax_l.plot([], [], color="orange", alpha=0.6, label="simulated sample draw")
    ax_l.legend(fontsize=6, loc="upper right")

    vl_h = ax_h.axvline(0.0, color="k", lw=1.2, alpha=0.7)
    vl_u = ax_u.axvline(0.0, color="k", lw=1.2, alpha=0.7)
    vl_l = ax_l.axvline(0.0, color="k", lw=1.2, alpha=0.7)

    title = fig.suptitle(f"Quadruped nominal-policy rollout ({gait_name})  ·  t=0.00s",
                          fontsize=10, fontweight="bold")

    n_frames = max(2, int(round(fps * sim_t[-1])) + 1)
    frame_idx = np.round(np.linspace(0, K, n_frames)).astype(int)
    sample_set = set(sample_idx.tolist())

    def update(fi):
        k = frame_idx[fi]
        t = sim_t[k]
        im.set_data(frames[k])
        vl_h.set_xdata([t, t]); vl_u.set_xdata([t, t]); vl_l.set_xdata([t, t])
        is_sample = k in sample_set
        sample_marker.set_text("SAMPLE POINT" if is_sample else "")
        phase = "burn-in" if k < burn_in else "sampling window"
        title.set_text(f"Quadruped nominal-policy rollout ({gait_name})  ·  "
                        f"t={t:.2f}/{sim_t[-1]:.2f}s  ·  {phase}")
        return im, vl_h, vl_u, vl_l, title, sample_marker

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
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
    traj, gait_name, stance_z = rollout_cpg(
        dyn, args.seed, args.num_steps, args.dt, args.gait, args.vary_height)
    height_note = f", stance_z={stance_z:.3f}" if stance_z is not None else ""
    print(f"Gait: {gait_name}{height_note}")
    traj_np = np.asarray(traj, dtype=np.float64)

    base_z = traj_np[:, 28]
    upright = 1.0 - 2.0 * (traj_np[:, 24] ** 2 + traj_np[:, 25] ** 2)
    l_vals = np.asarray(dyn.l(traj))
    sim_t = np.arange(traj_np.shape[0]) * args.dt

    sample_idx, n_steps = simulate_sample_indices(
        args.seed, args.burn_in, args.nominal_steps, args.samples_per_rollout)
    print(f"nominal_steps={n_steps}, snapshots at step(s) {sample_idx.tolist()}")

    mjcf_path = _find_render_mjcf(args.mjcf)
    print(f"Rendering {traj_np.shape[0]} frames from {mjcf_path} ...")
    frames = render_frames(
        traj_np, mjcf_path, args.width, args.height, args.camera,
        track=not args.no_track, cam_distance=args.cam_distance,
        cam_azimuth=args.cam_azimuth, cam_elevation=args.cam_elevation)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_animation(frames, sim_t, base_z, upright, l_vals, sample_idx, args.burn_in, args.fps, args.out, gait_name)


if __name__ == "__main__":
    main()
