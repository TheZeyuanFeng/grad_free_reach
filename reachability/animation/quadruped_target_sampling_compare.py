"""reachability/animation/quadruped_target_sampling_compare.py

Side-by-side comparison of two candidate fixes for the foot-penetration
problem in Quadruped's original (pre-fix) near-target sampling (see
quadruped_target_sampling.py's earlier run):

  A. analytic  -- physics-free. Joint/tilt noise as before, but base_z is
                  DERIVED via closed-form forward kinematics so the lowest
                  foot sits exactly on the ground (or above it, for a
                  deliberate "airborne" fraction of samples) -- never below.
  B. physics   -- settle exactly at the analytic standing pose, apply a
                  velocity "kick" (base_lin_vel/base_ang_vel perturbation),
                  then roll forward a few real MJX physics steps and take
                  whatever state results. Slower (real contact dynamics),
                  but the height/orientation excursion is whatever physics
                  actually produces under that disturbance, not constructed.

Historical: this comparison is what option A was chosen from and adopted as
Quadruped.sample (see quadruped.py) -- kept here as a record of the tradeoff
and the standalone speed/penetration measurements, not as an open decision.
Also prints the lowest-foot-height for every sample of both methods, to
directly confirm neither penetrates the ground (unlike the original bug).

Usage:
    python -m reachability.animation.quadruped_target_sampling_compare
    python -m reachability.animation.quadruped_target_sampling_compare --num_samples 20 --rollout_steps 3
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
    Quadruped, INIT_JOINT_ANGLES, INIT_HEIGHT, NUM_JOINTS, STABLE_HEIGHT_TOL,
    STABLE_DOT_MIN, STABLE_LIN_VEL_TOL, STABLE_ANG_VEL_TOL, CPG_THIGH_LEN, CPG_CALF_LEN,
    _find_mjcf,
)

# Full 3D FK constants (hip mount offsets + abduction rotation), order [FR, FL, RR, RL].
# Same formulas verified empirically against the real MJCF earlier this session.
_HIP_OFFSET = np.array([
    [ 0.183, -0.047, 0.0], [ 0.183,  0.047, 0.0],
    [-0.183, -0.047, 0.0], [-0.183,  0.047, 0.0],
], dtype=np.float32)
_ABDUCTION_Y = np.array([-0.08505, 0.08505, -0.08505, 0.08505], dtype=np.float32)
_FOOT_RADIUS = 0.02


def _quat_rotate(q_xyzw, v):
    qxyz = q_xyzw[..., :3]
    qw = q_xyzw[..., 3:4]
    t = 2.0 * jnp.cross(qxyz, v)
    return v + qw * t + jnp.cross(qxyz, t)


def _feet_rel_trunk(q):
    q4 = q.reshape(q.shape[:-1] + (4, 3))
    th0, th1, th2 = q4[..., 0], q4[..., 1], q4[..., 2]
    foot_x = -CPG_THIGH_LEN * jnp.sin(th1) - CPG_CALF_LEN * jnp.sin(th1 + th2)
    z_sagittal = -CPG_THIGH_LEN * jnp.cos(th1) - CPG_CALF_LEN * jnp.cos(th1 + th2)
    abd_y = jnp.asarray(_ABDUCTION_Y, q.dtype)
    foot_y = abd_y * jnp.cos(th0) - z_sagittal * jnp.sin(th0)
    foot_z = abd_y * jnp.sin(th0) + z_sagittal * jnp.cos(th0)
    foot_rel_hip = jnp.stack([foot_x, foot_y, foot_z], axis=-1)
    return foot_rel_hip + jnp.asarray(_HIP_OFFSET, q.dtype)


def lowest_foot_world_z(q, quat_xyzw, base_z):
    """World-frame height of the lowest foot's center. Subtract _FOOT_RADIUS
    to get clearance-to-ground (negative = penetrating).
    """
    feet_rel_trunk = _feet_rel_trunk(q)                                    # (n, 4, 3)
    feet_world_rel_origin = _quat_rotate(quat_xyzw[:, None, :], feet_rel_trunk)
    return base_z + jnp.min(feet_world_rel_origin[..., 2], axis=-1)        # (n,)


# ---------------------------------------------------------------------------
# Option A: analytic (physics-free)
# ---------------------------------------------------------------------------

def sample_option_a(dyn, key, n, joint_noise_std=0.15, tilt_std=0.08, lift_std=0.5 * STABLE_HEIGHT_TOL,
                     kick_mult=0.7):
    k_q, k_quat, k_lift, k_lin, k_ang = jax.random.split(key, 5)
    q = (jnp.asarray(INIT_JOINT_ANGLES, dyn.dtype)[None, :]
         + jax.random.normal(k_q, (n, NUM_JOINTS), dyn.dtype) * joint_noise_std)
    q = jnp.clip(q, dyn.state_low[0:12], dyn.state_high[0:12])
    dq = jnp.zeros((n, NUM_JOINTS), dyn.dtype)

    tilt_xy = jax.random.normal(k_quat, (n, 2), dyn.dtype) * tilt_std
    quat_xyzw = jnp.concatenate(
        [tilt_xy, jnp.zeros((n, 1), dyn.dtype), jnp.ones((n, 1), dyn.dtype)], axis=-1)
    quat_xyzw = quat_xyzw / jnp.linalg.norm(quat_xyzw, axis=-1, keepdims=True)

    # Ground-contact height for THIS (noisy) posture -- guaranteed non-penetrating.
    feet_rel_trunk = _feet_rel_trunk(q)
    feet_world_rel_origin = _quat_rotate(quat_xyzw[:, None, :], feet_rel_trunk)
    lowest_z = jnp.min(feet_world_rel_origin[..., 2], axis=-1)
    grounded_base_z = _FOOT_RADIUS - lowest_z

    # Most samples should sit exactly grounded (lift=0); only a fraction get
    # lifted, representing "airborne". A half-normal does this naturally --
    # max(0, Normal) is exactly 0 about half the time, with the raw uniform
    # draw used before being ALWAYS positive (never landing properly
    # grounded at all, hence every sample looking like it's hovering).
    lift = jnp.maximum(0.0, jax.random.normal(k_lift, (n,), dyn.dtype) * lift_std)
    base_z = (grounded_base_z + lift)[:, None]

    base_lin_vel = jax.random.normal(k_lin, (n, 3), dyn.dtype) * (kick_mult * STABLE_LIN_VEL_TOL)
    base_ang_vel = jax.random.normal(k_ang, (n, 3), dyn.dtype) * (kick_mult * STABLE_ANG_VEL_TOL)
    base_xy = jnp.zeros((n, 2), dyn.dtype)

    x = jnp.concatenate([q, dq, quat_xyzw, base_z, base_lin_vel, base_ang_vel, base_xy], axis=-1)
    return dyn.wrap_state(x)


# ---------------------------------------------------------------------------
# Option B: settle at exact standing pose, kick, roll forward with real physics
# ---------------------------------------------------------------------------

def sample_option_b(dyn, key, n, rollout_steps, dt=0.02, kick_mult=0.7):
    k_lin, k_ang = jax.random.split(key)
    q = jnp.broadcast_to(jnp.asarray(INIT_JOINT_ANGLES, dyn.dtype), (n, NUM_JOINTS))
    dq = jnp.zeros((n, NUM_JOINTS), dyn.dtype)
    quat = jnp.broadcast_to(jnp.array([0., 0., 0., 1.], dyn.dtype), (n, 4))
    bz = jnp.full((n, 1), INIT_HEIGHT, dyn.dtype)
    lin = jax.random.normal(k_lin, (n, 3), dyn.dtype) * (kick_mult * STABLE_LIN_VEL_TOL)
    ang = jax.random.normal(k_ang, (n, 3), dyn.dtype) * (kick_mult * STABLE_ANG_VEL_TOL)
    xy = jnp.zeros((n, 2), dyn.dtype)
    x = dyn.wrap_state(jnp.concatenate([q, dq, quat, bz, lin, ang, xy], axis=-1))

    q_hold = x[..., 0:12]
    u_hold = jnp.clip(2.0 * (q_hold - dyn._q_lo[None, :]) / (dyn._q_hi[None, :] - dyn._q_lo[None, :]) - 1.0, -1.0, 1.0)

    def body(carry, _):
        xc = carry
        x_next = dyn.step(xc, u_hold, None, dt)
        return x_next, None

    x_final, _ = jax.lax.scan(body, x, None, length=rollout_steps)
    return x_final


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare analytic vs. short-physics-rollout candidates for Quadruped.sample.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--num_samples", type=int, default=20)
    p.add_argument("--rollout_steps", type=int, default=3, help="Option B's physics rollout length.")
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--kick_mult", type=float, default=0.7,
                    help="Velocity-kick scale, in units of STABLE_LIN/ANG_VEL_TOL. "
                         "0.7 is the validated production tuning; use something larger "
                         "(e.g. 5.0) purely to exaggerate the disturbance for visualization.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=float, default=1.5)
    p.add_argument("--width", type=int, default=380)
    p.add_argument("--height", type=int, default=380)
    p.add_argument("--mjcf", default=None)
    p.add_argument("--out", default="/tmp/quadruped_target_sampling_compare.mp4")
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


def render_frames(states: np.ndarray, mjcf_path: str, width: int, height: int) -> np.ndarray:
    mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=height, width=width)
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


def save_comparison(frames_a, frames_b, l_a, l_b, bz_a, bz_b, up_a, up_b, band, fps, out_path):
    n = frames_a.shape[0]
    fig = plt.figure(figsize=(11, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.1], left=0.02, right=0.97,
                           top=0.85, bottom=0.10, wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0]); ax_a.axis("off")
    ax_b = fig.add_subplot(gs[0, 1]); ax_b.axis("off")
    ax_s = fig.add_subplot(gs[0, 2])

    im_a = ax_a.imshow(frames_a[0]); ax_a.set_title("A: analytic", fontsize=9)
    im_b = ax_b.imshow(frames_b[0]); ax_b.set_title("B: physics rollout", fontsize=9)

    ax_s.scatter(bz_a, up_a, c="tab:blue", s=30, alpha=0.7, label="A (analytic)")
    ax_s.scatter(bz_b, up_b, c="tab:orange", s=30, alpha=0.7, label="B (physics)")
    ax_s.axvline(INIT_HEIGHT - STABLE_HEIGHT_TOL, color="gray", lw=0.8, ls=":")
    ax_s.axvline(INIT_HEIGHT + STABLE_HEIGHT_TOL, color="gray", lw=0.8, ls=":")
    ax_s.axhline(STABLE_DOT_MIN, color="gray", lw=0.8, ls=":")
    ax_s.set_xlabel("base_z (m)", fontsize=8); ax_s.set_ylabel("upright", fontsize=8)
    ax_s.legend(fontsize=7, loc="lower left")
    ax_s.tick_params(labelsize=7)
    hi_a = ax_s.scatter([bz_a[0]], [up_a[0]], s=200, facecolors="none", edgecolors="tab:blue", linewidths=2.0)
    hi_b = ax_s.scatter([bz_b[0]], [up_b[0]], s=200, facecolors="none", edgecolors="tab:orange", linewidths=2.0)

    title = fig.suptitle("", fontsize=10, fontweight="bold")

    def update(i):
        im_a.set_data(frames_a[i]); im_b.set_data(frames_b[i])
        hi_a.set_offsets([[bz_a[i], up_a[i]]]); hi_b.set_offsets([[bz_b[i], up_b[i]]])
        sa = "IN" if l_a[i] < 0 else ("near" if abs(l_a[i]) < band else "out")
        sb = "IN" if l_b[i] < 0 else ("near" if abs(l_b[i]) < band else "out")
        title.set_text(f"sample {i+1}/{n}   A: l={l_a[i]:+.3f} ({sa})   |   B: l={l_b[i]:+.3f} ({sb})")
        return im_a, im_b, hi_a, hi_b, title

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
    ka, kb = jax.random.split(key)

    xa = sample_option_a(dyn, ka, args.num_samples, kick_mult=args.kick_mult)
    xb = sample_option_b(dyn, kb, args.num_samples, args.rollout_steps, args.dt, kick_mult=args.kick_mult)
    xa_np, xb_np = np.asarray(xa, dtype=np.float64), np.asarray(xb, dtype=np.float64)

    l_a, l_b = np.asarray(dyn.l(xa)), np.asarray(dyn.l(xb))
    bz_a, bz_b = xa_np[:, 28], xb_np[:, 28]
    up_a = 1.0 - 2.0 * (xa_np[:, 24] ** 2 + xa_np[:, 25] ** 2)
    up_b = 1.0 - 2.0 * (xb_np[:, 24] ** 2 + xb_np[:, 25] ** 2)

    clear_a = np.asarray(lowest_foot_world_z(xa[:, 0:12], xa[:, 24:28], xa[:, 28])) - _FOOT_RADIUS
    clear_b = np.asarray(lowest_foot_world_z(xb[:, 0:12], xb[:, 24:28], xb[:, 28])) - _FOOT_RADIUS
    band = 0.1
    print(f"A (analytic):     min foot clearance={clear_a.min():+.4f}m (negative=penetrating)  "
          f"in-target={100*np.mean(l_a<0):.1f}%  near-band={100*np.mean(np.abs(l_a)<band):.1f}%")
    print(f"B (physics, {args.rollout_steps} steps): min foot clearance={clear_b.min():+.4f}m  "
          f"in-target={100*np.mean(l_b<0):.1f}%  near-band={100*np.mean(np.abs(l_b)<band):.1f}%")

    mjcf_path = _find_render_mjcf(args.mjcf)
    print(f"Rendering {args.num_samples} frames x2 from {mjcf_path} ...")
    frames_a = render_frames(xa_np, mjcf_path, args.width, args.height)
    frames_b = render_frames(xb_np, mjcf_path, args.width, args.height)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_comparison(frames_a, frames_b, l_a, l_b, bz_a, bz_b, up_a, up_b, band, args.fps, args.out)


if __name__ == "__main__":
    main()
