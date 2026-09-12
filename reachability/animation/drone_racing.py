"""sim_quadrotor_multigates.py — Multi-gate closed-loop simulation.

Extends sim_quadrotor_gate.py to fly through N_GATES sequential gates spaced
at fixed intervals along the x-axis.

Gate-transition trigger (either condition):
  (a) x-coordinate in the current gate frame crosses 0  (passed through)
  (b) BRAT cost of the trajectory so far is ≤ 0

On transition the simulator:
  • shifts the state's x-position by -gate_spacing  (re-centres to next gate frame)
  • resets the time horizon back to T_train
  • resets the value-net search window

Everything else (velocity, attitude, angular rates) is preserved.

Usage:
    python -m reachability.animation.drone_racing --run_dir /path/to/run [options]
"""

from configs.constants import PROJECT_NAME
from reachability.training.functional import rollout_with_intermediate_checks
from reachability.data.sampling import sample_val_states_uniform
from utils import (
    attach_file_handler,
    decode_actions,
    find_latest_step,
    load_policy_net,
    load_value_net,
    make_dynamics,
    setup_logging,
)
from configs import get_cfg_defaults
from tqdm import tqdm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.pyplot as plt
import argparse
import logging
import os
import dataclasses
from typing import List, Optional

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import matplotlib
matplotlib.use("Agg")


setup_logging()
log = logging.getLogger(f"{PROJECT_NAME}.{__name__}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Simulate QuadrotorGateTraversal through multiple gates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run_dir",        required=True)
    p.add_argument("--ckpt_step",      type=int,   default=None)
    p.add_argument("--N",              type=int,   default=20,
                   help="Number of trajectories to simulate.")
    p.add_argument("--n_gates",        type=int,   default=3,
                   help="Number of gates to fly through.")
    p.add_argument("--gate_spacing",   type=float, default=5.0,
                   help="Distance (m) between consecutive gate planes along x.")
    p.add_argument("--seed",           type=int,   default=0)
    p.add_argument("--n_per_category", type=int,   default=5)
    p.add_argument("--fps",            type=int,   default=20)
    p.add_argument("--sim_time",       type=float, default=4.0,
                   help="Max time budget per gate (s). Resets on each transition.")
    p.add_argument("--window_half",    type=float, default=0.2)
    p.add_argument("--out_dir",        type=str,   default=None)
    return p


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def _load_cfg(run_dir: str):
    cfg = get_cfg_defaults()
    cfg_path = os.path.join(run_dir, "config.yaml")
    if os.path.isfile(cfg_path):
        cfg.merge_from_file(cfg_path)
    cfg.freeze()
    return cfg


# ---------------------------------------------------------------------------
# Gate geometry
# ---------------------------------------------------------------------------

def _gate_panels(hw: float, hh: float, t: float, x0: float = 0.0) -> list:
    ow, oh = hw + t, hh + t
    return [
        [[x0, -ow,  hh], [x0,  ow,  hh], [x0,  ow,  oh], [x0, -ow,  oh]],
        [[x0, -ow, -oh], [x0,  ow, -oh], [x0,  ow, -hh], [x0, -ow, -hh]],
        [[x0, -ow, -hh], [x0, -hw, -hh], [x0, -hw,  hh], [x0, -ow,  hh]],
        [[x0,  hw, -hh], [x0,  ow, -hh], [x0,  ow,  hh], [x0,  hw,  hh]],
    ]


def add_gate_to_axes(ax, dyn, x_offset: float = 0.0,
                     alpha: float = 0.55, label: Optional[str] = None) -> None:
    hw = dyn.gate_half_w
    hh = dyn.gate_half_h
    t  = dyn.gate_thickness
    coll = Poly3DCollection(
        _gate_panels(hw, hh, t, x0=x_offset),
        alpha=alpha, facecolor="slategray", edgecolor="k", linewidth=0.5,
    )
    ax.add_collection3d(coll)
    o = np.array([[x_offset, -hw, -hh], [x_offset, hw, -hh],
                  [x_offset, hw,  hh],  [x_offset, -hw, hh],
                  [x_offset, -hw, -hh]])
    kw = {"lw": 1.5, "alpha": 0.9}
    if label:
        kw["label"] = label
    ax.plot(o[:, 0], o[:, 1], o[:, 2], "g--", **kw)


# ---------------------------------------------------------------------------
# Fused, cached per-step voting logic (value-net search + policy-net vote):
# fixed-shape masking over the whole t_all array so the whole step is one
# nnx.jit-compiled call, cached per (value_net, policy_net, dyn).
# ---------------------------------------------------------------------------

_gate_step_jit_cache: dict = {}


def _get_gate_step_jit(value_net, policy_net, dyn):
    key = (id(value_net), id(policy_net), id(dyn))
    fn = _gate_step_jit_cache.get(key)
    if fn is None:
        @nnx.jit
        def gate_step(v_net, p_net, x_cur, t_all, t_lo_search, t_hi_search,
                      window_half, T_train):
            K = t_all.shape[0]

            x_batch = jnp.broadcast_to(x_cur, (K, x_cur.shape[-1]))
            V_all   = v_net(x_batch, t_all)["V"]

            in_search = (t_all >= t_lo_search) & (t_all <= t_hi_search)
            cond      = in_search & (V_all <= 0.0)
            any_hit   = jnp.any(cond)
            first_idx = jnp.argmax(cond)   # index of first True (0 if none)
            fallback  = jnp.max(jnp.where(in_search, t_all, -jnp.inf))
            t_mid     = jnp.where(any_hit, t_all[first_idx], fallback)

            t_hi_vote = jnp.minimum(t_mid + jnp.minimum(t_mid, window_half), T_train)
            in_vote   = (t_all >= t_mid) & (t_all <= t_hi_vote)

            nn_x   = dyn.nn_inputs(x_cur)
            nn_x_b = jnp.broadcast_to(nn_x, (K, nn_x.shape[-1]))
            out    = p_net(nn_x_b, t_all)
            u_batch, _ = decode_actions(out, dyn)

            if u_batch is None:
                u = jnp.zeros((1, dyn.control_dim), dtype=dyn.dtype)
            else:
                mask_f    = in_vote.astype(dyn.dtype)
                count     = jnp.maximum(jnp.sum(mask_f), 1.0)
                mean_sign = jnp.sum(u_batch * mask_f[:, None], axis=0, keepdims=True) / count
                mean_sign = jnp.sign(mean_sign)
                mean_sign = jnp.where(mean_sign == 0, 1.0, mean_sign)
                u = mean_sign * dyn.u_max.reshape(1, -1)

            new_lo = jnp.maximum(t_all[0], t_mid - window_half)
            new_hi = jnp.minimum(T_train,  t_hi_vote + window_half)
            return u, t_mid, t_hi_vote, new_lo, new_hi

        fn = gate_step
        _gate_step_jit_cache[key] = fn
    return fn


# ---------------------------------------------------------------------------
# Per-gate result container
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class GateResult:
    gate_idx:    int
    traj:        np.ndarray      # (K+1, state_dim)  in gate-local frame
    window_log:  list            # [(t_mid, t_hi), ...] length K
    brat_cost:   float
    passed:      bool            # True if transition was triggered
    trigger:     str             # "x_cross" | "brat" | "timeout"
    steps_taken: int             # number of dt-steps actually simulated


# ---------------------------------------------------------------------------
# BRAT cost helper
# ---------------------------------------------------------------------------

def brat_cost_fn(traj: jax.Array, dyn) -> float:
    return float(dyn.cost_fn(traj[None])[0])


# ---------------------------------------------------------------------------
# Single-gate segment rollout  (returns early on transition trigger)
# ---------------------------------------------------------------------------

def simulate_one_gate(
    x0_1:        jax.Array,   # (1, state_dim) in gate-local frame
    dyn,
    value_net,
    policy_net,
    cfg,
    T_sim:       float,          # max time budget for this gate
    T_train:     float,
    window_half: float,
) -> GateResult:
    """Roll out until gate passed, BRAT cost ≤ 0, or time budget exhausted."""
    dt           = cfg.GAME.TIME.DT
    K            = int(round(T_sim / dt))
    m            = cfg.GAME.TIME.NUM_SUBSTEPS
    problem_type = cfg.GAME.PROBLEM_TYPE

    t_all = jnp.arange(1, K + 1, dtype=dyn.dtype) * dt

    xs         = [x0_1]
    window_log = []
    x_cur      = x0_1

    t_lo_search = jnp.asarray(dt,      dtype=dyn.dtype)
    t_hi_search = jnp.asarray(T_train, dtype=dyn.dtype)

    trigger     = "timeout"
    steps_taken = K

    gate_step = _get_gate_step_jit(value_net, policy_net, dyn)

    for step in range(K):
        u, t_mid, t_hi_vote, t_lo_search, t_hi_search = gate_step(
            value_net, policy_net, x_cur, t_all,
            t_lo_search, t_hi_search, window_half, T_train)
        window_log.append((float(t_mid), float(t_hi_vote)))

        # Step dynamics
        x_cur, _ = rollout_with_intermediate_checks(
            x=x_cur, u=u, d=None,
            dyn=dyn, dt=dt, m=m,
            problem_type=problem_type,
        )
        xs.append(x_cur)

        # ── Transition checks ──────────────────────────────────────────────
        # (a) x crossed 0 (gate plane in gate-local frame)
        x_pos = float(x_cur[0, 0])
        if x_pos >= 0.0:
            trigger     = "x_cross"
            steps_taken = step + 1
            break

        # (b) BRAT cost ≤ 0 on partial trajectory so far
        traj_so_far = jnp.concatenate(xs, axis=0)
        if brat_cost_fn(traj_so_far, dyn) <= 0.0:
            trigger     = "brat"
            steps_taken = step + 1
            break

    traj_full = jnp.concatenate(xs, axis=0)
    cost      = brat_cost_fn(traj_full, dyn)
    passed    = trigger in ("x_cross", "brat")

    return GateResult(
        gate_idx    = -1,          # filled in by caller
        traj        = np.asarray(traj_full),
        window_log  = window_log,
        brat_cost   = cost,
        passed      = passed,
        trigger     = trigger,
        steps_taken = steps_taken,
    )


# ---------------------------------------------------------------------------
# Multi-gate trajectory rollout
# ---------------------------------------------------------------------------

def simulate_multi_gate(
    x0_1:        jax.Array,
    dyn,
    value_net,
    policy_net,
    cfg,
    n_gates:     int,
    gate_spacing: float,
    T_sim:       float,
    T_train:     float,
    window_half: float,
) -> List[GateResult]:
    """Fly through up to n_gates sequentially.

    After each gate transition the state x-position is shifted by -gate_spacing
    to re-centre the drone in front of the next gate.  All other state components
    (velocity, attitude, angular rates) are preserved exactly.
    """
    results   = []
    x_cur     = x0_1

    for g in range(n_gates):
        result         = simulate_one_gate(
            x0_1        = x_cur,
            dyn         = dyn,
            value_net   = value_net,
            policy_net  = policy_net,
            cfg         = cfg,
            T_sim       = T_sim,
            T_train     = T_train,
            window_half = window_half,
        )
        result.gate_idx = g
        results.append(result)

        if not result.passed:
            log.debug("  Trial: gate %d not passed (trigger=%s) — stopping.",
                      g, result.trigger)
            break

        # Re-centre: shift x by -gate_spacing, keep everything else
        last_state = jnp.asarray(result.traj[-1:], dtype=dyn.dtype)   # (1, state_dim)
        last_state = last_state.at[0, 0].add(-gate_spacing)
        x_cur = last_state

        log.debug("  Trial: gate %d passed via '%s', x shifted by -%.2f m",
                  g, result.trigger, gate_spacing)

    return results


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def gates_cleared(results: List[GateResult]) -> int:
    """Number of gates successfully passed."""
    return sum(1 for r in results if r.passed)


# ---------------------------------------------------------------------------
# Safety value helper
# ---------------------------------------------------------------------------

def eval_safety_along_traj(value_net, dyn,
                           traj_np: np.ndarray, T: float) -> np.ndarray:
    K  = traj_np.shape[0] - 1
    xs = jnp.clip(jnp.asarray(traj_np, dtype=dyn.dtype), dyn.state_low, dyn.state_high)
    ts = jnp.linspace(T, 0.0, K + 1, dtype=dyn.dtype)
    return np.asarray(value_net(xs, ts)["V"])


# ---------------------------------------------------------------------------
# Quadrotor body rendering helpers
# ---------------------------------------------------------------------------

_ARM_DIRS = np.array([
    [ 1,  0, 0],
    [-1,  0, 0],
    [ 0,  1, 0],
    [ 0, -1, 0],
], dtype=float)
_ARM_LEN = 0.15


def _quat_to_rotmat(qw, qx, qy, qz):
    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz),    2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),     1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),    1 - 2*(qx**2 + qy**2)],
    ])


def _rotor_positions(state: np.ndarray) -> np.ndarray:
    pos = state[:3]
    R   = _quat_to_rotmat(state[3], state[4], state[5], state[6])
    return pos + _ARM_LEN * (_ARM_DIRS @ R.T)


# ---------------------------------------------------------------------------
# Animation  (multi-gate aware)
# ---------------------------------------------------------------------------

_CAT_COLOR = {"success": "tab:green", "partial": "tab:orange", "failure": "tab:red"}
_CAT_LABEL = {
    "success": "FULL SUCCESS  (all gates cleared)",
    "partial":      "PARTIAL  (some gates cleared)",
    "failure":      "FAILURE  (no gates cleared)",
}


def _trajectory_category(results: List[GateResult], n_gates: int) -> str:
    n = gates_cleared(results)
    if n == n_gates:
        return "success"
    if n > 0:
        return "partial"
    return "failure"


def save_multi_gate_animation(
    results:      List[GateResult],
    n_gates:      int,
    gate_spacing: float,
    dyn,
    T_sim:        float,
    fps:          int,
    trial_idx:    int,
    out_path:     str,
    value_net,
) -> None:
    """Stitch gate-local trajectories into world frame and animate."""

    # ── Build world-frame trajectory ───────────────────────────────────────
    # Gate g has its plane at world x = g * gate_spacing.
    # A state in gate-local frame has x_local = x_world - g * gate_spacing.
    world_trajs = []
    gate_transition_times = []   # simulation time at each gate crossing
    cumulative_t = 0.0
    for g, res in enumerate(results):
        offset = g * gate_spacing
        traj_w = res.traj.copy()
        traj_w[:, 0] += offset          # shift x to world frame
        world_trajs.append(traj_w)

        K_g = res.steps_taken
        T_g = T_sim * K_g / max(1, res.traj.shape[0] - 1)
        cumulative_t += T_g
        if res.passed:
            gate_transition_times.append(cumulative_t)

    traj_world = np.concatenate(world_trajs, axis=0)   # (K_total+n_gates, 13)
    K_total    = traj_world.shape[0] - 1

    total_t = sum(
        T_sim * r.steps_taken / max(1, r.traj.shape[0] - 1) for r in results)
    sim_t   = np.linspace(0.0, total_t, K_total + 1)

    # Build stitched safety values (gate-local, so we can reuse value net)
    sv_all = []
    for res in results:
        sv = eval_safety_along_traj(value_net, dyn, res.traj, T_sim)
        sv_all.append(sv)
    safety_vals = np.concatenate(sv_all)

    n_cleared = gates_cleared(results)
    cat = _trajectory_category(results, n_gates)
    color = _CAT_COLOR[cat]

    px, py, pz = traj_world[:, 0], traj_world[:, 1], traj_world[:, 2]
    vx, vy, vz = traj_world[:, 7], traj_world[:, 8], traj_world[:, 9]
    wx, wy, wz = traj_world[:, 10], traj_world[:, 11], traj_world[:, 12]

    # ── Figure ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor("#f5f5f5")
    gs = fig.add_gridspec(3, 2, hspace=0.48, wspace=0.32,
                          left=0.06, right=0.97, top=0.88, bottom=0.07)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_v  = fig.add_subplot(gs[0, 1])
    ax_w  = fig.add_subplot(gs[1, 1])
    ax_sv = fig.add_subplot(gs[2, 1])

    costs_str = "  |  ".join(
        f"G{r.gate_idx}: {r.brat_cost:.3f}" for r in results)
    fig.suptitle(
        f"{_CAT_LABEL[cat]}   |   trial {trial_idx}"
        f"   |   gates cleared: {n_cleared}/{len(results)}"
        f"\nBRAT costs → {costs_str}",
        fontsize=10, fontweight="bold", color=color,
    )

    # ── 3-D view ───────────────────────────────────────────────────────────
    for g in range(n_gates):
        x_off = g * gate_spacing
        lbl = "gate opening" if g == 0 else None
        add_gate_to_axes(ax3d, dyn, x_offset=x_off,
                         alpha=0.4 if g > 0 else 0.6, label=lbl)

    ax3d.plot(px, py, pz, color="lightgray", lw=1.0, alpha=0.4, zorder=1)
    traj_line, = ax3d.plot([], [], [], color=color, lw=1.5, alpha=0.8, zorder=2)
    arm_lines  = [ax3d.plot([], [], [], color=color, lw=2.5, zorder=4)[0]
                  for _ in range(4)]
    rotor_dots = [ax3d.plot([], [], [], "o", color=color, ms=5,
                            markeredgecolor="k", markeredgewidth=0.5, zorder=5)[0]
                  for _ in range(4)]
    center_dot, = ax3d.plot([], [], [], "s", color=color, ms=5,
                            markeredgecolor="k", markeredgewidth=0.5, zorder=5)

    x_lo = float(px.min()) - 0.5
    x_hi = float(px.max()) + 0.5
    y_ext = (dyn.gate_half_w + dyn.gate_thickness) * 4
    z_ext = (dyn.gate_half_h + dyn.gate_thickness) * 4
    ax3d.set_xlim(x_lo, x_hi)
    ax3d.set_ylim(-y_ext, y_ext)
    ax3d.set_zlim(-z_ext, z_ext)
    ax3d.set_xlabel("x (m)", labelpad=2, fontsize=8)
    ax3d.set_ylabel("y (m)", labelpad=2, fontsize=8)
    ax3d.set_zlabel("z (m)", labelpad=2, fontsize=8)
    ax3d.set_title(f"3-D trajectory  ({n_gates} gates, spacing={gate_spacing}m)",
                   fontsize=9)
    ax3d.tick_params(labelsize=6)
    ax3d.legend(fontsize=7, loc="upper right")

    # ── Right-panel time series ────────────────────────────────────────────
    def _setup_ts_ax(ax, ylabel, signals):
        for sig, lbl, c in signals:
            ax.plot(sim_t, sig, color=c, lw=1.2, label=lbl)
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
        for tt in gate_transition_times:
            ax.axvline(tt, color="gold", lw=1.2, ls=":", alpha=0.8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.legend(fontsize=7, ncol=3, loc="upper right")
        ax.set_xlim(sim_t[0], sim_t[-1])
        ax.tick_params(labelsize=7)
        return ax.axvline(0, color="k", lw=1.5, alpha=0.7)

    vl_v  = _setup_ts_ax(ax_v,  "lin. vel (m/s)",
                          [(vx,"vx","tab:red"), (vy,"vy","tab:green"), (vz,"vz","tab:blue")])
    vl_w  = _setup_ts_ax(ax_w,  "ang. vel (rad/s)",
                          [(wx,"wx","tab:red"), (wy,"wy","tab:green"), (wz,"wz","tab:blue")])

    ax_sv.plot(sim_t, safety_vals, color="tab:purple", lw=1.5, label="V(x,t)")
    ax_sv.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5, label="V=0")
    for tt in gate_transition_times:
        ax_sv.axvline(tt, color="gold", lw=1.2, ls=":", alpha=0.8,
                      label="gate cross" if tt == gate_transition_times[0] else None)
    ax_sv.set_ylabel("safety V(x,t)", fontsize=8)
    ax_sv.set_xlabel("time (s)", fontsize=8)
    ax_sv.legend(fontsize=7, ncol=3, loc="upper right")
    ax_sv.set_xlim(sim_t[0], sim_t[-1])
    ax_sv.tick_params(labelsize=7)
    vl_sv = ax_sv.axvline(0, color="k", lw=1.5, alpha=0.7)

    # ── Animation ──────────────────────────────────────────────────────────
    n_frames    = max(2, min(K_total + 1, int(fps * total_t) + 1))
    frame_steps = np.round(np.linspace(0, K_total, n_frames)).astype(int)

    def update(fi: int):
        step  = frame_steps[fi]
        state = traj_world[step]
        pos   = state[:3]
        rotors = _rotor_positions(state)

        traj_line.set_data(px[:step + 1], py[:step + 1])
        traj_line.set_3d_properties(pz[:step + 1])
        center_dot.set_data([pos[0]], [pos[1]])
        center_dot.set_3d_properties([pos[2]])
        for j in range(4):
            arm_lines[j].set_data([pos[0], rotors[j, 0]],
                                  [pos[1], rotors[j, 1]])
            arm_lines[j].set_3d_properties([pos[2], rotors[j, 2]])
            rotor_dots[j].set_data([rotors[j, 0]], [rotors[j, 1]])
            rotor_dots[j].set_3d_properties([rotors[j, 2]])

        t = sim_t[step]
        vl_v.set_xdata([t, t])
        vl_w.set_xdata([t, t])
        vl_sv.set_xdata([t, t])
        return (traj_line, center_dot, vl_v, vl_w, vl_sv, *arm_lines, *rotor_dots)

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=1000 // fps, blit=False)
    anim.save(out_path, writer=FFMpegWriter(fps=fps, bitrate=2000))
    plt.close(fig)
    log.info("  Saved %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_argparser().parse_args()
    key  = jax.random.PRNGKey(args.seed)
    rngs = nnx.Rngs(args.seed)
    rng  = np.random.default_rng(args.seed)

    cfg = _load_cfg(args.run_dir)
    out_dir = args.out_dir or os.path.join(args.run_dir, "sim_multigates")
    os.makedirs(out_dir, exist_ok=True)
    attach_file_handler(os.path.join(out_dir, "sim_multigates.log"))

    dyn = make_dynamics(cfg)

    ckpt_dir   = os.path.join(cfg.IO.LOG_DIR, cfg.IO.CKPT_DIRNAME)
    final_step = args.ckpt_step or find_latest_step(ckpt_dir)
    dt         = cfg.GAME.TIME.DT
    T_train    = final_step * dt
    T_sim      = args.sim_time

    log.info(
        "Checkpoint step=%d  T_train=%.4f s  T_sim/gate=%.4f s  dt=%.4f s",
        final_step, T_train, T_sim, dt)
    log.info(
        "Gates: %d  spacing=%.2f m  window_half=%.3f s",
        args.n_gates, args.gate_spacing, args.window_half)

    value_net  = load_value_net(cfg,  dyn, rngs, ckpt_dir, final_step)
    policy_net = load_policy_net(cfg, dyn, rngs, ckpt_dir, final_step)

    # ── Sample initial states ──────────────────────────────────────────────
    key, sample_key = jax.random.split(key)
    x0_all = sample_val_states_uniform(sample_key, dyn, args.N)
    log.info("Sampled %d initial states", args.N)

    # ── Simulate ───────────────────────────────────────────────────────────
    all_results: List[List[GateResult]] = []
    cleared_counts = []

    for n in tqdm(range(args.N), desc="Simulate"):
        x0_1    = x0_all[n:n + 1]
        results = simulate_multi_gate(
            x0_1         = x0_1,
            dyn          = dyn,
            value_net    = value_net,
            policy_net   = policy_net,
            cfg          = cfg,
            n_gates      = args.n_gates,
            gate_spacing = args.gate_spacing,
            T_sim        = T_sim,
            T_train      = T_train,
            window_half  = args.window_half,
        )
        all_results.append(results)
        cleared_counts.append(gates_cleared(results))

    cleared_np = np.array(cleared_counts)

    # ── Console summary ────────────────────────────────────────────────────
    success = (cleared_np == args.n_gates).sum()
    any_success  = (cleared_np > 0).sum()
    log.info("Full success (all %d gates): %d / %d  (%.1f%%)",
             args.n_gates, success, args.N, 100 * success / args.N)
    log.info("Any gates cleared:           %d / %d  (%.1f%%)",
             any_success, args.N, 100 * any_success / args.N)
    log.info("Mean gates cleared: %.2f / %d", cleared_np.mean(), args.n_gates)
    for g in range(args.n_gates):
        n_g = (cleared_np > g).sum()
        log.info("  Gate %d cleared: %d / %d  (%.1f%%)",
                 g, n_g, args.N, 100 * n_g / args.N)

    # ── Gates-cleared bar chart ────────────────────────────────────────────
    palette = {args.n_gates: "tab:green", 0: "tab:red"}
    bar_colors = [palette.get(c, "tab:orange") for c in cleared_np]

    fig, ax = plt.subplots(figsize=(max(6, args.N // 4), 3))
    ax.bar(range(args.N), cleared_np, color=bar_colors)
    ax.axhline(args.n_gates, color="k", lw=1, ls="--", alpha=0.5,
               label=f"all {args.n_gates} gates")
    ax.set_xlabel("Trajectory index")
    ax.set_ylabel("Gates cleared")
    ax.set_title(
        f"Multi-gate sim (spacing={args.gate_spacing}m, window±{args.window_half}s)\n"
        f"Full success: {success}/{args.N}  |  "
        f"mean gates: {cleared_np.mean():.2f}/{args.n_gates}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    chart_path = os.path.join(out_dir, "sim_gate_clears.png")
    fig.savefig(chart_path, dpi=120)
    plt.close(fig)
    log.info("Saved bar chart → %s", chart_path)

    # ── Bucket and animate ─────────────────────────────────────────────────
    full_mask    = cleared_np == args.n_gates
    partial_mask = (cleared_np > 0) & ~full_mask
    failure_mask = cleared_np == 0

    buckets = {
        "success": np.where(full_mask)[0],
        "partial":      np.where(partial_mask)[0],
        "failure":      np.where(failure_mask)[0],
    }

    n_pick = args.n_per_category
    for cat, idxs in buckets.items():
        if len(idxs) == 0:
            log.info("No '%s' samples — skipping animation.", cat)
            continue
        chosen  = rng.choice(idxs, size=min(n_pick, len(idxs)), replace=False)
        cat_dir = os.path.join(out_dir, cat)
        os.makedirs(cat_dir, exist_ok=True)

        for trial_i, idx in enumerate(tqdm(chosen, desc=f"Animate {cat}")):
            save_multi_gate_animation(
                results      = all_results[idx],
                n_gates      = args.n_gates,
                gate_spacing = args.gate_spacing,
                dyn          = dyn,
                T_sim        = T_sim,
                fps          = args.fps,
                trial_idx    = int(idx),
                out_path     = os.path.join(cat_dir, f"{cat}_{trial_i:02d}.mp4"),
                value_net    = value_net,
            )

    log.info("Done. Results saved to %s", out_dir)


if __name__ == "__main__":
    main()
