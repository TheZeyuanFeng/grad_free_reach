"""
evade_pursuit_sim.py  —  Pursuit-evasion closed-loop simulation (JAX port).

Rolls out N independent trajectories for SIM_TIME seconds each. The pursuer
policy uses a value-net-adaptive time-window voting scheme; the evader votes over
the same window. After simulation it computes BRAT cost and pursuer success rate,
then animates a success and a failure trial. Optional sampling-based MPC/MPPI
agents can replace either side, and an optional BRT run provides a
least-restrictive safety filter.

JAX port of the torch reference (time_marching_reachability-vq_policy). Uses the
JAX PursuitEvasion dynamics and the repo's key-first / nnx conventions.

Usage:
    python evade_pursuit_sim.py --run_dir <run_dir> [options]
"""

import argparse
import logging
import os

import matplotlib
# Headless (file-saving) default for batch runs, but respect an explicitly
# requested backend (e.g. the keyboard sim sets MPLBACKEND=TkAgg for a live GUI).
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from tqdm import tqdm

from configs import get_cfg_defaults
from utils import (
    decode_actions,
    find_latest_step,
    load_policy_net,
    load_value_net,
    make_dynamics,
    setup_logging,
)
# Import a training module before reachability.data.* to prime the
# data<->training import order (avoids a partially-initialized-module circular import).
from reachability.training.functional import rollout_with_intermediate_checks
from reachability.data.sampling import sample_uniform_states

setup_logging()
log = logging.getLogger("sim")

# Jitted net calls: the closed-loop sim queries the value/policy nets thousands of
# times, so compile once per input shape and reuse (nnx.jit caches by net structure
# + arg shapes) instead of re-dispatching the multinet eagerly every step.
_V_JIT = nnx.jit(lambda net, x, t: net(x, t)["V"])
_POUT_JIT = nnx.jit(lambda net, x, t: net(x, t))


def _majority_sign(a):
    """Sign of the column-mean of a (M, dim) vote array; ties -> +1. Returns (1, dim)."""
    ms = jnp.sign(jnp.mean(a, axis=0, keepdims=True))
    return jnp.where(ms == 0, 1.0, ms)


def clamp_state(dyn, x):
    """Clamp/wrap a state into the valid set (positions, speed, angle wrap)."""
    return dyn.wrap_state(x)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Simulate pursuit-evasion closed-loop behaviour.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run_dir",  type=str, required=True)
    p.add_argument("--N",        type=int,   default=10, help="Number of trajectories.")
    p.add_argument("--sim_time", type=float, default=20.0, help="Simulation duration (s).")
    p.add_argument("--seed",     type=int,   default=0)
    p.add_argument("--fps",      type=int,   default=20, help="Animation FPS.")
    p.add_argument("--out_dir",  type=str,   default=None,
                   help="Directory to save animations. Defaults to <run_dir>/sim/.")
    p.add_argument("--animate_all", action="store_true",
                   help="Animate every trajectory instead of just one success and one failure.")
    p.add_argument("--animate_failures", action="store_true",
                   help="Animate all failure trajectories (cost > 0) instead of one random failure.")
    p.add_argument("--brt_dir", type=str, default=None,
                   help="Optional BRT run directory: applies a least-restrictive safety filter "
                        "(BRT controller overrides the pursuer only near an unsafe state).")
    p.add_argument("--mpc_evader", action="store_true",
                   help="Replace the learned evader disturbance with a sampling-based MPC agent.")
    p.add_argument("--mpc_horizon",   type=float, default=3.0, help="MPC planning horizon (s).")
    p.add_argument("--mpc_samples",   type=int,   default=512, help="Candidate control sequences per MPC step.")
    p.add_argument("--mpc_mode",      type=str,   default="MPPI", choices=["MPC", "MPPI"],
                   help="Greedy best-sample (MPC) or weighted mean (MPPI).")
    p.add_argument("--mpc_lambda",    type=float, default=0.05, help="MPPI temperature (lower = greedier).")
    p.add_argument("--mpc_pursuer", action="store_true",
                   help="Replace the learned pursuer control with a sampling-based MPC agent.")
    p.add_argument("--exact_t", action="store_true",
                   help="Fast mode: query the learned policy at the EXACT remaining time-to-go "
                        "each step (t = clip((K-k)*dt, dt, T)), no adaptive-window voting or "
                        "value-net queries. Fully jitted lax.scan rollout. Ignores MPC/BRT.")
    return p


# ---------------------------------------------------------------------------
# Config / checkpoint helpers
# ---------------------------------------------------------------------------

def load_cfg(run_dir: str):
    cfg = get_cfg_defaults()
    cfg_path = os.path.join(run_dir, "config.yaml")
    if os.path.isfile(cfg_path):
        cfg.merge_from_file(cfg_path)
    cfg.freeze()
    return cfg


def _all_steps(ckpt_dir: str) -> list:
    steps = []
    for name in os.listdir(ckpt_dir):
        if name.startswith("step_") and os.path.isdir(os.path.join(ckpt_dir, name)):
            try:
                steps.append(int(name.split("_")[-1]))
            except ValueError:
                pass
    if not steps:
        raise FileNotFoundError(f"No step_NNN dirs in {ckpt_dir}")
    return sorted(steps)


def load_final_nets(cfg, dyn, rngs, ckpt_dir: str):
    """Load value+policy nets from the final (highest) step. A single checkpoint
    holds the full time-conditioned V(x,t)/policy, so we just query at the desired t."""
    steps = _all_steps(ckpt_dir)
    final_step = steps[-1]
    value_net = load_value_net(cfg, dyn, rngs, ckpt_dir, final_step)
    policy_net = load_policy_net(cfg, dyn, rngs, ckpt_dir, final_step)
    log.info("Loaded final checkpoint: step %d  (T=%.3f s)", final_step, final_step * cfg.GAME.TIME.DT)
    return value_net, policy_net, steps, final_step


def load_brt_nets(brt_dir: str, dyn, rngs):
    """Load BRT value+policy nets from a separate BRT run directory."""
    brt_cfg = load_cfg(brt_dir)
    ckpt_dir = os.path.join(brt_dir, brt_cfg.IO.CKPT_DIRNAME)
    step = find_latest_step(ckpt_dir)
    brt_value_net = load_value_net(brt_cfg, dyn, rngs, ckpt_dir, step)
    brt_policy_net = load_policy_net(brt_cfg, dyn, rngs, ckpt_dir, step)
    brt_t_max = brt_cfg.GAME.TIME.DT * step
    log.info("Loaded BRT nets: step %d  (T=%.3f s)", step, brt_t_max)
    return brt_value_net, brt_policy_net, brt_t_max


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------

def steps_in_window(t_lo: float, t_hi: float, dt: float, all_steps: list) -> list:
    """Steps whose training horizon T_k = k*dt falls in [t_lo, t_hi]."""
    return [k for k in all_steps if t_lo <= k * dt <= t_hi]


def update_pursuer_window(x1, dyn, value_net, all_steps: list, dt: float, T: float) -> tuple:
    """Adaptive pursuer time window for the current state.

    Returns (t_lo, t_mid, t_hi): t_mid = min T_k with V(x,T_k) <= 0 (shortest
    catching horizon); t_hi = clip(t_mid+0.5, T); t_lo = t_hi - 1.0. Fallback when
    not catchable: (T-1, T-1, T)."""
    K = len(all_steps)
    x_batch = jnp.broadcast_to(x1, (K, x1.shape[-1]))
    t_batch = jnp.asarray([k * dt for k in all_steps], dtype=dyn.dtype)
    V_batch = np.asarray(_V_JIT(value_net, x_batch, t_batch))   # (K,)

    winning = np.nonzero(V_batch <= 0.0)[0]
    if winning.size == 0:
        return T - 1.0, T - 1.0, T   # not catchable
    t_mid = all_steps[int(winning[0])] * dt
    if t_mid <= 0.5:
        return 0.0, t_mid, 1.0
    t_max = min(t_mid + 0.5, T)
    return t_max - 1.0, t_mid, t_max


def poll_policy(x1, dyn, policy_net, window_steps: list, dt: float, all_steps: list):
    """Query the policy at all window_steps in ONE batched (jitted) call.

    Returns (u_all, d_all) each (M, dim) — the decoded bang-bang actions at each
    horizon T_k = k*dt in the window. Caller majority-votes over the M rows."""
    if not window_steps:
        window_steps = all_steps[-1:]  # fallback: final step
    nn_x = dyn.nn_inputs(x1)                       # (1, input_dim)
    M = len(window_steps)
    nn_batch = jnp.broadcast_to(nn_x, (M, nn_x.shape[-1]))
    t = jnp.asarray([k * dt for k in window_steps], dtype=dyn.dtype)
    out = _POUT_JIT(policy_net, nn_batch, t)
    return decode_actions(out, dyn)                # (M, ctrl), (M, dist)


# ---------------------------------------------------------------------------
# MPC evader (sampling-based, numpy)
# ---------------------------------------------------------------------------

class _EvaderMPCDyn:
    """Numpy 4D Dubins evader. State [x,y,theta,v]; control [omega, a]."""

    def __init__(self, dyn):
        self.v_min = dyn.v_min
        self.v_max = dyn.v_max_e
        self.omega_max = dyn.omega_max_e
        self.a_max = dyn.a_max
        self.robot_radius = dyn.robot_radius
        self.half = dyn.half
        self.obstacles = [(cx, cy, r + dyn.robot_radius) for cx, cy, r in dyn.circle_obstacles]
        self.control_init = np.zeros(2, dtype=np.float64)
        self.control_range_ = np.array([[-self.omega_max, self.omega_max],
                                        [-self.a_max, self.a_max]], dtype=np.float64)
        self.eps_var = np.array([(self.omega_max * 0.5) ** 2, (self.a_max * 0.5) ** 2], dtype=np.float64)

    def dsdt(self, s, u):
        theta, v = s[..., 2], s[..., 3]
        omega, accel = u[..., 0], u[..., 1]
        accel = np.where((v >= self.v_max) & (accel > 0), 0.0, accel)
        accel = np.where((v <= self.v_min) & (accel < 0), 0.0, accel)
        return np.stack([v * np.cos(theta), v * np.sin(theta), omega, accel], axis=-1)

    def clamp_control(self, s, u):
        return np.clip(u, self.control_range_[:, 0], self.control_range_[:, 1])

    def equivalent_wrapped_state(self, s):
        out = s.copy()
        out[..., 2] = (out[..., 2] + np.pi) % (2 * np.pi) - np.pi
        out[..., 3] = np.clip(out[..., 3], self.v_min, self.v_max)
        return out

    def cost_fn2(self, state_trajs, pursuers_xy):
        """state_trajs (A,N,H+1,4), pursuers_xy (P,2). Maximise min pursuer dist."""
        xy = state_trajs[..., :2]
        dists = np.linalg.norm(xy[:, :, :, None, :] - pursuers_xy[None, None, None, :, :], axis=-1)
        min_dist = dists.min(axis=-1)
        reward = min_dist.mean(axis=-1)
        penalty = np.zeros_like(reward)
        margin = np.minimum(
            np.minimum(xy[..., 0] + self.half, self.half - xy[..., 0]),
            np.minimum(xy[..., 1] + self.half, self.half - xy[..., 1]),
        ) - self.robot_radius
        penalty += np.sum(np.minimum(margin, 0.0) ** 2, axis=-1) * 2.0
        for cx, cy, r in self.obstacles:
            d_obs = np.linalg.norm(xy - np.array([cx, cy]), axis=-1) - r
            penalty += np.sum(np.minimum(d_obs, 0.0) ** 2, axis=-1) * 2.0
        return -reward + penalty


class MPC_Evader:
    """Sampling-based MPC/MPPI evader. get_action(x_full)->(1,2) [omega_e, a_e]."""

    def __init__(self, dyn, dt, horizon, num_samples=512, mode="MPPI", lambda_=0.05):
        self._dyn = dyn
        self._mpc = _EvaderMPCDyn(dyn)
        self.dt = dt
        self.num_samples = num_samples
        self.mode = mode
        self.lambda_ = lambda_
        self._H = max(1, int(round(horizon / dt)))
        self._controls = np.zeros((self._H, 2), dtype=np.float64)

    def get_action(self, x_full):
        x_np = np.asarray(x_full)   # (1, 10)
        evader_state = x_np[0, :4]
        pursuers_xy = np.stack([
            x_np[0, self._dyn._pursuer_base(j): self._dyn._pursuer_base(j) + 2]
            for j in range(self._dyn.num_pursuers)
        ], axis=0)

        init = evader_state[None, :]
        eps = np.random.randn(1, self.num_samples, self._H, 2) * np.sqrt(self._mpc.eps_var)
        eps[0, 0] = 0.0
        perturbed = np.clip(self._controls[None, None, :, :] + eps,
                            self._mpc.control_range_[:, 0], self._mpc.control_range_[:, 1])

        trajs = np.zeros((1, self.num_samples, self._H + 1, 4))
        trajs[:, :, 0, :] = init[:, None, :]
        for k in range(self._H):
            perturbed[:, :, k, :] = self._mpc.clamp_control(trajs[:, :, k, :], perturbed[:, :, k, :])
            dsdt = self._mpc.dsdt(trajs[:, :, k, :], perturbed[:, :, k, :])
            trajs[:, :, k + 1, :] = self._mpc.equivalent_wrapped_state(trajs[:, :, k, :] + dsdt * self.dt)

        costs = self._mpc.cost_fn2(trajs, pursuers_xy)
        if self.mode == "MPC":
            best_seq = perturbed[0, int(np.argmin(costs[0]))]
        else:
            w = np.exp(-costs[0] / self.lambda_); w = w / (w.sum() + 1e-12)
            best_seq = (perturbed[0] * w[:, None, None]).sum(axis=0)

        self._controls[:-1] = best_seq[1:]
        self._controls[-1] = self._mpc.control_init
        return jnp.asarray(best_seq[0], dtype=x_full.dtype)[None, :]   # (1, 2)


# ---------------------------------------------------------------------------
# MPC pursuer (sampling-based, numpy) — each pursuer optimises independently
# ---------------------------------------------------------------------------

class _SinglePursuerMPCDyn:
    """Numpy 3D Dubins pursuer (constant speed). State [x,y,theta]; control [omega]."""

    def __init__(self, dyn):
        self.v_const = dyn.v_const
        self.omega_max = dyn.omega_max_p
        self.robot_radius = dyn.robot_radius
        self.half = dyn.half
        self.obstacles = [(cx, cy, r + dyn.robot_radius) for cx, cy, r in dyn.circle_obstacles]
        self.control_init = np.zeros(1, dtype=np.float64)
        self.control_range_ = np.array([[-self.omega_max, self.omega_max]], dtype=np.float64)
        self.eps_var = np.array([(self.omega_max * 0.5) ** 2], dtype=np.float64)

    def dsdt(self, s, u):
        th, om = s[..., 2], u[..., 0]
        return np.stack([self.v_const * np.cos(th), self.v_const * np.sin(th), om], axis=-1)

    def clamp_control(self, s, u):
        return np.clip(u, self.control_range_[:, 0], self.control_range_[:, 1])

    def equivalent_wrapped_state(self, s):
        out = s.copy()
        out[..., 2] = (out[..., 2] + np.pi) % (2 * np.pi) - np.pi
        return out

    def cost_fn2(self, state_trajs, evader_xy):
        xy = state_trajs[..., :2]
        ev = evader_xy[None, None, None, :]
        dist = np.linalg.norm(xy - ev, axis=-1)
        cost = dist[..., -1] + dist.mean(axis=-1)
        margin = np.minimum(
            np.minimum(xy[..., 0] + self.half, self.half - xy[..., 0]),
            np.minimum(xy[..., 1] + self.half, self.half - xy[..., 1]),
        ) - self.robot_radius
        cost += np.sum(np.minimum(margin, 0.0) ** 2, axis=-1) * 10.0
        for cx, cy, r in self.obstacles:
            d_obs = np.linalg.norm(xy - np.array([cx, cy]), axis=-1) - r
            cost += np.sum(np.minimum(d_obs, 0.0) ** 2, axis=-1) * 10.0
        return cost


class MPC_Pursuer:
    """Each pursuer runs an independent MPPI to chase the evader.
    get_action(x_full)->(1,2) [omega_p1, omega_p2]."""

    def __init__(self, dyn, dt, horizon, num_samples=512, mode="MPPI", lambda_=0.05):
        self._dyn = dyn
        self._mpc = _SinglePursuerMPCDyn(dyn)
        self.dt = dt
        self._H = max(1, int(round(horizon / dt)))
        self.num_samples = num_samples
        self.mode = mode
        self.lambda_ = lambda_
        self._controls = [np.zeros((self._H, 1), dtype=np.float64) for _ in range(dyn.num_pursuers)]

    def _run_mppi(self, init3, evader_xy, controls):
        N, H = self.num_samples, self._H
        eps = np.random.randn(1, N, H, 1) * np.sqrt(self._mpc.eps_var)
        eps[0, 0] = 0.0
        perturbed = np.clip(controls[None, None, :, :] + eps,
                            self._mpc.control_range_[:, 0], self._mpc.control_range_[:, 1])
        trajs = np.zeros((1, N, H + 1, 3))
        trajs[:, :, 0, :] = init3[None, None, :]
        for k in range(H):
            perturbed[:, :, k, :] = self._mpc.clamp_control(trajs[:, :, k, :], perturbed[:, :, k, :])
            dsdt = self._mpc.dsdt(trajs[:, :, k, :], perturbed[:, :, k, :])
            trajs[:, :, k + 1, :] = self._mpc.equivalent_wrapped_state(trajs[:, :, k, :] + dsdt * self.dt)
        costs = self._mpc.cost_fn2(trajs, evader_xy)[0]
        if self.mode == "MPC":
            best_seq = perturbed[0, int(np.argmin(costs))]
        else:
            w = np.exp(-costs / self.lambda_); w = w / (w.sum() + 1e-12)
            best_seq = (perturbed[0] * w[:, None, None]).sum(axis=0)
        return best_seq, costs

    def get_action(self, x_full):
        x_np = np.asarray(x_full)
        evader_xy = x_np[0, :2]
        omegas = []
        for j in range(self._dyn.num_pursuers):
            base = self._dyn._pursuer_base(j)
            init3 = x_np[0, base: base + 3]
            best_seq, _ = self._run_mppi(init3, evader_xy, self._controls[j])
            self._controls[j][:-1] = best_seq[1:]
            self._controls[j][-1] = self._mpc.control_init
            omegas.append(float(best_seq[0, 0]))
        return jnp.asarray(np.array(omegas, dtype=np.float64), dtype=x_full.dtype)[None, :]  # (1, 2)


# ---------------------------------------------------------------------------
# Single-trajectory rollout
# ---------------------------------------------------------------------------

def simulate_one(x0_1, dyn, value_net, policy_net, all_steps, cfg, sim_time,
                 brt_value_net=None, brt_policy_net=None, brt_t_max=None,
                 mpc_evader=None, mpc_pursuer=None, desc=None):
    """Roll out one trajectory for sim_time seconds. Returns (traj (K+1,sd), window_log)."""
    dt = cfg.GAME.TIME.DT
    T = all_steps[-1] * dt
    m = cfg.GAME.TIME.NUM_SUBSTEPS
    K = int(round(sim_time / dt))
    use_brt = (brt_value_net is not None and brt_policy_net is not None and brt_t_max is not None)

    xs = [x0_1]
    window_log = []
    x_cur = x0_1

    need_pursuer_votes = mpc_pursuer is None
    need_evader_votes = mpc_evader is None

    step_iter = tqdm(range(K), desc=desc, leave=False) if desc is not None else range(K)
    for _ in step_iter:
        if need_pursuer_votes or need_evader_votes:
            _, pursuer_t_mid, pursuer_t_hi = update_pursuer_window(x_cur, dyn, value_net, all_steps, dt, T)
            window_log.append((pursuer_t_mid, pursuer_t_hi))
            shared_steps = steps_in_window(pursuer_t_mid, pursuer_t_hi, dt, all_steps)
            x_clamped = clamp_state(dyn, x_cur)
        else:
            window_log.append((0.0, 0.0))

        # One batched policy poll serves both the pursuer (u) and evader (d) votes.
        u_all = d_all = None
        if need_pursuer_votes or need_evader_votes:
            u_all, d_all = poll_policy(x_clamped, dyn, policy_net, shared_steps, dt, all_steps)

        u_scaled = dyn.u_max[None, :]
        d_scaled = dyn.d_max[None, :]

        if mpc_pursuer is not None:
            u = mpc_pursuer.get_action(x_cur)
        elif u_all is not None:
            u = _majority_sign(u_all) * u_scaled
        else:
            u = jnp.zeros((1, dyn.control_dim), dtype=dyn.dtype)

        if mpc_evader is not None:
            d = mpc_evader.get_action(x_cur)
        elif d_all is not None:
            d = _majority_sign(d_all) * d_scaled
        else:
            d = jnp.zeros((1, dyn.disturb_dim), dtype=dyn.dtype)

        if use_brt:
            t_brt = jnp.full((1,), brt_t_max, dtype=dyn.dtype)
            V_brt = float(_V_JIT(brt_value_net, clamp_state(dyn, x_cur), t_brt)[0])
            t_brat = jnp.full((1,), T, dtype=dyn.dtype)
            V_brat = float(_V_JIT(value_net, x_cur, t_brat)[0])
            if (V_brt <= 0.3) and (V_brat > 0.0):
                brt_out = brt_policy_net(dyn.nn_inputs(x_cur), t_brt)
                u, _ = decode_actions(brt_out, dyn)

        x_cur, _ = rollout_with_intermediate_checks(
            x=x_cur, u=u, d=d, dyn=dyn, dt=dt, m=m, problem_type="BRAT")
        xs.append(x_cur)

    return jnp.concatenate(xs, axis=0), window_log


def simulate_one_exact_t(x0_1, dyn, policy_net, cfg, sim_time, T):
    """Fast closed loop: at each step query the learned policy at the exact remaining
    time-to-go and apply its bang-bang (u for pursuers, d for evader). No voting, no
    value-net queries -> the whole rollout is one jitted lax.scan (compile once, then
    GPU-resident with no per-step host sync). Returns (traj (K+1, sd), [])."""
    dt = cfg.GAME.TIME.DT
    m = cfg.GAME.TIME.NUM_SUBSTEPS
    K = int(round(sim_time / dt))
    # time-to-go per step: step k -> (K-k)*dt, i.e. [sim_time .. dt], clipped to [dt, T].
    ts = jnp.clip(jnp.arange(K, 0, -1, dtype=dyn.dtype) * dt, dt, T)

    @nnx.jit
    def _rollout(net, x0, ts):
        def body(x, t):
            out = net(dyn.nn_inputs(x), jnp.full((1,), t, dtype=dyn.dtype))
            u, d = decode_actions(out, dyn)
            x2, _ = rollout_with_intermediate_checks(
                x=x, u=u, d=d, dyn=dyn, dt=dt, m=m, problem_type="BRAT")
            return x2, x2[0]
        _, xs = jax.lax.scan(body, x0, ts)          # xs: (K, sd)
        return jnp.concatenate([x0, xs], axis=0)    # (K+1, sd)

    return _rollout(policy_net, x0_1, ts), []


def brat_cost(traj, dyn) -> float:
    """BRAT cost for a single (K+1, state_dim) trajectory."""
    return float(dyn.cost_fn(traj[None])[0])


# ---------------------------------------------------------------------------
# Teacher control / disturbance slice visualisation
# ---------------------------------------------------------------------------

def plot_teacher_slices(value_net, dyn, t, state_slices=None, x_axis_idx=None,
                        y_axis_idx=None, resolution=60, out_path=None):
    """Plot teacher (optimal) control/disturbance signs over a 2D state slice, with
    the V=0 contour. Signs come from the value gradient: u* = -sign(dV/dtheta_pj),
    d* = +sign(dV/d[affected index])."""
    pc = dyn.plot_config()
    state_slices = state_slices if state_slices is not None else pc["state_slices"]
    x_axis_idx = x_axis_idx if x_axis_idx is not None else pc["x_axis_idx"]
    y_axis_idx = y_axis_idx if y_axis_idx is not None else pc["y_axis_idx"]
    R = resolution

    x_lo, x_hi = float(dyn.state_low[x_axis_idx]), float(dyn.state_high[x_axis_idx])
    y_lo, y_hi = float(dyn.state_low[y_axis_idx]), float(dyn.state_high[y_axis_idx])
    xs = np.linspace(x_lo, x_hi, R); ys = np.linspace(y_lo, y_hi, R)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")

    base = np.tile(np.asarray(state_slices, np.float32), (R * R, 1))
    base[:, x_axis_idx] = gx.ravel(); base[:, y_axis_idx] = gy.ravel()
    base = jnp.asarray(base)
    t_arr = jnp.full((R * R,), t, dtype=dyn.dtype)

    V_vals = value_net(base, t_arr)["V"]
    grad_V = jax.grad(lambda xx: jnp.sum(value_net(xx, t_arr)["V"]))(base)
    V_grid = np.asarray(V_vals).reshape(R, R)

    u_signs = np.stack([-np.sign(np.asarray(grad_V[:, dyn._pursuer_base(j) + 2]))
                        for j in range(dyn.num_pursuers)], axis=-1)
    d_signs = np.stack([np.sign(np.asarray(grad_V[:, 2])), np.sign(np.asarray(grad_V[:, 3]))], axis=-1)
    u_grids = u_signs.reshape(R, R, dyn.num_pursuers)
    d_grids = d_signs.reshape(R, R, 2)

    labels = pc.get("state_labels", [f"dim {i}" for i in range(dyn.state_dim)])
    x_label, y_label = labels[x_axis_idx], labels[y_axis_idx]

    n_actions = dyn.control_dim + dyn.disturb_dim
    fig, axes = plt.subplots(1, n_actions, figsize=(4 * n_actions, 4))
    if n_actions == 1:
        axes = [axes]
    action_grids = ([u_grids[..., j] for j in range(dyn.num_pursuers)]
                    + [d_grids[..., i] for i in range(2)])
    action_titles = ([f"u: w_p{j+1}" for j in range(dyn.num_pursuers)]
                     + ["d: w_e", "d: a_e"])
    for ax, grid, title in zip(axes, action_grids, action_titles):
        im = ax.imshow(grid.T, origin="lower", extent=[x_lo, x_hi, y_lo, y_hi],
                       vmin=-1, vmax=1, cmap="RdBu", aspect="auto")
        ax.contour(xs, ys, V_grid.T, levels=[0], colors="black", linewidths=1.5)
        ax.set_title(title, fontsize=9); ax.set_xlabel(x_label); ax.set_ylabel(y_label)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="sign")
    fig.suptitle(f"Teacher slices  t={t:.2f}  ({x_label} vs {y_label})", fontsize=10)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)
        log.info("Saved teacher slice plot -> %s", out_path)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def animate_trajectory(traj, dyn, title, fps, out_path, sim_dt=0.1, window_log=None,
                       value_net=None, final_T=None, brt_value_net=None, brt_t_max=None):
    """Save an MP4 (fallback GIF) animation of one pursuit-evasion trajectory.
    traj : (K+1, state_dim)."""
    traj_np = np.asarray(traj)
    K1 = traj_np.shape[0]
    half = dyn.half
    obs_cx, obs_cy, obs_r = dyn.circle_obstacles[0]

    traj_j = jnp.asarray(traj, dtype=dyn.dtype)
    l_target_np = np.asarray(dyn.target_l(traj_j))
    l_avoid_np = np.asarray(dyn.avoid_l(traj_j))
    pred_V_np = None
    if value_net is not None and final_T is not None:
        pred_V_np = np.asarray(value_net(traj_j, jnp.full((K1,), final_T, dtype=dyn.dtype))["V"])
    pred_V_brt_np = None
    if brt_value_net is not None and brt_t_max is not None:
        pred_V_brt_np = np.asarray(brt_value_net(traj_j, jnp.full((K1,), brt_t_max, dtype=dyn.dtype))["V"])
    times = np.arange(K1) * sim_dt

    fig, (ax, ax_l) = plt.subplots(2, 1, figsize=(6, 8), gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(title, fontsize=10); fig.tight_layout(pad=2.0)

    ax.set_xlim(-half - 0.3, half + 0.3); ax.set_ylim(-half - 0.3, half + 0.3)
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.add_patch(mpatches.Rectangle((-half, -half), dyn.field_size, dyn.field_size,
                                    linewidth=2, edgecolor="black", facecolor="none"))
    ax.add_patch(mpatches.Circle((obs_cx, obs_cy), obs_r,
                                 edgecolor="saddlebrown", facecolor="saddlebrown", alpha=0.5))
    ax.plot(traj_np[:, 0], traj_np[:, 1], color="blue", alpha=0.2, lw=1)
    for j in range(dyn.num_pursuers):
        base = dyn._pursuer_base(j)
        ax.plot(traj_np[:, base], traj_np[:, base + 1], color="red", alpha=0.2, lw=1)

    evader_body = mpatches.Circle((0, 0), dyn.robot_radius, color="blue", alpha=0.85, zorder=3)
    evader_arrow = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                               arrowprops=dict(arrowstyle="->", color="blue", lw=1.5))
    ax.add_patch(evader_body)
    pursuer_bodies, pursuer_catches, pursuer_arrows = [], [], []
    for _ in range(dyn.num_pursuers):
        body = mpatches.Circle((0, 0), dyn.robot_radius, color="red", alpha=0.85, zorder=3)
        catch = mpatches.Circle((0, 0), dyn.catch_radius, edgecolor="red", facecolor="none",
                                linestyle="--", alpha=0.4, zorder=2)
        arrow = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                            arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
        ax.add_patch(body); ax.add_patch(catch)
        pursuer_bodies.append(body); pursuer_catches.append(catch); pursuer_arrows.append(arrow)
    time_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=9, verticalalignment="top")

    ax_l.plot(times, l_target_np, color="blue", lw=1.5, label="target_l")
    ax_l.plot(times, l_avoid_np, color="red", lw=1.5, label="avoid_l")
    if pred_V_np is not None:
        ax_l.plot(times, pred_V_np, color="green", lw=1.5, linestyle="--", label=f"V(x, T={final_T:.1f})")
    if pred_V_brt_np is not None:
        ax_l.plot(times, pred_V_brt_np, color="orange", lw=1.5, linestyle="--", label=f"V_brt(x, T={brt_t_max:.1f})")
    ax_l.axhline(0, color="black", lw=0.8, linestyle="--")
    ax_l.set_xlabel("t (s)"); ax_l.set_ylabel("value")
    ax_l.legend(fontsize=8, loc="upper right"); ax_l.grid(True, alpha=0.3)
    cursor = ax_l.axvline(0.0, color="gray", lw=1.2, linestyle=":")

    def _update(frame):
        s = traj_np[frame]
        xe, ye, the = s[0], s[1], s[2]
        al = dyn.robot_radius * 1.8
        evader_body.center = (xe, ye)
        evader_arrow.xy = (xe + al * np.cos(the), ye + al * np.sin(the))
        evader_arrow.set_position((xe, ye))
        for j in range(dyn.num_pursuers):
            base = dyn._pursuer_base(j)
            xp, yp, thp = s[base], s[base + 1], s[base + 2]
            pursuer_bodies[j].center = (xp, yp)
            pursuer_catches[j].center = (xp, yp)
            pursuer_arrows[j].xy = (xp + al * np.cos(thp), yp + al * np.sin(thp))
            pursuer_arrows[j].set_position((xp, yp))
        win_str = ""
        if window_log and frame > 0:
            t_lo, t_hi = window_log[frame - 1]
            win_str = f"  win=[{t_lo:.1f},{t_hi:.1f}]"
        time_text.set_text(f"t = {frame * sim_dt:.2f} s{win_str}")
        cursor.set_xdata([frame * sim_dt, frame * sim_dt])
        return ([evader_body, evader_arrow, time_text, cursor]
                + pursuer_bodies + pursuer_catches + pursuer_arrows)

    ani = animation.FuncAnimation(fig, _update, frames=K1, interval=int(1000 / fps), blit=False)
    try:
        ani.save(out_path, writer=animation.FFMpegWriter(fps=fps, bitrate=1800))
        log.info("Saved animation -> %s", out_path)
    except Exception as exc:
        gif_path = os.path.splitext(out_path)[0] + ".gif"
        log.warning("MP4 save failed (%s) - falling back to GIF: %s", exc, gif_path)
        ani.save(gif_path, writer=animation.PillowWriter(fps=fps))
        log.info("Saved animation -> %s", gif_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = build_argparser().parse_args()
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    rngs = nnx.Rngs(args.seed)

    cfg = load_cfg(args.run_dir)
    dyn = make_dynamics(cfg)

    ckpt_dir = os.path.join(args.run_dir, cfg.IO.CKPT_DIRNAME)
    value_net, policy_net, all_steps, _ = load_final_nets(cfg, dyn, rngs, ckpt_dir)

    brt_value_net = brt_policy_net = brt_t_max = None
    if args.brt_dir:
        brt_value_net, brt_policy_net, brt_t_max = load_brt_nets(args.brt_dir, dyn, rngs)

    out_dir = args.out_dir or os.path.join(args.run_dir, "sim")
    os.makedirs(out_dir, exist_ok=True)

    dt = cfg.GAME.TIME.DT
    T = all_steps[-1] * dt

    # --- Sample N valid initial states (no terminal conditions at t=0) ---------
    valid, n_valid, attempts = [], 0, 0
    while n_valid < args.N:
        key, sk = jax.random.split(key)
        batch = sample_uniform_states(sk, dyn, args.N * 4)
        l_t = np.asarray(dyn.target_l(batch))   # <=0 => already caught
        l_a = np.asarray(dyn.avoid_l(batch))     # >=0 => pursuer already in collision
        keep = (l_t > 0) & (l_a < 0)
        if brt_value_net is not None:
            V_brt = np.asarray(brt_value_net(batch, jnp.full((batch.shape[0],), brt_t_max, dtype=dyn.dtype))["V"])
            keep = keep & (V_brt >= 1.5)
        vb = np.asarray(batch)[keep]
        if vb.shape[0] > 0:
            valid.append(vb); n_valid += vb.shape[0]
        attempts += 1
        if attempts > 500:
            raise RuntimeError("Could not sample enough valid initial states after 500 attempts.")
    x0_all = jnp.asarray(np.concatenate(valid, axis=0)[: args.N])
    log.info("Sampled %d valid initial states (no terminal conditions at t=0)", args.N)

    trajs, window_logs, costs = [], [], []
    log.info("Simulating %d trajectories x %.1f s  (T_train=%.2f s, dt=%.3f s)", args.N, args.sim_time, T, dt)
    if args.exact_t and (args.mpc_evader or args.mpc_pursuer or args.brt_dir):
        log.warning("--exact_t uses the learned policy at the exact time-to-go; "
                    "ignoring --mpc_* / --brt_dir.")
    for n in tqdm(range(args.N), desc="trajectories"):
        x0_1 = x0_all[n:n + 1]
        if args.exact_t:
            traj, wlog = simulate_one_exact_t(x0_1, dyn, policy_net, cfg, args.sim_time, T)
        else:
            mpc_agent = (MPC_Evader(dyn, dt, args.mpc_horizon, args.mpc_samples, args.mpc_mode, args.mpc_lambda)
                         if args.mpc_evader else None)
            mpc_pursuer_agent = (MPC_Pursuer(dyn, dt, args.mpc_horizon, args.mpc_samples, args.mpc_mode, args.mpc_lambda)
                                 if args.mpc_pursuer else None)
            traj, wlog = simulate_one(
                x0_1=x0_1, dyn=dyn, value_net=value_net, policy_net=policy_net,
                all_steps=all_steps, cfg=cfg, sim_time=args.sim_time,
                brt_value_net=brt_value_net, brt_policy_net=brt_policy_net, brt_t_max=brt_t_max,
                mpc_evader=mpc_agent, mpc_pursuer=mpc_pursuer_agent,
                desc=f"traj {n + 1}/{args.N}")
        c = brat_cost(traj, dyn)
        trajs.append(traj); window_logs.append(wlog); costs.append(c)
        log.info("    traj %d/%d  BRAT cost = %.4f  (%s)", n + 1, args.N, c, "SUCCESS" if c <= 0 else "failure")

    costs_np = np.array(costs)
    success_mask = costs_np <= 0.0
    success_rate = success_mask.mean()
    log.info("=" * 50)
    log.info("Success rate: %.1f%%  (%d / %d)", 100 * success_rate, success_mask.sum(), args.N)
    log.info("BRAT cost  - mean: %.4f  min: %.4f  max: %.4f", costs_np.mean(), costs_np.min(), costs_np.max())

    success_idxs = np.where(success_mask)[0].tolist()
    failure_idxs = np.where(~success_mask)[0].tolist()
    if args.animate_all:
        to_animate = [(i, "SUCCESS" if success_mask[i] else "FAILURE", f"traj{i:03d}.mp4") for i in range(args.N)]
    elif args.animate_failures:
        to_animate = [(idx, "FAILURE", f"failure_traj{idx}.mp4") for idx in failure_idxs]
        if not failure_idxs:
            log.warning("No failed trajectories to animate.")
    else:
        to_animate = []
        if success_idxs:
            idx = int(np.random.choice(success_idxs)); to_animate.append((idx, "SUCCESS", f"success_traj{idx}.mp4"))
        else:
            log.warning("No successful trajectories to animate.")
        if failure_idxs:
            idx = int(np.random.choice(failure_idxs)); to_animate.append((idx, "FAILURE", f"failure_traj{idx}.mp4"))
        else:
            log.warning("No failed trajectories to animate.")

    for (idx, label, fname) in to_animate:
        animate_trajectory(
            traj=trajs[idx], dyn=dyn,
            title=f"Pursuit-Evasion  [{label}]  cost={costs[idx]:.4f}",
            fps=args.fps, out_path=os.path.join(out_dir, fname),
            sim_dt=dt, window_log=window_logs[idx],
            value_net=value_net, final_T=T,
            brt_value_net=brt_value_net, brt_t_max=brt_t_max)


if __name__ == "__main__":
    main()
