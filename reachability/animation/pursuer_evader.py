"""
evade_pursuit_sim.py  —  Pursuit-evasion closed-loop simulation.

Rolls out N independent trajectories for SIM_TIME seconds each.
Pursuer policy uses a time-window voting scheme that adapts based on the
value network's assessment of game status.  Evader uses a fixed late-horizon
window.  After simulation, computes BRAT cost and pursuer success rate, then
animates one success and one failure trial.

Usage:
    python -m reachability.animation.pursuer_evader --run_dir <run_dir> [options]
"""

import argparse
import logging
import os
import random

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from configs import get_cfg_defaults, PROJECT_NAME
from utils import (
    decode_actions,
    find_latest_step,
    load_policy_net,
    load_value_net,
    make_dynamics,
    setup_logging,
)
from reachability.training.functional import rollout_with_intermediate_checks
from reachability.data.sampling import sample_uniform_states

setup_logging()
log = logging.getLogger(f"{PROJECT_NAME}.{__name__}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Simulate pursuit-evasion closed-loop behaviour.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run_dir",  type=str, required=True)
    p.add_argument("--N",        type=int,   default=10,
                   help="Number of trajectories.")
    p.add_argument("--sim_time", type=float, default=20.0,
                   help="Simulation duration (s).")
    p.add_argument("--seed",     type=int,   default=0)
    p.add_argument("--fps",      type=int,
                   default=20,  help="Animation FPS.")
    p.add_argument("--out_dir",  type=str,   default=None,
                   help="Directory to save animations. Defaults to <run_dir>/sim/.")
    p.add_argument("--animate_all", action="store_true",
                   help="Animate every trajectory instead of just one success and one failure.")
    p.add_argument("--brt_dir", type=str, default=None,
                   help="Optional BRT run directory. When given, applies a least-restrictive "
                        "safety filter: the BRT controller overrides the pursuer only when "
                        "the main policy is about to enter an unsafe state.")
    p.add_argument("--mpc_evader", action="store_true",
                   help="Replace the learned evader disturbance with a sampling-based MPC agent "
                        "that maximises distance from pursuers while avoiding obstacles.")
    p.add_argument("--mpc_horizon",   type=float, default=3.0,
                   help="MPC planning horizon in seconds.")
    p.add_argument("--mpc_samples",   type=int,   default=512,
                   help="Number of candidate control sequences per MPC step.")
    p.add_argument("--mpc_mode",      type=str,   default="MPPI",
                   choices=["MPC", "MPPI"],
                   help="Optimisation mode: greedy best-sample (MPC) or weighted mean (MPPI).")
    p.add_argument("--mpc_lambda",    type=float, default=0.05,
                   help="MPPI temperature λ (lower = more greedy).")
    p.add_argument("--mpc_pursuer", action="store_true",
                   help="Replace the learned pursuer control with a sampling-based MPC agent "
                        "that chases the evader while avoiding obstacles and self-collision.")
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


def load_final_nets(cfg, dyn, rngs: nnx.Rngs, ckpt_dir: str):
    """Load value and policy networks from the final (highest) step checkpoint.

    A single checkpoint contains the full time-conditioned V(x,t) and policy,
    so there is no need to load every intermediate step separately — just query
    with the desired t value.

    Returns
    -------
    value_net  : nnx.Module
    policy_net : nnx.Module
    steps      : sorted list of ints (all available step indices)
    final_step : int
    """
    steps = []
    for name in os.listdir(ckpt_dir):
        if name.startswith("step_") and os.path.isdir(os.path.join(ckpt_dir, name)):
            try:
                steps.append(int(name.split("_")[-1]))
            except ValueError:
                pass
    if not steps:
        raise FileNotFoundError(f"No step_NNN dirs in {ckpt_dir}")
    steps = sorted(steps)
    final_step = steps[-1]

    value_net = load_value_net(cfg, dyn, rngs, ckpt_dir, final_step)
    policy_net = load_policy_net(cfg, dyn, rngs, ckpt_dir, final_step)
    log.info("Loaded final checkpoint: step %d  (T=%.3f s)",
             final_step, final_step * cfg.GAME.TIME.DT)
    return value_net, policy_net, steps, final_step


def load_brt_nets(brt_dir: str, dyn, rngs: nnx.Rngs):
    """Load BRT value and policy nets from a separate BRT run directory.

    Returns
    -------
    brt_value_net  : nnx.Module
    brt_policy_net : nnx.Module
    brt_t_max      : float  — full training horizon of the BRT checkpoint
    """
    brt_cfg = load_cfg(brt_dir)
    ckpt_dir = os.path.join(brt_dir, brt_cfg.IO.CKPT_DIRNAME)
    step = find_latest_step(ckpt_dir)
    brt_value_net = load_value_net(brt_cfg, dyn, rngs, ckpt_dir, step)
    brt_policy_net = load_policy_net(brt_cfg, dyn, rngs, ckpt_dir, step)
    brt_t_max = brt_cfg.GAME.TIME.DT * step
    log.info("Loaded BRT nets: step %d  (T=%.3f s)", step, brt_t_max)
    return brt_value_net, brt_policy_net, brt_t_max


# ---------------------------------------------------------------------------
# Fused, cached per-step decision logic (pursuer window search + bang-bang
# vote for both u and d): fixed-shape masking over the whole t_all array in
# one nnx.jit-compiled call, cached per (value_net, policy_net, dyn).
# ---------------------------------------------------------------------------

_pe_step_jit_cache: dict = {}


def _get_pe_step_jit(value_net, policy_net, dyn):
    key = (id(value_net), id(policy_net), id(dyn))
    fn = _pe_step_jit_cache.get(key)
    if fn is None:
        @nnx.jit
        def pe_step(v_net, p_net, x_cur, t_all):
            K = t_all.shape[0]
            T = t_all[-1]

            x_batch = jnp.broadcast_to(x_cur, (K, x_cur.shape[-1]))
            V_all   = v_net(x_batch, t_all)["V"]

            cond            = V_all <= 0.0
            any_hit         = jnp.any(cond)
            first_idx       = jnp.argmax(cond)     # first True (0 if none)
            t_mid_catchable = t_all[first_idx]
            t_hi_catchable  = jnp.where(t_mid_catchable <= 0.5, 1.0,
                                        jnp.minimum(t_mid_catchable + 0.5, T))

            t_mid = jnp.where(any_hit, t_mid_catchable, T - 1.0)
            t_hi  = jnp.where(any_hit, t_hi_catchable,  T)

            in_window = (t_all >= t_mid) & (t_all <= t_hi)
            mask_f    = in_window.astype(dyn.dtype)
            # Fallback to the last available step when the window is empty
            # (mirrors the original's `window_steps = all_steps[-1:]`).
            empty     = jnp.sum(mask_f) == 0
            last_mask = jnp.zeros_like(mask_f).at[-1].set(1.0)
            mask_f    = jnp.where(empty, last_mask, mask_f)
            count     = jnp.sum(mask_f)

            nn_x   = dyn.nn_inputs(x_cur)
            nn_x_b = jnp.broadcast_to(nn_x, (K, nn_x.shape[-1]))
            out    = p_net(nn_x_b, t_all)
            u_batch, d_batch = decode_actions(out, dyn)

            def _vote(batch, dim):
                if batch is None:
                    return jnp.zeros((1, dim), dtype=dyn.dtype)
                mean = jnp.sum(batch * mask_f[:, None], axis=0, keepdims=True) / count
                sign = jnp.sign(mean)
                return jnp.where(sign == 0, 1.0, sign)

            u = _vote(u_batch, dyn.control_dim) * dyn.u_max.reshape(1, -1)
            d = _vote(d_batch, dyn.disturb_dim) * dyn.d_max.reshape(1, -1)
            return u, d, t_mid, t_hi

        fn = pe_step
        _pe_step_jit_cache[key] = fn
    return fn


# ---------------------------------------------------------------------------
# MPC evader / pursuer — sampling-based MPPI rolled out with the REAL
# dynamics (dyn.f / dyn._sdf_obstacle / dyn._sdf_boundary), not a hand-rolled
# numpy mirror. This both fixes the circle-vs-polygon obstacle mismatch and
# avoids a physics-duplication drift risk. Other robots' substates are held
# fixed for the duration of the rollout (matches the original's "cost is
# w.r.t. current pursuer/evader positions" design — it never re-simulated
# the other side's future motion either).
# ---------------------------------------------------------------------------

_evader_mpc_jit_cache: dict = {}


def _get_evader_mpc_jit(dyn, dt: float, H: int, n_samples: int, lam: float, mode: str):
    key = (id(dyn), dt, H, n_samples, lam, mode)
    fn = _evader_mpc_jit_cache.get(key)
    if fn is not None:
        return fn

    evader_idx = dyn.num_pursuers
    sl = slice(4 * evader_idx, 4 * evader_idx + 4)
    d_max = dyn.d_max

    @jax.jit
    def evader_mpc(x_full, controls_warm, rng_key):
        eps = jax.random.normal(rng_key, (n_samples, H, 2), dtype=dyn.dtype) * (d_max * 0.5)
        eps = eps.at[0].set(0.0)   # nominal candidate
        perturbed = jnp.clip(controls_warm[None] + eps, -d_max, d_max)   # (N, H, 2)

        x_frozen = jnp.broadcast_to(x_full, (n_samples, dyn.state_dim))
        own0     = x_frozen[:, sl]
        u_full   = jnp.zeros((n_samples, dyn.control_dim), dtype=dyn.dtype)

        def step(own, d_h):
            x_join       = x_frozen.at[:, sl].set(own)
            own_next     = own + dt * dyn.f(x_join, u_full, d_h)[:, sl]
            own_wrapped  = dyn.wrap_state(x_frozen.at[:, sl].set(own_next))[:, sl]
            return own_wrapped, own_wrapped

        _, owns_seq = jax.lax.scan(step, own0, jnp.swapaxes(perturbed, 0, 1))
        traj = jnp.swapaxes(jnp.concatenate([own0[None], owns_seq], axis=0), 0, 1)  # (N,H+1,4)

        pursuers_xy = jnp.stack(
            [x_full[0, 4 * j: 4 * j + 2] for j in range(dyn.num_pursuers)], axis=0)  # (P,2)
        xy = traj[..., :2]
        dists    = jnp.linalg.norm(xy[:, :, None, :] - pursuers_xy[None, None, :, :], axis=-1)
        min_dist = dists.min(axis=-1)
        reward   = min_dist.mean(axis=-1)   # (N,)

        flat = xy.reshape(-1, 2)
        penalty = jnp.sum(jnp.minimum(dyn._sdf_boundary(flat).reshape(n_samples, H + 1), 0.0) ** 2,
                          axis=-1) * 2.0
        for j in range(len(dyn.obstacles)):
            d_obs = dyn._sdf_obstacle(flat, j).reshape(n_samples, H + 1)
            penalty = penalty + jnp.sum(jnp.minimum(d_obs, 0.0) ** 2, axis=-1) * 2.0

        costs = -reward + penalty   # (N,)  lower is better

        if mode == "MPC":
            best_seq = perturbed[jnp.argmin(costs)]
        else:
            w = jax.nn.softmax(-costs / lam)
            best_seq = jnp.sum(perturbed * w[:, None, None], axis=0)

        next_warm = jnp.concatenate([best_seq[1:], jnp.zeros((1, 2), dtype=dyn.dtype)], axis=0)
        return best_seq[0], next_warm

    _evader_mpc_jit_cache[key] = evader_mpc
    return evader_mpc


class MPC_Evader:
    """Sampling-based MPC/MPPI controller for the evader.

    Call ``get_action(x_full)`` each sim step to get a (1, disturb_dim)
    disturbance in the same format as the learned disturbance.
    """

    def __init__(self, dyn, dt: float, horizon: float,
                 num_samples: int = 512, mode: str = "MPPI",
                 lambda_: float = 0.05, key: jax.Array = None):
        self.dyn = dyn
        self._H  = max(1, int(round(horizon / dt)))
        self._controls = jnp.zeros((self._H, dyn.disturb_dim), dtype=dyn.dtype)
        self.key = key if key is not None else jax.random.PRNGKey(0)
        self._step_fn = _get_evader_mpc_jit(dyn, dt, self._H, num_samples, lambda_, mode)

    def get_action(self, x_full: jax.Array) -> jax.Array:
        self.key, sub = jax.random.split(self.key)
        action, self._controls = self._step_fn(x_full, self._controls, sub)
        return action[None, :]


_pursuer_mpc_jit_cache: dict = {}


def _get_pursuer_mpc_jit(dyn, dt: float, H: int, n_samples: int, lam: float, mode: str,
                         pursuer_idx: int):
    key = (id(dyn), dt, H, n_samples, lam, mode, pursuer_idx)
    fn = _pursuer_mpc_jit_cache.get(key)
    if fn is not None:
        return fn

    sl      = slice(4 * pursuer_idx, 4 * pursuer_idx + 4)
    uc      = slice(2 * pursuer_idx, 2 * pursuer_idx + 2)
    u_max_j = dyn.u_max[uc]
    evader_idx = dyn.num_pursuers

    @jax.jit
    def pursuer_mpc(x_full, controls_warm, rng_key):
        eps = jax.random.normal(rng_key, (n_samples, H, 2), dtype=dyn.dtype) * (u_max_j * 0.5)
        eps = eps.at[0].set(0.0)
        perturbed = jnp.clip(controls_warm[None] + eps, -u_max_j, u_max_j)

        x_frozen = jnp.broadcast_to(x_full, (n_samples, dyn.state_dim))
        own0     = x_frozen[:, sl]
        d_full   = jnp.zeros((n_samples, dyn.disturb_dim), dtype=dyn.dtype)

        def step(own, u_h):
            x_join      = x_frozen.at[:, sl].set(own)
            u_full      = jnp.zeros((n_samples, dyn.control_dim), dtype=dyn.dtype).at[:, uc].set(u_h)
            own_next    = own + dt * dyn.f(x_join, u_full, d_full)[:, sl]
            own_wrapped = dyn.wrap_state(x_frozen.at[:, sl].set(own_next))[:, sl]
            return own_wrapped, own_wrapped

        _, owns_seq = jax.lax.scan(step, own0, jnp.swapaxes(perturbed, 0, 1))
        traj = jnp.swapaxes(jnp.concatenate([own0[None], owns_seq], axis=0), 0, 1)  # (N,H+1,4)

        evader_xy = x_full[0, 4 * evader_idx: 4 * evader_idx + 2]
        xy   = traj[..., :2]
        dist = jnp.linalg.norm(xy - evader_xy[None, None, :], axis=-1)   # (N, H+1)
        cost = dist[:, -1] + dist.mean(axis=-1)

        flat = xy.reshape(-1, 2)
        cost = cost + jnp.sum(jnp.minimum(dyn._sdf_boundary(flat).reshape(n_samples, H + 1), 0.0) ** 2,
                              axis=-1) * 10.0
        for j in range(len(dyn.obstacles)):
            d_obs = dyn._sdf_obstacle(flat, j).reshape(n_samples, H + 1)
            cost = cost + jnp.sum(jnp.minimum(d_obs, 0.0) ** 2, axis=-1) * 10.0

        if mode == "MPC":
            best_seq = perturbed[jnp.argmin(cost)]
        else:
            w = jax.nn.softmax(-cost / lam)
            best_seq = jnp.sum(perturbed * w[:, None, None], axis=0)

        next_warm = jnp.concatenate([best_seq[1:], jnp.zeros((1, 2), dtype=dyn.dtype)], axis=0)
        return best_seq[0], next_warm

    _pursuer_mpc_jit_cache[key] = pursuer_mpc
    return pursuer_mpc


class MPC_Pursuer:
    """Each pursuer runs an independent MPPI (2D control: omega + accel) to
    chase the evader. ``get_action(x_full)`` returns a (1, control_dim)
    tensor covering every pursuer."""

    def __init__(self, dyn, dt: float, horizon: float,
                 num_samples: int = 512, mode: str = "MPPI",
                 lambda_: float = 0.05, key: jax.Array = None):
        self.dyn = dyn
        self._H  = max(1, int(round(horizon / dt)))
        self.key = key if key is not None else jax.random.PRNGKey(1)
        self._controls = [jnp.zeros((self._H, 2), dtype=dyn.dtype)
                          for _ in range(dyn.num_pursuers)]
        self._step_fns = [
            _get_pursuer_mpc_jit(dyn, dt, self._H, num_samples, lambda_, mode, j)
            for j in range(dyn.num_pursuers)
        ]

    def get_action(self, x_full: jax.Array) -> jax.Array:
        actions = []
        for j, step_fn in enumerate(self._step_fns):
            self.key, sub = jax.random.split(self.key)
            action, self._controls[j] = step_fn(x_full, self._controls[j], sub)
            actions.append(action)
        return jnp.concatenate(actions, axis=0)[None, :]   # (1, control_dim)


# ---------------------------------------------------------------------------
# Single-trajectory rollout
# ---------------------------------------------------------------------------

def simulate_one(
    x0_1: jax.Array,   # (1, state_dim)
    dyn,
    value_net,
    policy_net,
    all_steps: list,
    cfg,
    sim_time: float,
    brt_value_net=None,
    brt_policy_net=None,
    brt_t_max: float = None,
    mpc_evader: "MPC_Evader | None" = None,
    mpc_pursuer: "MPC_Pursuer | None" = None,
):
    """Roll out one trajectory for sim_time seconds.

    If brt_value_net / brt_policy_net / brt_t_max are provided, a
    least-restrictive safety filter is applied at every step: the BRT
    controller overrides the pursuer control only when the probe rollout
    predicts an unsafe next state (same logic as engine.py).

    Returns
    -------
    traj       : (K+1, state_dim) array
    window_log : list of (t_mid, t_hi) floats, length K (one per step)
    """
    dt = cfg.GAME.TIME.DT
    T  = all_steps[-1] * dt
    m  = cfg.GAME.TIME.NUM_SUBSTEPS
    K  = int(round(sim_time / dt))
    use_brt = (brt_value_net is not None
               and brt_policy_net is not None
               and brt_t_max is not None)

    t_all = jnp.asarray([k * dt for k in all_steps], dtype=dyn.dtype)

    xs = [x0_1]
    window_log = []
    x_cur = x0_1

    pe_step = _get_pe_step_jit(value_net, policy_net, dyn)

    for _ in range(K):
        u_vote, d_vote, t_mid, t_hi = pe_step(value_net, policy_net, x_cur, t_all)
        window_log.append((float(t_mid), float(t_hi)))

        u = mpc_pursuer.get_action(x_cur) if mpc_pursuer is not None else u_vote
        d = mpc_evader.get_action(x_cur)  if mpc_evader  is not None else d_vote

        # ---- Step (with optional BRT least-restrictive filter) ------------
        if use_brt:
            t_brt  = jnp.full((1,), brt_t_max, dtype=dyn.dtype)
            V_brt  = brt_value_net(x_cur, t_brt)["V"]
            t_brat = jnp.full((1,), T, dtype=dyn.dtype)
            V_brat = value_net(x_cur, t_brat)["V"]
            unsafe = bool(V_brt[0] <= 0.3) and bool(V_brat[0] > 0.0)
            if unsafe:
                brt_out = brt_policy_net(dyn.nn_inputs(x_cur), t_brt)
                u, _ = decode_actions(brt_out, dyn)

        x_cur, _ = rollout_with_intermediate_checks(
            x=x_cur, u=u, d=d,
            dyn=dyn, dt=dt, m=m,
            problem_type="BRAT",
        )
        xs.append(x_cur)

    return jnp.concatenate(xs, axis=0), window_log   # (K+1, state_dim), K windows


# ---------------------------------------------------------------------------
# BRAT cost over a trajectory
# ---------------------------------------------------------------------------

def brat_cost(traj: jax.Array, dyn) -> float:
    """Compute BRAT cost for a single (K+1, state_dim) trajectory."""
    return float(dyn.cost_fn(traj[None])[0])


# ---------------------------------------------------------------------------
# Teacher control / disturbance slice visualisation
# ---------------------------------------------------------------------------

def plot_teacher_slices(
    value_net,
    dyn,
    t: float,
    state_slices: list = None,
    x_axis_idx: int = None,
    y_axis_idx: int = None,
    resolution: int = 60,
    out_path: str = None,
):
    """Plot teacher (optimal) control and disturbance over a 2D state slice.

    For each action dimension the teacher sign is determined by the value
    gradient: pursuer control minimises H so u*_j = -sign(∂V/∂θ_pj),
    evader disturbance maximises H so d*_i = +sign(∂V/∂(affected index)).
    The V=0 level-set is overlaid as a black contour.
    """
    cfg_plot = dyn.plot_config()
    if state_slices is None:
        state_slices = cfg_plot["state_slices"]
    if x_axis_idx is None:
        x_axis_idx = cfg_plot["x_axis_idx"]
    if y_axis_idx is None:
        y_axis_idx = cfg_plot["y_axis_idx"]

    dtype = dyn.dtype
    R = resolution

    x_lo, x_hi = float(dyn.state_low[x_axis_idx]), float(dyn.state_high[x_axis_idx])
    y_lo, y_hi = float(dyn.state_low[y_axis_idx]), float(dyn.state_high[y_axis_idx])

    xs = jnp.linspace(x_lo, x_hi, R, dtype=dtype)
    ys = jnp.linspace(y_lo, y_hi, R, dtype=dtype)
    gx, gy = jnp.meshgrid(xs, ys, indexing="ij")   # (R, R)

    base = jnp.broadcast_to(jnp.asarray(state_slices, dtype=dtype), (R * R, dyn.state_dim))
    base = base.at[:, x_axis_idx].set(gx.reshape(-1))
    base = base.at[:, y_axis_idx].set(gy.reshape(-1))

    t_arr = jnp.full((R * R,), t, dtype=dtype)

    V_vals = value_net(base, t_arr)["V"]                                  # (R*R,)
    grad_V = jax.grad(lambda x: value_net(x, t_arr)["V"].sum())(base)     # (R*R, state_dim)

    V_grid = np.asarray(V_vals).reshape(R, R)

    # Teacher actions (bang-bang, sign only):
    #   u* minimises H  → u*_j = -sign(∇V[θ_pj index]) * u_max_j
    #   d* maximises H  → d*_i = +sign(∇V[affected index]) * d_max_i
    #
    # For PursuitEvasion:
    #   ω_p1 → θ_p1 at state index 4*0+2
    #   ω_p2 → θ_p2 at state index 4*1+2
    #   ω_e  → θ_e  at state index 2
    #   a_e  → v_e  at state index 3
    u_signs = jnp.stack([
        -jnp.sign(grad_V[:, 4 * j + 2])
        for j in range(dyn.num_pursuers)
    ], axis=-1)   # (R*R, num_pursuers)

    d_signs = jnp.stack([
        jnp.sign(grad_V[:, 2]),   # ω_e
        jnp.sign(grad_V[:, 3]),   # a_e
    ], axis=-1)   # (R*R, 2)

    u_grids = np.asarray(u_signs).reshape(R, R, dyn.num_pursuers)
    d_grids = np.asarray(d_signs).reshape(R, R, 2)

    labels = cfg_plot.get(
        "state_labels", [f"dim {i}" for i in range(dyn.state_dim)])
    x_label = labels[x_axis_idx]
    y_label = labels[y_axis_idx]
    x_np = np.asarray(xs)
    y_np = np.asarray(ys)

    n_actions = dyn.control_dim + dyn.disturb_dim
    fig, axes = plt.subplots(1, n_actions, figsize=(4 * n_actions, 4))
    if n_actions == 1:
        axes = [axes]

    action_grids = ([u_grids[..., j] for j in range(dyn.num_pursuers)]
                    + [d_grids[..., i] for i in range(2)])
    action_titles = ([f"u: ω_p{j+1}  (pursuer {j+1})" for j in range(dyn.num_pursuers)]
                     + ["d: ω_e  (evader turn)", "d: a_e  (evader accel)"])

    for ax, grid, title in zip(axes, action_grids, action_titles):
        im = ax.imshow(
            grid.T,
            origin="lower",
            extent=[x_lo, x_hi, y_lo, y_hi],
            vmin=-1, vmax=1,
            cmap="RdBu",
            aspect="auto",
        )
        ax.contour(x_np, y_np, V_grid.T, levels=[0],
                   colors="black", linewidths=1.5)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="sign")

    fig.suptitle(f"Teacher slices  t={t:.2f}  "
                 f"({x_label} vs {y_label})", fontsize=10)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved teacher slice plot → %s", out_path)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def animate_trajectory(traj: jax.Array, dyn, title: str, fps: int, out_path: str,
                       sim_dt: float = 0.1, window_log: list = None,
                       value_net=None, final_T: float = None,
                       brt_value_net=None, brt_t_max: float = None):
    """Save an MP4 animation of one pursuit-evasion trajectory."""
    traj_np = np.asarray(traj)
    K1 = traj_np.shape[0]
    half = dyn.half

    traj_j = jnp.asarray(traj_np, dtype=dyn.dtype)
    l_target_np = np.asarray(dyn.target_l(traj_j))   # (K1,)
    l_avoid_np  = np.asarray(dyn.avoid_l(traj_j))     # (K1,)
    pred_V_np = None
    if value_net is not None and final_T is not None:
        t_final = jnp.full((K1,), final_T, dtype=dyn.dtype)
        pred_V_np = np.asarray(value_net(traj_j, t_final)["V"])   # (K1,)
    pred_V_brt_np = None
    if brt_value_net is not None and brt_t_max is not None:
        t_brt = jnp.full((K1,), brt_t_max, dtype=dyn.dtype)
        pred_V_brt_np = np.asarray(brt_value_net(traj_j, t_brt)["V"])   # (K1,)
    times = np.arange(K1) * sim_dt

    fig, (ax, ax_l) = plt.subplots(
        2, 1, figsize=(6, 8),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(pad=2.0)

    # ---- Field axes --------------------------------------------------------
    ax.set_xlim(-half - 0.3, half + 0.3)
    ax.set_ylim(-half - 0.3, half + 0.3)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    # Static elements
    ax.add_patch(mpatches.Rectangle(
        (-half, -half), dyn.field_size, dyn.field_size,
        linewidth=2, edgecolor="black", facecolor="none"))
    for verts in dyn.obstacle_vertices:
        ax.add_patch(mpatches.Polygon(
            np.asarray(verts), edgecolor="saddlebrown", facecolor="saddlebrown", alpha=0.5))

    # Trajectory ghost lines
    ax.plot(traj_np[:, 0], traj_np[:, 1], color="blue",  alpha=0.2, lw=1)
    for j in range(dyn.num_pursuers):
        base = 4 * j
        ax.plot(traj_np[:, base], traj_np[:, base + 1],
                color="red", alpha=0.2, lw=1)

    # Live patches
    evader_body = mpatches.Circle((0, 0), dyn.robot_radius,
                                  color="blue", alpha=0.85, zorder=3)
    evader_arrow = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                               arrowprops=dict(arrowstyle="->", color="blue", lw=1.5))
    ax.add_patch(evader_body)

    pursuer_bodies = []
    pursuer_catches = []
    pursuer_arrows = []
    for _ in range(dyn.num_pursuers):
        body = mpatches.Circle((0, 0), dyn.robot_radius,
                               color="red", alpha=0.85, zorder=3)
        catch = mpatches.Circle((0, 0), dyn.catch_radius,
                                edgecolor="red", facecolor="none",
                                linestyle="--", alpha=0.4, zorder=2)
        arrow = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                            arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
        ax.add_patch(body)
        ax.add_patch(catch)
        pursuer_bodies.append(body)
        pursuer_catches.append(catch)
        pursuer_arrows.append(arrow)

    time_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=9,
                        verticalalignment="top")

    # ---- Cost axes ---------------------------------------------------------
    ax_l.plot(times, l_target_np, color="blue",   lw=1.5, label="target_l")
    ax_l.plot(times, l_avoid_np,  color="red",    lw=1.5, label="avoid_l")
    if pred_V_np is not None:
        ax_l.plot(times, pred_V_np, color="green", lw=1.5,
                  linestyle="--", label=f"V(x, T={final_T:.1f})")
    if pred_V_brt_np is not None:
        ax_l.plot(times, pred_V_brt_np, color="orange", lw=1.5,
                  linestyle="--", label=f"V_brt(x, T={brt_t_max:.1f})")
    ax_l.axhline(0, color="black", lw=0.8, linestyle="--")
    ax_l.set_xlabel("t (s)")
    ax_l.set_ylabel("value")
    ax_l.legend(fontsize=8, loc="upper right")
    ax_l.grid(True, alpha=0.3)
    cursor = ax_l.axvline(0.0, color="gray", lw=1.2, linestyle=":")

    def _update(frame):
        s = traj_np[frame]
        xe, ye, the = s[0], s[1], s[2]
        al = dyn.robot_radius * 1.8

        evader_body.center = (xe, ye)
        evader_arrow.xy = (xe + al * np.cos(the), ye + al * np.sin(the))
        evader_arrow.xyann = (xe, ye)

        for j in range(dyn.num_pursuers):
            base = 4 * j
            xp, yp, thp = s[base], s[base + 1], s[base + 2]
            pursuer_bodies[j].center = (xp, yp)
            pursuer_catches[j].center = (xp, yp)
            pursuer_arrows[j].xy = (
                xp + al * np.cos(thp), yp + al * np.sin(thp))
            pursuer_arrows[j].xyann = (xp, yp)

        win_str = ""
        if window_log and frame > 0:
            t_lo, t_hi = window_log[frame - 1]
            win_str = f"  win=[{t_lo:.1f},{t_hi:.1f}]"
        time_text.set_text(f"t = {frame * sim_dt:.2f} s{win_str}")

        cursor.set_xdata([frame * sim_dt, frame * sim_dt])

        return ([evader_body, evader_arrow, time_text, cursor]
                + pursuer_bodies + pursuer_catches + pursuer_arrows)

    ani = animation.FuncAnimation(
        fig, _update, frames=K1, interval=int(1000 / fps), blit=False)

    writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
    ani.save(out_path, writer=writer)
    plt.close(fig)
    log.info("Saved animation → %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = build_argparser().parse_args()
    key = jax.random.PRNGKey(args.seed)
    random.seed(args.seed)
    rngs = nnx.Rngs(args.seed)

    cfg = load_cfg(args.run_dir)
    dyn = make_dynamics(cfg)

    ckpt_dir = os.path.join(args.run_dir, cfg.IO.CKPT_DIRNAME)
    value_net, policy_net, all_steps, _ = load_final_nets(cfg, dyn, rngs, ckpt_dir)

    brt_value_net, brt_policy_net, brt_t_max = None, None, None
    if args.brt_dir:
        brt_value_net, brt_policy_net, brt_t_max = load_brt_nets(
            args.brt_dir, dyn, rngs)

    out_dir = args.out_dir or os.path.join(args.run_dir, "sim")
    os.makedirs(out_dir, exist_ok=True)

    dt = cfg.GAME.TIME.DT
    T = all_steps[-1] * dt

    # --- Sample N valid initial states (oversample then filter) -------------
    valid, attempts = [], 0
    while len(valid) < args.N:
        key, sk = jax.random.split(key)
        batch = sample_uniform_states(sk, dyn, args.N * 4)
        l_t = dyn.target_l(batch)   # (B,)  <=0 means already caught
        l_a = dyn.avoid_l(batch)    # (B,)  >=0 means pursuer already in collision
        keep = (l_t > 0) & (l_a < 0)
        if brt_value_net is not None:
            t_brt_batch = jnp.full((batch.shape[0],), brt_t_max, dtype=dyn.dtype)
            V_brt_batch = brt_value_net(batch, t_brt_batch)["V"]
            keep = keep & (V_brt_batch >= 0.3)
        valid.append(batch[keep])
        attempts += 1
        if attempts > 50:
            raise RuntimeError(
                "Could not sample enough valid initial states after 50 attempts.")
    x0_all = jnp.concatenate(valid, axis=0)[: args.N]
    log.info("Sampled %d valid initial states (no terminal conditions at t=0)", args.N)

    trajs = []
    window_logs = []
    costs = []
    log.info("Simulating %d trajectories × %.1f s  (T_train=%.2f s, dt=%.3f s)",
             args.N, args.sim_time, T, dt)

    for n in range(args.N):
        x0_1 = x0_all[n:n+1]
        log.info("  trajectory %d / %d", n + 1, args.N)
        key, ek, pk = jax.random.split(key, 3)
        mpc_agent = (
            MPC_Evader(
                dyn=dyn, dt=dt,
                horizon=args.mpc_horizon,
                num_samples=args.mpc_samples,
                mode=args.mpc_mode,
                lambda_=args.mpc_lambda,
                key=ek,
            ) if args.mpc_evader else None
        )
        mpc_pursuer_agent = (
            MPC_Pursuer(
                dyn=dyn, dt=dt,
                horizon=args.mpc_horizon,
                num_samples=args.mpc_samples,
                mode=args.mpc_mode,
                lambda_=args.mpc_lambda,
                key=pk,
            ) if args.mpc_pursuer else None
        )
        traj, wlog = simulate_one(
            x0_1=x0_1, dyn=dyn,
            value_net=value_net, policy_net=policy_net,
            all_steps=all_steps, cfg=cfg,
            sim_time=args.sim_time,
            brt_value_net=brt_value_net,
            brt_policy_net=brt_policy_net,
            brt_t_max=brt_t_max,
            mpc_evader=mpc_agent,
            mpc_pursuer=mpc_pursuer_agent,
        )
        c = brat_cost(traj, dyn)
        trajs.append(traj)
        window_logs.append(wlog)
        costs.append(c)
        log.info("    BRAT cost = %.4f  (%s)",
                 c, "SUCCESS" if c <= 0 else "failure")

    costs_np = np.array(costs)
    success_mask = costs_np <= 0.0
    success_rate = success_mask.mean()
    log.info("=" * 50)
    log.info("Success rate: %.1f%%  (%d / %d)",
             100 * success_rate, success_mask.sum(), args.N)
    log.info("BRAT cost  — mean: %.4f  min: %.4f  max: %.4f",
             costs_np.mean(), costs_np.min(), costs_np.max())

    # --- Pick one success and one failure to animate -----------------------
    success_idxs = np.where(success_mask)[0].tolist()
    failure_idxs = np.where(~success_mask)[0].tolist()

    if args.animate_all:
        to_animate = [
            (i, "SUCCESS" if success_mask[i]
             else "FAILURE", f"traj{i:03d}.mp4")
            for i in range(args.N)
        ]
    else:
        to_animate = []
        if success_idxs:
            idx = random.choice(success_idxs)
            to_animate.append((idx, "SUCCESS", f"success_traj{idx}.mp4"))
        else:
            log.warning("No successful trajectories to animate.")

        if failure_idxs:
            idx = random.choice(failure_idxs)
            to_animate.append((idx, "FAILURE", f"failure_traj{idx}.mp4"))
        else:
            log.warning("No failed trajectories to animate.")

    for (idx, label, fname) in to_animate:
        title = f"Pursuit-Evasion  [{label}]  cost={costs[idx]:.4f}"
        out_path = os.path.join(out_dir, fname)
        animate_trajectory(
            traj=trajs[idx], dyn=dyn,
            title=title, fps=args.fps, out_path=out_path,
            sim_dt=dt, window_log=window_logs[idx],
            value_net=value_net, final_T=T,
            brt_value_net=brt_value_net, brt_t_max=brt_t_max,
        )

    # --- Summary plot of costs ---------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(args.N), costs_np,
           color=["steelblue" if s else "salmon" for s in success_mask])
    ax.axhline(0, color="black", lw=1, linestyle="--")
    ax.set_xlabel("Trajectory index")
    ax.set_ylabel("BRAT cost")
    ax.set_title(f"Simulation costs  (success rate {100*success_rate:.1f}%)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sim_costs.png"), dpi=120)
    plt.close(fig)
    log.info("Saved cost summary → %s", os.path.join(out_dir, "sim_costs.png"))


if __name__ == "__main__":
    main()
