"""Closed-loop rollout of the learned policy, and rollout-derived scores.

These produce the "true" side of every comparison in evaluation: the value
actually achieved by simulating the learned controller forward, as opposed to
what the value net predicts (:func:`predicted_value_at_initial_time`).
"""

import logging

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from scipy.interpolate import RegularGridInterpolator

from reachability.dynamics import Dynamics
from reachability.training.functional import rollout_with_intermediate_checks, substeps_for
from reachability.data.sampling import sample_uniform_states
from configs import PROJECT_NAME
from utils import decode_actions
from .ground_truth import gt_nd_value

log = logging.getLogger(f"{PROJECT_NAME}.{__name__}")


def rollout_trajectories(
    dyn: Dynamics,
    value_net: nnx.Module,
    policy_net: nnx.Module,
    t_final: float,
    dt: float,
    cfg,
    x0: jax.Array,
    policy_steps_per_dt: int = 10,
    action_fn=None,
) -> tuple:
    """Roll out trajectories using the explicit policy and compute true HJ scores.

    Args:
        action_fn: Optional ``(x, t) -> (u, d)`` override for the action source.
            Defaults to the policy net. Verification passes a probing-based
            adversary here (see ``reachability.eval.verification``) so the
            worst-case disturbance is used rather than the learned one, while
            reusing this function's BRT/BRS/BRAT scoring rather than
            duplicating it.

    Returns
    -------
    (true_score, Xtraj) where:
        true_score: (N,) array of true HJ scores.
        Xtraj:      (N, K+1, state_dim) trajectory array.
    """
    dtype = dyn.dtype
    N  = x0.shape[0]
    K  = int(round(t_final / dt))
    sub_dt = dt / policy_steps_per_dt
    problem_type = cfg.GAME.PROBLEM_TYPE
    num_substeps = substeps_for(cfg.GAME.TIME.DT, cfg)

    def _make_step_stats() -> dict:
        if problem_type == "BRAT":
            return {
                "lT_min": jnp.full((N,), jnp.inf,  dtype=dtype),
                "lA_max": jnp.full((N,), -jnp.inf, dtype=dtype),
            }
        return {"l_min": jnp.full((N,), jnp.inf, dtype=dtype)}

    def _accumulate_step_stats(step_stats: dict, sub_stats: dict) -> None:
        if problem_type == "BRAT":
            step_stats["lT_min"] = jnp.minimum(step_stats["lT_min"], sub_stats["lT_min"])
            step_stats["lA_max"] = jnp.maximum(step_stats["lA_max"], sub_stats["lA_max"])
        else:
            step_stats["l_min"] = jnp.minimum(step_stats["l_min"], sub_stats["l_min"])

    def _step(current_x, i):
        time_to_go = (K - i).astype(dtype) * dt
        step_stats = _make_step_stats()

        for sub_i in range(policy_steps_per_dt):
            t_remaining = time_to_go - sub_i * sub_dt
            perceived_t = jnp.full((N,), t_remaining, dtype=dtype)
            if action_fn is None:
                u_star, d_star = decode_actions(
                    policy_net(dyn.nn_inputs(current_x), perceived_t, shared_time=True), dyn)
            else:
                u_star, d_star = action_fn(current_x, perceived_t)
            current_x, sub_stats = rollout_with_intermediate_checks(
                x=current_x, u=u_star, d=d_star,
                dyn=dyn, dt=sub_dt,
                m=num_substeps,
                problem_type=problem_type,
            )
            _accumulate_step_stats(step_stats, sub_stats)

        return current_x, (current_x, step_stats)

    # A single fused lax.scan replaces the K * policy_steps_per_dt * num_substeps
    # sequence of eagerly-dispatched calls with one compiled, on-device loop.
    _, (xs_rest, seg_stats) = jax.lax.scan(_step, x0, jnp.arange(K))
    traj = Xtraj = jnp.concatenate(
        [x0[:, None, :], jnp.swapaxes(xs_rest, 0, 1)], axis=1
    )  # (N, K+1, state_dim)

    if problem_type == "BRAT":
        if not hasattr(dyn, "cost_fn"):
            raise AttributeError(
                f"Dynamics class {type(dyn).__name__} does not implement cost_fn, "
                "which is required for BRAT evaluation."
            )
        return dyn.cost_fn(Xtraj), traj

    Xflat  = Xtraj.reshape(-1, dyn.state_dim)
    t_zero = jnp.zeros(Xflat.shape[0], dtype=dtype)

    vterm_flat = value_net(Xflat, t_zero, shared_time=True)["V"].reshape(N, K + 1)
    suffix_min_vt = jax.lax.associative_scan(jnp.minimum, vterm_flat, axis=1, reverse=True)

    l_states = dyn.l(Xflat).reshape(N, K + 1)
    l_seg = jnp.swapaxes(seg_stats["l_min"], 0, 1)  # (N, K) — non-BRAT paths always populate "l_min"

    combined_l = l_states.at[:, :K].set(jnp.minimum(l_states[:, :K], l_seg))
    suffix_min_l = jax.lax.associative_scan(jnp.minimum, combined_l, axis=1, reverse=True)

    if problem_type == "BRS":
        return suffix_min_vt[:, 0], traj

    # BRT
    return jnp.minimum(suffix_min_l[:, 0], suffix_min_vt[:, 0]), traj


def predicted_value_at_initial_time(
    dyn: Dynamics,
    value_net: nnx.Module,
    t_final: float,
    cfg,
    x: jax.Array,
) -> jax.Array:
    """Evaluate V(x, T) — the network's prediction at the start of the horizon."""
    T = t_final
    t = jnp.full((x.shape[0],), T, dtype=dyn.dtype)
    return value_net(x, t, shared_time=True)["V"]


def rollout_brt_cost(
    policy_net: nnx.Module,
    dyn: Dynamics,
    x0: jax.Array,
    t_final: float,
    dt: float,
    num_substeps: int,
) -> np.ndarray:
    """Roll out the learned policy from x0 and return min_t l(x(t)).

    Returns (B,) numpy array of BRT trajectory cost.
    """
    dtype     = dyn.dtype
    B         = x0.shape[0]
    num_steps = max(1, round(t_final / dt))
    d_zeros   = jnp.zeros((B, dyn.disturb_dim), dtype=dtype)
    u_zeros   = jnp.zeros((B, dyn.control_dim), dtype=dtype)

    def _step(carry, step_i):
        x, l_running = carry
        t_remaining = (num_steps - step_i).astype(dtype) * dt
        t_vec = jnp.full((B,), t_remaining, dtype=dtype)

        out  = policy_net(dyn.nn_inputs(x), t_vec, shared_time=True)
        u, d = decode_actions(out, dyn)
        u = u_zeros if u is None else u
        d = d_zeros if d is None else d

        x, stats  = rollout_with_intermediate_checks(
            x, u, d, dyn, dt, num_substeps, problem_type="BRT"
        )
        l_running = jnp.minimum(l_running, stats["l_min"])
        return (x, l_running), None

    # Fused lax.scan instead of `num_steps` eagerly-dispatched Python-loop calls.
    (_, l_running), _ = jax.lax.scan(_step, (x0, dyn.l(x0)), jnp.arange(num_steps))

    return np.asarray(l_running)


def evaluate_rollout(
    value_net: nnx.Module,
    policy_net: nnx.Module,
    dyn: Dynamics,
    interp: RegularGridInterpolator,
    m_batches: int,
    batch_size: int,
    t_final: float,
    dt: float,
    num_substeps: int,
    key: jax.Array,
) -> dict:
    """Compare rollout BRT cost vs GT and vs value net prediction."""
    dtype = dyn.dtype
    sq_vs_gt,  abs_vs_gt  = [], []
    sq_vs_net, abs_vs_net = [], []
    n_brt_gt, n_brt_rollout, n_brt_net, n_total = 0, 0, 0, 0
    gaps = []

    for i in range(m_batches):
        key, sample_key = jax.random.split(key)
        x     = sample_uniform_states(sample_key, dyn, batch_size)
        t_vec = jnp.full((batch_size,), t_final, dtype=dtype)

        J     = rollout_brt_cost(
            policy_net, dyn, x, t_final, dt, num_substeps
        )
        V_gt  = gt_nd_value(interp, np.asarray(x))
        V_net = np.asarray(value_net(x, t_vec, shared_time=True)["V"])

        sq_vs_gt.append(((J - V_gt)  ** 2).mean())
        abs_vs_gt.append(np.abs(J - V_gt).mean())
        sq_vs_net.append(((J - V_net) ** 2).mean())
        abs_vs_net.append(np.abs(J - V_net).mean())
        gaps.append((J - V_gt).mean())

        n_brt_gt      += int((V_gt  < 0).sum())
        n_brt_rollout += int((J     < 0).sum())
        n_brt_net     += int((V_net < 0).sum())
        n_total       += len(J)

        log.info("  rollout batch %2d/%d: MSE_gt=%.6f  MAE_gt=%.6f  gap=%.6f",
                 i + 1, m_batches, sq_vs_gt[-1], abs_vs_gt[-1], gaps[-1])

    stats = {
        "mse_rollout_vs_gt":  float(np.mean(sq_vs_gt)),
        "mae_rollout_vs_gt":  float(np.mean(abs_vs_gt)),
        "mse_rollout_vs_net": float(np.mean(sq_vs_net)),
        "mae_rollout_vs_net": float(np.mean(abs_vs_net)),
        "mean_gap":           float(np.mean(gaps)),
        "vol_gt":             n_brt_gt      / max(1, n_total),
        "vol_rollout":        n_brt_rollout / max(1, n_total),
        "vol_net":            n_brt_net     / max(1, n_total),
    }

    log.info(
        "=== Rollout  MSE_gt=%.6f  MAE_gt=%.6f  mean_gap=%.6f  "
        "BRT_vol_GT=%.4f  BRT_vol_rollout=%.4f  BRT_vol_net=%.4f ===",
        stats["mse_rollout_vs_gt"], stats["mae_rollout_vs_gt"], stats["mean_gap"],
        stats["vol_gt"], stats["vol_rollout"], stats["vol_net"],
    )
    return stats
