import functools
import logging
import math

import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Tuple

from configs.constants import PROJECT_NAME
from reachability.dynamics import Dynamics

log = logging.getLogger(f"{PROJECT_NAME}.{__name__}")


# ---------------------------------------------------------------------------
# Time discretisation
#
# Three independent quantities, deliberately not derived from one another:
#   DT             -- control decision interval; a property of the system being
#                     certified, not a numerical knob.
#   PROBE_HORIZON  -- finite-difference step for measuring the Hamiltonian;
#                     floored by the value net's error, not by the dynamics.
#   MAX_SUBSTEP    -- integration tolerance; floored by dynamics accuracy and
#                     by the thinnest obstacle the BRT min has to catch.
# See the comments on each in configs/default_config.py.
# ---------------------------------------------------------------------------

def probe_horizon(cfg) -> float:
    """Horizon of one teacher probe, falling back to ``DT`` when unset."""
    h = cfg.GAME.TIME.PROBE_HORIZON
    if h is None or h <= 0.0:
        return float(cfg.GAME.TIME.DT)
    window = cfg.GAME.TIME.WINDOW_TIME
    if window is not None and window > 0.0 and h >= window:
        # The probe bootstraps V at t - h; past a window edge that is a
        # different window's net, so the margin stops meaning what it should.
        log.warning(
            "PROBE_HORIZON (%.4g) >= WINDOW_TIME (%.4g): probes bootstrap "
            "across a window boundary. Clamping to WINDOW_TIME / 2.", h, window)
        return float(window / 2.0)
    return float(h)


def substeps_for(horizon: float, cfg) -> int:
    """Euler substeps needed to integrate ``horizon`` within ``MAX_SUBSTEP``.

    ``MAX_SUBSTEP <= 0`` selects the deprecated ``NUM_SUBSTEPS`` count instead,
    which is how an archived config reproduces its original substep exactly.
    """
    max_sub = cfg.GAME.TIME.MAX_SUBSTEP
    if max_sub is None or max_sub <= 0.0:
        return int(cfg.GAME.TIME.NUM_SUBSTEPS)
    return max(1, int(math.ceil(float(horizon) / float(max_sub) - 1e-9)))


# ---------------------------------------------------------------------------
# Rollout helpers
# ---------------------------------------------------------------------------

def get_rbn_factor(x: jax.Array, dt: float, dyn: Dynamics) -> jax.Array:
    if hasattr(dyn, "get_gamma"):
        gamma = dyn.get_gamma(x)
    else:
        gamma = jnp.zeros(x.shape[0], dtype=x.dtype)
    denom = jnp.clip(1.0 - (gamma * dt), min=1e-2)
    return 1.0 / denom

def hj_bellman_update(
    v_next: jax.Array, 
    step_stats: dict, 
    problem_type: str, 
    gamma: jax.Array
) -> jax.Array:
    """
    The core HJ Bellman Operator.
    Combines future value with trajectory constraints (l_min, lA_max, etc.)
    """
    B = gamma.shape[0]
    N = v_next.shape[0]
    
    if N > B and N % B == 0:
        K = N // B
        gamma = jnp.repeat(gamma, K)
    elif gamma.shape == v_next.shape:
        pass
    else:
        raise ValueError(f"v_next shape {v_next.shape} is not compatible with gamma shape {gamma.shape}")

    v_future = v_next * gamma

    if problem_type == "BRS":
        return v_future
    
    if problem_type == "BRT":
        return jnp.minimum(step_stats["l_min"], v_future)
    
    if problem_type == "BRAT":
        return jnp.maximum(
            step_stats["lA_max"], 
            jnp.minimum(step_stats["lT_min"], v_future)
        )
    raise ValueError(f"Unknown problem type: {problem_type}")

@functools.lru_cache(maxsize=None)
def _rollout_length_base(N: int, lam: float, M: int) -> jax.Array:
    """Unshuffled per-sample rollout-length assignment for ``make_rollout_length_schedule``.

    N/lam/M are constant for a whole training run, so this (the only
    data-independent part of the schedule) is cached instead of rebuilt
    -- via a batch_size-length Python list -- on every call.
    """
    counts = []
    remaining = N
    for n in range(1, M + 1):
        c = min(int(round(N * 0.5 * (lam ** (n - 1)))), remaining)
        counts.append(c)
        remaining -= c
        if remaining <= 0:
            break

    # Overflow goes into the 1-step bucket.
    if remaining > 0:
        counts[0] += remaining

    # Pad to length M so the index arithmetic below is uniform.
    while len(counts) < M:
        counts.append(0)

    base = np.repeat(np.arange(1, M + 1, dtype=np.int32), np.array(counts, dtype=np.int64))
    return jnp.asarray(base)


def make_rollout_length_schedule(
    key: jax.Array,
    N: int,
    lam: float,
    M: int,
) -> jax.Array:
    """Build a geometrically-decaying TD(λ) rollout-length schedule.

    Assigns rollout lengths such that approximately ``N * 0.5 * lam^(n-1)``
    samples use n-step rollouts, for n = 1, …, M.  Any remainder is
    added to the 1-step bucket.

    Args:
        N:      Total number of samples (batch size).
        lam:    Decay factor in ``[0, 0.5]``; smaller → more 1-step samples.
        M:      Maximum rollout length.

    Returns:
        Shuffled integer tensor of shape (N,) with values in ``[1, M]``.

    Raises:
        AssertionError: If ``lam`` is outside ``[0, 0.5]``.
    """
    assert 0.0 <= lam <= 0.5, f"lam must be in [0, 0.5], got {lam}"
    return jax.random.permutation(key, _rollout_length_base(N, lam, M))


@functools.lru_cache(maxsize=None)
def rollout_length_live_counts(N: int, lam: float, M: int) -> Tuple[int, ...]:
    """Rows still needing work at each step of an ``N``-sample TD(λ) rollout.

    Entry ``n-1`` is ``#{samples with rollout length >= n}``, i.e. how many rows
    a TD scan actually has to advance at step ``n`` -- every other row already
    read its target off at ``n == n_steps`` and is only being carried along.
    Shuffling cannot change a count, so this is a property of (N, lam, M) alone
    and is safe to bake in as a static shape.
    """
    counts = np.bincount(np.asarray(_rollout_length_base(N, lam, M)), minlength=M + 1)
    return tuple(int(counts[n:].sum()) for n in range(1, M + 1))


def rollout_with_intermediate_checks(
    x: jax.Array,
    u: jax.Array,
    d: jax.Array,
    dyn: Dynamics,
    dt: float,
    m: int,
    problem_type: str,
):
    """Roll out for dt using m substeps with constant u, d; track running stats."""
    assert m >= 1
    dt_sub = dt / float(m)
    x_cur = x

    if problem_type in ("BRT", "BRS"):
        l_min = dyn.l(x_cur)
    elif problem_type == "BRAT":
        lT_min = dyn.target_l(x_cur)
        lA_max = dyn.avoid_l(x_cur)
    else:
        raise NotImplementedError(f"Unknown problem_type={problem_type}")

    for _ in range(m):
        x_cur = dyn.step(x_cur, u, d, dt_sub)
        if problem_type in ("BRT", "BRS"):
            l_min = jnp.minimum(l_min, dyn.l(x_cur))
        else:
            lT_min = jnp.minimum(lT_min, dyn.target_l(x_cur))
            lA_max = jnp.maximum(lA_max, dyn.avoid_l(x_cur))

    if problem_type in ("BRT", "BRS"):
        return x_cur, {"l_min": l_min}
    return x_cur, {"lT_min": lT_min, "lA_max": lA_max}


def backed_up_score_for_action(
    x: jax.Array,
    u: jax.Array,
    d: jax.Array,
    V_next_fn: Callable[[jax.Array], jax.Array],
    dyn: Dynamics,
    dt: float,
    m: int,
    problem_type: str,
    gamma: float = 1.0,
) -> jax.Array:
    """1-step DP-backed-up scalar for probing comparisons (includes intermediate checks)."""
    x_final, stats = rollout_with_intermediate_checks(x, u, d, dyn, dt, m, problem_type)
    v_next = V_next_fn(x_final)

    return hj_bellman_update(v_next, stats, problem_type, gamma), v_next

# ---------------------------------------------------------------------------
# Probing-based action inference
# ---------------------------------------------------------------------------

def infer_u_d_double_sided(
    x: jax.Array,
    V_next_fn: Callable[[jax.Array], jax.Array],
    dyn: Dynamics,
    dt: float,
    control_role: str,
    disturb_role: str,
    problem_type: str,
    num_substeps: int = 3,
    tau: float = 1e-5,
    return_conf: bool = False,
) -> Tuple[jax.Array, ...]:
    """Vectorized double-sided probing: probes +/- u_max and d_max per dimension.

    Args:
        x:            State batch, shape (B, state_dim).
        V_next_fn:    Callable mapping a state batch to scalar values. Must
                      accept exactly B * 2 * u_dim (or d_dim) rows — callers
                      are responsible for constructing a closure that expands
                      t_next consistently (e.g. via repeat_interleave).
        dyn:          Dynamics object.
        dt:           Timestep for the rollout.
        control_role: "max" or "min" — whether the controller maximises or
                      minimises the value.
        disturb_role: "max" or "min" — role of the disturbance.
        problem_type: "BRT", "BRS", or "BRAT".
        num_substeps: Number of Euler substeps per dt rollout.
        tau:          Confidence threshold; dimensions where |s+ - s-| < tau
                      use V_next tie-breaking instead of bang-bang on the
                      backup score.
        return_conf:  If True, also return the PER-CHANNEL probe margin
                      |s+ - s-| for u and d, shaped like the action.

    Returns:
        (u_star, d_star) if return_conf is False, else
        (u_star, d_star, u_conf, d_conf) with u_conf: (B, u_dim),
        d_conf: (B, d_dim).

    Note on the confidence shape
    ----------------------------
    |s+ - s-| is per channel because degeneracy is per channel. A state can
    have one channel decided by the probe and the rest tied -- on a singular
    arc that is the normal case, not an edge case. The previous return was a
    per-sample std over all 2*u_dim scores, which cannot express that: it is
    large whenever ANY one channel discriminates strongly, so the tied
    channels of a partially-decided state were scored as confident.
    """
    B = x.shape[0]
    gamma = get_rbn_factor(x, dt, dyn)
    u_dim, d_dim = dyn.control_dim, dyn.disturb_dim
    u_zeros_B = jnp.zeros((B, u_dim), dtype=x.dtype)
    d_zeros_B = jnp.zeros((B, d_dim), dtype=x.dtype)

    # --- Control probes: row 2*i = u_min[i] one-hot, row 2*i+1 = u_max[i] one-hot. ---
    idx = jnp.arange(u_dim)
    u_probe_mat = jnp.zeros((2 * u_dim, u_dim), dtype=x.dtype)
    u_probe_mat = u_probe_mat.at[2 * idx, idx].set(dyn.u_min)
    u_probe_mat = u_probe_mat.at[2 * idx + 1, idx].set(dyn.u_max)

    def u_body(carry, u_row):
        u_b = jnp.broadcast_to(u_row, (B, u_dim))
        score, v_next = backed_up_score_for_action(
            x, u_b, d_zeros_B, V_next_fn, dyn, dt, num_substeps, problem_type, gamma
        )
        return carry, (score, v_next)

    _, (scores_u_t, v_next_u_t) = jax.lax.scan(u_body, None, u_probe_mat)  # each (2*u_dim, B)
    scores_u = scores_u_t.T.reshape(B, u_dim, 2)
    v_next_u = v_next_u_t.T.reshape(B, u_dim, 2)

    s_minus, s_plus = scores_u[..., 0], scores_u[..., 1]
    vn_minus, vn_plus = v_next_u[..., 0], v_next_u[..., 1]

    confident = jnp.abs(s_plus - s_minus) >= tau
    choose_plus_bkp = s_plus > s_minus if control_role == "max" else s_plus < s_minus
    choose_plus_vn = vn_plus > vn_minus if control_role == "max" else vn_plus < vn_minus

    choose_plus = jnp.where(confident, choose_plus_bkp, choose_plus_vn)
    u_star = jnp.where(choose_plus, dyn.u_max, dyn.u_min)

    u_conf = jnp.abs(s_plus - s_minus)

    # --- Disturbance probes ---
    d_star = jnp.zeros((B, d_dim), dtype=x.dtype)
    d_conf = jnp.zeros((B, d_dim), dtype=x.dtype)
    if d_dim > 0:
        jdx = jnp.arange(d_dim)
        d_probe_mat = jnp.zeros((2 * d_dim, d_dim), dtype=x.dtype)
        d_probe_mat = d_probe_mat.at[2 * jdx, jdx].set(dyn.d_max)
        d_probe_mat = d_probe_mat.at[2 * jdx + 1, jdx].set(dyn.d_min)

        def d_body(carry, d_row):
            d_b = jnp.broadcast_to(d_row, (B, d_dim))
            score, v_next = backed_up_score_for_action(
                x, u_zeros_B, d_b, V_next_fn, dyn, dt, num_substeps, problem_type, gamma
            )
            return carry, (score, v_next)

        _, (scores_d_t, v_next_d_t) = jax.lax.scan(d_body, None, d_probe_mat)
        scores_d = scores_d_t.T.reshape(B, d_dim, 2)
        v_next_d = v_next_d_t.T.reshape(B, d_dim, 2)

        sd_plus, sd_minus = scores_d[..., 0], scores_d[..., 1]
        vn_plus_d, vn_minus_d = v_next_d[..., 0], v_next_d[..., 1]

        confident_d = jnp.abs(sd_plus - sd_minus) >= tau

        choose_plus_d_bkp = sd_plus > sd_minus if disturb_role == "max" else sd_plus < sd_minus
        choose_plus_d_vn = vn_plus_d > vn_minus_d if disturb_role == "max" else vn_plus_d < vn_minus_d
        choose_plus_d = jnp.where(confident_d, choose_plus_d_bkp, choose_plus_d_vn)
        d_star = jnp.where(choose_plus_d, dyn.d_max, dyn.d_min)
        d_conf = jnp.abs(sd_plus - sd_minus)

    return (u_star, d_star, u_conf, d_conf) if return_conf else (u_star, d_star)


def infer_u_d_triple_sided(
    x: jax.Array,
    V_next_fn: Callable[[jax.Array], jax.Array],
    dyn: Dynamics,
    dt: float,
    control_role: str,
    disturb_role: str,
    problem_type: str,
    num_substeps: int = 3,
    tau: float = 1e-5,
    return_conf: bool = False,
) -> Tuple[jax.Array, ...]:
    """Vectorized triple-sided probing: probes +u_max, -u_max, AND u_mid=0 per
    control dimension, selecting neutral when it is within ``tau`` of the better
    bang-bang extreme. Useful for stiff systems (e.g. the quadrotor) where
    bang-bang every step chatters and neutral control is often correct.

    Disturbance dimensions stay double-sided (adversaries rarely want a neutral
    action) -- identical to :func:`infer_u_d_double_sided`'s d-probe, including
    the V_next tie-break. Same signature/return contract as double-sided, so it
    is a drop-in behind ``cfg.GAME.TRIPLE_SIDED_PROBING``.

    The all-zero control probe is shared across dims: probe rows are
    ``[zero, +dim0, -dim0, +dim1, -dim1, ...]`` (1 + 2*u_dim rows).
    """
    B = x.shape[0]
    gamma = get_rbn_factor(x, dt, dyn)
    u_dim, d_dim = dyn.control_dim, dyn.disturb_dim
    u_zeros_B = jnp.zeros((B, u_dim), dtype=x.dtype)
    d_zeros_B = jnp.zeros((B, d_dim), dtype=x.dtype)

    # --- Control probes: row 0 = neutral (all zero); rows 1+2i / 1+2i+1 = +/- one-hot. ---
    idx = jnp.arange(u_dim)
    u_probe_mat = jnp.zeros((1 + 2 * u_dim, u_dim), dtype=x.dtype)
    u_probe_mat = u_probe_mat.at[1 + 2 * idx, idx].set(dyn.u_max)
    u_probe_mat = u_probe_mat.at[1 + 2 * idx + 1, idx].set(dyn.u_min)

    def u_body(carry, u_row):
        u_b = jnp.broadcast_to(u_row, (B, u_dim))
        score, _v_next = backed_up_score_for_action(
            x, u_b, d_zeros_B, V_next_fn, dyn, dt, num_substeps, problem_type, gamma
        )
        return carry, score

    _, scores_u_t = jax.lax.scan(u_body, None, u_probe_mat)  # (1+2*u_dim, B)
    s_mid = scores_u_t[0]                                    # (B,) shared neutral
    rest = scores_u_t[1:].T.reshape(B, u_dim, 2)             # (B, u_dim, 2)
    s_plus, s_minus = rest[..., 0], rest[..., 1]
    s_mid_exp = s_mid[:, None]                               # (B, 1) -> broadcast to (B, u_dim)

    if control_role == "max":
        best_bb = jnp.maximum(s_plus, s_minus)
        use_neutral = (best_bb - s_mid_exp) < tau
        choose_plus = s_plus >= s_minus
    else:
        best_bb = jnp.minimum(s_plus, s_minus)
        use_neutral = (s_mid_exp - best_bb) < tau
        choose_plus = s_plus <= s_minus

    u_bangbang = jnp.where(choose_plus, dyn.u_max, dyn.u_min)
    u_star = jnp.where(use_neutral, jnp.zeros_like(u_bangbang), u_bangbang)

    u_conf = jnp.abs(s_plus - s_minus)

    # --- Disturbance probes: double-sided with V_next tie-break (as double_sided). ---
    d_star = jnp.zeros((B, d_dim), dtype=x.dtype)
    d_conf = jnp.zeros((B, d_dim), dtype=x.dtype)
    if d_dim > 0:
        jdx = jnp.arange(d_dim)
        d_probe_mat = jnp.zeros((2 * d_dim, d_dim), dtype=x.dtype)
        d_probe_mat = d_probe_mat.at[2 * jdx, jdx].set(dyn.d_max)
        d_probe_mat = d_probe_mat.at[2 * jdx + 1, jdx].set(dyn.d_min)

        def d_body(carry, d_row):
            d_b = jnp.broadcast_to(d_row, (B, d_dim))
            score, v_next = backed_up_score_for_action(
                x, u_zeros_B, d_b, V_next_fn, dyn, dt, num_substeps, problem_type, gamma
            )
            return carry, (score, v_next)

        _, (scores_d_t, v_next_d_t) = jax.lax.scan(d_body, None, d_probe_mat)
        scores_d = scores_d_t.T.reshape(B, d_dim, 2)
        v_next_d = v_next_d_t.T.reshape(B, d_dim, 2)

        sd_plus, sd_minus = scores_d[..., 0], scores_d[..., 1]
        vn_plus_d, vn_minus_d = v_next_d[..., 0], v_next_d[..., 1]

        confident_d = jnp.abs(sd_plus - sd_minus) >= tau
        choose_plus_d_bkp = sd_plus > sd_minus if disturb_role == "max" else sd_plus < sd_minus
        choose_plus_d_vn = vn_plus_d > vn_minus_d if disturb_role == "max" else vn_plus_d < vn_minus_d
        choose_plus_d = jnp.where(confident_d, choose_plus_d_bkp, choose_plus_d_vn)
        d_star = jnp.where(choose_plus_d, dyn.d_max, dyn.d_min)
        d_conf = jnp.abs(sd_plus - sd_minus)

    return (u_star, d_star, u_conf, d_conf) if return_conf else (u_star, d_star)