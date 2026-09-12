"""Probabilistic (PAC-style) verification of a learned safe set.

Answers: *of the states this model claims are safe, what fraction actually are,
and how confident can we be in that number?* -- turning the point-estimate FPR
in :mod:`.metrics` into a bound of the form "true violation rate <= eps, with
confidence 1 - beta".

The bound is the standard one-sided Clopper-Pearson interval: observing ``k``
violations in ``N`` i.i.d. scenario draws gives ``eps`` via
``P(Binomial(N, eps) <= k) = beta``, i.e. ``1 - eps = Beta(N-k, k+1).ppf(beta)``.

Two caveats worth stating plainly, since they bound what this actually proves:

1. The adversary is approximate. ``ProbingStrategy`` is the greedy worst-case
   disturbance *with respect to the learned value net*, not a true worst case
   over all disturbance signals (that would mean solving the adversary's own
   optimal control problem exactly). It is a stronger and less self-referential
   adversary than the learned ``d_pred`` head, but a genuinely adversarial
   disturbance could still do better, which would push the true violation rate
   above the reported bound.
2. Only :func:`certify_at_threshold` carries the guarantee, and only for a
   threshold fixed *before* its sample is drawn. :func:`search_delta_level` is
   a compute-saving heuristic for *choosing* a threshold; running many adaptive
   tests and reporting whichever passed would inflate the effective beta. So
   :func:`run_verification` always re-certifies the chosen threshold against a
   fresh, independent draw, and reports that draw as the certificate.
"""

import logging

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from scipy.stats import beta as beta_dist
from tqdm import tqdm

from reachability.dynamics import Dynamics
from reachability.data.sampling import sample_val_states_uniform
from reachability.training.strategy import ProbingStrategy
from configs import PROJECT_NAME
from .rollout import rollout_trajectories

log = logging.getLogger(f"{PROJECT_NAME}.{__name__}")

# ---------------------------------------------------------------------------
# JIT caches
# ---------------------------------------------------------------------------

_rollout_jit_cache: dict = {}
_value_eval_jit_cache: dict = {}


def _get_rollout_jit(dyn, value_net, policy_net, cfg, t_final, dt,
                     policy_steps_per_dt, adversarial: bool):
    key = (id(dyn), id(value_net), id(policy_net), t_final, dt,
           policy_steps_per_dt, adversarial)
    fn = _rollout_jit_cache.get(key)
    if fn is None:
        action_fn = _probing_action_fn(dyn, value_net, cfg) if adversarial else None

        @nnx.jit
        def _rollout(x0):
            return rollout_trajectories(
                dyn=dyn, value_net=value_net, policy_net=policy_net,
                t_final=t_final, dt=dt, cfg=cfg, x0=x0,
                policy_steps_per_dt=policy_steps_per_dt, action_fn=action_fn,
            )

        _rollout_jit_cache[key] = _rollout
        fn = _rollout
    return fn


def _get_value_eval_jit(value_net):
    """Both callers (``collect_claimed_safe``, ``estimate_volumes``) pass a
    single ``t_final`` broadcast across the whole batch, so ``shared_time=True``
    is always valid here."""
    key = id(value_net)
    fn = _value_eval_jit_cache.get(key)
    if fn is None:
        @nnx.jit
        def _value_eval(x, t):
            return value_net(x, t, shared_time=True)["V"]

        _value_eval_jit_cache[key] = _value_eval
        fn = _value_eval
    return fn


# ---------------------------------------------------------------------------
# Scenario-bound statistics
# ---------------------------------------------------------------------------

def epsilon_of_k(N: int, beta: float, k: int) -> float:
    """eps* solving P(Binomial(N, eps*) <= k) = beta, i.e. 1-eps* = Beta(N-k,k+1).ppf(beta)."""
    return 1 - beta_dist.ppf(beta, N - k, k + 1)


def min_samples_for_target(beta: float, target_epsilon: float) -> int:
    """Smallest n for which even k=0 can certify ``target_epsilon``.

    ``epsilon_of_k(n, beta, 0) = 1 - beta**(1/n)``, so the requirement
    ``1 - beta**(1/n) <= eps`` inverts to ``n >= log(beta) / log(1 - eps)``.
    Below this n the test cannot pass no matter how clean the rollouts are --
    which is exactly the trap a delta search walks into as it raises the
    threshold and the claimed-safe region shrinks out from under the sample.
    """
    if not 0.0 < target_epsilon < 1.0:
        return 1
    return int(np.ceil(np.log(beta) / np.log(1.0 - target_epsilon)))


def max_allowed_violations(N: int, beta: float, target_epsilon: float):
    """Largest k in [0, N-1] with ``epsilon_of_k(N, beta, k) <= target_epsilon``.

    ``epsilon_of_k`` is non-decreasing in k, so this is a plain integer binary
    search -- pure math, no sampling. Returns None if even k=0 misses
    ``target_epsilon`` at this N (i.e. N is too small for the target).
    """
    if epsilon_of_k(N, beta, 0) > target_epsilon:
        return None
    lo, hi = 0, N - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if epsilon_of_k(N, beta, mid) <= target_epsilon:
            lo = mid
        else:
            hi = mid - 1
    return lo


# ---------------------------------------------------------------------------
# Problem-type orientation
#
# ``delta_level`` is always a threshold on the RAW value-net output V(x, T) --
# nothing is ever sign-flipped, so a printed delta means exactly what it says.
# The three functions below carry the whole convention, deliberately kept
# separate rather than collapsed into one reorientation:
#
#   tightens_by_increasing  which direction of delta SHRINKS the claimed set
#   value_keep_mask         which side of delta is the claimed-success region
#   is_violation            which sign of the RAW rollout cost is a failure
#
# The previous ``safety_score`` reorientation (v for 'max', -v for 'min') was
# equivalent but made a reported delta mean V <= -delta for reach problems,
# which is exactly the kind of hidden flip that produced the mis-signed
# quantile in ``metrics.calculate_inflation_delta``.
# ---------------------------------------------------------------------------

def tightens_by_increasing(cfg) -> bool:
    """Does RAISING ``delta_level`` shrink (tighten) the claimed-success region?

    True  -- avoidance (BRT/BRS, CONTROL_ROLE=max): region is ``{V >= delta}``.
    False -- reachability (BRT/BRS, CONTROL_ROLE=min) and BRAT (reach-avoid):
             region is ``{V <= delta}``, so LOWERING delta tightens.
    """
    problem_type = cfg.GAME.PROBLEM_TYPE
    if problem_type == "BRAT":
        return False
    if problem_type in ("BRT", "BRS"):
        return cfg.GAME.CONTROL_ROLE == "max"
    raise NotImplementedError(f"Unknown PROBLEM_TYPE={problem_type!r}")


def value_keep_mask(raw_values, delta_level: float, cfg):
    """Mask of raw predicted values inside the claimed-success region at ``delta_level``."""
    if tightens_by_increasing(cfg):
        return raw_values >= delta_level
    return raw_values <= delta_level


def is_violation(raw_costs, cfg):
    """Violation mask from the RAW (unoriented) rollout scenario cost.

    avoidance: ``cost < 0``;  reachability / BRAT: ``cost > 0``.
    """
    problem_type = cfg.GAME.PROBLEM_TYPE
    if problem_type == "BRAT":
        return raw_costs > 0
    if problem_type in ("BRT", "BRS"):
        return raw_costs < 0 if cfg.GAME.CONTROL_ROLE == "max" else raw_costs > 0
    raise NotImplementedError(f"Unknown PROBLEM_TYPE={problem_type!r}")


def order_statistic_jump(values, costs, cfg, k_target: int):
    """Threshold estimated to leave about ``k_target`` violations, from this sample.

    Which end of the sorted violator values is the "least clearly violating"
    residual depends on the tightening direction: for avoidance
    (region ``{V >= delta}``) raising delta drops the LOWEST-value violators
    first, so the residual is the top ``k_target``; for reachability/BRAT
    (region ``{V <= delta}``) lowering delta drops the HIGHEST-value violators
    first, so the residual is the bottom ``k_target`` -- the opposite end.

    Returns None when there are already <= ``k_target`` violations.
    """
    violation_values = np.sort(values[is_violation(costs, cfg)])  # ascending
    m = len(violation_values)
    if m <= k_target:
        return None
    if tightens_by_increasing(cfg):
        idx = m - 1 if k_target == 0 else m - k_target
    else:
        idx = 0 if k_target == 0 else k_target - 1
    return float(violation_values[idx])


# ---------------------------------------------------------------------------
# Sampling + adversarial rollout
# ---------------------------------------------------------------------------

def sample_states(key, dyn: Dynamics, n: int, pins: dict = None) -> jax.Array:
    """Draw ``n`` validation states, with any pinned dimensions held fixed.

    ``pins`` maps a state-vector index to the value every drawn state must
    carry there, which restricts the whole certificate to that slice of the
    state space -- see :func:`describe_pins` for what that does to the claim.
    """
    x = sample_val_states_uniform(key, dyn, n)
    if not pins:
        return x
    for i, v in pins.items():
        x = x.at[:, i].set(jnp.asarray(v, dtype=x.dtype))
    return dyn.wrap_state(x)


def validate_pins(key, dyn: Dynamics, pins: dict) -> None:
    """Fail fast if ``wrap_state`` refuses to hold a pin at its requested value.

    A pin outside the dynamics' own state box comes back clipped rather than
    honoured, which would silently certify a different slice than the one
    asked for -- so probe a small batch up front instead of finding out from a
    number that looks fine.
    """
    if not pins:
        return
    x = np.asarray(sample_states(key, dyn, 8, pins))
    for i, v in pins.items():
        got = x[:, i]
        if not np.allclose(got, v, atol=1e-5):
            raise ValueError(
                f"pinned state dim {i} to {v}, but wrap_state returned "
                f"{np.unique(got)[:4]} -- the value is outside what "
                f"{type(dyn).__name__} accepts there "
                f"(state box [{float(dyn.state_low[i])}, {float(dyn.state_high[i])}])."
            )


def describe_pins(pins: dict) -> str:
    return ", ".join(f"x[{i}]={v:g}" for i, v in pins.items()) if pins else "(none)"


def _probing_action_fn(dyn: Dynamics, value_net, cfg):
    """Worst-case ``(u, d)`` from double-sided probing of the value net.

    ``shared_time=True`` is safe here: every caller of this action_fn (the
    rollout loop in :mod:`.rollout`) invokes it with one time value
    broadcast across the whole batch, never per-sample-varying times.
    """
    strategy = ProbingStrategy(value_net, cfg)
    return lambda x, t: strategy(dyn, x, t, shared_time=True)


def collect_claimed_safe(
    key, dyn: Dynamics, value_net, cfg, t_final: float, delta_level: float,
    target_n: int, batch_size: int, max_reject_batches: int, pins: dict = None,
):
    """Rejection-sample ``target_n`` states the model claims safe at ``delta_level``.

    Returns ``(states, values, n_drawn)`` with ``values`` the RAW network output;
    ``states`` may be shorter than ``target_n`` if the claimed-safe region is too
    small to fill it within ``max_reject_batches`` (including empty, when the
    region has vanished). With ``pins``, both the region and the draw are
    restricted to that slice (see :func:`sample_states`).
    """
    dtype = dyn.dtype
    kept_x, kept_s, n_drawn = [], [], 0

    for _ in range(max_reject_batches):
        if sum(len(s) for s in kept_s) >= target_n:
            break
        key, sub = jax.random.split(key)
        cand = sample_states(sub, dyn, batch_size, pins)
        n_drawn += batch_size

        t_vec = jnp.full((cand.shape[0],), t_final, dtype=dtype)
        value = np.asarray(_get_value_eval_jit(value_net)(cand, t_vec))
        keep = value_keep_mask(value, delta_level, cfg)
        if keep.any():
            kept_x.append(np.asarray(cand)[keep])
            kept_s.append(value[keep])

    if not kept_x:
        return np.zeros((0, dyn.state_dim)), np.zeros(0), n_drawn
    return (np.concatenate(kept_x)[:target_n],
            np.concatenate(kept_s)[:target_n],
            n_drawn)


def _shape_bucket(m: int, cap: int) -> int:
    """Round ``m`` up to the next power of two, capped at ``cap``.

    Every distinct batch shape makes XLA compile a fresh executable for the
    whole ``T/dt``-step rollout scan, and each resident CUBIN needs GPU memory
    OUTSIDE JAX's preallocated arena -- so a search that visits 50 different
    ``n_collected`` values compiles 50 executables and eventually dies with
    ``RESOURCE_EXHAUSTED: Failed to load in-memory CUBIN``.

    Bucketing bounds the number of shapes to O(log cap) (~18 at cap=100k) while
    never padding by more than 2x. Padding to ``cap`` unconditionally would fix
    the shape count too, but a 433-state chunk would then cost a 100000-state
    rollout.
    """
    return min(max(1 << max(m - 1, 0).bit_length(), 1), cap)


def rollout_costs(
    dyn: Dynamics, value_net, policy_net, cfg, states, t_final: float, dt: float,
    policy_steps_per_dt: int, batch_size: int, adversarial: bool = True,
):
    """Closed-loop rollout of ``states``; returns RAW, unoriented scenario costs.

    Interpret them with :func:`is_violation` -- nothing is sign-flipped here.

    ``adversarial=True`` drives the action source with probing (worst case
    w.r.t. the value net) instead of the trained policy net.
    """
    rollout_jit = _get_rollout_jit(
        dyn, value_net, policy_net, cfg, t_final, dt, policy_steps_per_dt, adversarial)
    out = []
    for s in tqdm(range(0, len(states), batch_size), desc="Verify rollout", leave=False):
        raw = np.asarray(states[s:s + batch_size])
        m = len(raw)
        padded = _shape_bucket(m, batch_size)
        if padded > m:
            raw = np.concatenate([raw, np.repeat(raw[-1:], padded - m, axis=0)])
        true_score, _ = rollout_jit(jnp.asarray(raw, dtype=dyn.dtype))
        out.append(np.asarray(true_score)[:m])
    return np.concatenate(out) if out else np.zeros(0)


# ---------------------------------------------------------------------------
# Certification
# ---------------------------------------------------------------------------

def certify_at_threshold(key, delta_level: float, *, dyn, value_net, policy_net,
                         cfg, t_final, dt, va) -> tuple:
    """One PAC test of a FIXED, pre-chosen ``delta_level`` against a fresh draw.

    This is the only function here that produces a statistically valid
    certificate -- see the module docstring on why the search must not be the
    thing reported.

    Returns ``(record, scores, costs)``. The raw per-sample ``scores``
    (predicted safety) and ``costs`` (realized safety) are handed back so
    callers can reuse this one expensive rollout -- for the order-statistic
    jump in :func:`search_delta_level`, and for the whole
    :func:`epsilon_delta_curve` -- instead of paying for another.
    """
    key, ck = jax.random.split(key)
    states, scores, n_drawn = collect_claimed_safe(
        ck, dyn, value_net, cfg, t_final, delta_level, va["N"],
        va["sample_batch_size"], va["max_reject_batches"],
        va.get("pin_state_dims"),
    )
    n = len(states)
    if n == 0:
        return ({"delta_level": delta_level, "k": 0, "n_collected": 0,
                 "n_drawn": n_drawn, "epsilon": float("nan"), "success": False,
                 "note": "claimed-safe region empty at this threshold"},
                np.zeros(0), np.zeros(0))

    costs = rollout_costs(dyn, value_net, policy_net, cfg, states, t_final, dt,
                          va["policy_steps_per_dt"],
                          va["rollout_batch_size"], va["adversarial"])
    k = int(is_violation(costs, cfg).sum())
    epsilon = float(epsilon_of_k(n, va["beta"], k))
    record = {"delta_level": delta_level, "k": k, "n_collected": n,
              "n_drawn": n_drawn, "epsilon": epsilon,
              "success": epsilon <= va["target_epsilon"]}
    return record, scores, costs


def epsilon_delta_curve(scores, costs, cfg, beta: float, n_points: int = 25) -> list:
    """Post-hoc eps(delta) trade-off curve, swept from a SINGLE rollout sample.

    For any threshold delta, the claimed-safe set is exactly the already-rolled-out
    subset selected by :func:`value_keep_mask` -- so the entire curve comes free
    from one sample, no extra rollouts. Thresholds are placed at value quantiles
    so each point retains a meaningful sample size, ordered LOOSEST FIRST (which
    is ascending for avoidance and descending for reachability/BRAT, per
    :func:`tightens_by_increasing`), so ``retained_frac`` always decreases along
    the curve regardless of problem type.

    This is a COMPARISON AID, NOT a certificate: every point reuses the same
    sample, so picking the best-looking one is exactly the adaptive selection
    the module docstring warns about. Use it to compare two models' curves
    (lower-and-righter dominates); use :func:`certify_at_threshold` on a fresh
    draw for any number you intend to claim.

    Expect the curve to be NON-MONOTONIC at the high-delta end: raising delta
    removes violators (pushing eps down) but also shrinks ``n`` (pushing the
    bound back up, since even k=0 only certifies ``1 - beta**(1/n)``). The
    turning point is real information -- past it you are giving up volume AND
    getting a looser bound -- not an artifact to smooth over.
    """
    if len(scores) == 0:
        return []
    tighten = tightens_by_increasing(cfg)
    qs = np.linspace(0.0, 0.95, n_points) if tighten else np.linspace(1.0, 0.05, n_points)
    levels = np.unique(np.quantile(scores, qs))
    if not tighten:
        levels = levels[::-1]  # np.unique sorts ascending; loosest is the largest here
    curve = []
    for d in levels:
        mask = value_keep_mask(scores, float(d), cfg)
        n = int(mask.sum())
        if n == 0:
            continue
        k = int(is_violation(costs[mask], cfg).sum())
        curve.append({"delta_level": float(d), "n": n, "k": k,
                      "epsilon": float(epsilon_of_k(n, beta, k)),
                      "retained_frac": n / len(scores)})
    return curve


def search_delta_level(key, *, dyn, value_net, policy_net, cfg, t_final, dt,
                       va, k_target) -> tuple:
    """Search for the LOOSEST ``delta_level`` that passes (largest kept volume).

    Direction-agnostic: "tighter"/"looser" are defined by
    :func:`tightens_by_increasing`, not by numeric position, so the search
    mirrors itself for reachability/BRAT (where tighter means numerically
    smaller) without any sign trick on the values themselves.

    NOT a certificate (see module docstring) -- ``run_verification`` re-tests
    the winner on a fresh draw. Jumps via the violators' order statistic while
    failing, then bisects once a passing threshold is known.
    """
    tighten = tightens_by_increasing(cfg)

    def is_tighter(a, b):
        """Is threshold ``a`` strictly more restrictive than ``b``?"""
        return a > b if tighten else a < b

    loosest_grid = va["delta_grid_min"] if tighten else va["delta_grid_max"]
    tightest_grid = va["delta_grid_max"] if tighten else va["delta_grid_min"]

    bad_bound = None   # tightest threshold known to FAIL
    good_bound = None  # loosest threshold known to PASS
    candidate = va["initial_delta_level"]
    history = []
    best_record = None
    n_min = min_samples_for_target(va["beta"], va["target_epsilon"])

    for it in tqdm(range(va["max_search_iters"]), desc="Delta search"):
        key, ik = jax.random.split(key)
        rec, scores, costs = certify_at_threshold(
            ik, candidate, dyn=dyn, value_net=value_net, policy_net=policy_net,
            cfg=cfg, t_final=t_final, dt=dt, va=va)
        rec["iter"] = it
        history.append(rec)
        log.info("  iter %d: delta=%.6f  n=%d  k=%d  eps=%.6g  %s", it, candidate,
                 rec["n_collected"], rec["k"], rec["epsilon"],
                 "pass" if rec["success"] else "fail")

        if rec["success"]:
            if good_bound is None or is_tighter(good_bound, candidate):
                good_bound, best_record = candidate, rec
            if bad_bound is None or abs(good_bound - bad_bound) < va["convergence_tolerance"]:
                break
            candidate = (bad_bound + good_bound) / 2
        elif rec["n_collected"] < n_min:
            # region vanished or too thin to ever pass -- too TIGHT, so loosen
            rec["note"] = f"n={rec['n_collected']} < n_min={n_min}; uncertifiable at any k"
            reference = bad_bound if bad_bound is not None else loosest_grid
            candidate = (candidate + reference) / 2
        else:
            if bad_bound is None or is_tighter(candidate, bad_bound):
                bad_bound = candidate
            reference = good_bound if good_bound is not None else tightest_grid
            jump = order_statistic_jump(scores, costs, cfg, k_target)
            if jump is not None and is_tighter(jump, bad_bound) and is_tighter(reference, jump):
                candidate = jump
            else:
                candidate = (bad_bound + reference) / 2
            if good_bound is not None and abs(good_bound - bad_bound) < va["convergence_tolerance"]:
                break

    return good_bound, bad_bound, best_record, history


def estimate_volumes(key, *, dyn, value_net, cfg, t_final, delta_level, va) -> dict:
    """Fraction of the validation state space that is CLAIMED safe.

    Reported at both the raw zero level set and the calibrated ``delta_level``
    -- the conservatism cost of the certificate. Cheap: value-net evals only,
    no rollouts. Says nothing about whether those claims are CORRECT: that is
    what ``epsilon`` bounds, and what the rollout-based confusion metrics in
    ``eval.py``'s standard mode measure.

    Under ``pin_state_dims`` the denominator is the pinned SLICE, not the full
    validation box, so these volumes are only comparable against other runs
    measured on the same slice.
    """
    n_learned = n_calib = n_total = 0
    remaining = va["N_volume_estimation"]
    pbar = tqdm(total=remaining, desc="Volume estimate")
    while remaining > 0:
        n = min(va["volume_batch_size"], remaining)
        key, sk = jax.random.split(key)
        x = sample_states(sk, dyn, n, va.get("pin_state_dims"))
        t_vec = jnp.full((n,), t_final, dtype=dyn.dtype)
        value = np.asarray(_get_value_eval_jit(value_net)(x, t_vec))
        n_learned += int(value_keep_mask(value, 0.0, cfg).sum())
        n_calib += int(value_keep_mask(value, delta_level, cfg).sum())
        n_total += n
        remaining -= n
        pbar.update(n)
    pbar.close()
    return {"N_volume_estimation": n_total,
            "learned_safe_volume": n_learned / max(1, n_total),
            "calibrated_safe_volume": n_calib / max(1, n_total)}


def estimate_policy_success_rate(key, *, dyn, value_net, policy_net, cfg,
                                 t_final, dt, va) -> dict:
    """Fraction of UNIFORMLY sampled states from which the rollout succeeds.

    No claimed-safe filter and no threshold: this is the overall denominator,
    directly comparable to ``eval.py``'s accuracy metrics and generally smaller
    than the calibrated/learned volumes, which are measured only over the states
    the value net claims. Unlike ``epsilon`` it is defined whether or not a
    certificate was found, which makes it the one number that can be compared
    across models that certify and models that do not.

    Costs one rollout pass over ``N_policy_success_rate_estimation`` states.
    """
    n = va["N_policy_success_rate_estimation"]
    if n <= 0:
        return {}
    key, sk = jax.random.split(key)
    x = np.asarray(sample_states(sk, dyn, n, va.get("pin_state_dims")))
    costs = rollout_costs(dyn, value_net, policy_net, cfg, x, t_final, dt,
                          va["policy_steps_per_dt"], va["rollout_batch_size"],
                          va["adversarial"])
    n_success = int((~is_violation(costs, cfg)).sum())
    return {"N_policy_success_rate_estimation": n,
            "policy_success_rate": n_success / max(1, n)}


def run_verification(key, *, dyn: Dynamics, value_net, policy_net, cfg,
                     t_final: float, dt: float, va: dict) -> dict:
    """Search for a threshold, then certify it against a fresh independent draw."""
    tighten = tightens_by_increasing(cfg)
    keep_dir = ">=" if tighten else "<="
    k_target = max_allowed_violations(va["N"], va["beta"], va["target_epsilon"])
    pins = va.get("pin_state_dims") or {}
    logs = {"beta": va["beta"], "target_epsilon": va["target_epsilon"],
            "N": va["N"], "k_target": k_target, "adversarial": va["adversarial"],
            "control_role": cfg.GAME.CONTROL_ROLE,
            "problem_type": cfg.GAME.PROBLEM_TYPE,
            "tightens_by_increasing": tighten,
            "pin_state_dims": {str(i): v for i, v in pins.items()}}

    if pins:
        key, pin_key = jax.random.split(key)
        validate_pins(pin_key, dyn, pins)

    log.info("problem_type=%s  control_role=%s  claimed-success region = {V %s delta}  "
             "(tightens_by_increasing=%s)",
             cfg.GAME.PROBLEM_TYPE, cfg.GAME.CONTROL_ROLE, keep_dir, tighten)
    if pins:
        log.info("state pins: %s -- every number below is conditional on this "
                 "slice, NOT the full validation range.", describe_pins(pins))
    log.info("N=%d  target_epsilon=%g  beta=%g -> k_target=%s (max violations/draw)",
             va["N"], va["target_epsilon"], va["beta"], k_target)
    if k_target is None:
        log.warning("target_epsilon unachievable at N=%d even with 0 violations -- "
                    "increase N or relax target_epsilon.", va["N"])
        return logs

    if va["search"]:
        key, sk = jax.random.split(key)
        delta_level, bad_bound, search_best, history = search_delta_level(
            sk, dyn=dyn, value_net=value_net, policy_net=policy_net, cfg=cfg,
            t_final=t_final, dt=dt, va=va, k_target=k_target)
        logs.update({"search_history": history, "search_bad_bound": bad_bound,
                     "search_certificate": search_best})
        if delta_level is None:
            log.warning("No threshold in [%g, %g] passed within %d iterations.",
                        va["delta_grid_min"], va["delta_grid_max"], va["max_search_iters"])
            logs["threshold_found"] = False
            key, vk = jax.random.split(key)
            logs.update(estimate_volumes(vk, dyn=dyn, value_net=value_net, cfg=cfg,
                                         t_final=t_final, delta_level=0.0, va=va))
            logs["calibrated_safe_volume"] = 0.0
            key, pk = jax.random.split(key)
            logs.update(estimate_policy_success_rate(
                pk, dyn=dyn, value_net=value_net, policy_net=policy_net, cfg=cfg,
                t_final=t_final, dt=dt, va=va))
            log.info("[uncalibrated, no certificate] learned_safe_volume=%.6f  "
                     "policy_success_rate=%s",
                     logs["learned_safe_volume"], logs.get("policy_success_rate"))
            return logs
        logs["threshold_found"] = True
    else:
        delta_level = va["initial_delta_level"]
        logs["threshold_found"] = True

    # Fresh, independent draw at the now-fixed threshold: THIS is the certificate.
    key, fk = jax.random.split(key)
    cert, cert_scores, cert_costs = certify_at_threshold(
        fk, delta_level, dyn=dyn, value_net=value_net, policy_net=policy_net,
        cfg=cfg, t_final=t_final, dt=dt, va=va)
    logs["certificate"] = cert
    logs["delta_level"] = delta_level
    # Free by-product of the certification rollout -- comparison aid only.
    logs["epsilon_delta_curve"] = epsilon_delta_curve(
        cert_scores, cert_costs, cfg, va["beta"], va["curve_points"])

    log.info("--------Verification-----------")
    log.info("delta_level: %.6f   (claimed-success region = {V %s %.6f})",
             delta_level, keep_dir, delta_level)
    log.info("epsilon: %.6g (target %g)  k=%d / n=%d  -> %s",
             cert["epsilon"], va["target_epsilon"], cert["k"], cert["n_collected"],
             "CERTIFIED" if cert["success"] else "FAILED on fresh draw")
    sb = logs.get("search_certificate")
    if sb is not None:
        log.info("epsilon (search-selected, no fresh draw -- reference pipeline's "
                 "headline number): %.6g  k=%d / n=%d   selection effect: %+.6g",
                 sb["epsilon"], sb["k"], sb["n_collected"],
                 cert["epsilon"] - sb["epsilon"])

    key, vk = jax.random.split(key)
    logs.update(estimate_volumes(vk, dyn=dyn, value_net=value_net, cfg=cfg,
                                 t_final=t_final, delta_level=delta_level, va=va))
    log.info("claimed safe volume  @0=%.6f  @delta=%.6f",
             logs["learned_safe_volume"], logs["calibrated_safe_volume"])

    key, pk = jax.random.split(key)
    logs.update(estimate_policy_success_rate(
        pk, dyn=dyn, value_net=value_net, policy_net=policy_net, cfg=cfg,
        t_final=t_final, dt=dt, va=va))
    if "policy_success_rate" in logs:
        log.info("policy success rate (uniform over %s, N=%d): %.6f",
                 f"the slice {describe_pins(pins)}" if pins
                 else "the whole validation range",
                 logs["N_policy_success_rate_estimation"], logs["policy_success_rate"])

    return logs
