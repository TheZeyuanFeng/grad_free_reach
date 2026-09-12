import logging
import math

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from configs.constants import PROJECT_NAME
from reachability.dynamics import Dynamics
from .functional import (
    rollout_with_intermediate_checks, hj_bellman_update, get_rbn_factor,
    make_rollout_length_schedule, rollout_length_live_counts, substeps_for,
)
from .strategy import ActionStrategy, ProbingStrategy, PolicyStrategy

log = logging.getLogger(f"{PROJECT_NAME}.{__name__}")

_STEP_TOL = 1e-2  # Tolerance for "active" rollout steps, to avoid numerical issues with t=0.0

def _select_state(mask, new_state, old_state):
    """Per-sample select ``new_state`` where ``mask`` else ``old_state``.
    """
    def _leaf(a, b):
        m = mask.reshape(mask.shape + (1,) * (jnp.ndim(a) - mask.ndim))
        return jnp.where(m, a, b)
    return jax.tree_util.tree_map(_leaf, new_state, old_state)


class HJSolver:
    """TD-lambda target computer for HJ reachability.

    Owns two strategies — a probing teacher and an explicit-policy student —
    and exposes ``compute_targets`` as the single entry point for generating
    training targets from either.

    Strategy call convention
    ------------------------
    All ``ActionStrategy`` subclasses accept ``(dyn, x, t)`` only; the value /
    policy network is captured at construction time, not passed at call time.
    """

    def __init__(
        self,
        dyn: Dynamics,
        value_net: nnx.Module,
        policy_net: nnx.Module,
        cfg,
    ) -> None:
        self.dyn = dyn
        self.v_net = value_net
        self.p_net = policy_net
        self.cfg = cfg

        self.teacher: ActionStrategy = ProbingStrategy(value_net, cfg)
        self.student: ActionStrategy = PolicyStrategy(policy_net)

        self._targets_jit_cache: dict = {}
        self._live_counts_cache: dict = {}
        _m = getattr(value_net, "net", value_net)
        self._num_windows = int(getattr(_m, "num_windows", 1))

    def _full_live_counts(self, n_steps: jax.Array, B: int) -> tuple:
        """Static live-row counts for a full-batch TD(λ) ``n_steps``, or None.

        ``rollout_length_live_counts`` is only the right answer if ``n_steps``
        really is a permutation of the (B, LAMBDA, M) schedule, and being wrong
        would silently drop rows from their last step. So check it against the
        real array -- once per (B, λ, M), since the counts cannot vary after
        that -- and fall back to the full-batch scan if it does not hold.
        """
        lam, M = float(self.cfg.TRAIN.LAMBDA), int(self.cfg.TRAIN.M)
        key = (B, lam, M)
        if key not in self._live_counts_cache:
            expected = rollout_length_live_counts(B, lam, M)
            arr = np.asarray(jax.device_get(n_steps))
            actual = tuple(int((arr >= n).sum()) for n in range(1, M + 1))
            if actual == expected and (arr.size == 0 or int(arr.max()) <= M):
                self._live_counts_cache[key] = expected
            else:
                log.warning(
                    "n_steps is not the (B=%d, lam=%g, M=%d) TD(lambda) schedule "
                    "(live counts %s, expected %s); using the full-batch TD scan.",
                    B, lam, M, actual, expected)
                self._live_counts_cache[key] = None
        return self._live_counts_cache[key]

    def compute_targets(
        self,
        key: jax.Array,
        x: jax.Array,
        t: jax.Array,
        n_steps: jax.Array,
        teacher_frac: float = 0.0,
        window_range: tuple = None,
        v_net: nnx.Module = None,
        p_net: nnx.Module = None,
    ) -> jax.Array:
        """Compute TD-lambda targets for every sample in the batch.

        The teacher rollout length is controlled by ``TRAIN.TEACHER_ROLLOUT_STEPS``:
          k>0 : teacher rollout lengths are drawn from the same TD(lambda)
                schedule as the student (make_rollout_length_schedule), capped
                at k instead of TRAIN.M -- e.g. k=5 gives a mix of 1..5-step
                teacher rollouts weighted by TRAIN.LAMBDA, not a fixed 5 for
                every sample.
          0   : teacher uses the exact same per-sample lengths as the student
                (n_steps), rather than an independently-drawn schedule.

        Args:
            x:            State batch, shape (B, state_dim).
            t:            Time batch, shape (B,).
            n_steps:      Per-sample rollout lengths, shape (B,).
            teacher_frac: Fraction of the batch to roll out with the teacher.
                          Typically scheduled from high to low over the curriculum.
            v_net, p_net: Optional overrides for the value/policy nets used in the
                          rollout (e.g. a cheap ``WindowSlice`` view scoped to just
                          the windows ``window_range`` can reach), in place of the
                          full multi-window nets captured at construction. Falls
                          back to ``self.v_net``/``self.p_net`` when omitted.

        Returns:
            y: Target values, shape (B,).
        """
        assert 0.0 <= teacher_frac <= 1.0, "teacher_frac must be in [0, 1]"
        B = x.shape[0]
        v_net = v_net if v_net is not None else self.v_net
        p_net = p_net if p_net is not None else self.p_net

        cap = int(math.ceil(B * self.cfg.TRAIN.TEACHER_FRAC_INIT))
        cap = max(0, min(cap, B))
        n_teacher = min(int(round(B * teacher_frac)), cap)

        key, sched_key = jax.random.split(key)
        perm        = jax.random.permutation(key, B)
        teacher_idx = perm[:cap]      # (cap,) candidate teacher rows, dynamic content, static size
        student_idx = perm[cap:]      # (B-cap,) rows the teacher never touches, static size

        teacher_n_fixed = self.cfg.TRAIN.TEACHER_ROLLOUT_STEPS
        wr = (0, self._num_windows - 1) if window_range is None else (int(window_range[0]), int(window_range[1]))

        def _gather(a, idx):
            return jax.tree_util.tree_map(lambda v: v[idx], a)

        full_live = self._full_live_counts(n_steps, B)
        def _clamp(live, rows):
            return None if live is None else tuple(min(rows, c) for c in live)

        tx = _gather(x, teacher_idx)
        tt = t[teacher_idx]
        if teacher_n_fixed > 0:
            tn = make_rollout_length_schedule(sched_key, cap, self.cfg.TRAIN.LAMBDA, teacher_n_fixed)
            tlive = rollout_length_live_counts(cap, float(self.cfg.TRAIN.LAMBDA), teacher_n_fixed)
        else:
            tn = n_steps[teacher_idx]
            tlive = _clamp(full_live, cap)
        y_teacher_cap = self._get_teacher_targets_jit(wr, tlive)(v_net, tx, tt, tn)  # (cap,)

        if n_teacher == cap:
            y_student_rest = self._get_student_targets_jit(wr, _clamp(full_live, B - cap))(
                v_net, p_net, _gather(x, student_idx), t[student_idx], n_steps[student_idx]
            )
            y = jnp.zeros((B,), dtype=y_teacher_cap.dtype)
            y = y.at[student_idx].set(y_student_rest)
            y = y.at[teacher_idx].set(y_teacher_cap)
            return y

        teacher_valid = jnp.arange(cap) < n_teacher       # (cap,) which cap-rows are real teacher
        y_student = self._get_student_targets_jit(wr, full_live)(v_net, p_net, x, t, n_steps)
        y_sel = jnp.where(teacher_valid, y_teacher_cap, y_student[teacher_idx])
        return y_student.at[teacher_idx].set(y_sel)

    def _get_student_targets_jit(self, window_range: tuple, n_live: tuple = None):
        key = ("student", window_range, n_live)
        fn = self._targets_jit_cache.get(key)
        if fn is None:
            wrng, live = window_range, n_live
            def student_core(v_net, p_net, x, t, n_steps):
                return self._compute_targets_for_batch(
                    x, t, n_steps, self.student, v_net, p_net, self.cfg.TRAIN.M, wrng, live)
            fn = nnx.jit(student_core)
            self._targets_jit_cache[key] = fn
        return fn

    def _get_teacher_targets_jit(self, window_range: tuple, n_live: tuple = None):
        key = ("teacher", window_range, n_live)
        fn = self._targets_jit_cache.get(key)
        if fn is None:
            wrng, live = window_range, n_live
            teacher_max_n = self.cfg.TRAIN.TEACHER_ROLLOUT_STEPS
            if teacher_max_n <= 0:
                teacher_max_n = self.cfg.TRAIN.M
            def teacher_core(v_net, x, t, n_steps):
                return self._compute_targets_for_batch(
                    x, t, n_steps, self.teacher, v_net, v_net, teacher_max_n, wrng, live)
            fn = nnx.jit(teacher_core)
            self._targets_jit_cache[key] = fn
        return fn

    def _compute_targets_for_batch(
        self,
        x: jax.Array,
        t: jax.Array,
        n_steps: jax.Array,
        strategy: ActionStrategy,
        v_net: nnx.Module,
        action_net: nnx.Module,
        max_n: int,
        window_range: tuple = None,
        n_live: tuple = None,
    ) -> jax.Array:
        """Compute per-sample TD(n) targets for a batch with a single strategy.

        ``None`` keeps the original ``lax.scan`` over the full batch, which
        stays the cheaper choice to compile when there is nothing to trim.
        """
        cfg          = self.cfg
        dyn          = self.dyn
        problem_type = cfg.GAME.PROBLEM_TYPE
        dt           = cfg.GAME.TIME.DT
        num_substeps = substeps_for(dt, cfg)
        B            = x.shape[0]
        max_n        = int(max_n)

        # Running reachability stats (same initialisation as a fresh rollout).
        if problem_type == "BRAT":
            stats0: dict = {
                "lT_min": jnp.full((B,), jnp.inf, dtype=x.dtype),
                "lA_max": jnp.full((B,), -jnp.inf, dtype=x.dtype),
            }
        else:  # BRT or BRS
            stats0 = {"l_min": jnp.full((B,), jnp.inf, dtype=x.dtype)}

        def advance(carry, gamma):
            """One TD step for whatever rows are in ``carry``; shape-agnostic."""
            curr_x, curr_t, stats, discount = carry
            active = curr_t > dt * (1.0 - _STEP_TOL)

            # Strategy signature: (dyn, x, t) -> (u, d)
            u, d = strategy(dyn, curr_x, curr_t, net=action_net, window_range=window_range)

            next_x, step_stats = rollout_with_intermediate_checks(
                x=curr_x, u=u, d=d, dyn=dyn, dt=dt,
                m=num_substeps, problem_type=problem_type,
            )
            curr_x = _select_state(active, next_x, curr_x)
            stats  = self._update_stats(stats, step_stats, active, discount)
            curr_t = jnp.maximum(curr_t - dt, 0.0)
            discount = discount * gamma  # after this line, discount == gamma**n

            # After n steps this matches the old `_rollout(x, t, n, strategy)`.
            V_terminal = v_net(curr_x, curr_t, window_range=window_range)["V"]
            y_val = hj_bellman_update(V_terminal, stats, problem_type, discount)
            return (curr_x, curr_t, stats, discount), y_val

        discount0 = jnp.ones((B,), dtype=x.dtype)
        y0        = jnp.zeros((B,), dtype=x.dtype)

        if n_live is None:
            gamma = get_rbn_factor(x, dt, dyn)

            def body(carry, n):
                core, y_val = advance(carry[:4], gamma)
                return (*core, jnp.where(n_steps == n, y_val, carry[4])), None

            init_carry = (x, t, stats0, discount0, y0)
            (_, _, _, _, y), _ = jax.lax.scan(body, init_carry, jnp.arange(1, max_n + 1))
            return y

        order = jnp.argsort(-n_steps, stable=True)
        ns    = n_steps[order]
        gamma = get_rbn_factor(x[order], dt, dyn)
        carry = (x[order], t[order], stats0, discount0)
        y     = y0
        for n in range(1, max_n + 1):
            k = int(n_live[n - 1])
            if k <= 0:
                break   # every remaining row already has its target
            carry = jax.tree_util.tree_map(lambda v: v[:k], carry)
            carry, y_val = advance(carry, gamma[:k])
            y = y.at[:k].set(jnp.where(ns[:k] == n, y_val, y[:k]))
        return jnp.zeros_like(y).at[order].set(y)

    
    # ---------------------------------------------------------------------------
    # Stats accumulation
    # ---------------------------------------------------------------------------

    def _update_stats(
        self,
        stats: dict,
        step_stats: dict,
        active: jax.Array,
        discount: jax.Array,
    ) -> dict:
        """Accumulate running reachability stats (in-place) for the active sub-batch."""
        new_stats = {}
        problem_type = self.cfg.GAME.PROBLEM_TYPE
        if problem_type == "BRAT":
            lT_cand = jnp.minimum(stats["lT_min"], discount * step_stats["lT_min"])
            lA_cand = jnp.maximum(stats["lA_max"], step_stats["lA_max"])
            # Same barrier as the BRT branch below -- see that comment for why.
            lT_cand, lA_cand = jax.lax.optimization_barrier((lT_cand, lA_cand))

            new_stats["lT_min"] = jnp.where(active, lT_cand, stats["lT_min"])
            new_stats["lA_max"] = jnp.where(active, lA_cand, stats["lA_max"])

        else: # BRT or BRS
            l_cand = jnp.minimum(stats["l_min"], discount * step_stats["l_min"])
            l_cand = jax.lax.optimization_barrier(l_cand)
            new_stats["l_min"] = jnp.where(active, l_cand, stats["l_min"])

        return new_stats