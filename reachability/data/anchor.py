from typing import Tuple, List, Dict, Any, Optional
import logging
import time

import jax
import jax.numpy as jnp
from flax import nnx
from dataclasses import dataclass

from configs.constants import PROJECT_NAME
from reachability.dynamics import Dynamics
from reachability.training.functional import (
    rollout_with_intermediate_checks, hj_bellman_update, get_rbn_factor, substeps_for,
)
from .sampling import sample_uniform_states
from utils import decode_actions

log = logging.getLogger(f"{PROJECT_NAME}.{__name__}")


# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------

@dataclass
class AnchorDataset:
    """Flattened anchor supervision gathered from policy rollouts.

    Attributes:
        X: States of shape (N * W, state_dim).
        Y: Targets of shape (N * W, 1).
        T: Times of shape (N * W, 1).
    """
    X: jax.Array
    Y: jax.Array
    T: jax.Array

# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

_rollout_jit_cache: dict = {}


def _build_rollout_policy_traj_jit(dyn: Dynamics, cfg, chunk_size: int, num_chunks: int, dtype: jnp.dtype = jnp.float32):
    window_time = cfg.GAME.TIME.WINDOW_TIME
    dt = cfg.GAME.TIME.DT
    num_substeps = substeps_for(dt, cfg)
    problem_type = cfg.GAME.PROBLEM_TYPE
    W = int(round(window_time / dt))

    @nnx.jit
    def rollout(x0_chunks, policy, value_fn, terminal_time):

        def rollout_chunk(x0):
            xs: List[jax.Array] = [x0]
            seg_stats: List[dict] = []

            curr_x = x0
            t_steps = jnp.arange(W, 0, -1, dtype=dtype) * dt + terminal_time

            for i in range(W):
                phi = dyn.nn_inputs(curr_x)
                out = policy(phi, t_steps[i])
                u, d = decode_actions(out, dyn)
                curr_x, stats = rollout_with_intermediate_checks(
                    x=curr_x, u=u, d=d, dyn=dyn, dt=dt, m=num_substeps, problem_type=problem_type,
                )

                xs.append(curr_x)
                seg_stats.append(stats)

            ys = []
            t_terminal = jnp.full(x0.shape[:-1], terminal_time, dtype=dtype)
            v_prev = value_fn(xs[W], t_terminal)["V"]

            for i in range(W - 1, -1, -1):
                gamma = get_rbn_factor(xs[i], dt, dyn)
                v_i = hj_bellman_update(v_prev, seg_stats[i], problem_type, gamma)
                ys.insert(0, v_i)
                v_prev = v_i

            # Flatten trajectories for supervised finetuning.
            Xs = jnp.stack(xs[:W], axis=1).reshape(-1, x0.shape[-1])
            Ys = jnp.stack(ys, axis=1).reshape(-1)
            Ts = jnp.tile(t_steps[None, :], (chunk_size, 1)).reshape(-1)

            y0 = ys[0]
            t0 = jnp.full((chunk_size,), t_steps[0], dtype=dtype)
            return Xs, Ys, Ts, x0, y0, t0

        def scan_body(carry, x0_chunk):
            return carry, rollout_chunk(x0_chunk)

        _, (Xs, Ys, Ts, x0, y0, t0) = jax.lax.scan(scan_body, None, x0_chunks)

        snapshots = {
            "x0": x0.reshape(-1, x0.shape[-1]),
            "y0": y0.reshape(-1),
            "t0": t0.reshape(-1),
        }
        return Xs.reshape(-1, Xs.shape[-1]), Ys.reshape(-1), Ts.reshape(-1), snapshots

    return rollout


def _get_rollout_policy_traj_jit(dyn: Dynamics, cfg, chunk_size: int, num_chunks: int, dtype: jnp.dtype = jnp.float32):
    """Cached per (dyn, chunk_size, num_chunks): compiles once for the whole
    training run, since ANCHOR_BATCHES/TRAIN.BATCH_SIZE are constant."""
    key = (id(dyn), chunk_size, num_chunks, dtype)
    fn = _rollout_jit_cache.get(key)
    if fn is None:
        fn = _build_rollout_policy_traj_jit(dyn, cfg, chunk_size, num_chunks, dtype)
        _rollout_jit_cache[key] = fn
    return fn


def _build_rollout_full_horizon_jit(dyn: Dynamics, cfg, chunk_size: int, num_chunks: int,
                                    n_full: int, W_harvest: int, dtype: jnp.dtype = jnp.float32):
    """Full-horizon Monte-Carlo anchor targets (no intermediate bootstrap).

    Rolls the student policy for the ENTIRE remaining horizon (``n_full`` macro
    steps, from time-to-go ``t0 = n_full*dt`` down to 0) and backs the HJ value
    up over the whole trajectory, bootstrapping only on the exact ``t=0``
    boundary (``value_fn`` there ~ the boundary condition). Only the first
    ``W_harvest`` states -- the current window's slice, times in
    ``(t0-window_time, t0]`` -- are returned for supervision, but each carries
    its true full-horizon reach cost rather than a one-window bootstrap. This
    removes the cross-window bootstrap that lets an over-optimistic earlier
    window inflate the next one's targets.
    """
    dt = cfg.GAME.TIME.DT
    num_substeps = substeps_for(dt, cfg)
    problem_type = cfg.GAME.PROBLEM_TYPE

    @nnx.jit
    def rollout(x0_chunks, policy, value_fn):

        def rollout_chunk(x0):
            # state i is queried at time-to-go t0 - i*dt = (n_full - i)*dt
            t_steps = jnp.arange(n_full, 0, -1, dtype=dtype) * dt   # (n_full,)

            # ---- forward rollout as a scan (compiles the step body ONCE, instead
            # of inlining n_full copies). Emits the pre-step state x_i and the
            # step's running stats; the final carry is x_{n_full}. ----
            def fwd_body(curr_x, t_i):
                phi = dyn.nn_inputs(curr_x)
                out = policy(phi, t_i)
                u, d = decode_actions(out, dyn)
                next_x, stats = rollout_with_intermediate_checks(
                    x=curr_x, u=u, d=d, dyn=dyn, dt=dt, m=num_substeps, problem_type=problem_type,
                )
                return next_x, (curr_x, stats)

            x_final, (xs, seg_stats) = jax.lax.scan(fwd_body, x0, t_steps)
            # xs: (n_full, chunk, sd) = [x_0 .. x_{n_full-1}];  seg_stats: pytree of (n_full, chunk)

            # ---- backward value backup as a reverse scan; bootstrap on the exact
            # t=0 boundary only. reverse=True walks i = n_full-1 .. 0 and writes
            # each v_i back at forward index i, so ys[i] == v_i as before. ----
            t_terminal = jnp.zeros(x0.shape[:-1], dtype=dtype)
            v_terminal = value_fn(x_final, t_terminal)["V"]

            def bwd_body(v_next, xi_stats):
                x_i, stats_i = xi_stats
                gamma = get_rbn_factor(x_i, dt, dyn)
                v_i = hj_bellman_update(v_next, stats_i, problem_type, gamma)
                return v_i, v_i

            _, ys = jax.lax.scan(bwd_body, v_terminal, (xs, seg_stats), reverse=True)
            # ys: (n_full, chunk) in forward order (ys[i] = v_i)

            # Harvest only the current window's slice (first W_harvest states).
            # swapaxes(0,1) reproduces the original stack(..., axis=1) chunk-major layout.
            Xs = jnp.swapaxes(xs[:W_harvest], 0, 1).reshape(-1, x0.shape[-1])
            Ys = jnp.swapaxes(ys[:W_harvest], 0, 1).reshape(-1)
            Ts = jnp.tile(t_steps[:W_harvest][None, :], (chunk_size, 1)).reshape(-1)

            y0 = ys[0]
            t0 = jnp.full((chunk_size,), t_steps[0], dtype=dtype)
            return Xs, Ys, Ts, x0, y0, t0

        def scan_body(carry, x0_chunk):
            return carry, rollout_chunk(x0_chunk)

        _, (Xs, Ys, Ts, x0, y0, t0) = jax.lax.scan(scan_body, None, x0_chunks)
        snapshots = {
            "x0": x0.reshape(-1, x0.shape[-1]),
            "y0": y0.reshape(-1),
            "t0": t0.reshape(-1),
        }
        return Xs.reshape(-1, Xs.shape[-1]), Ys.reshape(-1), Ts.reshape(-1), snapshots

    return rollout


def _get_rollout_full_horizon_jit(dyn: Dynamics, cfg, chunk_size: int, num_chunks: int,
                                  n_full: int, W_harvest: int, dtype: jnp.dtype = jnp.float32):
    """Cached per (dyn, chunk_size, num_chunks, n_full): full-horizon MC rollout.
    n_full varies per window, so this recompiles once per window (cheap)."""
    key = (id(dyn), chunk_size, num_chunks, n_full, W_harvest, dtype)
    fn = _rollout_jit_cache.get(key)
    if fn is None:
        fn = _build_rollout_full_horizon_jit(dyn, cfg, chunk_size, num_chunks, n_full, W_harvest, dtype)
        _rollout_jit_cache[key] = fn
    return fn


def build_anchor_dataset_window(
    key: jax.Array,
    num_data_batches: int,
    dyn: Any,
    policy: nnx.Module,
    value_fn: nnx.Module,
    cfg: Any,
    terminal_time: float,
    x0: Optional[jax.Array] = None,
    full_horizon: bool = False,
) -> Tuple[AnchorDataset, Dict[str, jax.Array]]:
    """Build anchor supervision by rolling out `num_data_batches` chunks of
    TRAIN.BATCH_SIZE states, one chunk at a time, and concatenating them.

    ``x0`` is the caller's choice of start states, ``(num_data_batches *
    TRAIN.BATCH_SIZE, state_dim)``; ``None`` falls back to a uniform draw.

    ``full_horizon`` rolls the policy for the entire remaining horizon and backs
    the value up over the whole trajectory (bootstrapping only on the t=0
    boundary), yielding true Monte-Carlo reach-cost targets for the current
    window instead of a one-window bootstrap. ``policy`` and ``value_fn`` must
    then span windows [0, current] (the rollout crosses earlier windows and
    bootstraps window 0 at t=0).
    """
    B = cfg.TRAIN.BATCH_SIZE
    if x0 is None:
        keys = jax.random.split(key, num_data_batches)
        x0 = jnp.stack([sample_uniform_states(k, dyn, B) for k in keys], axis=0)
    else:
        x0 = jnp.reshape(x0, (num_data_batches, B, -1))

    if full_horizon:
        dt = cfg.GAME.TIME.DT
        W_harvest = int(round(cfg.GAME.TIME.WINDOW_TIME / dt))
        n_full = int(round((terminal_time + cfg.GAME.TIME.WINDOW_TIME) / dt))
        _n_cached = len(_rollout_jit_cache)
        rollout_jit = _get_rollout_full_horizon_jit(dyn, cfg, B, num_data_batches, n_full, W_harvest)
        fresh = len(_rollout_jit_cache) > _n_cached   # cache grew -> this call triggers a compile
        t0 = time.perf_counter()
        Xs, Ys, Ts, snapshots = rollout_jit(x0, policy, value_fn)
        jax.block_until_ready((Xs, Ys, Ts))
        log.info("full-horizon rollout: n_full=%d chunks=%d unrolled_steps=%d %s -> %.1fs %s",
                 n_full, num_data_batches, n_full, "FRESH-COMPILE" if fresh else "cached",
                 time.perf_counter() - t0, "(compile+run)" if fresh else "(run-only)")
    else:
        _n_cached = len(_rollout_jit_cache)
        rollout_jit = _get_rollout_policy_traj_jit(dyn, cfg, B, num_data_batches)
        fresh = len(_rollout_jit_cache) > _n_cached
        t0 = time.perf_counter()
        Xs, Ys, Ts, snapshots = rollout_jit(
            x0, policy, value_fn, jnp.asarray(terminal_time, dtype=dyn.dtype)
        )
        jax.block_until_ready((Xs, Ys, Ts))
        log.info("one-window rollout: chunks=%d %s -> %.1fs %s",
                 num_data_batches, "FRESH-COMPILE" if fresh else "cached",
                 time.perf_counter() - t0, "(compile+run)" if fresh else "(run-only)")

    return AnchorDataset(X=Xs, Y=Ys, T=Ts), snapshots


# ---------------------------------------------------------------------------
# Mini-batch sampler
# ---------------------------------------------------------------------------

def sample_anchor_minibatch(
    key: jax.Array,
    ds: AnchorDataset,
    batch_size: int,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Randomly sample a mini-batch from one offset of an ``AnchorDataset``.

    Args:
        ds:         The anchor dataset.
        batch_size: Number of (x, y, t) triples to return.
        key:        The random key for sampling.

    Returns:
        ``(x, y, t)`` tensors with leading batch dimension ``batch_size``.
    """
    n_total = ds.X.shape[0]
    idx = jax.random.choice(key, n_total, shape=(batch_size,), replace=False)
    return ds.X[idx], ds.Y[idx], ds.T[idx]