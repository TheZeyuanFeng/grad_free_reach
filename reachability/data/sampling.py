"""reachability/data/sampling.py -- training-state sampling.

BoundaryAwareSampler draws each training batch as a weighted mixture of
regions ("buckets"), so the network sees a mix of arbitrary states, states
near the boundaries it needs to get right, and (optionally) states visited
by realistic behaviour, rather than only uniform draws:

    uniform         -- state_low/state_high box (dyn.sample() if defined)
    target          -- near dyn.target_l's zero level set, else uniform
    avoid           -- near dyn.avoid_l's zero level set, else uniform
    user_target     -- interior of the user goal region via
                       dyn.sample_target_states, else uniform
    value_boundary  -- near the CURRENT value net's zero level set (|V|<band)
    safe            -- the correct-sign side of V, |V|>=band (see unsafe_mask)
    unsafe          -- the wrong-sign side of V, |V|>=band
    nominal         -- states visited by rolling out dyn.nominal_policy

value_boundary/safe/unsafe are served from per-region ring buffers, each
refreshed by update_boundary_buffer() classifying training states by V.

None of this file is dynamics-specific: every bucket beyond "uniform" is
driven entirely by optional capabilities (see reachability/dynamics/
capabilities.py), so a new Dynamics subclass gets a bucket "for free" the
moment it implements the matching capability, with zero changes needed here.
capabilities.py is the authoritative list of these hooks and exactly what
each one needs to return -- this file only describes how they're USED.
"""

import logging
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from configs.constants import PROJECT_NAME

from reachability.dynamics import Dynamics
from reachability.dynamics.capabilities import (
    SupportsAvoidRegion,
    SupportsCustomSample,
    SupportsNominalParams,
    SupportsNominalPolicy,
    SupportsNominalSeed,
    SupportsTargetRegion,
)

log = logging.getLogger(f"{PROJECT_NAME}.{__name__}")

# ---------------------------------------------------------------------------
# Uniform samplers (unchanged public helpers)
# ---------------------------------------------------------------------------

def sample_uniform_states(key: jax.Array, dyn: Dynamics, batch_size: int) -> jax.Array:
    if isinstance(dyn, SupportsCustomSample):
        return dyn.sample(key, batch_size)
    r = jax.random.uniform(key, shape=(batch_size, dyn.state_dim), dtype=dyn.dtype)
    x = jnp.expand_dims(dyn.state_low, 0) + jnp.expand_dims(dyn.state_high - dyn.state_low, 0) * r
    return dyn.wrap_state(x)


def sample_val_states_uniform(key: jax.Array, dyn: Dynamics, batch_size: int) -> jax.Array:
    if isinstance(dyn, SupportsCustomSample):
        return dyn.sample(key, batch_size)
    r = jax.random.uniform(key, shape=(batch_size, dyn.state_dim), dtype=dyn.dtype)
    x = (jnp.expand_dims(dyn.val_state_low, 0)
         + jnp.expand_dims(dyn.val_state_high - dyn.val_state_low, 0) * r)
    return dyn.wrap_state(x)


class BoundaryAwareSampler:
    """Mixture sampler over several state-space regions — fully jitted, fixed-shape.

    See this module's docstring for what the five regions are and which
    optional Dynamics hook drives each one.
    """

    def __init__(self, dyn, value_net=None, w_uniform=0.40, w_target=0.25,
                 w_avoid=0.20, w_value_boundary=0.15,
                 w_user_target=0.0, w_safe=0.0, w_unsafe=0.0,
                 unsafe_is_positive=True,
                 boundary_band=0.1, buffer_size=4096,
                 pool_size=131072, pool_max_draws=200_000_000, pool_chunk=65536,
                 pool_key=None,
                 nominal_strategy=None, w_nominal=0.0, nominal_seed_fn=None,
                 nominal_steps=30, nominal_burn_in_steps=0,
                 nominal_param_sampler=None, nominal_samples_per_rollout=1) -> None:
        self.dyn = dyn
        self.value_net = value_net
        self.boundary_band = float(boundary_band)
        self.buffer_size = int(buffer_size)
        self._dtype = dyn.dtype
        # V-classified buckets: which sign of V is "unsafe" (matches eval's
        # _unsafe_mask). control_role='max' (BRT): V<=0 unsafe -> False;
        # control_role='min' (reach/BRS): V>=0 unsafe -> True.
        self.unsafe_is_positive = bool(unsafe_is_positive)
        self._has_user_target = hasattr(dyn, "sample_target_states")

        if nominal_strategy is not None:
            self.nominal_strategy = nominal_strategy
        elif isinstance(dyn, SupportsNominalPolicy):
            from reachability.training.strategy import NominalPolicyStrategy
            self.nominal_strategy = NominalPolicyStrategy(lambda d, x, t, **kw: d.nominal_policy(x, t, **kw))
        else:
            self.nominal_strategy = None

        if nominal_param_sampler is not None:
            self.nominal_param_sampler = nominal_param_sampler
        elif isinstance(dyn, SupportsNominalParams):
            self.nominal_param_sampler = dyn.nominal_params
        else:
            self.nominal_param_sampler = None
        self.nominal_samples_per_rollout = max(1, int(nominal_samples_per_rollout))

        self.w = self._resolve_weights(w_uniform, w_target, w_avoid, w_value_boundary,
                                       w_nominal, w_user_target, w_safe, w_unsafe)

        self.nominal_steps = int(nominal_steps)
        self.nominal_burn_in_steps = int(nominal_burn_in_steps)
        assert self.nominal_samples_per_rollout <= self.nominal_steps, (
            f"nominal_samples_per_rollout ({self.nominal_samples_per_rollout}) must be <= "
            f"nominal_steps ({self.nominal_steps}) -- each rollout draws that many DISTINCT "
            "snapshot points from a window of nominal_steps positions."
        )
        if nominal_seed_fn is not None:
            self.nominal_seed_fn = nominal_seed_fn
        elif isinstance(dyn, SupportsNominalSeed):
            self.nominal_seed_fn = dyn.nominal_seed
        else:
            self.nominal_seed_fn = self._uniform

        self._state_range = dyn.state_high - dyn.state_low

        # Static target/avoid boundary pools, banked once (see _build_pool).
        pk = jax.random.PRNGKey(0) if pool_key is None else pool_key
        kt, ka = jax.random.split(pk)
        self.pool_size = int(pool_size)
        self._target_pool = (
            self._build_pool(dyn.target_l, "target", kt, pool_size, pool_max_draws, pool_chunk)
            if isinstance(dyn, SupportsTargetRegion) and self.w["target"] > 0.0 else None)
        self._avoid_pool = (
            self._build_pool(dyn.avoid_l, "avoid", ka, pool_size, pool_max_draws, pool_chunk)
            if isinstance(dyn, SupportsAvoidRegion) and self.w["avoid"] > 0.0 else None)

        # Fixed-capacity ring buffers (device arrays), one per V-classified
        # region: value_boundary (|V|<band), safe, unsafe. Each has one extra
        # trailing row -- a write-only trash slot for out-of-region states, so
        # they never consume capacity (see _build_update_jit).
        self._ring_names = ("value_boundary", "safe", "unsafe")
        self._rings = {name: self._new_ring(dyn) for name in self._ring_names}
        self._pos = {name: 0 for name in self._ring_names}

        self._kernel_cache: dict = {}
        self._update_jit = self._build_update_jit()

    def _new_ring(self, dyn):
        return {
            "vbuf": jnp.zeros((self.buffer_size + 1, dyn.state_dim), self._dtype),
            "tbuf": jnp.zeros((self.buffer_size + 1,), self._dtype),
            "vvalid": jnp.zeros((self.buffer_size + 1,), jnp.bool_),
        }

    def _resolve_weights(self, w_uniform, w_target, w_avoid, w_value_boundary, w_nominal,
                         w_user_target=0.0, w_safe=0.0, w_unsafe=0.0):
        """Redistribute any bucket's weight into uniform if this sampler
        can't actually serve it (dynamics lacks the capability, or no
        value_net was given), then normalize.

        Used by both __init__ and set_weights, so neither can produce a
        weight assignment that crashes at sample() time -- e.g. setting
        w_nominal > 0 for a dynamics with no nominal_policy silently falls
        back to uniform instead of calling self.nominal_strategy=None later.
        """
        has_target = isinstance(self.dyn, SupportsTargetRegion)
        has_avoid = isinstance(self.dyn, SupportsAvoidRegion)
        has_value = self.value_net is not None
        has_nominal = self.nominal_strategy is not None
        has_user_target = getattr(self, "_has_user_target", hasattr(self.dyn, "sample_target_states"))
        if not has_target:
            w_uniform += w_target; w_target = 0.0
        if not has_avoid:
            w_uniform += w_avoid; w_avoid = 0.0
        if not has_value:
            # value_boundary/safe/unsafe all need V to classify states.
            w_uniform += w_value_boundary + w_safe + w_unsafe
            w_value_boundary = w_safe = w_unsafe = 0.0
        if not has_nominal:
            w_uniform += w_nominal; w_nominal = 0.0
        if not has_user_target:
            w_uniform += w_user_target; w_user_target = 0.0

        total = (w_uniform + w_target + w_avoid + w_value_boundary + w_nominal
                 + w_user_target + w_safe + w_unsafe)
        return {"uniform": w_uniform/total, "target": w_target/total,
                "avoid": w_avoid/total, "value_boundary": w_value_boundary/total,
                "nominal": w_nominal/total, "user_target": w_user_target/total,
                "safe": w_safe/total, "unsafe": w_unsafe/total}

    # ------------------------------------------------------------------
    # Fixed-shape building blocks
    # ------------------------------------------------------------------

    def _uniform(self, key, n):
        if isinstance(self.dyn, SupportsCustomSample):
            return self.dyn.sample(key, n)
        r = jax.random.uniform(key, (n, self.dyn.state_dim), dtype=self._dtype)
        x = self.dyn.state_low[None, :] + r * (self.dyn.state_high - self.dyn.state_low)[None, :]
        return self.dyn.wrap_state(x)

    def _sample_t(self, key, n, T, dt, step, t_min, snap):
        t_max = jnp.clip(step * dt, t_min, T)
        t = jax.random.uniform(key, (n,), minval=t_min, maxval=t_max)
        if snap:
            t = jnp.round(t / dt) * dt
            t = jnp.clip(t, dt, T)
        return t

    def _project_to_band(self, level_fn, key, n, iters=8, step_frac=0.5):
        """Newton-project uniform draws onto ``{|level_fn| < band}``.
        """
        k1, k2 = jax.random.split(key)
        lo, hi = self.dyn.state_low, self.dyn.state_high
        x = self._uniform(k1, n)
        c = jax.random.uniform(k2, (n,), dtype=self._dtype,
                               minval=-self.boundary_band, maxval=self.boundary_band)
        grad_l = jax.grad(lambda xx: jnp.sum(level_fn(xx)))
        max_step = step_frac * self._state_range[None, :]

        def body(_, x):
            g = grad_l(x)
            gn2 = jnp.sum(g * g, axis=-1, keepdims=True)
            raw = jnp.where(gn2 > 1e-12, (level_fn(x) - c)[:, None] * g / jnp.clip(gn2, min=1e-12), 0.0)
            x = x - jnp.clip(raw, -max_step, max_step)
            return self.dyn.wrap_state(jnp.clip(x, lo[None, :], hi[None, :]))

        return jax.lax.fori_loop(0, iters, body, x)

    def _build_pool(self, level_fn, label, key, pool_size, max_draws, chunk):
        """Bank ``pool_size`` states inside ``level_fn``'s band, once, at construction.
        """
        # chunk is closed over, not passed: it sizes an array, so jit needs it
        # static or jax.random.uniform sees a traced shape.
        project = jax.jit(lambda k: self._project_to_band(level_fn, k, chunk))
        kept, got, drawn = [], 0, 0
        while got < pool_size and drawn < max_draws:
            key, k = jax.random.split(key)
            cand = project(k)
            hit = cand[jnp.abs(level_fn(cand)) < self.boundary_band]
            if hit.shape[0]:
                kept.append(hit)
                got += int(hit.shape[0])
            drawn += chunk

        if got == 0:
            log.warning(
                "%s boundary pool is EMPTY after %d draws (band=%.3g). This bucket "
                "will fall back to uniform sampling -- widen SAMPLING.BOUNDARY_BAND "
                "or give the dynamics a SupportsCustomSample.sample override.",
                label, drawn, self.boundary_band)
            return None

        pool = jnp.concatenate(kept, axis=0)[:pool_size]
        acc = got / max(drawn, 1)
        if pool.shape[0] < pool_size:
            log.info(
                "%s boundary pool stopped at the draw cap: %d/%d states from %d "
                "draws (acceptance %.4g). Fine if that is >= ~10x TRAIN.BATCH_SIZE "
                "-- otherwise raise SAMPLING.POOL_MAX_DRAWS or widen BOUNDARY_BAND.",
                label, pool.shape[0], pool_size, drawn, acc)
        else:
            log.info("%s boundary pool: %d states from %d draws (acceptance %.4g).",
                     label, pool.shape[0], drawn, acc)
        return pool

    def _from_pool(self, pool, key, n, T, dt, step, t_min, snap):
        """Draw n states from a static boundary pool (no jitter, no rejection)."""
        k1, k2 = jax.random.split(key)
        idx = jax.random.randint(k1, (n,), 0, pool.shape[0])
        return pool[idx], self._sample_t(k2, n, T, dt, step, t_min, snap)

    def _uniform_bucket(self, key, n, T, dt, step, t_min, snap):
        ku, kt = jax.random.split(key)
        x = self._uniform(ku, n)
        t = self._sample_t(kt, n, T, dt, step, t_min, snap)
        return x, t

    def _target_bucket(self, key, n, T, dt, step, t_min, snap):
        """Target-region samples for one bucket call.

        Draws from the static pool banked at construction; falls back to plain
        uniform when the dynamics has no target_l (which for a target-biased
        SupportsCustomSample.sample -- e.g. Quadruped -- still lands on the
        target region via that override) or when the pool came up empty.
        """
        if self._target_pool is not None:
            return self._from_pool(self._target_pool, key, n, T, dt, step, t_min, snap)
        return self._uniform_bucket(key, n, T, dt, step, t_min, snap)

    def _avoid_bucket(self, key, n, T, dt, step, t_min, snap):
        """Avoid-region samples for one bucket call: static pool, else uniform."""
        if self._avoid_pool is not None:
            return self._from_pool(self._avoid_pool, key, n, T, dt, step, t_min, snap)
        return self._uniform_bucket(key, n, T, dt, step, t_min, snap)

    def _user_target_bucket(self, key, n, T, dt, step, t_min, snap):
        """User-designated goal region via dyn.sample_target_states (the interior
        of the goal, distinct from the target_l~=0 boundary band of _target_bucket);
        falls back to plain uniform when the dynamics doesn't define it."""
        if self._has_user_target:
            kx, kt = jax.random.split(key)
            x = self.dyn.wrap_state(self.dyn.sample_target_states(kx, n))
            t = self._sample_t(kt, n, T, dt, step, t_min, snap)
            return x, t
        return self._uniform_bucket(key, n, T, dt, step, t_min, snap)

    def _nominal(self, key, n, dt):
        """Roll a nominal (hand-designed) policy forward for a FIXED horizon
        (nominal_steps, after nominal_burn_in_steps) and harvest several
        snapshots of the visited state per rollout.
        """
        spr = self.nominal_samples_per_rollout
        n_traj = -(-n // spr)  # ceil division; static Python ints (n, spr) at kernel-build time

        k_seed, k_params, k_pick = jax.random.split(key, 3)
        x0 = self.nominal_seed_fn(k_seed, n_traj)

        rollout_kwargs = {}
        if self.nominal_param_sampler is not None:
            rollout_kwargs = self.nominal_param_sampler(k_params, n_traj)

        burn_in = self.nominal_burn_in_steps
        horizon = self.nominal_steps
        total_steps = burn_in + horizon

        def body(carry, i):
            x, t_elapsed = carry
            u, d = self.nominal_strategy(self.dyn, x, t_elapsed, **rollout_kwargs)
            x_next = self.dyn.step(x, u, d, dt)
            return (x_next, t_elapsed + dt), x_next

        init = (x0, jnp.zeros((n_traj,), self._dtype))
        _, x_history = jax.lax.scan(body, init, jnp.arange(total_steps))   # (total_steps, n_traj, state_dim)

        pick_keys = jax.random.split(k_pick, n_traj)
        pick_within = jax.vmap(lambda k: jax.random.choice(k, horizon, (spr,), replace=False))(pick_keys)
        pick_step = burn_in + pick_within                                  # (n_traj, spr), all within [burn_in, burn_in+horizon)
        traj_idx = jnp.broadcast_to(jnp.arange(n_traj)[:, None], (n_traj, spr))
        picked = x_history[pick_step, traj_idx]                            # (n_traj, spr, state_dim)
        picked = picked.reshape(n_traj * spr, self.dyn.state_dim)[:n]
        return self.dyn.wrap_state(picked)

    def _nominal_bucket(self, key, n, T, dt, step, t_min, snap):
        kx, kt = jax.random.split(key)
        x = self._nominal(kx, n, dt)
        t = self._sample_t(kt, n, T, dt, step, t_min, snap)
        return x, t

    def _from_buffer(self, key, n, vbuf, tbuf, vvalid, T, dt, step, t_min, snap):
        k1, k3, k4 = jax.random.split(key, 3)
        any_valid = vvalid[:self.buffer_size].any()

        def ring_branch(_):
            w = jnp.where(vvalid[:self.buffer_size], 1.0, 0.0).astype(self._dtype)
            idx = jax.random.choice(k1, self.buffer_size, (n,), replace=True, p=w / w.sum())
            return vbuf[idx], tbuf[idx]

        def uniform_branch(_):
            x = self._uniform(k3, n)
            t = self._sample_t(k4, n, T, dt, step, t_min, snap)
            return x, t

        return jax.lax.cond(any_valid, ring_branch, uniform_branch, operand=None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _counts(self, batch_size):
        """Split ``batch_size`` across buckets in proportion to ``self.w``.
        """
        enabled = [k for k, w in self.w.items() if w > 0.0]
        if not enabled:
            return {k: (batch_size if k == "uniform" else 0) for k in self.w}

        sink = max(enabled, key=lambda k: self.w[k])
        counts = {k: 0 for k in self.w}
        for k in enabled:
            if k != sink:
                counts[k] = int(round(self.w[k] * batch_size))
        counts[sink] = batch_size - sum(counts.values())

        if counts[sink] < 0:
            counts = {k: 0 for k in self.w}
            counts[sink] = batch_size
        return counts

    def _get_kernel(self, batch_size, T, dt, snap):
        counts = self._counts(batch_size)
        key = (batch_size, bool(snap), tuple(sorted(counts.items())))
        if key in self._kernel_cache:
            return self._kernel_cache[key]

        n_u, n_t, n_a = counts["uniform"], counts["target"], counts["avoid"]
        n_v, n_n = counts["value_boundary"], counts["nominal"]
        n_ut, n_s, n_us = counts["user_target"], counts["safe"], counts["unsafe"]

        def kernel(key, step, t_min, rings):
            ks = jax.random.split(key, 9)
            xs, ts = [], []
            if n_u > 0:
                x, t = self._uniform_bucket(ks[0], n_u, T, dt, step, t_min, snap)
                xs.append(x); ts.append(t)
            if n_t > 0:
                x, t = self._target_bucket(ks[1], n_t, T, dt, step, t_min, snap)
                xs.append(x); ts.append(t)
            if n_a > 0:
                x, t = self._avoid_bucket(ks[2], n_a, T, dt, step, t_min, snap)
                xs.append(x); ts.append(t)
            if n_v > 0:
                r = rings["value_boundary"]
                x, t = self._from_buffer(ks[3], n_v, r["vbuf"], r["tbuf"], r["vvalid"], T, dt, step, t_min, snap)
                xs.append(x); ts.append(t)
            if n_n > 0:
                x, t = self._nominal_bucket(ks[5], n_n, T, dt, step, t_min, snap)
                xs.append(x); ts.append(t)
            if n_ut > 0:
                x, t = self._user_target_bucket(ks[6], n_ut, T, dt, step, t_min, snap)
                xs.append(x); ts.append(t)
            if n_s > 0:
                r = rings["safe"]
                x, t = self._from_buffer(ks[7], n_s, r["vbuf"], r["tbuf"], r["vvalid"], T, dt, step, t_min, snap)
                xs.append(x); ts.append(t)
            if n_us > 0:
                r = rings["unsafe"]
                x, t = self._from_buffer(ks[8], n_us, r["vbuf"], r["tbuf"], r["vvalid"], T, dt, step, t_min, snap)
                xs.append(x); ts.append(t)
            x_out = jnp.concatenate(xs, axis=0); t_out = jnp.concatenate(ts, axis=0)
            perm = jax.random.permutation(ks[4], batch_size)
            return x_out[perm], t_out[perm]

        jitted = nnx.jit(kernel)
        self._kernel_cache[key] = jitted
        return jitted

    def sample(self, key, batch_size, T, dt, step, t_min=0.0, snap_time_to_dt=True):
        kernel = self._get_kernel(int(batch_size), float(T), float(dt), bool(snap_time_to_dt))
        step_arr = jnp.asarray(step, dtype=self._dtype)
        tmin_arr = jnp.asarray(t_min, dtype=self._dtype)
        return kernel(key, step_arr, tmin_arr, self._rings)

    def _build_update_jit(self):
        """Classify each incoming state by V and write it into the matching ring
        (value_boundary / safe / unsafe), compacted.

        Each ring's usable capacity equals BUFFER_SIZE (the rule is just
        ``BUFFER_SIZE >= ~10 x BATCH_SIZE``): the previous version wrote the whole
        batch and flagged the in-band subset, so capacity was ``acceptance *
        BUFFER_SIZE`` for an acceptance rate that moves with V.

        Fixed-shape trick: per region, rank the matching rows with a cumsum and
        send them to consecutive slots; every non-matching row is routed to a
        single trash slot at index ``cap`` that nothing ever reads. No dynamic
        shapes, no host sync inside the kernel.
        """
        band = self.boundary_band; cap = self.buffer_size
        unsafe_pos = self.unsafe_is_positive

        @nnx.jit
        def upd(net, x, t, rings, pos):
            V = net(x, t)["V"]
            is_boundary = jnp.abs(V) < band
            is_unsafe = (V >= band) if unsafe_pos else (V <= -band)
            is_safe = (~is_boundary) & (~is_unsafe)
            masks = {"value_boundary": is_boundary, "safe": is_safe, "unsafe": is_unsafe}
            new_rings, counts = {}, {}
            for name, m in masks.items():
                r = rings[name]
                rank = jnp.cumsum(m) - 1                          # slot among matching rows
                dest = jnp.where(m, (pos[name] + rank) % cap, cap)  # cap == trash slot
                new_rings[name] = {
                    "vbuf": r["vbuf"].at[dest].set(x),
                    "tbuf": r["tbuf"].at[dest].set(t),
                    "vvalid": r["vvalid"].at[dest].set(m),
                }
                counts[name] = jnp.sum(m)
            return new_rings, counts
        return upd

    def update_boundary_buffer(self, x, t, value_net=None):
        net = value_net or self.value_net
        if net is None:
            return 0
        if (self.w["value_boundary"] + self.w["safe"] + self.w["unsafe"]) == 0.0:
            return 0
        pos = {name: jnp.asarray(self._pos[name], jnp.int32) for name in self._ring_names}
        self._rings, counts = self._update_jit(net, x, t, self._rings, pos)
        # Advance each ring by the number actually stored (only matching rows
        # consumed a slot). Costs one host sync per call.
        for name in self._ring_names:
            self._pos[name] = (self._pos[name] + int(counts[name])) % self.buffer_size
        return int(counts["value_boundary"])

    def seed_boundary_buffer(self, key, value_net, t, n):
        """Warm-start the value-boundary ring by projecting onto ``{V ~ 0}``.

        The ring is populated as a side effect of ordinary training batches, so
        it costs nothing -- but it starts EMPTY, and stays empty for as long as
        no sampled state happens to land in the band. Measured on an untrained
        net, ``|V| < 0.1`` matched 0.0000 of a uniform batch, so the
        value_boundary bucket silently serves its uniform fallback until V
        first crosses zero somewhere the sampler happens to look.

        Seeding removes that dead period by *constructing* in-band states
        directly: the same Newton projection the target/avoid pools use, with
        the value net as the level function. Unlike those pools this cannot be
        done once, because V moves -- so it is re-run whenever occupancy drops
        below what the batch needs.

        Uses autodiff of V, which is free here (training time, workstation);
        nothing about the deployment path changes.

        Returns the number of newly-valid states written.
        """
        if self.w["value_boundary"] <= 0.0:
            return 0
        net = value_net or self.value_net
        if net is None:
            return 0
        k_t, k_x = jax.random.split(key)
        t_vec = (jnp.full((n,), t, dtype=self._dtype) if jnp.ndim(t) == 0
                 else jnp.asarray(t, dtype=self._dtype))
        x = self._project_to_band(lambda xx: net(xx, t_vec)["V"], k_x, n)
        return int(self.update_boundary_buffer(x, t_vec, value_net=net))

    def clear_buffer(self):
        self._rings = {name: self._new_ring(self.dyn) for name in self._ring_names}
        self._pos = {name: 0 for name in self._ring_names}

    @property
    def buffer_occupancy(self):
        # value-boundary ring occupancy (used to gate seeding).
        return int(jnp.sum(self._rings["value_boundary"]["vvalid"][:self.buffer_size]))

    @staticmethod
    def curriculum_weights(progress, **spans):
        """Linear schedule as a function of ``progress`` in [0, 1] -- how far
        through the schedule training is, however the caller chooses to
        define that (e.g. Trainer._compute_sampling_weight_updates derives
        it from elapsed reachability time-horizon coverage, not raw step
        count; clamped to 1.0 past that point). This function itself doesn't
        care what progress means, only that 0 = start, 1 = fully ramped.

        Each kwarg is one bucket, keyed exactly as ``set_weights`` expects it
        (``w_uniform``, ``w_target``, ...), given either as a scalar (held
        constant) or as a ``(start, end)`` pair. A NEGATIVE ``end`` is the
        "no ramp" sentinel and means "hold at ``start``", so a curriculum is
        opt-in per bucket purely by supplying a non-negative end -- there is
        no separate enable flag. Note that ``end=0.0`` is therefore a real
        target (decay the bucket away), not "unset".
        """
        alpha = min(1.0, max(0.0, progress))
        out = {}
        for key, span in spans.items():
            start, end = span if isinstance(span, (tuple, list)) else (span, -1.0)
            if end < 0.0:
                end = start
            out[key] = start + alpha * (end - start)
        return out

    def set_weights(self, **kwargs):
        w = self.w
        self.w = self._resolve_weights(
            kwargs.get("w_uniform", w["uniform"]),
            kwargs.get("w_target", w["target"]),
            kwargs.get("w_avoid", w["avoid"]),
            kwargs.get("w_value_boundary", w["value_boundary"]),
            kwargs.get("w_nominal", w["nominal"]),
            kwargs.get("w_user_target", w["user_target"]),
            kwargs.get("w_safe", w["safe"]),
            kwargs.get("w_unsafe", w["unsafe"]),
        )