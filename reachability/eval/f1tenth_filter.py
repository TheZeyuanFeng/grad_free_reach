"""BEV-conditioned F1Tenth safety filters.

Two least-intrusive filters wrap a *nominal* controller with the learned HJ
safety value/policy so the car stays inside the drivable set:

* :class:`LeastRestrictiveF1TenthFilter` -- pass ``u_nom`` through while the value
  ``V(x) >= threshold``; switch to the learned safety policy the moment it drops
  below.
* :class:`SamplingF1TenthFilter` -- discrete-time sampling CBF: sample candidate
  controls, roll each one forward, keep the candidate closest to ``u_nom`` whose
  next-state value satisfies ``V(x') >= max((1-gamma*dt) V(x), 0)`` (all values
  taken as ``V - threshold``); fall back to the safety policy if none is feasible.

Safety is decided purely by ``V - threshold`` (no separate inflation margin).

``F1TenthBEV``'s value/policy nets render the ego-BEV *internally* from the state
(``nn_inputs`` folds in the render), so filters just call ``value_net(x, t)`` /
``policy_net(nn_inputs(x), t)`` on the full 8-D state
``[x, y, delta, v, theta, omega, slip, track_idx]`` -- no separate BEV plumbing.

Filters operate on the 7-D *physical* state used by the numpy simulator; the
active ``track_idx`` is held by the filter and appended before each net call.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from utils import decode_actions


class _BaseF1TenthFilter:
    def __init__(self, dyn, value_net, policy_net, track_idx, eval_t, threshold=0.0):
        self.dyn = dyn
        self.value_net = value_net
        self.policy_net = policy_net
        self.track_idx = float(track_idx)
        self.eval_t = float(eval_t)
        self.threshold = float(threshold)
        self._dtype = dyn.dtype

        # Jit the (raw) value forward -- renders the BEV internally -- once; reused
        # each step. Safety is decided by V - threshold, so no separate inflation.
        @nnx.jit
        def _v(vnet, x, t):
            return vnet(x, t)["V"]
        self._v = _v

        @nnx.jit
        def _pi_action(pnet, x, t):
            out = pnet(dyn.nn_inputs(x), t)
            u, _ = decode_actions(out, dyn)
            return u
        self._pi_action = _pi_action

    # -- state helpers -------------------------------------------------
    def _x8(self, s_np, n=1):
        """(n, 8) jax state from a 7-D physical numpy state + active track_idx."""
        x = np.broadcast_to(np.asarray(s_np, np.float32), (n, 7))
        col = np.full((n, 1), self.track_idx, np.float32)
        return jnp.asarray(np.concatenate([x, col], axis=1))

    def _t(self, n=1):
        return jnp.full((n,), self.eval_t, dtype=self._dtype)

    def get_value(self, s_np) -> float:
        """Calibrated safety value V(s) - threshold (positive => safe)."""
        V = float(self._v(self.value_net, self._x8(s_np), self._t())[0])
        return V - self.threshold

    def _safety_action(self, s_np) -> np.ndarray:
        x = self.dyn.wrap_state(self._x8(s_np))
        u = self._pi_action(self.policy_net, x, self._t())
        return np.asarray(u[0])


class LeastRestrictiveF1TenthFilter(_BaseF1TenthFilter):
    def filter_control(self, u_nom: np.ndarray, state: np.ndarray) -> dict:
        V = float(self._v(self.value_net, self._x8(state), self._t())[0])
        Vc = V - self.threshold
        if Vc < 0.0:
            u = self._safety_action(state)
            return {"u": np.asarray(u, np.float32), "v": V, "active": 1.0}
        return {"u": np.asarray(u_nom, np.float32).copy(), "v": V, "active": 0.0}


class SamplingF1TenthFilter(_BaseF1TenthFilter):
    """Discrete-time sampling CBF filter.

    Each candidate control is rolled forward ``filter_rollout_dt`` seconds using
    ``filter_rollout_steps`` internal Euler substeps (e.g. 0.1 s / 3 steps =
    0.0333 s each), and the CBF constraint is evaluated on the resulting state:
    ``V(x') >= max((1 - gamma * filter_rollout_dt) * V(x), 0)``. This internal
    look-ahead is independent of the outer control/sim step.
    """

    _N_NOM, _N_BANG, _N_GAUSS = 1, 4, 16

    def __init__(self, dyn, value_net, policy_net, track_idx, eval_t,
                 filter_rollout_dt=0.1, filter_rollout_steps=3, gamma=2.0,
                 threshold=0.0, n_samples=32, noise_std=0.3, turn_weight=0.25,
                 seed=0):
        super().__init__(dyn, value_net, policy_net, track_idx, eval_t, threshold)
        self.filter_rollout_dt = float(filter_rollout_dt)
        self.filter_rollout_steps = int(filter_rollout_steps)
        self.gamma = float(gamma)
        self.n_samples = int(n_samples)
        self.noise_std = float(noise_std)
        self._rng = np.random.default_rng(seed)
        self._u_max = np.asarray(dyn.u_max, np.float32)          # (2,)
        # Deviation-from-nominal weights on the [-1,1]-normalised control
        # [steering_rate, accel]; turning is higher priority so its deviation
        # is penalised less (smaller weight) when picking the closest candidate.
        self._sel_w = np.array([float(turn_weight), 1.0], np.float32)
        signs = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], np.float32)
        self._bang = signs * self._u_max                         # (4, 2)

        dt_sub = self.filter_rollout_dt / self.filter_rollout_steps
        n_sub = self.filter_rollout_steps

        @nnx.jit
        def _rollout(x8, u_batch):
            # x8: (B,8), u_batch: (B,2). Roll forward filter_rollout_dt with the JAX dyn.
            d = jnp.zeros((u_batch.shape[0], 0), dtype=x8.dtype)
            def body(x, _):
                return dyn.wrap_state(x + dt_sub * dyn.f(x, u_batch, d)), None
            xf, _ = jax.lax.scan(body, x8, None, length=n_sub)
            return xf
        self._rollout = _rollout

    def _sample_controls(self, u_nom):
        # nom(1) + bang(4) + gauss(16) + uniform(rest); truncate to n_samples if
        # n_samples < 21 so the batch size is always exactly n_samples.
        n_uni = max(0, self.n_samples - self._N_NOM - self._N_BANG - self._N_GAUSS)
        gauss = np.clip(u_nom + self._rng.standard_normal((self._N_GAUSS, 2))
                        * self._u_max * self.noise_std, -self._u_max, self._u_max)
        uni = (self._rng.random((n_uni, 2)) * 2.0 - 1.0) * self._u_max
        cat = np.concatenate([u_nom[None], self._bang, gauss, uni], axis=0)
        return cat[: self.n_samples].astype(np.float32)

    def filter_control(self, u_nom: np.ndarray, state: np.ndarray) -> dict:
        u_nom = np.asarray(u_nom, np.float32)
        V_curr = float(self._v(self.value_net, self._x8(state), self._t())[0]) - self.threshold
        V_thr = max((1.0 - self.gamma * self.filter_rollout_dt) * V_curr, 0.0)

        u_batch = self._sample_controls(u_nom)                    # (B,2)
        B = u_batch.shape[0]
        x_next = self._rollout(self._x8(state, B), jnp.asarray(u_batch))
        V_next = np.asarray(self._v(self.value_net, x_next, self._t(B))) - self.threshold

        feasible = V_next >= V_thr
        if feasible.any():
            fu = u_batch[feasible]
            # Closest feasible candidate under a weighted norm on the
            # [-1,1]-normalised control (turning penalised less than accel).
            dev = (fu - u_nom) / self._u_max
            dist = np.sqrt((dev ** 2 * self._sel_w).sum(axis=-1))
            u = fu[np.argmin(dist)]
        else:
            u = self._safety_action(state)
        u = np.asarray(u, np.float32)
        # Filter is "active" whenever it changed the action from nominal -- either a
        # feasible-but-not-u_nom pick (u_nom violated the CBF) or the recovery policy.
        active = float(not np.allclose(u, np.asarray(u_nom, np.float32), atol=1e-6))
        return {"u": u, "v": V_curr + self.threshold, "active": active}
