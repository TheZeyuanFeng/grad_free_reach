"""F1Tenth bicycle dynamics conditioned on an ego-centric BEV occupancy image.

State (8D): ``[x, y, delta, v, theta, omega, slip, track_idx]`` -- world pose,
steering angle, speed, yaw, yaw rate, slip, and an integer ``track_idx`` (a
carried label, constant through integration, not a physical DOF nor an NN input).
Control (2D): ``[sv, a]`` (steering rate, accel; ``a`` clamped above ``V_SW``).
Speed-switched kinematic (|v|<1.5) / dynamic single-track bicycle; no disturbance.

``nn_inputs(x)`` renders the ego BEV (128x128) on-device from the state and
concatenates 4 normalised proprio dims ([delta, v, omega, slip]) -> 16388 features,
so the jitted rollout/TD loop needs no BEV plumbing. ``l(x)`` is a bilinear SDF
lookup (positive inside the drivable region) minus a velocity margin.

Multi-track maps/SDFs are stacked on a leading track axis (see ``datasets.ego_bev``)
and selected per sample by ``track_idx``. Pass ``track_ids=[...]`` for multi-track
or a single ``track_id``.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import jax
import jax.numpy as jnp

from datasets.ego_bev import (
    load_tracks,
    build_lx_stack,
    free_world_coords_stack,
    ego_bev_multi,
    sample_lx_multi,
)

from .base import Dynamics
from .capabilities import SupportsCustomSample, SupportsAvoidRegion

# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------

_MU = 1.0489
_C_SF = 4.718
_C_SR = 5.4562
_LF = 0.15875
_LR = 0.17145
_H_CG = 0.074
_M = 3.74
_I_ZZ = 0.04712
_G = 9.81
_LWB = _LF + _LR
_V_SW = 7.319               # velocity switch threshold for the accel limit
_KINEMATIC_V = 1.5          # |v| below this uses the kinematic model

# ---------------------------------------------------------------------------
# State / control bounds
# ---------------------------------------------------------------------------

_S_MIN,  _S_MAX = -0.4189,  0.4189
_SV_MIN, _SV_MAX = -3.2,     3.2
_A_MAX = 9.51
_V_MIN,  _V_MAX = 1.0,    10.0
_OMEGA_MAX = 4.5
_SLIP_MIN, _SLIP_MAX = -0.8,  0.8

_BEV_H = _BEV_W = 128
_BEV_DIM = _BEV_H * _BEV_W  # 16384
_PROP_DIM = 4               # delta, v, omega, slip
_OBS_EMBED_DIM = 128

_L_CORRECTION_MAX = 0.15
_PLOT_MARGIN_M = 2.0


class F1TenthBEV(Dynamics, SupportsCustomSample, SupportsAvoidRegion):
    """F1Tenth 7-DOF bicycle dynamics with an ego-BEV observation (multi-track)."""

    state_dim:     int = 8
    control_dim:   int = 2
    disturb_dim:   int = 0
    input_dim:     int = _OBS_EMBED_DIM + _PROP_DIM   # post-encode phi width
    obs_dim:       int = _BEV_DIM                       # raw flattened BEV
    obs_ch_dim:    int = 1
    obs_embed_dim: int = _OBS_EMBED_DIM
    obs_kind:      str = "bev"                          # selects the 2D conv encoder
    TRACK_IDX_COL: int = 7

    def __init__(
        self,
        tracks_dir: str,
        track_id: str = "001",
        track_ids: Optional[List[str]] = None,
        vis_track_idx: str = "",
        out_size: int = _BEV_H,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.out_size = int(out_size)

        # track_ids (non-empty) selects multi-track; else the single track_id.
        ids = list(track_ids) if track_ids else [track_id]
        self.track_ids = ids
        self.n_tracks = len(ids)
        self.vis_track_id = vis_track_idx if vis_track_idx in ids else ids[0]
        self.vis_track_idx = ids.index(self.vis_track_id)

        # ---- host-side track load (once) ----
        maps, mpps, sizes = load_tracks(tracks_dir, ids)   # (N,H,W) uint8, (N,), (N,2)
        lx = build_lx_stack(maps, mpps, sizes)             # (N,H,W) SDF, positive = safe
        free_xy, free_cnt = free_world_coords_stack(maps, mpps, sizes)

        self._maps_f = jnp.asarray(maps, dtype=dtype) / 255.0  # (N,H,W) float in [0,1]
        self._lx = jnp.asarray(lx)                             # (N,H,W)
        self._mpps = jnp.asarray(mpps, dtype=dtype)            # (N,)
        self._free_xy = jnp.asarray(free_xy)                   # (N, Nmax, 2)
        self._free_cnt = jnp.asarray(free_cnt)                 # (N,)
        self._sizes = sizes

        # Per-track world extent (metres), keyed by track id -- used by the
        # visualizer to ZOOM the value-function grid to the active track instead
        # of the global (all-tracks) bounding box.
        self.track_bounds = {
            tid: {"x_min": 0.0, "x_max": float(sizes[i, 1] * mpps[i]),
                  "y_min": 0.0, "y_max": float(sizes[i, 0] * mpps[i])}
            for i, tid in enumerate(ids)
        }

        # ---- state bounds (x,y span the largest track extent) ----
        x_max = float((sizes[:, 1] * mpps).max())
        y_max = float((sizes[:, 0] * mpps).max())
        lo = [0.0,   0.0,   _S_MIN, _V_MIN, -math.pi, -_OMEGA_MAX, _SLIP_MIN, 0.0]
        hi = [x_max, y_max, _S_MAX, _V_MAX,  math.pi,  _OMEGA_MAX, _SLIP_MAX,
              float(self.n_tracks - 1)]
        self.state_low = jnp.array(lo, dtype=dtype)
        self.state_high = jnp.array(hi, dtype=dtype)
        self.val_state_low = self.state_low
        self.val_state_high = self.state_high

        self.u_max = jnp.array([_SV_MAX, _A_MAX], dtype=dtype)
        self.u_min = -self.u_max
        self.d_max = jnp.zeros(0, dtype=dtype)
        self.d_min = -self.d_max

        # Proprio normalisation: [delta, v, omega, slip] = x[..., [2, 3, 5, 6]]
        self._prop_lo = jnp.array([_S_MIN, _V_MIN, -_OMEGA_MAX, _SLIP_MIN], dtype=dtype)
        self._prop_hi = jnp.array([_S_MAX, _V_MAX,  _OMEGA_MAX, _SLIP_MAX], dtype=dtype)

        self._plot_bounds = {
            "x_min": -_PLOT_MARGIN_M, "x_max": x_max + _PLOT_MARGIN_M,
            "y_min": -_PLOT_MARGIN_M, "y_max": y_max + _PLOT_MARGIN_M,
        }

    def _get_track_idx(self, x: jax.Array) -> jax.Array:
        """Integer track label carried in state column TRACK_IDX_COL.

        Clamped to a valid track so a dummy / out-of-range value (e.g. at
        deployment for monitoring/filtering, where the caller doesn't know or
        care which library track the observation came from) is always safe.
        """
        idx = jnp.round(x[..., self.TRACK_IDX_COL]).astype(jnp.int32)
        return jnp.clip(idx, 0, self.n_tracks - 1)

    # ------------------------------------------------------------------
    # Control clamp + dynamics
    # ------------------------------------------------------------------

    def _clamp_control(self, x: jax.Array, u: jax.Array) -> jax.Array:
        """Velocity-dependent upper bound on longitudinal acceleration."""
        v = x[..., 3]
        a_upper = jnp.where(v > _V_SW, _A_MAX * _V_SW / v, _A_MAX)
        a = jnp.minimum(u[..., 1], a_upper)
        return jnp.stack([u[..., 0], a], axis=-1)

    def f(self, x: jax.Array, u: jax.Array, d: jax.Array) -> jax.Array:
        """Continuous-time dynamics dx/dt; blends kinematic/dynamic by |v|<1.5."""
        u = self._clamp_control(x, u)
        delta, v, theta, omega, slip = (x[..., i] for i in (2, 3, 4, 5, 6))
        sv, a = u[..., 0], u[..., 1]

        # ---- kinematic branch ----
        k_x = v * jnp.cos(theta)
        k_y = v * jnp.sin(theta)
        k_theta = v / _LWB * jnp.tan(delta)
        k_omega = (a / _LWB * jnp.tan(delta)
                   + v / (_LWB * jnp.cos(delta) ** 2) * sv)
        k_slip = jnp.zeros_like(v)

        # ---- dynamic branch ----
        v_safe = jnp.where(jnp.abs(v) < _KINEMATIC_V, _KINEMATIC_V, v)  # guard 1/v
        d_x = v * jnp.cos(slip + theta)
        d_y = v * jnp.sin(slip + theta)
        d_theta = omega
        d_omega = _omega_dot(v_safe, delta, omega, slip, a)
        d_slip = _slip_dot(v_safe, delta, omega, slip, a)

        kin = jnp.abs(v) < _KINEMATIC_V
        xdot = jnp.where(kin, k_x, d_x)
        ydot = jnp.where(kin, k_y, d_y)
        theta_dot = jnp.where(kin, k_theta, d_theta)
        omega_dot = jnp.where(kin, k_omega, d_omega)
        slip_dot = jnp.where(kin, k_slip, d_slip)

        # track_idx is a discrete label -> zero derivative keeps it constant.
        track_dot = jnp.zeros_like(v)
        return jnp.stack([xdot, ydot, sv, a, theta_dot, omega_dot, slip_dot, track_dot],
                         axis=-1)

    def wrap_state(self, x: jax.Array) -> jax.Array:
        theta = (x[..., 4] + jnp.pi) % (2 * jnp.pi) - jnp.pi
        x = x.at[..., 4].set(theta)
        lo, hi = self.state_low.astype(x.dtype), self.state_high.astype(x.dtype)
        for i in (2, 3, 5, 6):   # delta, v, omega, slip (x,y,track_idx untouched)
            x = x.at[..., i].set(jnp.clip(x[..., i], lo[i], hi[i]))
        return x

    def step(self, x, u, d, dt) -> jax.Array:
        return self.wrap_state(x + dt * self.f(x, u, d))

    # ------------------------------------------------------------------
    # Boundary / cost
    # ------------------------------------------------------------------

    def l_correction(self, x: jax.Array) -> jax.Array:
        v = x[..., 3]
        return jnp.clip(jnp.abs(v) * 0.02, max=_L_CORRECTION_MAX)

    def l(self, x: jax.Array) -> jax.Array:
        """Signed distance to the track boundary (positive inside drivable)."""
        track_idx = self._get_track_idx(x)
        sdf = sample_lx_multi(self._lx, track_idx, x[..., 0], x[..., 1], self._mpps)
        return sdf - self.l_correction(x)

    def avoid_l(self, x: jax.Array) -> jax.Array:
        return -self.l(x)

    def cost_fn(self, state_traj: jax.Array, gamma: Optional[float] = None) -> jax.Array:
        """min_t l(x_t) along a trajectory (B, T, 8) -> (B,)."""
        return jnp.min(self.l(state_traj), axis=-1)

    # ------------------------------------------------------------------
    # NN feature map — render the ego BEV from state, then pack proprio
    # ------------------------------------------------------------------

    def nn_inputs(self, x: jax.Array) -> jax.Array:
        track_idx = self._get_track_idx(x)
        wx, wy, v, theta = x[..., 0], x[..., 1], x[..., 3], x[..., 4]
        bev = ego_bev_multi(self._maps_f, track_idx, wx, wy, theta, v,
                            self._mpps, self.out_size)
        bev_flat = bev.reshape(*bev.shape[:-2], -1)          # (..., 16384)

        prop = jnp.stack([x[..., 2], x[..., 3], x[..., 5], x[..., 6]], axis=-1)
        lo, hi = self._prop_lo.astype(x.dtype), self._prop_hi.astype(x.dtype)
        prop_n = 2.0 * (prop - lo) / (hi - lo) - 1.0         # (..., 4) in [-1, 1]

        return jnp.concatenate([bev_flat, prop_n], axis=-1)  # (..., 16388)

    # ------------------------------------------------------------------
    # Sampling — pick a track, then a drivable pose on it (SupportsCustomSample)
    # ------------------------------------------------------------------

    def sample(self, key: jax.Array, n: int) -> jax.Array:
        k1, k2, k3 = jax.random.split(key, 3)
        track_idx = jax.random.randint(k1, (n,), 0, self.n_tracks)

        # free-pixel index within the chosen track's valid count
        u = jax.random.uniform(k2, (n,), dtype=self.dtype)
        idx = jnp.floor(u * self._free_cnt[track_idx].astype(self.dtype)).astype(jnp.int32)
        pos = self._free_xy[track_idx, idx]                  # (n, 2) world metres

        R = jax.random.uniform(k3, (n, 5), dtype=self.dtype)
        theta = (R[:, 0] * 2.0 - 1.0) * jnp.pi
        v = _V_MIN + R[:, 1] * (_V_MAX - _V_MIN)
        delta = _S_MIN + R[:, 2] * (_S_MAX - _S_MIN)
        omega = -_OMEGA_MAX + R[:, 3] * (2.0 * _OMEGA_MAX)
        slip = _SLIP_MIN + R[:, 4] * (_SLIP_MAX - _SLIP_MIN)

        state = jnp.stack([pos[:, 0], pos[:, 1], delta, v, theta, omega, slip,
                           track_idx.astype(self.dtype)], axis=-1)
        return self.wrap_state(state)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_config(self):
        b = self._plot_bounds
        return {
            "state_slices": [0.0, 0.0, 0.0, 5.0, math.pi / 2, 0.0, 0.0,
                             float(self.vis_track_idx)],
            "state_labels": [
                "x (m)", "y (m)", "delta (rad)", "v (m/s)",
                "theta (rad)", "omega (rad/s)", "slip (rad)", "track_idx",
            ],
            "x_axis_idx": 0,
            "y_axis_idx": 1,
            "z_axis_idx": 3,
            "z_vals": [1.0, 4.0, 7.0],
            "track_idx": self.vis_track_id,
            **b,
        }


# ---------------------------------------------------------------------------
# Dynamic-bicycle derivative helpers (Pacejka-free linear tyre model)
# ---------------------------------------------------------------------------

def _omega_dot(v, delta, omega, slip, a):
    Ff_eff = _G * _LR - a * _H_CG
    Fr_eff = _G * _LF + a * _H_CG
    return (
        -_MU * _M / (v * _I_ZZ * _LWB)
        * (_LF ** 2 * _C_SF * Ff_eff + _LR ** 2 * _C_SR * Fr_eff) * omega
        + _MU * _M / (_I_ZZ * _LWB)
        * (_LR * _C_SR * Fr_eff - _LF * _C_SF * Ff_eff) * slip
        + _MU * _M / (_I_ZZ * _LWB) * _LF * _C_SF * Ff_eff * delta
    )


def _slip_dot(v, delta, omega, slip, a):
    Ff_eff = _G * _LR - a * _H_CG
    Fr_eff = _G * _LF + a * _H_CG
    return (
        (_MU / (v ** 2 * _LWB) * (_C_SR * Fr_eff * _LR - _C_SF * Ff_eff * _LF) - 1) * omega
        - _MU / (v * _LWB) * (_C_SR * Fr_eff + _C_SF * Ff_eff) * slip
        + _MU / (v * _LWB) * _C_SF * Ff_eff * delta
    )
