"""Quadrotor gate avoidance — BRT safety problem.

State (13D):   [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
Control (4D):  [thrust_delta, dwx, dwy, dwz]
    thrust_delta ∈ [-thrust_half, +thrust_half]
    Physical collective thrust = thrust_delta + thrust_center
    dwx, dwy ∈ [-dwx_max, dwx_max],  dwz ∈ [-dwz_max, dwz_max]
Disturbance:   none (disturb_dim = 0)

NN features (13D): all state dims normalised to [-1, 1]
    [x_n, y_n, z_n, qw, qx, qy, qz, vx_n, vy_n, vz_n, wx_n, wy_n, wz_n]
    (quaternion kept as unit vector; position and velocities linearly scaled)

Gate geometry (infinite wall with rectangular opening at x = 0)
---------------------------------------------------------------
Flying direction: +x (quadrotor starts at x < 0, flies toward gate at x = 0)

    Opening:         |y| ≤ gate_half_w,  |z| ≤ gate_half_h
    The wall extends infinitely in y and z outside the opening.
    Gate slab depth: ±gate_slab_half in x

The quadrotor body is modelled as a ball of radius ``quad_radius``.

Unsafe set (avoid_l ≥ 0  ⟺  l ≤ 0)
--------------------------------------
1. Gate collision: ball centre within quad_radius of the infinite wall
   (x-slab ±gate_slab_half, y-z plane outside the opening).

2. Any velocity component outside its declared safe range
   (vel_safe_lo / vel_safe_hi for [vx, vy, vz, wx, wy, wz]).

The safety value function l(x) = -avoid_l(x), so l > 0 ⟺ safe.
cost_fn = min_t l(x_t); cost < 0 means the trajectory entered the unsafe set.

Problem type: BRT   CONTROL_ROLE=max  (maximiser keeps the system safe)

Usage
-----
    python train.py --opt \\
        DYNAMICS.CLASS QuadrotorGateAvoidance \\
        GAME.PROBLEM_TYPE BRT \\
        GAME.CONTROL_ROLE max \\
        IO.EXP_NAME exp_quadrotor_avoid
"""

from typing import Optional

import jax
import jax.numpy as jnp

from .base import Dynamics


class QuadrotorGateAvoidance(Dynamics):
    """Quadrotor gate avoidance safety (BRT) problem.

    All physical and geometry parameters are constructor kwargs so they can
    be overridden via ``DYNAMICS.KWARGS.*`` in the YAML / --opt flags.
    """

    def __init__(
        self,
        dtype: jnp.dtype = jnp.float32,
        # --- Thrust control ---
        thrust_min: float = 0.0,
        thrust_max: float = 9.8 * 2,
        # --- Angular-rate control bounds ---
        dwx_max: float = 8.0,
        dwy_max: float = 8.0,
        dwz_max: float = 4.0,
        # --- Physical constants ---
        mass: float = 1.0,
        CT: float = 1.0,
        gravity: float = 9.8,
        # --- State space bounds ---
        x_lim: tuple = (-4.0, 1.0),
        y_lim: tuple = (-1.5, 1.5),
        z_lim: tuple = (-1.5, 1.5),
        ang_vel_lim: tuple = (-5.0, 5.0),
        # --- Gate geometry (infinite wall with rectangular opening at x = 0) ---
        gate_half_w: float = 0.5,      # inner half-width  (y direction)
        gate_half_h: float = 0.5,      # inner half-height (z direction)
        gate_slab_half: float = 0.05,   # half-depth of gate slab in x
        # --- Quadrotor body model ---
        quad_radius: float = 0.15,      # collision ball radius
        # --- Parameters used only in sample_target_states ---
        x_behind_min: float = 0.2,
        x_behind_max: float = 0.4,
        vx_pass_min: float = 1.0,
        vx_pass_max: float = 4.5,
        target_v_weight: float = 0.2,   # kept for API compatibility
        # --- Velocity safety limits: [vx, vy, vz, wx, wy, wz] ---
        vel_safe_lo: tuple = (0.5, -2.0, -2.0, -4.5, -4.5, -4.5),
        vel_safe_hi: tuple = (4.5,  2.0,  2.0,  4.5,  4.5,  4.5),
        # --- Gate SDF shaping ---
        # slope = gate_sdf_scale within ±gate_sdf_margin of the gate boundary,
        # slope = 1 beyond (Lipschitz-continuous with constant = gate_sdf_scale).
        gate_sdf_scale: float = 10.0,
        gate_sdf_margin: float = 0.5,
    ):
        
        self.state_dim:   int = 13
        self.control_dim: int = 4   # [thrust_delta, dwx, dwy, dwz]
        self.disturb_dim: int = 0
        self.input_dim:   int = 13  # all state dims, normalised

        super().__init__()
        self.dtype = dtype

        self.thrust_min = thrust_min
        self.thrust_max = thrust_max
        self.thrust_center = (thrust_max + thrust_min) / 2.0
        self.thrust_half = (thrust_max - thrust_min) / 2.0
        self.dwx_max = dwx_max
        self.dwy_max = dwy_max
        self.dwz_max = dwz_max

        self.mass = mass
        self.CT = CT
        self.gravity = gravity

        self.gate_half_w = gate_half_w
        self.gate_half_h = gate_half_h
        self.gate_slab_half = gate_slab_half
        self.quad_radius = quad_radius

        self.x_behind_min = x_behind_min
        self.x_behind_max = x_behind_max
        self.vx_pass_min = vx_pass_min
        self.vx_pass_max = vx_pass_max
        self.target_v_weight = target_v_weight
        self.vel_safe_lo = jnp.array(vel_safe_lo, dtype=dtype)
        self.vel_safe_hi = jnp.array(vel_safe_hi, dtype=dtype)
        self.gate_sdf_scale = gate_sdf_scale
        self.gate_sdf_margin = gate_sdf_margin

        lo = [x_lim[0], y_lim[0], z_lim[0],
              -1.0, -1.0, -1.0, -1.0,
               0.0, -2.5, -2.5,
              ang_vel_lim[0], ang_vel_lim[0], ang_vel_lim[0]]
        hi = [x_lim[1], y_lim[1], z_lim[1],
               1.0,  1.0,  1.0,  1.0,
               5.0,  2.5,  2.5,
              ang_vel_lim[1], ang_vel_lim[1], ang_vel_lim[1]]

        self.state_low = jnp.array(lo, dtype=dtype)
        self.state_high = jnp.array(hi, dtype=dtype)
        self.val_state_low = self.state_low
        self.val_state_high = self.state_high

        self.u_max = jnp.array([self.thrust_half, dwx_max, dwy_max, dwz_max], dtype=dtype)
        self.d_max = jnp.zeros(0, dtype=dtype)
        self.u_min = -self.u_max
        self.d_min = -self.d_max

    # ------------------------------------------------------------------
    # State utilities
    # ------------------------------------------------------------------

    def wrap_state(self, state: jax.Array) -> jax.Array:
        # 1. Pos/Vel clamping
        pos = jnp.clip(state[..., :3], self.state_low[:3], self.state_high[:3])
        vel = jnp.clip(state[..., 7:], self.state_low[7:], self.state_high[7:])
        
        # 2. Quaternion Normalization (eps inside sqrt keeps norm/divide gradients
        #    finite even at a zero-norm quaternion; a bare norm() has a NaN gradient there).
        q = state[..., 3:7]
        q_norm = jnp.sqrt(jnp.sum(q ** 2, axis=-1, keepdims=True) + 1e-12)
        q = q / q_norm
        
        return jnp.concatenate([pos, q, vel], axis=-1)

    def nn_inputs(self, x: jax.Array) -> jax.Array:
        x_wrapped = self.wrap_state(x)
        rng = (self.state_high - self.state_low)
        # Linear normalization to [-1, 1]
        normed = 2.0 * (x_wrapped - self.state_low) / jnp.maximum(rng, 1e-6) - 1.0
        
        # Overwrite quaternion indices with normalized unit quaternion (don't scale q)
        return normed.at[..., 3:7].set(x_wrapped[..., 3:7])

    def sample_target_states(self, key: jax.Array, num_states: int) -> jax.Array:
        """Sample states near gate for boundary initialization."""
        # Create a modified bounding box for sampling
        lo = self.state_low.at[0].set(-4.0) \
                          .at[1].set(-self.gate_half_w - 0.2) \
                          .at[2].set(-self.gate_half_h - 0.2) \
                          .at[7].set(self.vx_pass_min - 0.2)
        
        hi = self.state_high.at[0].set(0.4) \
                           .at[1].set(self.gate_half_w + 0.2) \
                           .at[2].set(self.gate_half_h + 0.2) \
                           .at[7].set(self.vx_pass_max + 0.2)

        r = jax.random.uniform(key, (num_states, self.state_dim), dtype=self.dtype)
        x = lo + (hi - lo) * r
        # Half the states are forced upright (quaternion = identity [1,0,0,0], indices
        # 3:7); the other dims stay uniform. The remaining half keep a uniform-random
        # quaternion (normalized by wrap_state below).
        n_up = num_states // 2
        upright = jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0, 0.0], dtype=self.dtype),
                                   (num_states, 4))
        row_mask = (jnp.arange(num_states) < n_up)[:, None]
        x = x.at[:, 3:7].set(jnp.where(row_mask, upright, x[:, 3:7]))
        return self.wrap_state(x)

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------
    
    def f(self, x: jax.Array, u: jax.Array, d: jax.Array) -> jax.Array:
        x = self.wrap_state(x)
        
        # Unpack
        qw, qx, qy, qz = x[..., 3], x[..., 4], x[..., 5], x[..., 6]
        vx, vy, vz = x[..., 7], x[..., 8], x[..., 9]
        wx, wy, wz = x[..., 10], x[..., 11], x[..., 12]
        
        thrust = u[..., 0] + self.thrust_center
        dwx, dwy, dwz = u[..., 1], u[..., 2], u[..., 3]
        k = self.CT / self.mass

        # 1. Position Derivatives
        p_dot = x[..., 7:10]

        # 2. Quaternion Kinematics
        q_dot = jnp.stack([
            -(wx * qx + wy * qy + wz * qz) / 2.0,
             (wx * qw + wz * qy - wy * qz) / 2.0,
             (wy * qw - wz * qx + wx * qz) / 2.0,
             (wz * qw + wy * qx - wx * qy) / 2.0
        ], axis=-1)

        # 3. Linear Accelerations
        v_dot = jnp.stack([
            2.0 * (qw * qy + qx * qz) * k * thrust,
            2.0 * (-qw * qx + qy * qz) * k * thrust,
            -self.gravity + (1.0 - 2.0 * qx**2 - 2.0 * qy**2) * k * thrust
        ], axis=-1)

        # 4. Angular Accelerations (Simplified diagonal inertia)
        w_dot = jnp.stack([
            dwx - 5.0 * wy * wz / 9.0,
            dwy + 5.0 * wx * wz / 9.0,
            dwz
        ], axis=-1)

        return jnp.concatenate([p_dot, q_dot, v_dot, w_dot], axis=-1)

    # ------------------------------------------------------------------
    # Safety set functions
    # ------------------------------------------------------------------

    def avoid_l(self, x: jax.Array) -> jax.Array:
        px, py, pz = x[..., 0], x[..., 1], x[..., 2]
        hw, hh = self.gate_half_w, self.gate_half_h

        # --- 1. Gate collision SDF ---
        # 2D SDF of the opening (positive inside opening)
        sdf2d = jnp.minimum(hw - jnp.abs(py), hh - jnp.abs(pz))
        # X-depth SDF
        dx = jnp.abs(px) - self.gate_slab_half
        
        outside_2d = jnp.maximum(sdf2d, 0.0)
        outside_x  = jnp.maximum(dx, 0.0)
        interior   = jnp.minimum(jnp.maximum(sdf2d, dx), 0.0)
        sdf3d = jnp.sqrt(outside_2d**2 + outside_x**2) + interior
        
        gate_val = self.quad_radius - sdf3d
        # Scale penalties for violation
        gate_val = jnp.where(gate_val >= 0, gate_val * 5.0, gate_val)
        avoid_gate = jnp.tanh(gate_val)

        # --- 2. Velocity constraints ---
        vel = x[..., 7:]
        exceed_hi = vel - self.vel_safe_hi
        exceed_lo = self.vel_safe_lo - vel
        avoid_vel = jnp.maximum(jnp.max(exceed_hi, axis=-1), jnp.max(exceed_lo, axis=-1))

        # Combined cost (weighted)
        return jnp.maximum(avoid_gate, avoid_vel * 2.0)

    def l(self, x: jax.Array) -> jax.Array:
        """Safety function: l > 0 ⟺ safe, l ≤ 0 ⟺ unsafe (in avoid set)."""
        return -self.avoid_l(x)

    def cost_fn(self, state_traj: jax.Array, gamma: Optional[float] = None) -> jax.Array:
        l_vals = self.l(state_traj)   # (B, T)

        if gamma is not None and gamma < 1.0:
            T = l_vals.shape[-1]
            weights = gamma ** jnp.arange(T, dtype=l_vals.dtype)
            l_vals = l_vals * weights

        return jnp.min(l_vals, axis=-1)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_config(self):
        """Plot x vs y at fixed z=0, hovering attitude, moderate forward speed."""
        return {
            "state_slices": [
                0.0, 0.0, 0.0,           # x, y, z
                1.0, 0.0, 0.0, 0.0,      # qw, qx, qy, qz (identity)
                2.5, 0.0, 0.0,           # vx, vy, vz
                0.0, 0.0, 0.0,           # wx, wy, wz
            ],
            "state_labels": [
                "x", "y", "z",
                "qw", "qx", "qy", "qz",
                "vx", "vy", "vz",
                "wx", "wy", "wz",
            ],
            "x_axis_idx": 0,    # x (approach direction)
            "y_axis_idx": 1,    # y (lateral)
            "z_axis_idx": 7,    # vx
            "z_vals": [0.5, 1.0, 2.0, 3.5, 4.5],
        }