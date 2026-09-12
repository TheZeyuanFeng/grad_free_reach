from typing import Optional

import jax
import jax.numpy as jnp

from .base import Dynamics


class QuadrotorCylinderAvoidance(Dynamics):
    """Quadrotor gate avoidance safety (BRT) problem.

    All physical and geometry parameters are constructor kwargs so they can
    be overridden via ``DYNAMICS.KWARGS.*`` in the YAML / --opt flags.
    """

    def __init__(
        self,
        dtype: jnp.dtype = jnp.float32,
        # --- Thrust control ---
        thrust_min: float = -20.0,
        thrust_max: float = 20.0,
        # --- Angular-rate control bounds ---
        dwx_max: float = 8.0,
        dwy_max: float = 8.0,
        dwz_max: float = 4.0,
        # --- Physical constants ---
        mass: float = 1.0,
        CT: float = 1.0,
        gravity: float = 9.8,
        # --- State space bounds ---
        x_lim: tuple = (-3.0, 3.0),
        y_lim: tuple = (-3.0, 3.0),
        z_lim: tuple = (-3.0, 3.0),
        lin_vel_lim: tuple = (-5.0, 5.0),
        ang_vel_lim: tuple = (-5.0, 5.0),
        # --- Quadrotor body model ---
        arm_l: float = 0.17,            # quad collision disc radius
        collisionR: float = 0.5,        # cylinder radius
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
        self.arm_l = arm_l
        self.collisionR = collisionR

        
        lo = [x_lim[0], y_lim[0], z_lim[0],
              -1.0, -1.0, -1.0, -1.0,
               lin_vel_lim[0], lin_vel_lim[0], lin_vel_lim[0],
              ang_vel_lim[0], ang_vel_lim[0], ang_vel_lim[0]]
        hi = [x_lim[1], y_lim[1], z_lim[1],
               1.0,  1.0,  1.0,  1.0,
               lin_vel_lim[1],  lin_vel_lim[1], lin_vel_lim[1],
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
        # Quaternion Normalization (eps inside sqrt keeps norm/divide gradients
        #    finite even at a zero-norm quaternion; a bare norm() has a NaN gradient there).
        q = state[..., 3:7]
        q_norm = jnp.sqrt(jnp.sum(q ** 2, axis=-1, keepdims=True) + 1e-12)
        q = q / q_norm
        
        return jnp.concatenate([state[..., :3], q, state[..., 7:]], axis=-1)

    def nn_inputs(self, x: jax.Array) -> jax.Array:
        x_wrapped = self.wrap_state(x)
        rng = (self.state_high - self.state_low)
        # Linear normalization to [-1, 1]
        normed = 2.0 * (x_wrapped - self.state_low) / jnp.maximum(rng, 1e-6) - 1.0
        
        # Overwrite quaternion indices with normalized unit quaternion (don't scale q)
        return normed.at[..., 3:7].set(x_wrapped[..., 3:7])

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
    def dist_to_cylinder(self, state, a, b):
        '''for cylinder with full body collision'''
        state_ = self.wrap_state(state)
        # vector from cylinder axis (a, b) to the quadrotor centre (x, y)
        px = state_[..., 0] - a
        py = state_[..., 1] - b

        # body z-axis (thrust direction) expressed in the world frame -- the
        # rotation of [0, 0, 1] by the unit quaternion q = [qw, qx, qy, qz].
        qw, qx, qy, qz = (state_[..., 3], state_[..., 4], state_[..., 5], state_[..., 6])
        vx = 2.0 * (qw * qy + qx * qz)
        vy = 2.0 * (-qw * qx + qy * qz)
        vz = 1.0 - 2.0 * qx ** 2 - 2.0 * qy ** 2

        # full-body distance: centre-to-axis distance minus the arm's projected
        # extent toward the axis (denominator floored to keep the gradient finite).
        dist = jnp.sqrt(px ** 2 + py ** 2)
        denom = jnp.maximum(
            px ** 2 * vx ** 2 + px ** 2 * vz ** 2 + 2 * px * py * vx * vy
            + py ** 2 * vy ** 2 + py ** 2 * vz ** 2,
            1e-12,
        )
        dist = dist - jnp.sqrt(
            (self.arm_l ** 2 * px ** 2 * vz ** 2) / denom
            + (self.arm_l ** 2 * py ** 2 * vz ** 2) / denom
        )
        return jnp.maximum(dist, jnp.zeros_like(dist)) - self.collisionR

    def avoid_l(self, x: jax.Array) -> jax.Array:
        return -self.dist_to_cylinder(x, 0.0, 0.0)

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
            "z_vals": [0.0, 1.0, 2.0, 3.5, 5.0],
        }