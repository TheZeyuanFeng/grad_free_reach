import math
import logging
from typing import List, Optional, Tuple, Dict

import jax
import jax.numpy as jnp

from .base import Dynamics

log = logging.getLogger(__name__)


class PursuitEvasion(Dynamics):
    """Pursuit-evasion game: one 4D Dubins evader vs two 3D Dubins pursuers.

    State layout (dim = 10):
        [x_e, y_e, theta_e, v_e,   x_p1, y_p1, theta_p1,   x_p2, y_p2, theta_p2]
         indices 0-3                indices 4-6               indices 7-9

    Control     (pursuers, minimises): [omega_p1, omega_p2]  dim = 2
    Disturbance (evader,   maximises): [omega_e,  a_e]        dim = 2

    Reach-avoid formulation:
        target set  l(x) <= 0 : evader caught by any pursuer OR evader hits obstacle/wall
        failure set g(x) >  0 : any pursuer hits obstacle/wall, or pursuers collide
    """

    def __init__(
        self,
        dtype:           jnp.dtype = jnp.float32,
        v_min:           float = 0.0,
        v_max_evader:    float = 1.5,
        a_max:           float = 2.0,
        omega_max_e:     float = 1.2,
        v_const:         float = 1.0,
        omega_max_p:     float = 1.0,
        catch_radius:    float = 0.75,
        robot_radius:    float = 0.25,
        field_size:      float = 10.0,
        wall_width:      float = 0.0,
        obstacle_radius: float = 1.0,
    ):
        self.num_pursuers = 2
        self.num_evaders = 1

        self.state_dim   = 4 + 3 * self.num_pursuers   # 10
        self.control_dim = self.num_pursuers           # 2: omega_p1, omega_p2
        self.disturb_dim = 2                           # 2: omega_e, a_e
        # 7 scalars (x,y per robot + evader v) + 2*3 (sin/cos per heading angle)
        self.input_dim   = 5 + 4 * self.num_pursuers

        super().__init__()

        self.dtype = dtype
        self.half = field_size / 2.0
        self.wall_width = wall_width
        self.v_min = v_min
        self.v_max_e = v_max_evader
        self.a_max = a_max
        self.omega_max_e = omega_max_e
        self.v_const = v_const
        self.omega_max_p = omega_max_p
        self.catch_radius = catch_radius
        self.robot_radius = robot_radius
        self.field_size = field_size
        self.obstacle_radius = obstacle_radius

        # Single circular obstacle at the centre of the field.
        self.circle_obstacles: List[Tuple[float, float, float]] = [
            (0.0, 0.0, obstacle_radius)
        ]

        self.u_max = jnp.array([self.omega_max_p] * self.num_pursuers, dtype=dtype)
        self.d_max = jnp.array([self.omega_max_e, self.a_max], dtype=dtype)
        self.u_min = -self.u_max
        self.d_min = -self.d_max

        lo_e = jnp.array([-self.half, -self.half, -math.pi, v_min], dtype=dtype)
        hi_e = jnp.array([self.half,  self.half,  math.pi, v_max_evader], dtype=dtype)
        lo_p = jnp.array([-self.half, -self.half, -math.pi], dtype=dtype)
        hi_p = jnp.array([self.half,  self.half,  math.pi], dtype=dtype)

        self.state_low = jnp.concatenate([lo_e] + [lo_p] * self.num_pursuers)
        self.state_high = jnp.concatenate([hi_e] + [hi_p] * self.num_pursuers)
        self.val_state_low = self.state_low
        self.val_state_high = self.state_high

        # Angle indices: theta_e at 2, theta_p1 at 6, theta_p2 at 9.
        self._angle_idxs = [2] + [self._pursuer_base(j) + 2 for j in range(self.num_pursuers)]
        self._scalar_idxs = [i for i in range(self.state_dim) if i not in self._angle_idxs]
        # Position indices (x,y for evader + each pursuer), for clamping.
        self._pos_idxs = [0, 1] + [self._pursuer_base(j) + k
                                   for j in range(self.num_pursuers) for k in range(2)]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pursuer_base(self, j: int) -> int:
        return 4 + 3 * j

    def _sdf_circle(self, xy: jax.Array, cx: float, cy: float, r: float) -> jax.Array:
        """Exact analytic SDF to a filled circle. Positive = outside (safe)."""
        c = jnp.array([cx, cy], dtype=xy.dtype)
        return jnp.linalg.norm(xy - c, axis=-1) - r - self.robot_radius

    def _sdf_boundary(self, xy: jax.Array) -> jax.Array:
        """Signed distance to field boundary. Positive = inside (safe)."""
        return jnp.min(jnp.stack([
            xy[:, 0] + self.half,
            self.half - xy[:, 0],
            xy[:, 1] + self.half,
            self.half - xy[:, 1],
        ], axis=-1), axis=-1) - self.robot_radius - self.wall_width

    def _evader_obstacle_sdf(self, x: jax.Array) -> jax.Array:
        """Min SDF of the evader to all obstacles + boundary (>0 = safe)."""
        xy_e = x[:, :2]
        sdfs = [self._sdf_boundary(xy_e)]
        for cx, cy, r in self.circle_obstacles:
            sdfs.append(self._sdf_circle(xy_e, cx, cy, r))
        return jnp.min(jnp.stack(sdfs, axis=-1), axis=-1)

    def _pursuer_obstacle_sdf(self, x: jax.Array, j: int) -> jax.Array:
        """Min SDF of pursuer j to all obstacles + boundary (>0 = safe)."""
        base = self._pursuer_base(j)
        xy_p = x[:, base:base + 2]
        sdfs = [self._sdf_boundary(xy_p)]
        for cx, cy, r in self.circle_obstacles:
            sdfs.append(self._sdf_circle(xy_p, cx, cy, r))
        return jnp.min(jnp.stack(sdfs, axis=-1), axis=-1)

    # ------------------------------------------------------------------
    # State wrapping
    # ------------------------------------------------------------------

    def wrap_state(self, x: jax.Array) -> jax.Array:
        pos = jnp.asarray(self._pos_idxs)
        ang = jnp.asarray(self._angle_idxs)
        # Clamp x, y positions for evader and all pursuers.
        x = x.at[..., pos].set(jnp.clip(x[..., pos], -self.half, self.half))
        # Wrap angles to [-pi, pi].
        x = x.at[..., ang].set((x[..., ang] + math.pi) % (2 * math.pi) - math.pi)
        # Clamp evader speed.
        x = x.at[..., 3].set(jnp.clip(x[..., 3], self.v_min, self.v_max_e))
        return x

    def nn_inputs(self, x: jax.Array) -> jax.Array:
        x = self.wrap_state(x)
        norm_x = 2.0 * (x - self.state_low) / (self.state_high - self.state_low) - 1.0
        scalars = norm_x[..., jnp.asarray(self._scalar_idxs)]
        # Encode headings as (sin, cos) of the raw wrapped angle.
        th = x[..., jnp.asarray(self._angle_idxs)]
        angles = jnp.concatenate([jnp.sin(th), jnp.cos(th)], axis=-1)
        return jnp.concatenate([scalars, angles], axis=-1)

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def f(self, x: jax.Array, u: jax.Array, d: jax.Array) -> jax.Array:
        """u: (B, 2) = [omega_p1, omega_p2] (control); d: (B, 2) = [omega_e, a_e] (disturbance)."""
        x = self.wrap_state(x)
        parts = []

        # Evader -- 4D Dubins, driven by disturbance d.
        theta_e = x[:, 2]
        v_e = x[:, 3]
        omega_e = d[:, 0]
        accel = d[:, 1]
        accel = jnp.where((v_e >= self.v_max_e) & (accel > 0), 0.0, accel)
        accel = jnp.where((v_e <= self.v_min) & (accel < 0), 0.0, accel)
        parts.append(jnp.stack(
            [v_e * jnp.cos(theta_e), v_e * jnp.sin(theta_e), omega_e, accel], axis=-1))

        # Pursuers -- 3D Dubins with constant speed, driven by control u.
        for j in range(self.num_pursuers):
            base = self._pursuer_base(j)
            theta_p = x[:, base + 2]
            omega_p = u[:, j]
            parts.append(jnp.stack(
                [self.v_const * jnp.cos(theta_p),
                 self.v_const * jnp.sin(theta_p),
                 omega_p], axis=-1))

        return jnp.concatenate(parts, axis=-1)

    # ------------------------------------------------------------------
    # Reach / avoid objectives
    # ------------------------------------------------------------------

    def target_l(self, x: jax.Array) -> jax.Array:
        """Reach condition (<=0 = target reached, pursuers win): evader caught by any
        pursuer OR evader collides with obstacle/wall."""
        x = self.wrap_state(x)
        xy_e = x[:, :2]
        catch_margins = jnp.min(jnp.stack([
            jnp.linalg.norm(xy_e - x[:, self._pursuer_base(j):self._pursuer_base(j) + 2], axis=-1)
            - self.catch_radius
            for j in range(self.num_pursuers)
        ], axis=-1), axis=-1)
        gx = jnp.minimum(catch_margins, self._evader_obstacle_sdf(x))
        gx = jnp.where(gx <= 0, gx * 3, gx)
        return gx

    def avoid_l(self, x: jax.Array) -> jax.Array:
        """Failure condition (>0 = failure): any pursuer collides with obstacle/wall
        OR pursuers collide with each other."""
        x = self.wrap_state(x)
        terms = [-self._pursuer_obstacle_sdf(x, j) for j in range(self.num_pursuers)]
        for i in range(self.num_pursuers):
            for j in range(i + 1, self.num_pursuers):
                xy_i = x[:, self._pursuer_base(i):self._pursuer_base(i) + 2]
                xy_j = x[:, self._pursuer_base(j):self._pursuer_base(j) + 2]
                terms.append(2 * self.robot_radius - jnp.linalg.norm(xy_i - xy_j, axis=-1))
        lx = jnp.max(jnp.stack(terms, axis=-1), axis=-1) - 0.1
        lx = jnp.where(lx >= 0, lx * 10, lx)  # heavy penalty on collision
        return jnp.clip(lx, max=2.0)

    def l(self, x: jax.Array) -> jax.Array:
        return -self.avoid_l(x)

    # ------------------------------------------------------------------
    # Trajectory cost (BRAT)
    # ------------------------------------------------------------------

    def cost_fn(self, state_traj: jax.Array, gamma: Optional[float] = None) -> jax.Array:
        """BRAT cost: min_t max( l(x(t)), max_{s<=t} g(x(s)) ).

        <=0: pursuers catch the evader (or force it into an obstacle) before any pursuer
             hits an obstacle. >0: evader escapes or a pursuer hits an obstacle first.
        """
        B, T, _ = state_traj.shape
        flat = state_traj.reshape(B * T, -1)
        reach_vals = self.target_l(flat).reshape(B, T)
        avoid_vals = self.avoid_l(flat).reshape(B, T)

        avoid_max_so_far = jax.lax.cummax(avoid_vals, axis=avoid_vals.ndim - 1)
        cost_at_t = jnp.maximum(reach_vals, avoid_max_so_far)

        if gamma is not None and gamma < 1.0:
            weights = gamma ** jnp.arange(T, dtype=state_traj.dtype)
            cost_at_t = cost_at_t * weights

        return jnp.min(cost_at_t, axis=-1)

    def render(self, ax) -> None:
        pass

    # ------------------------------------------------------------------
    # Plot config
    # ------------------------------------------------------------------

    def plot_config(self) -> Dict:
        v_mid = (self.v_min + self.v_max_e) / 2.0
        slices = [0.0, 0.0, 0.0, v_mid]
        pursuers = [[2.0, -2.0, -math.pi], [-3.0, 0.0, -math.pi / 2]]
        for j in range(self.num_pursuers):
            slices += pursuers[j]

        labels = ["x_e", "y_e", r"$\theta_e$", "v_e"]
        for j in range(self.num_pursuers):
            labels += [f"x_p{j+1}", f"y_p{j+1}", rf"$\theta_{{p{j+1}}}$"]

        return {
            "state_slices": slices,
            "state_labels": labels,
            "x_axis_idx":   0,
            "y_axis_idx":   1,
            "z_axis_idx":   7,
            "z_vals":       [-3, -1.5, 0, 1.5, 3],
        }
