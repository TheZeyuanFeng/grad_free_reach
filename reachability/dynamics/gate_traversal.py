"""Quadrotor gate traversal — reach-avoid problem.

State (13D):   [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
Control (4D):  [thrust_delta, dwx, dwy, dwz]
    thrust_delta ∈ [-thrust_half, +thrust_half]
    Physical collective thrust = thrust_delta + thrust_center
    dwx, dwy ∈ [-dwx_max, dwx_max],  dwz ∈ [-dwz_max, dwz_max]
Disturbance:   none (disturb_dim = 0)

NN features (13D): all state dims normalised to [-1, 1]
    [x_n, y_n, z_n, qw, qx, qy, qz, vx_n, vy_n, vz_n, wx_n, wy_n, wz_n]
    (quaternion kept as unit vector; position and velocities linearly scaled)

Gate geometry (rectangular frame in the x=0 plane)
---------------------------------------------------
Flying direction: +x (quadrotor starts at x < 0, flies toward gate at x = 0)

    Inner opening:   |y| ≤ gate_half_w,  |z| ≤ gate_half_h
    Frame material:  gate_thickness wide border around the opening
    Gate slab depth: ±gate_slab_half in x

The quadrotor body is modelled as a ball of radius ``quad_radius``.

Reach set (target_l ≤ 0)
--------------------------
    x  ∈ [x_behind_min,  x_behind_max]   (slightly behind gate, x > 0)
    |y| ≤ gate_half_w                     (within gate opening)
    |z| ≤ gate_half_h
    vx ∈ [vx_pass_min,   vx_pass_max]    (desired forward speed)

Avoid set (avoid_l ≥ 0)
--------------------------
The ball collides with the gate frame when the 3D signed distance from the
ball centre to the gate structure is less than ``quad_radius``.

Problem type: BRAT   CONTROL_ROLE=min  (minimiser flies through the gate)

Usage
-----
    python train.py --opt \\
        DYNAMICS.CLASS QuadrotorGateTraversal \\
        GAME.PROBLEM_TYPE BRAT \\
        GAME.CONTROL_ROLE min \\
        IO.EXP_NAME exp_quadrotor_gate
"""

from typing import Optional

import jax
import jax.numpy as jnp

from configs.constants import PROJECT_NAME
from .base import Dynamics

class QuadrotorGateTraversal(Dynamics):
    """Quadrotor gate traversal reach-avoid problem.

    All physical and geometry parameters are constructor kwargs so they can
    be overridden via ``DYNAMICS.KWARGS.*`` in the YAML / --opt flags.
    """

    state_dim:   int = 13
    control_dim: int = 4   # [thrust_delta, dwx, dwy, dwz]
    disturb_dim: int = 0
    input_dim:   int = 13  # all state dims, normalised

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
        x_lim: tuple = (-9.0, 1.0),
        y_lim: tuple = (-3.0, 3.0),
        z_lim: tuple = (-3.0, 3.0),
        ang_vel_lim: tuple = (-5.0, 5.0),
        # --- Gate geometry (rectangular frame at x = 0) ---
        gate_half_w: float = 0.4,       # inner half-width  (y direction)
        gate_half_h: float = 0.4,       # inner half-height (z direction)
        gate_thickness: float = 0.15,   # frame border width (y and z)
        gate_slab_half: float = 0.05,   # half-depth of gate in x
        # --- Quadrotor body model ---
        quad_radius: float = 0.15,      # collision ball radius
        # --- Target zone (behind the gate, x > 0) ---
        x_behind_min: float = 0.2,      # minimum x behind gate
        x_behind_max: float = 0.4,      # maximum x behind gate
        vx_pass_min: float = 1.0,       # minimum forward speed through gate
        vx_pass_max: float = 4.5,       # maximum forward speed through gate
        target_v_weight: float = 0.2,   # adjusting v diff for target lx computation
        # --- Additional safety constraints ---
        vx_gate_min: float = 0.3,       # minimum forward speed required inside the gate
        # velocities must stay within this fraction of state bounds
        vel_limit_fraction: float = 0.9,
    ):
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
        self.gate_thickness = gate_thickness
        self.gate_slab_half = gate_slab_half
        self.quad_radius = quad_radius

        self.x_behind_min = x_behind_min
        self.x_behind_max = x_behind_max
        self.vx_pass_min = vx_pass_min
        self.vx_pass_max = vx_pass_max
        self.target_v_weight = target_v_weight
        self.vx_gate_min = vx_gate_min
        self.vel_limit_fraction = vel_limit_fraction

        lo = [x_lim[0], y_lim[0], z_lim[0],
              -1.0, -1.0, -1.0, -1.0,
              -1.0, -5.0, -5.0,
              ang_vel_lim[0], ang_vel_lim[0], ang_vel_lim[0]]
        hi = [x_lim[1], y_lim[1], z_lim[1],
              1.0,   1.0,  1.0,  1.0,
              5.0, 5.0, 5.0,
              ang_vel_lim[1], ang_vel_lim[1], ang_vel_lim[1]]

        self.state_low = jnp.array(lo, dtype=dtype)
        self.state_high = jnp.array(hi, dtype=dtype)

        lo_val = [-5.0, y_lim[0], z_lim[0],
                  -1.0, -1.0, -1.0, -1.0,
                  -1.0, -5.0, -5.0,
                  ang_vel_lim[0], ang_vel_lim[0], ang_vel_lim[0]]
        hi_val = [-0.5, y_lim[1], z_lim[1],
                  1.0,   1.0,  1.0,  1.0,
                  5.0, 5.0, 5.0,
                  ang_vel_lim[1], ang_vel_lim[1], ang_vel_lim[1]]

        self.val_state_low = jnp.array(lo_val, dtype=dtype)
        self.val_state_high = jnp.array(hi_val, dtype=dtype)

        self.u_max = jnp.array(
            [self.thrust_half, dwx_max, dwy_max, dwz_max],
            dtype=dtype,
        )
        self.d_max = jnp.zeros(0, dtype=dtype)
        self.u_min = -self.u_max
        self.d_min = -self.d_max

    # ------------------------------------------------------------------
    # State utilities
    # ------------------------------------------------------------------

    def wrap_state(self, state: jax.Array) -> jax.Array:
        pos = jnp.clip(state[..., :3], self.state_low[:3], self.state_high[:3])
        # Guarded quaternion normalization: an unguarded q/||q|| gives 0/0 = NaN at a
        # zero-norm quaternion, and d(q/||q||)/dq ~ 1/||q|| explodes for tiny norms ->
        # NaN gradients through the differentiated value path. Adding eps INSIDE the
        # sqrt keeps both the norm and the division (and their gradients) finite.
        qn = state[..., 3:7]
        norm = jnp.sqrt(jnp.sum(qn ** 2, axis=-1, keepdims=True) + 1e-12)
        q = qn / norm
        vel = jnp.clip(state[..., 7:], self.state_low[7:], self.state_high[7:])
        return jnp.concatenate([pos, q, vel], axis=-1)

    def nn_inputs(self, x: jax.Array) -> jax.Array:
        x_wrapped = self.wrap_state(x)
        lo, hi = self.state_low, self.state_high
        normed = 2.0 * (x_wrapped - lo) / (hi - lo) - 1.0
        return normed.at[..., 3:7].set(x_wrapped[..., 3:7])

    def sample_target_states(self, key: jax.Array, num_states: int) -> jax.Array:
        """Sample states uniformly from the target zone (behind gate, in hole)."""
        r = jax.random.uniform(key, (num_states, self.state_dim), dtype=self.dtype)
        # Restrict x, y, z, vx to the target zone (JAX arrays are immutable -> .at[].set).
        lo = self.state_low.astype(self.dtype).at[0].set(-1.0) \
                 .at[1].set(-self.gate_half_w).at[2].set(-self.gate_half_h) \
                 .at[7].set(self.vx_pass_min - 0.2)
        hi = self.state_high.astype(self.dtype).at[0].set(0.4) \
                 .at[1].set(self.gate_half_w).at[2].set(self.gate_half_h) \
                 .at[7].set(self.vx_pass_max + 0.2)
        x = lo[None, :] + (hi - lo)[None, :] * r
        # Half the states flat/upright: quaternion = identity [1,0,0,0] (indices 3:7),
        # other dims uniform; the other half keep a uniform-random quaternion.
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
        qw, qx, qy, qz = x[..., 3], x[..., 4], x[..., 5], x[..., 6]
        vx, vy, vz = x[..., 7], x[..., 8], x[..., 9]
        wx, wy, wz = x[..., 10], x[..., 11], x[..., 12]

        thrust = u[..., 0] + self.thrust_center
        k = self.CT / self.mass

        # Compute derivatives individually
        p_dot = x[..., 7:10]
        q_dot = 0.5 * jnp.stack([
            -(wx * qx + wy * qy + wz * qz),
            (wx * qw + wz * qy - wy * qz),
            (wy * qw - wz * qx + wx * qz),
            (wz * qw + wy * qx - wx * qy)
        ], axis=-1)

        v_dot = jnp.stack([
            2.0 * (qw * qy + qx * qz) * k * thrust,
            2.0 * (-qw * qx + qy * qz) * k * thrust,
            -self.gravity + (1.0 - 2.0 * qx**2 - 2.0 * qy**2) * k * thrust
        ], axis=-1)

        w_dot = jnp.stack([
            u[..., 1] - 5.0 * wy * wz / 9.0,
            u[..., 2] + 5.0 * wx * wz / 9.0,
            u[..., 3]
        ], axis=-1)

        return jnp.concatenate([p_dot, q_dot, v_dot, w_dot], axis=-1)

    # ------------------------------------------------------------------
    # Reach-avoid set functions
    # ------------------------------------------------------------------

    def target_l(self, x: jax.Array) -> jax.Array:
        px, py, pz, vx = x[..., 0], x[..., 1], x[..., 2], x[..., 7]
        x_mid = (self.x_behind_min + self.x_behind_max) / 2.0
        x_half = (self.x_behind_max - self.x_behind_min) / 2.0
        vx_mid = (self.vx_pass_min + self.vx_pass_max) / 2.0
        vx_half = (self.vx_pass_max - self.vx_pass_min) / 2.0

        terms = jnp.stack([
            jnp.abs(px - x_mid) - x_half,
            jnp.abs(py) - self.gate_half_w + self.quad_radius,
            jnp.abs(pz) - self.gate_half_h + self.quad_radius,
            jnp.abs(vx - vx_mid) - vx_half
        ], axis=-1)
        target_l_raw = jnp.max(terms, axis=-1)
        # Scale each side so the raw range lands in ~[-5, 2] (neg side x10, pos side
        # x0.2), then tanh-bound. Sign-preserving, so the target set is unchanged.
        scaled = jnp.where(target_l_raw <= 0, target_l_raw * 20.0, target_l_raw * 0.2)
        return jnp.tanh(scaled)

    def avoid_l(self, x: jax.Array) -> jax.Array:
        px, py, pz, vx = x[..., 0], x[..., 1], x[..., 2], x[..., 7]
        hw, hh, t = self.gate_half_w, self.gate_half_h, self.gate_thickness

        # 1. Gate SDF
        sdf_outer = jnp.maximum(jnp.abs(py) - (hw + t), jnp.abs(pz) - (hh + t))
        sdf_inner = jnp.maximum(jnp.abs(py) - hw, jnp.abs(pz) - hh)
        sdf2d = jnp.maximum(sdf_outer, -sdf_inner)
        dx = jnp.abs(px) - self.gate_slab_half
        sdf3d = jnp.sqrt(jnp.clip(sdf2d, 0)**2 + jnp.clip(dx, 0)**2) + jnp.minimum(jnp.maximum(sdf2d, dx), 0)
        avoid_gate = self.quad_radius - sdf3d

        # 2. Forward speed in gate
        in_gate_margin = jnp.min(jnp.stack([self.gate_slab_half - jnp.abs(px), hw - jnp.abs(py), hh - jnp.abs(pz)], axis=-1), axis=-1)
        avoid_vx = jnp.minimum(in_gate_margin, self.vx_gate_min - vx)

        # 3. Crossing outside opening
        avoid_forward = jnp.min(jnp.stack([t - jnp.abs(px), vx*0.8, jnp.maximum(jnp.abs(py)-hw, jnp.abs(pz)-hh)], axis=-1), axis=-1)

        avoid_raw = jnp.maximum(jnp.maximum(avoid_gate, avoid_vx), avoid_forward)
        # Scale each side so the raw range lands in ~[-2, 2] (neg side /8, pos side
        # x10), then tanh-bound. Sign-preserving, so the avoid set is unchanged.
        scaled = jnp.where(avoid_raw < 0, avoid_raw*2, avoid_raw * 10.0)
        return jnp.tanh(scaled)

    def l(self, x: jax.Array) -> jax.Array:
        """Alias for target_l (used by BRS/BRT compatibility)."""
        return self.avoid_l(x)

    def cost_fn(self, state_traj: jax.Array, gamma: Optional[float] = None) -> jax.Array:
        avoid_vals = self.avoid_l(state_traj)
        avoid_max = jax.lax.cummax(avoid_vals, axis=avoid_vals.ndim - 1)
        cost = jnp.maximum(self.target_l(state_traj), avoid_max)
        if gamma is not None:
            cost *= (gamma ** jnp.arange(cost.shape[-1]))
        return jnp.min(cost, axis=-1)

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
            "z_axis_idx": 8,    # vy
            "z_vals": [-2.0, -1.0, 0.0, 1.0, 2.0],
        }

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

    def _create_state_grid(
        self, 
        resolution: int = 50,
        x_slice: Optional[tuple] = None,
        y_slice: Optional[tuple] = None,
    ) -> tuple:
        """Create a grid of states in x-y plane (z=0, hovering attitude).
        
        Args:
            resolution: number of points per axis
            x_slice: (min, max) for x axis, default uses state bounds
            y_slice: (min, max) for y axis, default uses state bounds
            
        Returns:
            (X, Y, states) where X, Y are meshgrids and states is (N, 13)
        """
        import numpy as np
        
        if x_slice is None:
            x_slice = (self.state_low[0], self.state_high[0])
        if y_slice is None:
            y_slice = (self.state_low[1], self.state_high[1])
        
        x_vals = np.linspace(x_slice[0], x_slice[1], resolution)
        y_vals = np.linspace(y_slice[0], y_slice[1], resolution)
        X, Y = np.meshgrid(x_vals, y_vals)
        
        # Flatten for batch processing
        x_flat = X.flatten()
        y_flat = Y.flatten()
        
        # Create state with fixed other dimensions (hovering, moderate forward speed)
        n_states = len(x_flat)
        states_np = np.zeros((n_states, self.state_dim))
        states_np[:, 0] = x_flat  # x
        states_np[:, 1] = y_flat  # y
        states_np[:, 2] = 0.0     # z
        states_np[:, 3] = 1.0     # qw (identity quaternion)
        states_np[:, 4:7] = 0.0   # qx, qy, qz
        states_np[:, 7] = 2.5     # vx (moderate forward speed)
        states_np[:, 8:13] = 0.0  # vy, vz, wx, wy, wz
        
        states = jnp.array(states_np, dtype=self.dtype)
        return X, Y, states

    def plot_domination_regions(
        self,
        V_pred: Optional[jax.Array] = None,
        resolution: int = 50,
        figsize: tuple = (10, 8),
    ):
        """Plot BRAT domination regions on the x-y state space.
        
        Single plot showing where avoid, target, and reachability costs dominate.
        
        Args:
            V_pred: (N,) tensor of predicted values. Required for domination analysis.
            resolution: grid resolution
            figsize: figure size
            
        Returns:
            (fig, ax) tuple
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import patches
        
        # Create state grid
        X, Y, states = self._create_state_grid(resolution=resolution)
        
        # Compute losses
        lA = self.avoid_l(states)
        lT = self.target_l(states)
        
        lA_np = np.asanyarray(lA).reshape(X.shape)
        lT_np = np.asanyarray(lT).reshape(X.shape)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if V_pred is None:
            # Without value function: show raw constraints
            raise ValueError("plot_domination_regions requires V_pred. Use plot_constraints instead.")
        
        # Convert V_pred to numpy
        V_pred_np = np.asanyarray(V_pred).reshape(X.shape) if isinstance(V_pred, jax.Array) else V_pred.reshape(X.shape)
        
        # BRAT domination analysis
        avoid_dominates = (lA_np > V_pred_np).astype(float)
        target_dominates = ((lT_np < V_pred_np) & (lA_np < V_pred_np)).astype(float)
        reachability_dominates = 1.0 - avoid_dominates - target_dominates
        
        # Create RGB image: Red=avoid, Blue=target, Green=reachable
        rgb_map = np.zeros((*X.shape, 3))
        rgb_map[..., 0] = avoid_dominates           # Red
        rgb_map[..., 2] = target_dominates          # Blue
        rgb_map[..., 1] = reachability_dominates    # Green
        
        # Display the domination map
        ax.imshow(rgb_map, extent=[X.min(), X.max(), Y.min(), Y.max()],
                 origin='lower', aspect='auto', alpha=0.85, interpolation='nearest')
        
        # Add contours separating regions
        contour1 = ax.contour(X, Y, avoid_dominates, levels=[0.5], colors='darkred', 
                             linewidths=2.5, linestyles='-', label='Avoid boundary')
        contour2 = ax.contour(X, Y, target_dominates, levels=[0.5], colors='darkblue', 
                             linewidths=2.5, linestyles='-', label='Target boundary')
        contour3 = ax.contour(X, Y, reachability_dominates, levels=[0.5], colors='darkgreen', 
                             linewidths=2.5, linestyles='-', label='Reachability boundary')
        
        ax.clabel(contour1, inline=True, fontsize=8)
        ax.clabel(contour2, inline=True, fontsize=8)
        ax.clabel(contour3, inline=True, fontsize=8)
        
        ax.set_xlabel('x (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('y (m)', fontsize=12, fontweight='bold')
        ax.set_title('BRAT Cost Domination Map', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.85, edgecolor='darkred', linewidth=2, label='Avoid-dominates'),
            Patch(facecolor='green', alpha=0.85, edgecolor='darkgreen', linewidth=2, label='Reachability-dominates'),
            Patch(facecolor='blue', alpha=0.85, edgecolor='darkblue', linewidth=2, label='Target-dominates'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.95, 
                 title='Cost Dominance', title_fontsize=12)
        
        plt.tight_layout()
        return fig, ax

    def plot_constraints(self, resolution: int = 50, figsize: tuple = (14, 6)):
        """Plot raw constraint landscapes (avoid_l and target_l) without value function.
        
        Args:
            resolution: grid resolution
            figsize: figure size
            
        Returns:
            (fig, axes) tuple
        """
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Create state grid
        X, Y, states = self._create_state_grid(resolution=resolution)
        
        # Compute losses
        lA = self.avoid_l(states)
        lT = self.target_l(states)
        
        lA_np = np.asanyarray(lA).reshape(X.shape)
        lT_np = np.asanyarray(lT).reshape(X.shape)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: avoid_l
        ax = axes[0]
        contour_a = ax.contourf(X, Y, lA_np, levels=20, cmap='Reds', alpha=0.7)
        contour_lines_a = ax.contour(X, Y, lA_np, levels=[0], colors='darkred', linewidths=2.5)
        ax.clabel(contour_lines_a, inline=True, fontsize=8)
        plt.colorbar(contour_a, ax=ax, label='avoid_l')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title('Obstacle Constraint (avoid_l)\nRed region = collision')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: target_l
        ax = axes[1]
        contour_t = ax.contourf(X, Y, lT_np, levels=20, cmap='Blues', alpha=0.7)
        contour_lines_t = ax.contour(X, Y, lT_np, levels=[0], colors='darkblue', linewidths=2.5)
        ax.clabel(contour_lines_t, inline=True, fontsize=8)
        plt.colorbar(contour_t, ax=ax, label='target_l')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title('Target Constraint (target_l)\nBlue region = in target')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, axes

    def plot_target_vs_avoid_dominance(self, resolution: int = 50, figsize: tuple = (10, 8)):
        """Plot which constraint (target or avoid) is more violated across the state space.
        
        Shows target-dominated regions (where target_l > avoid_l) vs 
        avoid-dominated regions (where avoid_l > target_l).
        
        Args:
            resolution: grid resolution
            figsize: figure size
            
        Returns:
            (fig, ax) tuple
        """
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Create state grid
        X, Y, states = self._create_state_grid(resolution=resolution)
        
        # Compute losses
        lA = self.avoid_l(states)
        lT = self.target_l(states)
        
        lA_np = np.asanyarray(lA).reshape(X.shape)
        lT_np = np.asanyarray(lT).reshape(X.shape)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Comparison: which constraint is more violated?
        avoid_dominates = (lA_np > lT_np).astype(float)
        target_dominates = 1.0 - avoid_dominates
        
        # Create RGB image: Red=avoid-dominates, Blue=target-dominates
        rgb_map = np.zeros((*X.shape, 3))
        rgb_map[..., 0] = avoid_dominates      # Red
        rgb_map[..., 2] = target_dominates     # Blue
        
        # Display the map
        ax.imshow(rgb_map, extent=[X.min(), X.max(), Y.min(), Y.max()],
                 origin='lower', aspect='auto', alpha=0.85, interpolation='nearest')
        
        # Add contour at the boundary
        contour = ax.contour(X, Y, avoid_dominates, levels=[0.5], colors='black', 
                            linewidths=2.5, linestyles='-')
        ax.clabel(contour, inline=True, fontsize=8)
        
        ax.set_xlabel('x (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('y (m)', fontsize=12, fontweight='bold')
        ax.set_title('Target vs Avoid Constraint Dominance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.85, edgecolor='darkred', linewidth=2, label='Avoid-dominates'),
            Patch(facecolor='blue', alpha=0.85, edgecolor='darkblue', linewidth=2, label='Target-dominates'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.95, 
                 title='Constraint Dominance', title_fontsize=12)
        
        plt.tight_layout()
        return fig, ax

    def render(self, ax) -> None:
        """Render the gate traversal environment (x-y plane, z=0).
        
        Shows the rectangular gate frame and the reachable region.
        """
        from matplotlib import patches
        import matplotlib.pyplot as plt

        # Draw outer boundary of state space in x-y
        x_min, x_max = self.state_low[0].item(), self.state_high[0].item()
        y_min, y_max = self.state_low[1].item(), self.state_high[1].item()
        
        ax.add_patch(patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2.0, edgecolor="black", facecolor="none",
            linestyle="--", alpha=0.3, label="State bounds",
        ))

        # Draw gate frame (at x=0, rectangular opening)
        # Outer frame boundary
        hw = self.gate_half_w
        hh = self.gate_half_h
        t = self.gate_thickness
        sx = self.gate_slab_half

        # Frame: four rectangles forming the border around the opening
        frame_rects = [
            # Top bar
            patches.Rectangle((-sx, hh), 2*sx, t,
                             linewidth=1.5, edgecolor="darkred", facecolor="red", alpha=0.4),
            # Bottom bar
            patches.Rectangle((-sx, -hh - t), 2*sx, t,
                             linewidth=1.5, edgecolor="darkred", facecolor="red", alpha=0.4),
            # Left bar
            patches.Rectangle((-sx, -hh), t, 2*hh,
                             linewidth=1.5, edgecolor="darkred", facecolor="red", alpha=0.4),
            # Right bar
            patches.Rectangle((hw, -hh), t, 2*hh,
                             linewidth=1.5, edgecolor="darkred", facecolor="red", alpha=0.4),
        ]
        for rect in frame_rects:
            ax.add_patch(rect)

        # Draw gate opening (target region y-z bounds projected)
        opening = patches.Rectangle(
            (-sx, -hw), 2*sx, 2*hw,
            linewidth=2.0, edgecolor="green", facecolor="green", alpha=0.1,
            label="Gate opening (y-direction)",
        )
        ax.add_patch(opening)

        # Draw target zone (behind gate)
        target_rect = patches.Rectangle(
            (self.x_behind_min, -hw), 
            self.x_behind_max - self.x_behind_min, 2*hw,
            linewidth=2.0, edgecolor="blue", facecolor="blue", alpha=0.1,
            label="Target zone (x, y)",
        )
        ax.add_patch(target_rect)

        # Draw quadrotor collision radius (as a circle)
        quad_circle = patches.Circle(
            (0, 0), self.quad_radius,
            linewidth=1.0, edgecolor="purple", facecolor="none",
            linestyle=":", alpha=0.6, label=f"Quad radius={self.quad_radius:.2f}",
        )
        ax.add_patch(quad_circle)

        ax.set_xlim(x_min - 0.5, x_max + 0.5)
        ax.set_ylim(y_min - 0.5, y_max + 0.5)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m) — approach direction")
        ax.set_ylabel("y (m) — lateral")
        ax.set_title("Quadrotor Gate Traversal (top view, z=0)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)


if __name__ == "__main__":
    import logging
    import os
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(f"{PROJECT_NAME}.{__name__}")
    
    # Test the QuadrotorGateTraversal dynamics
    dyn = QuadrotorGateTraversal(
        dtype=jnp.float32,
    )
    
    logger.info(f"Quadrotor Gate Traversal Dynamics")
    logger.info(f"  State dim: {dyn.state_dim}")
    logger.info(f"  Control dim: {dyn.control_dim}")
    logger.info(f"  Disturbance dim: {dyn.disturb_dim}")
    logger.info(f"  Input (NN) dim: {dyn.input_dim}")
    
    # Create state grid
    resolution = 500
    X, Y, states = dyn._create_state_grid(resolution=resolution)
    logger.info(f"Created {len(states)} state grid points (resolution={resolution})")
    
    # Value-network overlay is optional; left unloaded here (plots the constraint
    # landscape only). Load a trained value net and set V_pred to overlay V.
    V_pred = None

    # Plot domination regions
    logger.info("Plotting constraint landscape...")
    os.makedirs('./tmp', exist_ok=True)
    
    if V_pred is not None:
        fig, ax = dyn.plot_domination_regions(V_pred=V_pred, resolution=resolution)
        fig.savefig('./tmp/constraint_landscape.png', dpi=150, bbox_inches='tight')
        logger.info("✓ Saved BRAT domination map to ./tmp/constraint_landscape.png")
    else:
        # Show both: raw constraints and target vs avoid comparison
        fig1, axes = dyn.plot_constraints(resolution=resolution)
        fig1.savefig('./tmp/constraint_landscape_raw.png', dpi=150, bbox_inches='tight')
        logger.info("✓ Saved raw constraint landscape to ./tmp/constraint_landscape_raw.png")
        
        fig2, ax = dyn.plot_target_vs_avoid_dominance(resolution=resolution)
        fig2.savefig('./tmp/constraint_landscape.png', dpi=150, bbox_inches='tight')
        logger.info("✓ Saved target vs avoid comparison to ./tmp/constraint_landscape.png")