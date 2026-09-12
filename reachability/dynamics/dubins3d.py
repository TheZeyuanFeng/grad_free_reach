from .base import Dynamics
from configs.constants import PROJECT_NAME

import math
import jax
import jax.numpy as jnp
import numpy as np
import logging
import os
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches


log = logging.getLogger(f"{PROJECT_NAME}.{__name__}")

class Dubins3D(Dynamics):
    """Dubins car (3D state) safety dynamics: x, y, theta."""

    state_dim: int = 3
    control_dim: int = 1
    disturb_dim: int = 0
    input_dim: int = 4

    def __init__(
        self,
        v: float = 1.0,
        omega_max: float = 1.0,
        dx_max: float = 0.0,
        dy_max: float = 0.0,
        x_lim: tuple = (-1.5, 1.5),
        y_lim: tuple = (-1.5, 1.5),
        target_center: tuple = (0.0, 0.0),
        target_r: float = 0.5,
        obstacle_center: tuple = (-0.7, 0.2),
        obstacle_r: float = 0.3,
        dtype=jnp.float32,
    ):
        super().__init__()
        
        self.dtype = dtype
        self.v = v
        self.omega_max = omega_max
        self.dx_max = dx_max
        self.dy_max = dy_max

        self.x_min, self.x_max = x_lim
        self.y_min, self.y_max = y_lim
        self.theta_min = -jnp.pi
        self.theta_max = jnp.pi

        self.state_low = jnp.array([self.x_min, self.y_min, self.theta_min], dtype=dtype)
        self.state_high = jnp.array([self.x_max, self.y_max, self.theta_max], dtype=dtype)
        self.val_state_low = self.state_low
        self.val_state_high = self.state_high
        self.u_max = jnp.array([omega_max], dtype=dtype)
        self.d_max = jnp.array([dx_max, dy_max], dtype=dtype)
        self.u_min = -self.u_max
        self.d_min = -self.d_max

        self._xy_low = self.state_low[:2]
        self._xy_high = self.state_high[:2]
        self._target_center = jnp.array(list(target_center), dtype=dtype)
        self._target_r = jnp.array(target_r, dtype=dtype)
        self._obstacle_center = jnp.array(list(obstacle_center), dtype=dtype)
        self._obstacle_r = jnp.array(obstacle_r, dtype=dtype)

    def wrap_state(self, state: jax.Array) -> jax.Array:
        theta = (state[..., -1] + jnp.pi) % (2 * jnp.pi) - jnp.pi
        return state.at[..., -1].set(theta)

    def normalize_xy(self, xy: jax.Array) -> jax.Array:
        """Normalize x,y to [-1, 1] using state bounds."""
        return 2.0 * (xy - self._xy_low) / (self._xy_high - self._xy_low) - 1.0

    def nn_inputs(self, x: jax.Array) -> jax.Array:
        """NN inputs: [x_norm, y_norm, sin(theta), cos(theta)].

        Args:
            x: (..., 3) unnormalized state
        Returns:
            (..., 4) features
        """
        x = self.wrap_state(x)
        xy_n = self.normalize_xy(x[..., :2])
        theta = x[..., 2]
        return jnp.concatenate([
            xy_n, 
            jnp.sin(theta)[..., None], 
            jnp.cos(theta)[..., None]
        ], axis=-1)

    def f(self, x: jax.Array, u: jax.Array, d: jax.Array) -> jax.Array:
        """Continuous-time dynamics dx/dt = f(x, u, d).

        Args:
            x: (B, 3)
            u: (B, 1) — [omega]
            d: (B, 2) — [dx, dy] disturbances
        Returns:
            xdot: (B, 3)
        """
        dx = self.v * jnp.cos(x[:, 2])
        dy = self.v * jnp.sin(x[:, 2])
        dx += d[:, 0] if self.disturb_dim > 0 and self.dx_max > 0 else 0.0
        dy += d[:, 1] if self.disturb_dim > 0 and self.dy_max > 0 else 0.0
        dtheta = u[:, 0]
        return jnp.stack([dx, dy, dtheta], axis=-1)

    def target_l(self, x: jax.Array) -> jax.Array:
        """Signed distance to target set (negative = inside)."""
        dist = jnp.linalg.norm(x[..., :2] - self._target_center, axis=-1)
        return dist - self._target_r

    def avoid_l(self, x: jax.Array) -> jax.Array:
        """Signed distance to obstacle (negative = inside)."""
        dist = jnp.linalg.norm(x[..., :2] - self._obstacle_center, axis=-1)
        return -(dist - self._obstacle_r)

    def l(self, x: jax.Array) -> jax.Array:
        return self.target_l(x)

    def cost_fn(self, state_traj: jax.Array, gamma: Optional[float] = None) -> jax.Array:
        """
        Calculates the Reach-Avoid cost: min_t [ max(Target(x_t), max_{s<=t} Avoid(x_s)) ]
        """
        avoid_vals = self.avoid_l(state_traj) # (Batch, T)
        target_vals = self.target_l(state_traj) # (Batch, T)
        
        # Running max of avoid violations
        avoid_max_so_far = jax.lax.cummax(avoid_vals, axis=avoid_vals.ndim - 1)
        
        # Reach-Avoid objective
        cost_at_each_step = jnp.maximum(target_vals, avoid_max_so_far)
        
        if gamma is not None and gamma < 1.0:
            T = cost_at_each_step.shape[-1]
            weights = gamma ** jnp.arange(T, dtype=self.dtype)
            cost_at_each_step = cost_at_each_step * weights

        return jnp.min(cost_at_each_step, axis=-1)

    def plot_config(self):
        return {
            'state_slices': [0, 0, 0],
            'state_labels': ['x', 'y', r'$\theta$'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
            'z_vals': [-math.pi, -math.pi / 2, 0, math.pi / 2, math.pi],
        }

    def render(self, ax):
        # State space boundary
        rect = patches.Rectangle(
            (self.x_min, self.y_min),
            self.x_max - self.x_min,
            self.y_max - self.y_min,
            linewidth=1.5, edgecolor="black", facecolor="none", linestyle="--",
            label="State bounds"
        )
        ax.add_patch(rect)
        
        # Target set (green)
        target = plt.Circle(
            np.array(self._target_center),
            float(self._target_r),
            color="green", alpha=0.3, label="Target"
        )
        ax.add_patch(target)
        
        # Obstacle (red)
        obstacle = plt.Circle(
            np.array(self._obstacle_center),
            float(self._obstacle_r),
            color="red", alpha=0.3, label="Obstacle"
        )
        ax.add_patch(obstacle)
        
        ax.set_xlim(self.x_min - 0.1, self.x_max + 0.1)
        ax.set_ylim(self.y_min - 0.1, self.y_max + 0.1)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="upper right", fontsize=8)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test Dubins3D
    dyn = Dubins3D(
        device="cpu",
        v=1.0,
        omega_max=1.0,
        x_lim=(-1.5, 1.5),
        y_lim=(-1.5, 1.5),
        target_center=(0.0, 0.0),
        target_r=0.5,
        obstacle_center=(-0.7, 0.2),
        obstacle_r=0.3,
        dtype=jnp.float32,
    )
    
    # Sample and check initial state
    init_state = dyn.sample_initial_state(1) if hasattr(dyn, 'sample_initial_state') else jnp.array([[-1.0, -1.0, 0.0]], dtype=jnp.float32)
    print(f"\nDubins3D")
    print(f"  State dim: {dyn.state_dim}")
    print(f"  Control dim: {dyn.control_dim}")
    print(f"  Input (NN) dim: {dyn.input_dim}")
    print(f"  Sample state shape: {init_state.shape}")
    print(f"  Target loss: {dyn.target_l(init_state).item():.4f}")
    print(f"  Avoid loss: {dyn.avoid_l(init_state).item():.4f}")
    
    # Save render
    out_path = "./tmp/dubins3d.png"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    dyn.save_env_render(out_path, title="Dubins 3D: Target & Obstacle")