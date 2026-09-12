import jax
import jax.numpy as jnp
import numpy as np
import logging
import os
from typing import Optional, Dict
import matplotlib.patches as patches

from .base import Dynamics
from configs.constants import PROJECT_NAME


log = logging.getLogger(f"{PROJECT_NAME}.{__name__}")

class NarrowPassage(Dynamics):
    """
    Two-vehicle narrow passage, 10D state:
    x = [x1, y1, th1, v1, phi1,  x2, y2, th2, v2, phi2]
    Control: u = [a1, phidot1, a2, phidot2]
    """

    def __init__(
        self,
        avoid_fn_weight: float = 10.0,
        avoid_only: bool = False,
        freeze_on_collision: bool = False,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.state_dim: int = 10
        self.control_dim: int = 4
        self.disturb_dim: int = 0
        self.input_dim: int = 12  # 8 scalars + 2*(sin, cos)

        # State Indices for readability
        self.X1, self.Y1, self.TH1, self.V1, self.PHI1 = 0, 1, 2, 3, 4
        self.X2, self.Y2, self.TH2, self.V2, self.PHI2 = 5, 6, 7, 8, 9
        
        super().__init__()

        self.dtype = dtype
        self.num_robots = 2

        self.L = 2.0  # Wheelbase

        # Targets
        self.goal1 = jnp.array([6.0, -1.2], dtype=dtype)
        self.goal2 = jnp.array([-6.0, 1.2], dtype=dtype)

        # Environment Constraints
        self.curb_low, self.curb_high = -2.8, 2.8
        self.stranded_car = jnp.array([0.0, -1.8], dtype=dtype)
        self.lat_min, self.lat_max = -10.0, 10.0

        # Bounds
        self.v_min, self.v_max = 0.1, 7.0
        self.phi_min, self.phi_max = -0.3 * jnp.pi, 0.3 * jnp.pi
        self.a_max = 2.0
        self.phidot_max = 3.0 * jnp.pi

        # Normalization Setup
        # Scalar indices are everything except the headings (TH1=2, TH2=7)
        self._scalar_idxs = jnp.array([0, 1, 3, 4, 5, 6, 8, 9], dtype=jnp.int32)

        low = jnp.array([-10.0, -3.8, -jnp.pi, self.v_min, self.phi_min] * 2, dtype=dtype)
        high = jnp.array([10.0, 3.8, jnp.pi, self.v_max, self.phi_max] * 2, dtype=dtype)
        self.state_low, self.state_high = low, high

        # Validation Range (used for sampling initial states)
        self.val_state_low = jnp.array([-8, -1.7, 0, 0.1, 0, 6, -1.7, -jnp.pi, 0.1, 0], dtype=dtype)
        self.val_state_high = jnp.array([-6, 1.7, 0, 0.1, 0, 8, 1.7, -jnp.pi, 0.1, 0], dtype=dtype)
        # self.val_state_low = jnp.array([-8, -1.7, -0.6, self.v_min, self.phi_min, 6, -1.7, -jnp.pi-0.6, self.v_min, self.phi_min], dtype=dtype)
        # self.val_state_high = jnp.array([-6, 1.7, 0.6, self.v_max, self.phi_max, 8, 1.7, -jnp.pi+0.6, self.v_max, self.phi_max], dtype=dtype)

        self.u_max = jnp.array([self.a_max, self.phidot_max] * 2, dtype=dtype)
        self.d_max = jnp.zeros((0,), dtype=dtype)
        self.u_min = -self.u_max
        self.d_min = -self.d_max

        self.avoid_fn_weight = avoid_fn_weight
        self.avoid_only = avoid_only
        self.freeze_on_collision = freeze_on_collision
        self.target_state_low = jnp.array([3.5, -2.1, -jnp.pi, self.v_min, self.phi_min, -8.5, -2.1, -jnp.pi, self.v_min, self.phi_min], dtype=dtype)
        self.target_state_high = jnp.array([8.5, 2.1, jnp.pi, self.v_max, self.phi_max, -3.5, 2.1, jnp.pi, self.v_max, self.phi_max], dtype=dtype)

    def wrap_state(self, x: jax.Array) -> jax.Array:
        # Split cars
        car1 = x[..., 0:5]
        car2 = x[..., 5:10]

        def process_car(c):
            px, py, th, v, phi = c[..., 0], c[..., 1], c[..., 2], c[..., 3], c[..., 4]
            return jnp.stack([
                jnp.clip(px, self.lat_min, self.lat_max),
                jnp.clip(py, self.curb_low - 1.0, self.curb_high + 1.0),
                (th + jnp.pi) % (2.0 * jnp.pi) - jnp.pi,
                jnp.clip(v, self.v_min, self.v_max),
                jnp.clip(phi, self.phi_min, self.phi_max)
            ], axis=-1)

        return jnp.concatenate([process_car(car1), process_car(car2)], axis=-1)

    def nn_inputs(self, x: jax.Array) -> jax.Array:
        """Produces normalized features for the Neural Network."""
        x = self.wrap_state(x)
        # Normalize scalar states to [-1, 1]
        norm_x = 2.0 * (x - self.state_low) / (self.state_high - self.state_low) - 1.0
        scalars = norm_x[..., self._scalar_idxs]
        
        # Heading as Sin/Cos
        th1, th2 = x[..., self.TH1], x[..., self.TH2]
        angles = jnp.stack([
            jnp.sin(th1), jnp.cos(th1),
            jnp.sin(th2), jnp.cos(th2)
        ], axis=-1)
        
        return jnp.concatenate([scalars, angles], axis=-1)
    
    def f(self, x: jax.Array, u: jax.Array, d: jax.Array = None) -> jax.Array:
        x = self.wrap_state(x)
        
        # Vehicle 1
        v1, th1, phi1 = x[..., self.V1], x[..., self.TH1], x[..., self.PHI1]
        a1, pdot1 = u[..., 0], u[..., 1]
        
        # Vehicle 2
        v2, th2, phi2 = x[..., self.V2], x[..., self.TH2], x[..., self.PHI2]
        a2, pdot2 = u[..., 2], u[..., 3]

        # Construct dx/dt using stacking (JAX arrays are immutable)
        dsdt = jnp.stack([
            v1 * jnp.cos(th1),
            v1 * jnp.sin(th1),
            v1 * jnp.tan(phi1) / self.L,
            a1,
            pdot1,
            v2 * jnp.cos(th2),
            v2 * jnp.sin(th2),
            v2 * jnp.tan(phi2) / self.L,
            a2,
            pdot2
        ], axis=-1)

        if self.freeze_on_collision:
            # Use jnp.where instead of masked assignment
            collision_mask = (self.avoid_l(x) >= 0)[..., None]
            dsdt = jnp.where(collision_mask, 0.0, dsdt)

        return dsdt

    def target_l(self, x: jax.Array) -> jax.Array:
        """Signed distance to goal. Success if <= 0."""
        dist1 = jnp.linalg.norm(x[..., [self.X1, self.Y1]] - self.goal1, axis=-1) - self.L
        dist2 = jnp.linalg.norm(x[..., [self.X2, self.Y2]] - self.goal2, axis=-1) - self.L
        raw_target = jnp.maximum(dist1, dist2)
        # JAX arrays are immutable: rescale each side of 0 via where, not masked assignment.
        raw_target = jnp.where(raw_target > 0, raw_target / 20, raw_target * 2)
        return jnp.tanh(raw_target)

    def _avoid_fn(self, x: jax.Array) -> jax.Array:
        """Minimum distance to safety (SDF). Safety if > 0."""
        hl = 0.5 * self.L
        p1, p2 = x[..., 0:2], x[..., 5:7]

        # Constraints (all should be positive for safety)
        dists = jnp.stack([
            # Curbs
            x[..., self.Y1] - self.curb_low - hl,
            x[..., self.Y2] - self.curb_low - hl,
            self.curb_high - x[..., self.Y1] - hl,
            self.curb_high - x[..., self.Y2] - hl,
            # Lateral Bounds
            x[..., self.X1] - self.lat_min - hl,
            x[..., self.X2] - self.lat_min - hl,
            self.lat_max - x[..., self.X1] - hl,
            self.lat_max - x[..., self.X2] - hl,
            # Obstacles
            jnp.linalg.norm(p1 - self.stranded_car, axis=-1) - self.L,
            jnp.linalg.norm(p2 - self.stranded_car, axis=-1) - self.L,
            # Inter-vehicle collision
            jnp.linalg.norm(p1 - p2, axis=-1) - self.L
        ], axis=0)

        return jnp.tanh(self.avoid_fn_weight * jnp.min(dists, axis=0))

    def avoid_l(self, x: jax.Array) -> jax.Array:
        """Failure if >= 0."""
        return -self._avoid_fn(x)

    def l(self, x: jax.Array) -> jax.Array:
        return self.avoid_l(x) if self.avoid_only else self.target_l(x)

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

    def plot_config(self) -> Dict:
        return {
            "state_slices": [0.0, 0.0, 0.0, 3.0, 0.0, 6.0, 0.0, -jnp.pi, 1.0, 0.0],
            "state_labels": ["x1", "y1", r"$\theta_1$", "v1", r"$\phi_1$",
                             "x2", "y2", r"$\theta_2$", "v2", r"$\phi_2$"],
            "x_axis_idx": self.X1,
            "y_axis_idx": self.Y1,
            "z_axis_idx": self.V1,
            "z_vals": [0.0, 1.0, 2.0, 3.0, 4.0],
        }

    def render(self, ax):
        # Road bounds
        road = patches.Rectangle(
            (self.lat_min, self.curb_low),
            self.lat_max - self.lat_min,
            self.curb_high - self.curb_low,
            linewidth=1.5, edgecolor="gray", facecolor="lightyellow", alpha=0.4,
            label="Road"
        )
        ax.add_patch(road)

        # Curb lines
        ax.axhline(self.curb_low,  color="gray", linewidth=1.5, linestyle="-")
        ax.axhline(self.curb_high, color="gray", linewidth=1.5, linestyle="-")

        # Stranded car (obstacle)
        sc = np.asarray(self.stranded_car)
        stranded = patches.Rectangle(
            (sc[0] - 1.0, sc[1] - 0.5), 2.0, 1.0,
            linewidth=1.5, edgecolor="red", facecolor="red", alpha=0.4,
            label="Stranded car"
        )
        ax.add_patch(stranded)

        # Goal regions
        g1 = np.asarray(self.goal1)
        g2 = np.asarray(self.goal2)
        ax.plot(*g1, marker="*", markersize=12, color="green", label="Goal 1")
        ax.plot(*g2, marker="*", markersize=12, color="blue",  label="Goal 2")

        ax.set_xlim(self.lat_min, self.lat_max)
        ax.set_ylim(self.curb_low - 0.5, self.curb_high + 0.5)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend(loc="upper right", fontsize=8)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test NarrowPassage
    dyn = NarrowPassage(
        avoid_fn_weight=10.0,
        avoid_only=False,
        freeze_on_collision=False,
        device="cpu",
        dtype=jnp.float32,
    )
    
    # Create a sample initial state from validation bounds
    init_state = dyn.val_state_low[None, :]
    print(f"\nNarrowPassage")
    print(f"  State dim: {dyn.state_dim}")
    print(f"  Control dim: {dyn.control_dim}")
    print(f"  Input (NN) dim: {dyn.input_dim}")
    print(f"  Sample state shape: {init_state.shape}")
    print(f"  Target loss: {dyn.target_l(init_state).item():.4f}")
    print(f"  Avoid loss: {dyn.avoid_l(init_state).item():.4f}")
    
    # Save render
    out_path = "./tmp/narrow_passage.png"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    dyn.save_env_render(out_path, title="Narrow Passage: Two-Vehicle Reach-Avoid")