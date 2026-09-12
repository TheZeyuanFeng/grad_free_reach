import jax
import jax.numpy as jnp
from typing import Optional

from .base import Dynamics


class LessLinearND(Dynamics):
    """N-dimensional weakly nonlinear system for BRT reachability.

    State   x  ∈ [-1, 1]^N
    Control u  ∈ [-u_max, u_max]^{N-1}   (controller, minimises V)
    No disturbance (disturb_dim = 0).

    Continuous-time dynamics:
        ẋ = A x + B u + nl(x)

    where
        A[i,i]  = -0.5  (diagonal damping)
        A[i,0]  = -1    for i = 1 … N-1  (coupling to first state)
        B       = [0; 0.4 I_{N-1}]       (control enters states 1 … N-1)

    Nonlinear term (default params make nl[0] ≡ 0):
        nl[0]   = mu * sin(alpha * x_0) * x_0^2
        nl[1:]  = -gamma * x_0^2 * x[1:]

    Target set (BRT reach condition, l ≤ 0):
        0.5 * (‖diag(ellipse_params) x‖^2 − goalR^2) ≤ 0
    with ellipse_params = [sqrt(N-1), 1, …, 1] and goalR = sqrt(N-1) * goalR_2d.
    """

    # Set dynamically in __init__ (N-dependent); see PursuitEvasion for precedent.
    state_dim:   int
    control_dim: int
    disturb_dim: int
    input_dim:   int

    def __init__(
        self,
        N: int = 2,
        u_max: float = 0.5,
        gamma: float = 20.0,
        mu: float = 0.0,
        alpha: float = 0.0,
        goalR_2d: float = 0.25,
        sample_inflation: float = 1.0,
        dtype: jnp.dtype = jnp.float32,
    ):
        # Class-level attributes must be set before super().__init__()
        LessLinearND.state_dim = N
        LessLinearND.control_dim = N - 1
        LessLinearND.disturb_dim = 0
        # state already in [-1,1]^N; returned as-is
        LessLinearND.input_dim = N

        super().__init__()

        self.state_dim = N
        self.N = N            # exposed for the LessLinear paper-slice / eval_lesslinear
        self.dtype = dtype
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha

        # Target ellipsoid
        self.goalR_2d = goalR_2d
        self.goalR = ((N - 1) ** 0.5) * goalR_2d
        self.sample_inflation = float(sample_inflation)
        self.ellipse_params = jnp.concatenate(
            [((N - 1) ** 0.5) * jnp.ones(1), jnp.ones(N - 1)],
            axis=0,
        ).astype(dtype)

        # System matrix A: -0.5*I with extra -1 in first column for rows 1..N-1
        A_diag = -0.5 * jnp.eye(N, dtype=dtype)
        A_col0 = jnp.concatenate(
            [jnp.zeros((1, 1)), jnp.ones((N - 1, 1))], axis=0)  # [0,1,…,1]^T
        # first-col-only matrix
        A_extra = jnp.concatenate([A_col0, jnp.zeros((N, N - 1))], axis=1)
        self.A = (A_diag - A_extra).astype(dtype)

        # Control matrix B: [0; 0.4 I_{N-1}]
        self.B = jnp.concatenate(
            [jnp.zeros((1, N - 1), dtype=dtype), 0.4 *
             jnp.eye(N - 1, dtype=dtype)],
            axis=0,
        ).astype(dtype)

        # State bounds
        self.state_low = jnp.full((N,), -1.0, dtype=dtype)
        self.state_high = jnp.full((N,),  1.0, dtype=dtype)
        self.val_state_low = self.state_low
        self.val_state_high = self.state_high

        # Control / disturbance bounds
        self.u_max = jnp.full((N - 1,), u_max, dtype=dtype)
        self.d_max = jnp.zeros((0,), dtype=dtype)
        self.u_min = -self.u_max
        self.d_min = -self.d_max

    # ------------------------------------------------------------------
    # NN features: state is already normalised to [-1, 1]^N
    # ------------------------------------------------------------------

    def wrap_state(self, x: jax.Array) -> jax.Array:
        return jnp.clip(x, self.state_low, self.state_high)

    def nn_inputs(self, x: jax.Array) -> jax.Array:
        return self.wrap_state(x).astype(dtype=self.dtype)

    # ------------------------------------------------------------------
    # Continuous-time dynamics
    # ------------------------------------------------------------------

    def f(self, x: jax.Array, u: jax.Array, d: jax.Array) -> jax.Array:
        """dx/dt = A x + B u + nl(x).

        Args:
            x: (B, N)
            u: (B, N-1)
            d: (B, 0)  — disturbance unused for BRT
        Returns:
            xdot: (B, N)
        """
        A = self.A.astype(x.dtype)
        B = self.B.astype(x.dtype)

        x0 = x[..., 0]    # (B,)
        x_rest = x[..., 1:]   # (B, N-1)

        # Nonlinear term
        nl0 = (self.mu * jnp.sin(self.alpha * x0)
               * x0 ** 2)[..., None]  # (B,1)
        nl_rest = -self.gamma * \
            x0[..., None] ** 2 * x_rest                   # (B,N-1)
        # (B,N)
        nl = jnp.concatenate([nl0, nl_rest], axis=-1)

        return x @ A.T + u @ B.T + nl

    # ------------------------------------------------------------------
    # Target set signed distance (BRT: l ≤ 0 = inside reach set)
    # ------------------------------------------------------------------

    def l(self, x: jax.Array) -> jax.Array:
        """0.5 * (‖diag(ellipse_params) x‖^2 − goalR^2).

        Negative inside the ellipsoid (target reached).
        """
        ep = self.ellipse_params.astype(x.dtype)
        return 0.5 * (jnp.sum((ep * x) ** 2, axis=-1) - self.goalR ** 2)

    def target_l(self, x: jax.Array) -> jax.Array:
        """BRT target set signed level: reach condition is target_l <= 0, which
        for this system is exactly l(x). Exposing it makes the dynamics a
        SupportsTargetRegion (enables the target boundary pool + target buckets)."""
        return self.l(x)

    def sample_target_states(self, key: jax.Array, num_states: int) -> jax.Array:
        """Sample states (roughly) inside the target ellipsoid {l(x) <= 0}.

        Uniform-on-sphere direction, linear radius (NOT U^{1/N}, which for large N
        concentrates near the surface so almost nothing lands inside), stretched by
        the ellipsoid semi-axes goalR/ep. With r = U * sample_inflation the fraction
        inside the target is 1/sample_inflation (=1.0 for inflation=1.0). Some x_i
        (i>0) may exceed the box; clamped to state bounds."""
        k1, k2 = jax.random.split(key)
        z = jax.random.normal(k1, (num_states, self.state_dim), dtype=self.dtype)
        z_hat = z / jnp.clip(jnp.linalg.norm(z, axis=-1, keepdims=True), min=1e-8)
        r = jax.random.uniform(k2, (num_states, 1), dtype=self.dtype) * self.sample_inflation
        ep = self.ellipse_params.astype(self.dtype)
        x = r * z_hat * (self.goalR / ep)
        return jnp.clip(x, self.state_low, self.state_high)

    # ------------------------------------------------------------------
    # Trajectory cost (BRT: min_t l(x(t)))
    # ------------------------------------------------------------------

    def cost_fn(self, state_traj: jax.Array,
                gamma: Optional[float] = None) -> jax.Array:
        """BRT trajectory cost: min_t l(x(t)).

        Args:
            state_traj: (B, T, N)
            gamma:      optional discount factor
        Returns:
            (B,) minimum value over the trajectory
        """
        values = self.l(state_traj)   # (B, T)
        if gamma is not None and gamma < 1.0:
            T = values.shape[-1]
            weights = gamma ** jnp.arange(T, dtype=values.dtype)  # (T,)
            values = values * weights
        return jnp.min(values, axis=-1)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, ax) -> None:
        """Draw the target ellipsoid and state bounds.
        
        For N=2: draw the target ellipse and state box.
        For N>2: plot the 2D projection defined by plot_config.
        """
        from matplotlib.patches import Ellipse, Rectangle
        
        if self.state_dim == 2:
            # Draw target ellipse
            a = self.goalR / self.ellipse_params[0]  # semi-axis x
            b = self.goalR / self.ellipse_params[1]  # semi-axis y
            ellipse = Ellipse(
                (0, 0), width=2*a, height=2*b,
                fill=False, edgecolor='red', linewidth=2, label='Target'
            )
            ax.add_patch(ellipse)
            
            # Draw state bounds box
            rect = Rectangle(
                (-1, -1), width=2, height=2,
                fill=False, edgecolor='black', linewidth=1.5, linestyle='--',
                label='State bounds'
            )
            ax.add_patch(rect)
            
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_xlabel('x[0]')
            ax.set_ylabel('x[1]')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_aspect('equal')
        else:
            # For higher dimensions, just draw state bounds in 2D projection
            rect = Rectangle(
                (-1, -1), width=2, height=2,
                fill=False, edgecolor='black', linewidth=1.5, linestyle='--'
            )
            ax.add_patch(rect)
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_xlabel('x[0]')
            ax.set_ylabel('x[1]')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')

    # ------------------------------------------------------------------
    # Plot config
    # ------------------------------------------------------------------

    def plot_config(self):
        return {
            "state_slices": [0.0] * self.state_dim,
            "state_labels": ["x0"] + [f"x{i}" for i in range(1, self.state_dim)],
            "x_axis_idx":   0,
            "y_axis_idx":   1,
            "z_axis_idx":   2 if self.state_dim > 2 else 1,
            'z_vals': [-1.0, 0.5, 0.0, 0.5, 1.0],
        }