from abc import ABC, abstractmethod
import logging
from typing import Optional

import jax
from matplotlib import pyplot as plt
from configs.constants import PROJECT_NAME


log = logging.getLogger(f"{PROJECT_NAME}.{__name__}")


class Dynamics(ABC):
    """Abstract base class for HJ reachability dynamics.

    Subclasses must implement :meth:`f`, :meth:`l`, and :meth:`nn_inputs`.
    All other methods have sensible defaults that subclasses may override.

    Required class-level attributes
    --------------------------------
    state_dim   : int  – dimensionality of the state vector
    control_dim : int  – dimensionality of the control input
    disturb_dim : int  – dimensionality of the disturbance input
    input_dim   : int  – dimensionality of the NN feature vector (output of nn_inputs)

    Required instance attributes (set in subclass __init__ before super())
    -----------------------------------------------------------------------
    dtype       : jnp.dtype
    state_low   : jax.Array, shape (state_dim,)
    state_high  : jax.Array, shape (state_dim,)
    u_max       : jax.Array, shape (control_dim,)
    d_max       : jax.Array, shape (disturb_dim,)

    Also expected (not abstract, but called unconditionally by the standard
    training loop -- omitting them means training/plotting crashes the first
    time it's exercised, not a silent fallback):
    -----------------------------------------------------------------------
    plot_config() -> dict
        Used by reachability/plotting.py for training-progress value-function
        slices.
    val_state_low / val_state_high : jax.Array, shape (state_dim,)
        Validation-time sampling bounds. Fine to just alias state_low/state_high.

    Beyond this required core, see reachability/dynamics/capabilities.py for
    the full menu of optional, pluggable behaviours (custom sampling, target/
    avoid regions, nominal-policy rollout sampling, ...) -- each is a named
    Protocol class documenting exactly which method to implement and what it
    unlocks.
    """

    # Subclasses must set these at the class level.
    state_dim:   int
    control_dim: int
    disturb_dim: int
    input_dim:   int
    obs_dim:     Optional[int] = None
    obs_ch_dim:  Optional[int] = None
    obs_embed_dim: Optional[int] = None

    def __init__(self) -> None:
        # Verify required class-level attributes are present.
        for attr in ("state_dim", "control_dim", "disturb_dim", "input_dim"):
            if not hasattr(self, attr):
                raise TypeError(
                    f"{type(self).__name__} must define class attribute '{attr}'."
                )

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def f(
        self,
        x: jax.Array,   # (B, state_dim)
        u: jax.Array,   # (B, control_dim)
        d: jax.Array,   # (B, disturb_dim)
    ) -> jax.Array:
        """Continuous-time dynamics: returns dx/dt, shape (B, state_dim)."""

    @abstractmethod
    def l(self, x: jax.Array) -> jax.Array:
        """Signed-distance cost/value function, shape (B,).

        Sign convention: negative inside the target/safe set, positive outside.
        """

    @abstractmethod
    def nn_inputs(self, x: jax.Array) -> jax.Array:
        """Map raw state x to normalised NN feature vector, shape (B, input_dim)."""

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def wrap_state(self, x: jax.Array) -> jax.Array:
        """Project x back into the valid state manifold (e.g. angle wrapping).

        Default: identity. Override in subclasses that have periodic dimensions
        or hard state constraints.
        """
        return x

    def step(
        self,
        x: jax.Array,
        u: jax.Array,
        d: jax.Array,
        dt: float,
    ) -> jax.Array:
        """Forward Euler step followed by state wrapping."""
        return self.wrap_state(x + dt * self.f(x, u, d))

    def save_env_render(self, out_path: str, title: str = "") -> None:
        """Save a standalone figure of the environment geometry."""
        fig, ax = plt.subplots(figsize=(8, 8))
        self.render(ax)
        if title:
            ax.set_title(title, fontsize=12)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved environment render → %s", out_path)