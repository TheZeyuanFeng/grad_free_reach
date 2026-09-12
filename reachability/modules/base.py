from abc import ABC, abstractmethod
from typing import Dict, Optional

import jax
import jax.numpy as jnp
from flax import nnx

# ---------------------------------------------------------------------------
# Return-type aliases
# ---------------------------------------------------------------------------

# Value network output: "V" is always present; action keys are optional.
ValueNetOutput = Dict[str, Optional[jax.Array]]

# Policy network output: both keys always present, value may be None.
PolicyNetOutput = Dict[str, Optional[jax.Array]]


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------

class BoundaryAwareNet(nnx.Module):
    def __init__(self, net, dyn, problem_type, exact_bc=True):
        super().__init__()
        self.net = net
        self.dyn = dyn
        self.problem_type = problem_type
        self.exact_bc = exact_bc

    def boundary(self, x):
        if self.problem_type == "BRAT":
            return jnp.maximum(self.dyn.target_l(x), self.dyn.avoid_l(x))
        else:
            return self.dyn.l(x)

    def __call__(self, x, t, window_range: tuple = None, apply_inflation: bool = False,
                 shared_time: bool = False) -> Dict[str, jax.Array]:
        eps = 1e-6
        batch_size = x.shape[0]
        t_flat = t.reshape(-1)
        if t_flat.size == 1:
            t_flat = jnp.broadcast_to(t_flat, (batch_size,))
        elif t_flat.size != batch_size:
            raise ValueError(
                f"Expected t to have 1 or {batch_size} elements, got {t_flat.size}."
            )
        phi = self.dyn.nn_inputs(x)
        out = self.net(phi, t_flat, window_range=window_range, apply_inflation=apply_inflation,
                       shared_time=shared_time)

        g_x = self.boundary(x).reshape(-1)

        if self.exact_bc:
            residual = out["V"].reshape(-1)
            out["V"] = g_x + t_flat * residual
        else:
            near_zero = t_flat <= eps
            out["V"] = jnp.where(near_zero, g_x, out["V"].reshape(-1))
        return out
        
class ReachabilityValueNet(nnx.Module, ABC):
    """Abstract base for all time-conditioned value networks.

    Subclasses must:
      - Set ``self.control_dim`` and ``self.disturb_dim`` as instance attributes
        in ``__init__`` before calling ``super().__init__()``.
      - Implement :meth:`forward` with the signature defined below.
      - Follow the standard constructor signature::

            __init__(phi_dim, **kwargs)

        where ``**kwargs`` carry architecture-specific hyperparameters with
        sensible defaults so the net can be instantiated without a config object.
    """

    obs_dim: Optional[int]
    obs_ch_dim: Optional[int]
    obs_embed_dim: int

    def __init__(self) -> None:
        super().__init__()
        for attr in ("obs_dim", "obs_ch_dim", "obs_embed_dim"):
            if not hasattr(self, attr):
                raise TypeError(
                    f"{type(self).__name__} must set 'self.{attr}' in __init__."
                )

    @abstractmethod
    def __call__(
        self,
        phi_x: jax.Array,
        t: jax.Array,
    ) -> ValueNetOutput:
        """Compute V(phi_x, t) and optional action heads.

        Args:
            phi_x:  Normalised NN feature vector, shape (B, phi_dim).
            t:      Time-to-go, shape (B,) or (B, 1).

        Returns:
            Dict always containing ``"V"`` (shape (B,)).
        """


class ReachabilityPolicyNet(nnx.Module, ABC):
    """Abstract base for all time-conditioned policy networks.

    Subclasses must:
      - Set ``self.control_dim`` and ``self.disturb_dim`` as instance attributes
        in ``__init__`` before calling ``super().__init__()``.
      - Implement :meth:`forward` with the signature defined below.

    Raises:
        TypeError: At instantiation time if ``control_dim`` or ``disturb_dim``
            are not set on the instance.
    """

    control_dim: int
    disturb_dim: int
    obs_dim: Optional[int]
    obs_ch_dim: Optional[int]
    obs_embed_dim: int

    def __init__(self) -> None:
        super().__init__()
        for attr in ("control_dim", "disturb_dim", "obs_dim", "obs_ch_dim", "obs_embed_dim"):
            if not hasattr(self, attr):
                raise TypeError(
                    f"{type(self).__name__} must set 'self.{attr}' in __init__."
                )

    @abstractmethod
    def __call__(
        self,
        phi_x: jax.Array,
        t: jax.Array,
    ) -> PolicyNetOutput:
        """Return continuous policy action heads.

        Args:
            phi_x: Normalised NN feature vector, shape (B, phi_dim).
            t:     Time-to-go, shape (B,) or (B, 1).

        Returns:
            Dict with keys ``"u_raw"`` (shape (B, control_dim) or None)
            and ``"d_raw"`` (shape (B, disturb_dim) or None).
            A None value means the corresponding dim is 0.
        """