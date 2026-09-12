import jax
import jax.numpy as jnp
from flax import nnx
from typing import Tuple

from reachability.dynamics import Dynamics
from utils import decode_actions
from .functional import (
    infer_u_d_double_sided,
    infer_u_d_triple_sided,
    probe_horizon,
    substeps_for,
)


class ActionStrategy:
    """Base class for action selection strategies.

    Call convention
    ---------------
    All subclasses are called as::

        u, d = strategy(dyn, x, t)

    The value / policy network is captured at construction time and must
    **not** be passed at call time. This keeps call sites uniform and avoids
    accidentally passing a stale or wrong network.
    """

    def __call__(
        self, dyn: Dynamics, x: jax.Array, t: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        raise NotImplementedError


class ProbingStrategy(ActionStrategy):
    """Infers optimal actions by querying the value network (implicit policy).

    Used at rollout / evaluation time. The ``v_next_fn`` closure silently
    handles the batch expansion that ``infer_u_d_double_sided`` applies
    internally via ``repeat_interleave``.

    Returns:
        (u_star, d_star)
    """

    def __init__(self, value_net: nnx.Module, cfg) -> None:
        self.v_net = value_net
        self.cfg = cfg

    def __call__(
        self, dyn: Dynamics, x: jax.Array, t: jax.Array, return_conf: bool = False, net: nnx.Module = None,
        window_range: tuple = None, shared_time: bool = False,
    ) -> Tuple[jax.Array, jax.Array]:
        """
        shared_time: True iff every row of ``t`` is identical (e.g. a
            verification rollout step, where the whole batch shares one
            time-to-go). Lets the value net dispatch to exactly the one
            window every row needs instead of evaluating every window in
            range and discarding the rest. Must NOT be set from training,
            where different rows of a sampled batch sit at different times
            and genuinely need different windows.
        """
        cfg = self.cfg
        v_net = net if net is not None else self.v_net
        h = probe_horizon(cfg)
        t_next = jnp.clip(t - h, min=0.0)

        def v_next_fn(x_query: jax.Array) -> jax.Array:
            # infer_u_d_double_sided expands the batch by repeat_interleave,
            # so x_query will be an integer multiple of the original batch.
            qB, refB = x_query.shape[0], t_next.shape[0]
            if qB == refB:
                t_exp = t_next
            else:
                t_exp = jnp.repeat(t_next, qB // refB, axis=0)

            t_exp = t_exp[:qB] # truncate to match x_query
            out = v_net(x_query, t_exp, window_range=window_range, shared_time=shared_time)["V"]
            return out

        probe_fn = (
            infer_u_d_triple_sided
            if getattr(cfg.GAME, "TRIPLE_SIDED_PROBING", False)
            else infer_u_d_double_sided
        )
        return probe_fn(
            x=x,
            V_next_fn=v_next_fn,
            dyn=dyn,
            dt=h,
            control_role=cfg.GAME.CONTROL_ROLE,
            disturb_role=cfg.GAME.DISTURB_ROLE,
            problem_type=cfg.GAME.PROBLEM_TYPE,
            num_substeps=substeps_for(h, cfg),
            return_conf=return_conf,
        )

class NominalPolicyStrategy(ActionStrategy):
    """Wraps a hand-designed (non-learned) control law as an ActionStrategy.s

    Args:
        policy_fn: Callable ``(dyn, x, t) -> u`` or ``(dyn, x, t) -> (u, d)``.
            Any extra parameters (gait phase/frequency/amplitude, ...) should
            already be bound into this callable (e.g. via functools.partial)
            before construction -- this class is just the ActionStrategy
            adapter, not where gait-specific math should live.

    Returns:
        (u, d) -- d is None unless policy_fn itself returns one.
    """

    def __init__(self, policy_fn) -> None:
        self.policy_fn = policy_fn

    def __call__(
        self, dyn: Dynamics, x: jax.Array, t: jax.Array, net: nnx.Module = None,
        window_range: tuple = None, **kwargs,
    ) -> Tuple[jax.Array, jax.Array]:
        # kwargs: optional per-rollout parameters (e.g. gait, stance height)
        # sampled once per rollout and forwarded through unchanged -- opaque
        # to this class, meaningful only to policy_fn.
        out = self.policy_fn(dyn, x, t, **kwargs)
        if isinstance(out, tuple):
            return out
        return out, None


class PolicyStrategy(ActionStrategy):
    """Uses a trained policy network directly.

    Returns:
        (u_star, d_star)
    """

    def __init__(self, policy_net: nnx.Module) -> None:
        self.net = policy_net

    def __call__(
        self, dyn: Dynamics, x: jax.Array, t: jax.Array, net: nnx.Module = None,
        window_range: tuple = None,
    ) -> Tuple[jax.Array, jax.Array]:
        net = net if net is not None else self.net
        out = net(dyn.nn_inputs(x), t, window_range=window_range)
        return decode_actions(out, dyn)