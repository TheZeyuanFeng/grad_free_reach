"""reachability/dynamics/capabilities.py -- optional Dynamics behaviours.

The base Dynamics ABC (see base.py) only requires f/l/nn_inputs. Everything
here is a NAMED, OPTIONAL capability that a Dynamics subclass can add on top:
implement the method(s) below with the exact signature shown and the
matching piece of the framework (mostly reachability/data/sampling.py) picks
it up automatically -- no changes needed there.

These are typing.Protocol classes, so this is still fully structural/duck
typed: a Dynamics subclass does NOT need to inherit from any of these to be
detected (isinstance(dyn, SupportsNominalPolicy) is True for any object with
a matching nominal_policy method, whether or not it says so in its class
list). Inheriting from one is optional, purely to make intent explicit in
the class definition and get type-checker support:

    class MyDynamics(Dynamics, SupportsNominalPolicy):
        def nominal_policy(self, x, t, **kwargs): ...

Caveat: runtime_checkable Protocol's isinstance() check only verifies that a
method with the right NAME exists -- it does not check argument names, types,
or return shape. Matching the signature shown in each docstring is still on
you; this catches "you forgot to implement it", not "you implemented it
wrong".
"""

from typing import Protocol, runtime_checkable

import jax


@runtime_checkable
class SupportsCustomSample(Protocol):
    """Opt out of the generic uniform-over-[state_low, state_high] sampler.

    Used by: sample_uniform_states, sample_val_states_uniform,
    BoundaryAwareSampler's "uniform" bucket and as the default nominal-policy
    seed source when nothing else is defined.
    """

    def sample(self, key: jax.Array, n: int) -> jax.Array:
        """Return n fresh states, shape (n, state_dim)."""
        ...


@runtime_checkable
class SupportsTargetRegion(Protocol):
    """Defines a target region distinct from l(x) for boundary-aware sampling.

    Used by: BoundaryAwareSampler's "target" bucket (biases sampling toward
    target_l's zero level set).
    """

    def target_l(self, x: jax.Array) -> jax.Array:
        """Signed-distance-like function, shape (B,). Sign convention matches
        Dynamics.l: negative inside the target, positive outside.
        """
        ...


@runtime_checkable
class SupportsAvoidRegion(Protocol):
    """Defines an avoid region distinct from l(x) for boundary-aware sampling.

    Used by: BoundaryAwareSampler's "avoid" bucket (biases sampling toward
    avoid_l's zero level set).
    """

    def avoid_l(self, x: jax.Array) -> jax.Array:
        """Signed-distance-like function, shape (B,). Sign convention matches
        Dynamics.l: negative inside the avoid set, positive outside.
        """
        ...


@runtime_checkable
class SupportsNominalPolicy(Protocol):
    """A hand-designed, non-learned control law for generating realistic
    rollout states to sample from (e.g. a CPG gait).

    NOT used inside HJSolver -- that needs the value-optimal control (see
    ProbingStrategy/PolicyStrategy in reachability/training/strategy.py), not
    a fixed policy. Wrapped by NominalPolicyStrategy and driven by
    BoundaryAwareSampler's "nominal" bucket.
    """

    def nominal_policy(self, x: jax.Array, t: jax.Array, **kwargs) -> jax.Array:
        """Return control u, shape (B, control_dim) (or (u, d) if disturb_dim > 0).

        t is fed as *elapsed rollout time*, not HJ time-to-go -- a periodic
        policy needs its own clock to advance, not one that counts down to 0.
        **kwargs receives whatever SupportsNominalParams.nominal_params (if
        implemented) returns for this rollout, forwarded unchanged.
        """
        ...


@runtime_checkable
class SupportsNominalSeed(Protocol):
    """Seed distribution for SupportsNominalPolicy rollouts.

    Define this when a generic uniform or target-focused sample is a poor
    starting point for a multi-step rollout under your nominal_policy -- e.g.
    off the policy's own limit cycle, or not physically settled/consistent.
    Falls back to SupportsCustomSample.sample (or plain uniform) if absent.
    """

    def nominal_seed(self, key: jax.Array, n: int) -> jax.Array:
        """Return n seed states, shape (n, state_dim)."""
        ...


@runtime_checkable
class SupportsNominalParams(Protocol):
    """Optional per-rollout parameters for SupportsNominalPolicy.

    Sampled once per rollout (not per step) and held fixed for its duration,
    then forwarded to nominal_policy as keyword arguments -- e.g. which gait
    to walk, or what height to walk at. Lets one nominal_policy represent a
    family of behaviours instead of a single fixed one.
    """

    def nominal_params(self, key: jax.Array, n: int) -> dict:
        """Return a dict of per-rollout arrays, each shaped (n, ...)."""
        ...
