import jax
import jax.numpy as jnp
from typing import Optional

def _weighted_mean(err, conf, eps=1e-6):
    """Confidence-weighted mean of ``err``.

    ``conf`` may be per-sample (B,) or per-element, matching ``err``'s shape.
    A per-sample weight has to give every channel of a state the same say; a
    per-channel weight can zero out the channels the teacher never resolved
    while still training the ones it did.

    Normalization is per-channel (over the batch axis), not global: raw probe
    margins scale with each control channel's physical leverage over the
    value function (e.g. thrust vs. angular rate), so a single global mean
    would let the highest-leverage channel dominate the loss regardless of
    per-sample confidence. Per-sample ``conf`` broadcasts identically across
    channels, so this reduces to the old behavior in that case.
    """
    w = jnp.clip(conf.astype(err.dtype), min=0)
    if w.ndim < err.ndim:
        w = jnp.expand_dims(w, axis=tuple(range(w.ndim, err.ndim)))
    w = jnp.broadcast_to(w, err.shape)
    w_norm = w / jnp.clip(w.mean(axis=0, keepdims=True), min=eps)
    return (err * w_norm).mean()


def vq_policy_loss(
    out: dict,
    teacher_u: jax.Array,
    teacher_d: jax.Array,
    u_conf: jax.Array,
    d_conf: Optional[jax.Array],
    beta: float = 0.25,
    eps: float = 1e-6,
) -> jax.Array:
    """Loss for a VQ policy network (mimic + commitment toward teacher target).

    Must only be called when ``out`` contains the VQ-specific keys
    ``"z_e_u"`` / ``"z_q_u"`` (and ``"z_e_d"`` / ``"z_q_d"`` for
    disturbed systems).

    The teacher target ``e_target = sign(teacher)`` is used for both the mimic
    and commitment terms so that both losses pull ``z_e`` toward the same
    codeword value.  This works for binary VQ (teacher ∈ {±u_max}) and for
    ternary VQ where the teacher may output zero (triple-sided probing).

    Parameters
    ----------
    out        : forward-pass output dict from VQPolicyNet/VQPolicyMultiNet.
    teacher_u  : (B, control_dim) teacher control actions.
    teacher_d  : (B, disturb_dim) teacher disturbance actions (may be empty).
    u_conf     : confidence weights for control, (B,) or (B, control_dim).
    d_conf     : confidence weights for disturbance, (B,) or (B, disturb_dim).
    beta       : commitment weight; total loss = (1 + beta) * mimic (default 0.25).
    eps        : small constant for weight normalisation.

    Returns
    -------
    Scalar loss tensor.
    """
    loss = jnp.array(0.0, dtype=teacher_u.dtype)

    for key, teacher, conf in (("z_e_u", teacher_u, u_conf),
                               ("z_e_d", teacher_d, d_conf)):
        if key not in out or teacher is None or teacher.size == 0 or conf is None:
            continue
        z_e = out[key]
        e_target = jnp.sign(teacher).astype(z_e.dtype)

        loss = loss + (1.0 + beta) * _weighted_mean(
            jnp.square(z_e - e_target), conf, eps=eps)

    return loss


def fp_weighted_anchor_loss(
    pred: jax.Array,
    target: jax.Array,
    control_role: str,
    fp_lambda: float,
    reduction: str = 'mean'
) -> jax.Array:
    sq_error = jnp.square(pred - target)
    if control_role == "max":
        is_fp = (pred > 0) & (target <= 0)
    else:
        is_fp = (pred < 0) & (target >= 0)

    weights = jnp.where(is_fp, fp_lambda, 1.0).astype(pred.dtype)
    if reduction == 'none':
        return sq_error * weights
    return (sq_error * weights).mean()