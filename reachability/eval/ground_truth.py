"""Analytical ground-truth comparison (LessLinear pairwise decomposition).

``LessLinear``'s N-D value function decomposes exactly into a sum of 2-D
pairwise terms, so a precomputed pairwise grid gives analytical GT to compare
the learned net against. Only applicable to dynamics with that structure --
every entry point here is gated on a ``--pairwise_path`` being supplied.
"""

import logging

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from scipy.interpolate import RegularGridInterpolator

from reachability.dynamics import Dynamics
from reachability.data.sampling import sample_uniform_states
from configs import PROJECT_NAME
from .metrics import _accumulate_confusion_counts, _rates_from_counts

_PW_LO, _PW_HI = -1.0, 1.0  # Normalized state bounds for LessLinear grids

log = logging.getLogger(f"{PROJECT_NAME}.{__name__}")


def _load_pairwise(path: str) -> RegularGridInterpolator:
    values = np.load(path)
    pts    = np.linspace(_PW_LO, _PW_HI, values.shape[0])
    return RegularGridInterpolator(
        (pts, pts), values,
        method="linear", bounds_error=False, fill_value=None,
    )


def _query_pairwise(interp: RegularGridInterpolator, x0, xi) -> np.ndarray:
    """Bilinear lookup V_2d(x0, xi)."""
    pts = np.stack([np.asarray(x0, dtype=float), np.asarray(xi, dtype=float)], axis=1)
    return interp(pts)


def gt_nd_value(interp: RegularGridInterpolator, states: np.ndarray) -> np.ndarray:
    """V_ND(x) = Σᵢ₌₁^{N-1} V_2d(x₀, xᵢ).  states: (B, N)."""
    states = np.asarray(states)
    x0    = states[:, 0]
    total = np.zeros(len(x0))
    for i in range(1, states.shape[1]):
        total += _query_pairwise(interp, x0, states[:, i])
    return total


def gt_nd_value_array(
    interp: RegularGridInterpolator, states: "jax.Array | np.ndarray"
) -> jax.Array:
    """Array-returning wrapper around gt_nd_value (matches ``states`` dtype)."""
    v = gt_nd_value(interp, np.asarray(states))
    return jnp.asarray(v, dtype=jnp.asarray(states).dtype)


def evaluate_mse(
    value_net: nnx.Module,
    dyn: Dynamics,
    interp: RegularGridInterpolator,
    m_batches: int,
    batch_size: int,
    t_final: float,
    key: jax.Array,
    control_role: str = "min",
) -> tuple:
    """Return (mse, mae, conf) over m_batches × batch_size random states.

    conf is a dict with TPR/FPR/TNR/FNR computed from pooled counts.
    """
    dtype = dyn.dtype
    sq_errors, abs_errors = [], []
    counts = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    n_gt_neg, n_pred_neg, n_total = 0, 0, 0
    gt_min, gt_max = np.inf, -np.inf

    for i in range(m_batches):
        key, sample_key = jax.random.split(key)
        x     = sample_uniform_states(sample_key, dyn, batch_size)
        t_vec = jnp.full((batch_size,), t_final, dtype=dtype)

        V_net = np.asarray(value_net(x, t_vec)["V"])
        V_gt  = gt_nd_value(interp, np.asarray(x))

        sq_errors.append(((V_net - V_gt) ** 2).mean())
        abs_errors.append(np.abs(V_net - V_gt).mean())
        _accumulate_confusion_counts(counts, V_net, V_gt, control_role)

        n_gt_neg   += int((V_gt  < 0).sum())
        n_pred_neg += int((V_net < 0).sum())
        n_total    += len(V_gt)
        gt_min = min(gt_min, float(V_gt.min()))
        gt_max = max(gt_max, float(V_gt.max()))

        log.info("  batch %2d / %d:  MSE = %.6f   MAE = %.6f",
                 i + 1, m_batches, sq_errors[-1], abs_errors[-1])

    mse = float(np.mean(sq_errors))
    mae = float(np.mean(abs_errors))

    rates    = _rates_from_counts(counts)
    vol_gt   = n_gt_neg   / max(1, n_total)
    vol_pred = n_pred_neg / max(1, n_total)
    conf     = {**rates, "vol_gt": vol_gt, "vol_pred": vol_pred,
                "gt_min": gt_min, "gt_max": gt_max}

    log.info(
        "=== Total  MSE=%.6f  MAE=%.6f  "
        "TPR=%.4f  FPR=%.4f  TNR=%.4f  FNR=%.4f  "
        "BRT_vol_GT=%.4f  BRT_vol_pred=%.4f  "
        "GT_range=[%.4f, %.4f] ===",
        mse, mae,
        conf["TPR"], conf["FPR"], conf["TNR"], conf["FNR"],
        vol_gt, vol_pred, gt_min, gt_max,
    )
    return mse, mae, conf
