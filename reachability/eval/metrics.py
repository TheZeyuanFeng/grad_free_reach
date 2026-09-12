"""Confusion-matrix metrics and safety-threshold calibration.

Sign convention throughout (see :func:`_unsafe_mask`): whether "unsafe" means
score <= 0 or score >= 0 depends on ``control_role``, so every function here
takes it explicitly rather than assuming a fixed convention.
"""

import numpy as np

import jax


def _unsafe_mask(score, control_role: str):
    """Return a boolean mask of unsafe states.

    Sign convention:
        BRT (control_role='max'): score <= 0 → unsafe (inside BRT)
        BRS/reach (control_role='min'): score >= 0 → unsafe (outside BRS)
    """
    if control_role == "max":
        return score <= 0.0
    return score >= 0.0


def calculate_inflation_delta(true_score, pred_score, control_role, target_fpr=0.001):
    """Buffer (delta) on the predicted score needed to hit a target false-positive rate.

    A false positive is "predicted safe, actually unsafe". WHICH TAIL of the
    predicted-score distribution that is flips with ``control_role``, so the
    quantile and the comparison have to flip together with it:

        BRT  ('max'): safe means pred > delta  -> trim the UPPER tail
        reach ('min'): safe means pred < delta -> trim the LOWER tail

    Hardcoding the 'max' form (upper quantile + ``pred > delta``) does not
    merely mis-sign the reach case, it inverts it: the returned delta would
    admit ``1 - target_fpr`` of the truly-unsafe states instead of ``target_fpr``.
    """
    true_np = np.asarray(true_score)
    pred_np = np.asarray(pred_score)

    true_unsafe_mask = _unsafe_mask(true_np, control_role)
    n_true_unsafe = int(true_unsafe_mask.sum())
    if n_true_unsafe == 0:
        return None

    pred_unsafe_samples = pred_np[true_unsafe_mask]
    if control_role == "max":
        q, is_fp = 1.0 - target_fpr, lambda p, d: p > d
    else:
        q, is_fp = target_fpr, lambda p, d: p < d

    delta = float(np.quantile(pred_unsafe_samples, q))
    achieved_fpr = float(is_fp(pred_unsafe_samples, delta).sum()) / n_true_unsafe

    return {
        "delta": delta,
        "target_fpr": target_fpr,
        "achieved_fpr": achieved_fpr,
        "n_true_unsafe": n_true_unsafe
    }


def confusion_metrics(
    pred_score: jax.Array,
    true_score: jax.Array,
    control_role: str,
) -> dict:
    """Compute confusion matrix metrics from array scores."""
    pred_unsafe = np.asarray(_unsafe_mask(pred_score, control_role))
    true_unsafe = np.asarray(_unsafe_mask(true_score, control_role))

    pred_safe = ~pred_unsafe
    true_safe = ~true_unsafe

    TP = int(( pred_safe  &  true_safe).sum())
    FP = int(( pred_safe  & ~true_safe).sum())
    TN = int((~pred_safe  & ~true_safe).sum())
    FN = int((~pred_safe  &  true_safe).sum())
    N  = int(pred_safe.size)

    return {
        "N":         float(N),
        "TP":        float(TP),
        "FP":        float(FP),
        "TN":        float(TN),
        "FN":        float(FN),
        "TPR":       TP / max(1, TP + FN),
        "FPR":       FP / max(1, FP + TN),
        "TNR":       TN / max(1, TN + FP),
        "FNR":       FN / max(1, FN + TP),
        "ACC":       (TP + TN) / max(1, N),
        "n_success": int((~true_unsafe).sum()),
    }


def _accumulate_confusion_counts(
    counts: dict,
    pred_score: np.ndarray,
    true_score: np.ndarray,
    control_role: str,
) -> None:
    """Add per-batch TP/FP/TN/FN into a running totals dict (in-place)."""
    pred_unsafe = _unsafe_mask(pred_score, control_role)
    true_unsafe = _unsafe_mask(true_score, control_role)
    pred_safe   = ~pred_unsafe
    true_safe   = ~true_unsafe

    counts["TP"] += int(( pred_safe  &  true_safe).sum())
    counts["FP"] += int(( pred_safe  & ~true_safe).sum())
    counts["TN"] += int((~pred_safe  & ~true_safe).sum())
    counts["FN"] += int((~pred_safe  &  true_safe).sum())


def _rates_from_counts(counts: dict) -> dict:
    TP, FP, TN, FN = counts["TP"], counts["FP"], counts["TN"], counts["FN"]
    return {
        "TPR": TP / max(1, TP + FN),
        "FPR": FP / max(1, FP + TN),
        "TNR": TN / max(1, TN + FP),
        "FNR": FN / max(1, FN + TP),
    }
