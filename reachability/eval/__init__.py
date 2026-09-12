"""Evaluation of trained HJ reachability models.

Mirrors the ``reachability/training`` layout: ``eval.py`` at the repo root is a
thin CLI entry point, and the actual work lives here, split by concern:

    rollout.py       closed-loop trajectory simulation + rollout-derived scores
    metrics.py       confusion matrix / FPR / threshold calibration
    ground_truth.py  analytical (LessLinear pairwise) GT lookup + MSE evaluation
    slices.py        paper-slice heatmap grids
    modes.py         top-level evaluation modes wired together
"""

import reachability.training.functional  # noqa: F401  (import-order guard)
