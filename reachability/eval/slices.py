"""Paper-slice heatmap grids: learned V vs. analytical GT on a 2-D slice.

The slice is ``[x₀, y, y, …, y]`` -- one free "first" coordinate against a
second coordinate shared by every remaining dimension -- which is the slice the
LessLinear pairwise decomposition makes analytically tractable
(``V_ND(x₀, y, …, y) = (N-1)·V_2d(x₀, y)``).
"""

import logging
import os

import numpy as np
import jax.numpy as jnp
from flax import nnx
from scipy.interpolate import RegularGridInterpolator

from reachability.plotting import plot_paper_slice
from configs import PROJECT_NAME
from .ground_truth import _PW_LO, _PW_HI, _query_pairwise

log = logging.getLogger(f"{PROJECT_NAME}.{__name__}")


def _gt_paper_slice(
    interp: RegularGridInterpolator, N: int, grid_res: int
) -> tuple:
    """GT: V_ND(x₀, y, …, y) = (N-1)·V_2d(x₀, y)."""
    x0_lin = np.linspace(_PW_LO, _PW_HI, grid_res)
    xi_lin = np.linspace(_PW_LO, _PW_HI, grid_res)
    X0, Xi = np.meshgrid(x0_lin, xi_lin, indexing="ij")
    v_pair = _query_pairwise(interp, X0.ravel(), Xi.ravel())
    return X0, Xi, ((N - 1) * v_pair).reshape(grid_res, grid_res)


def _learned_paper_slice(
    value_net: nnx.Module,
    N: int,
    grid_res: int,
    t_final: float,
    dtype,
    chunk: int = 8192,
) -> tuple:
    """Learned: query value_net on [x₀, y, y, …, y] grid."""
    x0_lin = np.linspace(_PW_LO, _PW_HI, grid_res)
    xi_lin = np.linspace(_PW_LO, _PW_HI, grid_res)
    X0, Xi = np.meshgrid(x0_lin, xi_lin, indexing="ij")

    x0_f = X0.ravel().astype(np.float32)
    xi_f = Xi.ravel().astype(np.float32)

    states          = np.empty((len(x0_f), N), dtype=np.float32)
    states[:, 0]    = x0_f
    states[:, 1:]   = xi_f[:, None]

    x_t   = jnp.asarray(states, dtype=dtype)
    t_vec = jnp.full((len(x0_f),), t_final, dtype=dtype)

    v_parts = [
        np.asarray(value_net(x_t[s:s + chunk], t_vec[s:s + chunk])["V"])
        for s in range(0, len(x0_f), chunk)
    ]
    return X0, Xi, np.concatenate(v_parts).reshape(grid_res, grid_res)


def _maybe_plot_paper_slice(
    value_net, dyn, interp, save_path: str,
    grid_res: int, t_final: float,
    mse=None, mae=None,
) -> None:
    """Compute and save paper-slice heatmap when the dynamics expose `.N`."""
    if not hasattr(dyn, "N"):
        return
    N = dyn.N
    log.info("Computing learned paper slice (N=%d, res=%d) …", N, grid_res)
    X0, Xi, v_learned = _learned_paper_slice(
        value_net, N, grid_res, t_final, dyn.dtype
    )
    v_gt = None
    if interp is not None:
        log.info("Computing GT paper slice …")
        _, _, v_gt = _gt_paper_slice(interp, N, grid_res)

    plot_paper_slice(
        X0, Xi, v_learned, v_gt,
        out_path=os.path.join(save_path, "paper_slice.png"),
        N=N, t_final=t_final,
        mse=mse, mae=mae,
    )
