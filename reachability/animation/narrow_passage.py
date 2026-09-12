import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Polygon

from reachability.dynamics import Dynamics
from configs.constants import PROJECT_NAME
    

log = logging.getLogger(f"{PROJECT_NAME}.{__name__}")


# ---------------------------------------------------------------------------
# NarrowPassage state layout
# [x1, y1, th1, v1, phi1,  x2, y2, th2, v2, phi2]
#   0    1    2   3    4    5    6    7   8    9
# ---------------------------------------------------------------------------
_DEFAULT_IDX: Dict[str, int] = dict(x1=0, y1=1, th1=2, x2=5, y2=6, th2=7)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def car_triangle(
    x: float, y: float, heading: float, L: float = 0.8, W: float = 0.45
) -> np.ndarray:
    """Return (3, 2) vertices of a triangle pointing along ``heading`` (radians)."""
    pts_local = np.array([
        [ L / 2,  0.0   ],  # tip
        [-L / 2,  W / 2 ],  # rear-left
        [-L / 2, -W / 2 ],  # rear-right
    ], dtype=np.float32)
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    pts = pts_local @ R.T
    pts[:, 0] += x
    pts[:, 1] += y
    return pts


def _pose(x_t: jax.Array, idx: Dict[str, int]) -> Tuple[float, ...]:
    """Extract (x1, y1, th1, x2, y2, th2) as Python floats from a 1-D state tensor."""
    return tuple(float(x_t[idx[k]]) for k in ("x1", "y1", "th1", "x2", "y2", "th2"))


# ---------------------------------------------------------------------------
# Trajectory selection
# ---------------------------------------------------------------------------

def choose_traj_index(
    true_score: jax.Array,
    mode: str = "best",
    index: int = 0,
    seed: int = 0,
) -> int:
    """Return the index of the trajectory to animate.

    Args:
        true_score: (N,) cost tensor — lower is better (negative = success).
        mode:       One of "best", "worst", "random", "random_fail", "index".
        index:      Used when mode="index".
        seed:       RNG seed for random modes.
    """
    N = int(true_score.size)
    mode = mode.lower()

    if mode == "best":
        return int(jnp.argmin(true_score))
    if mode == "worst":
        return int(jnp.argmax(true_score))
    if mode == "random":
        rng = np.random.default_rng(seed)
        return int(rng.integers(0, N))
    if mode == "random_fail":
        fail_idx = jnp.nonzero(true_score > 0)[0]
        rng = np.random.default_rng(seed)
        if fail_idx.size == 0:
            log.warning("random_fail: no failing trajectories found; picking randomly.")
            return int(rng.integers(0, N))
        return int(fail_idx[rng.integers(0, fail_idx.size)])
    if mode == "index":
        if index < 0 or index >= N:
            raise ValueError(f"index {index} out of range for {N} trajectories.")
        return index

    raise ValueError(f"Unknown select mode '{mode}'. Choose from: best, worst, random, random_fail, index.")


# ---------------------------------------------------------------------------
# Value slice
# ---------------------------------------------------------------------------

class _TimeConditionedValue:
    """Callable wrapper: evaluates V(x, t) on a batch via the value network."""

    def __init__(self, net: nnx.Module, t_scalar: float):
        self.net = net
        self.t_scalar = t_scalar

    def __call__(self, x: jax.Array) -> jax.Array:
        B = x.shape[0]
        t = jnp.full((B,), self.t_scalar, dtype=x.dtype)
        return self.net(x, t)["V"]


def _value_slice(
    V_fn: _TimeConditionedValue,
    dyn: Dynamics,
    x_ref: jax.Array,
    idx: Dict[str, int],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    nx: int,
    ny: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate V, avoid_l, and target_l on a (ny, nx) grid over car-1 (x, y).

    All other state dimensions are held fixed at ``x_ref``.

    Returns:
        xs:   (nx,) x-axis grid values
        ys:   (ny,) y-axis grid values
        Z:    (ny, nx) value heatmap
        lxs:  (ny, nx) avoid_l values
        gxs:  (ny, nx) target_l values
    """
    dtype = dyn.dtype
    xs = jnp.linspace(xlim[0], xlim[1], nx, dtype=dtype)
    ys = jnp.linspace(ylim[0], ylim[1], ny, dtype=dtype)
    Xg, Yg = jnp.meshgrid(xs, ys, indexing="xy")  # (nx, ny)

    grid = jnp.stack([Xg.reshape(-1), Yg.reshape(-1)], axis=1)  # (nx*ny, 2)
    S = jnp.broadcast_to(jnp.asarray(x_ref, dtype=dtype)[None, :], (grid.shape[0], dyn.state_dim))
    S = S.at[:, idx["x1"]].set(grid[:, 0])
    S = S.at[:, idx["y1"]].set(grid[:, 1])

    Z   = V_fn(S).reshape(ny, nx)
    lxs = dyn.avoid_l(S).reshape(ny, nx)
    gxs = dyn.target_l(S).reshape(ny, nx)

    return (
        np.asarray(xs), np.asarray(ys),
        np.asarray(Z), np.asarray(lxs), np.asarray(gxs),
    )


# ---------------------------------------------------------------------------
# Failure scatter
# ---------------------------------------------------------------------------

def _save_failure_scatter(
    state_traj: jax.Array,
    true_score: jax.Array,
    idx: Dict[str, int],
    out_path: str,
) -> None:
    """Save a scatter plot of all failing trajectory poses to ``out_path``."""
    fail_mask = true_score > 0
    if not bool(jnp.any(fail_mask)):
        log.info("No failing trajectories — skipping failure scatter.")
        return

    ft = np.asarray(state_traj[fail_mask])  # (F, T, state_dim)

    # Initial positions (stars) and full trajectory poses (dots)
    x1_init = ft[:, 0, idx["x1"]]
    y1_init = ft[:, 0, idx["y1"]]
    x2_init = ft[:, 0, idx["x2"]]
    y2_init = ft[:, 0, idx["y2"]]

    x1_all = ft[:, :, idx["x1"]].reshape(-1)
    y1_all = ft[:, :, idx["y1"]].reshape(-1)
    x2_all = ft[:, :, idx["x2"]].reshape(-1)
    y2_all = ft[:, :, idx["y2"]].reshape(-1)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.scatter(x1_init, y1_init, s=8, c="tab:blue",   alpha=0.4, marker="*", label="car1 start")
    ax.scatter(x2_init, y2_init, s=8, c="tab:orange", alpha=0.4, marker="*", label="car2 start")
    ax.scatter(x1_all,  y1_all,  s=1, c="tab:blue",   alpha=0.15, label="car1 traj")
    ax.scatter(x2_all,  y2_all,  s=1, c="tab:orange", alpha=0.15, label="car2 traj")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Failure trajectory poses  (N={int(jnp.sum(fail_mask))})")
    ax.legend(loc="best", markerscale=3)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    log.info("Saved failure scatter → %s", out_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def animate_two_car_with_heatmap(
    *,
    dyn: Dynamics,
    model: nnx.Module,
    state_traj: jax.Array,
    true_score: jax.Array,
    out_path: str,
    select_mode: str = "best",
    select_index: int = 0,
    seed: int = 0,
    idx: Optional[Dict[str, int]] = None,
    fps: int = 20,
    car_L: float = 0.8,
    car_W: float = 0.45,
    heat_nx: int = 141,
    heat_ny: int = 141,
    heat_pad: float = 0.0,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    dt: float = 0.0,
    heat_alpha: float = 0.85,
) -> None:
    """Render and save a two-car trajectory animation with a value-function heatmap.

    Args:
        dyn:          Dynamics object.
        model:        Trained value network.
        state_traj:   Trajectory tensor, shape (N, K+1, state_dim) or (K+1, state_dim).
        true_score:   Per-trajectory cost, shape (N,).  Negative = success.
        out_path:     Output file path (.mp4 or .gif).
        select_mode:  How to pick the trajectory: "best", "worst", "random",
                      "random_fail", or "index".
        select_index: Used when select_mode="index".
        seed:         RNG seed for random selection modes.
        idx:          Override state-column mapping.  Defaults to NarrowPassage layout.
        fps:          Frames per second.
        car_L / car_W: Triangle dimensions for car visualisation.
        heat_nx/ny:   Heatmap resolution.
        heat_pad:     Inset the heatmap bounds by this amount on each side.
        xlim / ylim:  Plot bounds.  Auto-computed from trajectory if None.
        dt:           Time step; used to compute the correct time-to-go each frame.
        heat_alpha:   Opacity of the heatmap layer.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    dtype = dyn.dtype
    col_idx = idx if idx is not None else _DEFAULT_IDX

    # ------------------------------------------------------------------
    # Failure scatter (saved before we modify/select state_traj)
    # ------------------------------------------------------------------
    if state_traj.ndim == 3:
        scatter_path = os.path.splitext(out_path)[0] + "_failure_scatter.png"
        _save_failure_scatter(state_traj, true_score, col_idx, scatter_path)

    # ------------------------------------------------------------------
    # Select trajectory
    # ------------------------------------------------------------------
    if state_traj.ndim == 3:
        n_idx = choose_traj_index(true_score, mode=select_mode, index=select_index, seed=seed)
        X     = jnp.asarray(state_traj[n_idx], dtype=dtype)  # (T, state_dim)
        cost  = true_score[n_idx]

        # Trim at first success for successful trajectories
        if cost <= 0:
            reach_avoid = jnp.maximum(dyn.target_l(X), dyn.avoid_l(X))
            success_steps = jnp.nonzero(reach_avoid < 0)[0]
            if success_steps.size > 0:
                X = X[: int(success_steps[0]) + 1]

    elif state_traj.ndim == 2:
        X     = jnp.asarray(state_traj, dtype=dtype)
        cost  = true_score
        n_idx = 0
    else:
        raise ValueError(
            f"state_traj must be (N, T, D) or (T, D), got shape {tuple(state_traj.shape)}"
        )

    T = X.shape[0]
    log.info("Animating trajectory #%d  cost=%.4f  T=%d", n_idx, float(cost), T)

    # ------------------------------------------------------------------
    # Plot bounds
    # ------------------------------------------------------------------
    x1s = np.asarray(X[:, col_idx["x1"]])
    y1s = np.asarray(X[:, col_idx["y1"]])
    x2s = np.asarray(X[:, col_idx["x2"]])
    y2s = np.asarray(X[:, col_idx["y2"]])

    _pad = 1.5
    if xlim is None:
        xlim = (float(np.concatenate([x1s, x2s]).min()) - _pad,
                float(np.concatenate([x1s, x2s]).max()) + _pad)
    if ylim is None:
        ylim = (float(np.concatenate([y1s, y2s]).min()) - _pad,
                float(np.concatenate([y1s, y2s]).max()) + _pad)

    heat_xlim = (xlim[0] + heat_pad, xlim[1] - heat_pad)
    heat_ylim = (ylim[0] + heat_pad, ylim[1] - heat_pad)

    # ------------------------------------------------------------------
    # Figure and initial frame
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    t0_scalar = (T - 1) * dt
    V0 = _TimeConditionedValue(model, t0_scalar)
    xs, ys, Z, lxs, gxs = _value_slice(V0, dyn, X[0], col_idx, heat_xlim, heat_ylim, heat_nx, heat_ny)

    extent = [xs.min(), xs.max(), ys.min(), ys.max()]
    im = ax.imshow(
        Z, origin="lower", extent=extent,
        interpolation="nearest", alpha=heat_alpha,
        aspect="auto", cmap="coolwarm_r",
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="V(x, t)  [car-1 x,y slice]")

    def _draw_contours(Z_data, lxs_data, gxs_data):
        cs_v = ax.contour(Z_data,   levels=[0.0], origin="lower", extent=extent, linewidths=1.5)
        cs_l = ax.contour(lxs_data, levels=[0.0], origin="lower", extent=extent,
                          linestyles="--", colors="saddlebrown", linewidths=1.0)
        cs_g = ax.contour(gxs_data, levels=[0.0], origin="lower", extent=extent,
                          linestyles="--", colors="green", linewidths=1.0)
        return cs_v, cs_l, cs_g

    def _remove_contours(*contour_sets):
        # matplotlib >=3.8: a ContourSet is itself a removable artist and no
        # longer exposes `.collections`. Fall back to that for older versions.
        for cs in contour_sets:
            if hasattr(cs, "collections"):
                for coll in cs.collections:
                    coll.remove()
            else:
                cs.remove()

    cs_v, cs_l, cs_g = _draw_contours(Z, lxs, gxs)

    (trail1,) = ax.plot([], [], lw=2, color="steelblue",  label="car 1")
    (trail2,) = ax.plot([], [], lw=2, color="darkorange", label="car 2")

    p1x, p1y, p1h, p2x, p2y, p2h = _pose(X[0], col_idx)
    poly1 = Polygon(car_triangle(p1x, p1y, p1h, L=car_L, W=car_W),
                    closed=True, facecolor="steelblue",  edgecolor="k", alpha=0.95)
    poly2 = Polygon(car_triangle(p2x, p2y, p2h, L=car_L, W=car_W),
                    closed=True, facecolor="darkorange", edgecolor="k", alpha=0.95)
    ax.add_patch(poly1)
    ax.add_patch(poly2)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")
    title = ax.set_title("")

    # ------------------------------------------------------------------
    # Animation callbacks
    # ------------------------------------------------------------------

    def _init():
        trail1.set_data([], [])
        trail2.set_data([], [])
        return im, trail1, trail2, poly1, poly2, title

    def _update(t: int):
        nonlocal cs_v, cs_l, cs_g

        t_scalar = (T - t - 1) * dt
        Vt = _TimeConditionedValue(model, t_scalar)
        xs, ys, Z, lxs, gxs = _value_slice(Vt, dyn, X[t], col_idx, heat_xlim, heat_ylim, heat_nx, heat_ny)

        im.set_data(Z)
        im.set_clim(Z.min(), Z.max())

        _remove_contours(cs_v, cs_l, cs_g)
        cs_v, cs_l, cs_g = _draw_contours(Z, lxs, gxs)

        trail1.set_data(x1s[: t + 1], y1s[: t + 1])
        trail2.set_data(x2s[: t + 1], y2s[: t + 1])

        px1, py1, ph1, px2, py2, ph2 = _pose(X[t], col_idx)
        poly1.set_xy(car_triangle(px1, py1, ph1, L=car_L, W=car_W))
        poly2.set_xy(car_triangle(px2, py2, ph2, L=car_L, W=car_W))

        avoid_val  = float(dyn.avoid_l(X[t][None])[0])
        target_val = float(dyn.target_l(X[t][None])[0])
        title.set_text(
            f"traj #{n_idx}  t={t}/{T - 1}  cost={float(cost):.3f}"
            f"  lx={avoid_val:.3f}  gx={target_val:.3f}"
        )
        return im, trail1, trail2, poly1, poly2, title

    anim = FuncAnimation(fig, _update, frames=T, init_func=_init, blit=False)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    try:
        anim.save(out_path, writer=FFMpegWriter(fps=fps, bitrate=4000))
        log.info("Saved animation → %s", out_path)
    except Exception as exc:
        gif_path = os.path.splitext(out_path)[0] + ".gif"
        log.warning("MP4 save failed (%s) — falling back to GIF: %s", exc, gif_path)
        anim.save(gif_path, writer=PillowWriter(fps=fps))
        log.info("Saved animation → %s", gif_path)

    plt.close(fig)