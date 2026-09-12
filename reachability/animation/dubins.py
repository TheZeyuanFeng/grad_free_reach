import logging
import os
from typing import Optional, Tuple

import numpy as np
from configs.constants import PROJECT_NAME
import jax
import jax.numpy as jnp
from flax import nnx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Polygon

from reachability.dynamics import Dynamics

log = logging.getLogger(f"{PROJECT_NAME}.{__name__}")

# Dubins3D state layout: [x, y, theta]
_X, _Y, _TH = 0, 1, 2


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _vehicle_triangle(
    x: float, y: float, heading: float, L: float = 0.12, W: float = 0.07
) -> np.ndarray:
    """Return (3, 2) triangle vertices pointing along ``heading`` (radians)."""
    pts = np.array([
        [ L / 2,  0.0   ],
        [-L / 2,  W / 2 ],
        [-L / 2, -W / 2 ],
    ], dtype=np.float32)
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    pts = pts @ R.T
    pts[:, 0] += x
    pts[:, 1] += y
    return pts


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
        true_score: (N,) cost array — negative = success.
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
        return int(np.random.default_rng(seed).integers(0, N))
    if mode == "random_fail":
        fail_idx = jnp.nonzero(true_score > 0)[0]
        rng = np.random.default_rng(seed)
        if fail_idx.size == 0:
            log.warning("random_fail: no failing trajectories; picking randomly.")
            return int(rng.integers(0, N))
        return int(fail_idx[rng.integers(0, fail_idx.size)])
    if mode == "index":
        if index < 0 or index >= N:
            raise ValueError(f"index {index} out of range for {N} trajectories.")
        return index

    raise ValueError(
        f"Unknown mode '{mode}'. Choose from: best, worst, random, random_fail, index."
    )


# ---------------------------------------------------------------------------
# Value / set slices
# ---------------------------------------------------------------------------

class _ValueFn:
    """Evaluates V(x, t) on a batch via the value network."""

    def __init__(self, net: nnx.Module, t_scalar: float):
        self.net = net
        self.t_scalar = t_scalar

    def __call__(self, x: jax.Array) -> jax.Array:
        B = x.shape[0]
        t = jnp.full((B,), self.t_scalar, dtype=x.dtype)
        return self.net(x, t)["V"]


def _xy_slice(
    V_fn: _ValueFn,
    dyn: Dynamics,
    theta_fixed: float,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    nx: int,
    ny: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate V, target_l, and avoid_l on a (ny, nx) grid over (x, y) at fixed θ.

    Returns:
        xs:    (nx,) x-axis grid
        ys:    (ny,) y-axis grid
        Z:     (ny, nx) value heatmap
        t_lvl: (ny, nx) target_l values
        a_lvl: (ny, nx) avoid_l values
    """
    dtype = dyn.dtype
    xs = jnp.linspace(xlim[0], xlim[1], nx, dtype=dtype)
    ys = jnp.linspace(ylim[0], ylim[1], ny, dtype=dtype)
    Xg, Yg = jnp.meshgrid(xs, ys, indexing="xy")

    S = jnp.zeros((nx * ny, dyn.state_dim), dtype=dtype)
    S = S.at[:, _X].set(Xg.reshape(-1))
    S = S.at[:, _Y].set(Yg.reshape(-1))
    S = S.at[:, _TH].set(theta_fixed)

    Z     = V_fn(S).reshape(ny, nx)
    t_lvl = dyn.target_l(S).reshape(ny, nx)
    a_lvl = dyn.avoid_l(S).reshape(ny, nx)

    return (
        np.asarray(xs), np.asarray(ys),
        np.asarray(Z), np.asarray(t_lvl), np.asarray(a_lvl),
    )


# ---------------------------------------------------------------------------
# Failure scatter
# ---------------------------------------------------------------------------

def _save_failure_scatter(
    state_traj: jax.Array,
    true_score: jax.Array,
    out_path: str,
) -> None:
    """Save an (x, y) scatter plot of all failing trajectory poses."""
    fail_mask = true_score > 0
    if not bool(jnp.any(fail_mask)):
        log.info("No failing trajectories — skipping failure scatter.")
        return

    ft = np.asarray(state_traj[fail_mask])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(ft[:, 0, _X], ft[:, 0, _Y], s=8, c="tab:red", alpha=0.5,
               marker="*", label="start")
    x_all = ft[:, :, _X].reshape(-1)
    y_all = ft[:, :, _Y].reshape(-1)
    ax.scatter(x_all, y_all, s=0.5, c="tab:red", alpha=0.1, label="traj")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Failure poses  (N={int(jnp.sum(fail_mask))})")
    ax.legend(loc="best", markerscale=4)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    log.info("Saved failure scatter → %s", out_path)


# ---------------------------------------------------------------------------
# Contour helpers
# ---------------------------------------------------------------------------

def _draw_contours(ax, Z, t_lvl, a_lvl, extent):
    cs_v = ax.contour(Z,     levels=[0.0], origin="lower", extent=extent,
                      colors="black",      linewidths=1.5)
    cs_t = ax.contour(t_lvl, levels=[0.0], origin="lower", extent=extent,
                      colors="green",      linewidths=1.0, linestyles="--")
    cs_a = ax.contour(a_lvl, levels=[0.0], origin="lower", extent=extent,
                      colors="saddlebrown",linewidths=1.0, linestyles="--")
    return cs_v, cs_t, cs_a


def _remove_contours(*contour_sets):
    for cs in contour_sets:
        # Matplotlib >=3.8: ContourSet is itself an Artist; older versions expose
        # a `.collections` list. Support both.
        if hasattr(cs, "collections"):
            for coll in cs.collections:
                coll.remove()
        else:
            cs.remove()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def animate_dubins(
    *,
    dyn: Dynamics,
    model: nnx.Module,
    state_traj: jax.Array,
    true_score: jax.Array,
    out_path: str,
    select_mode: str = "best",
    select_index: int = 0,
    seed: int = 0,
    fps: int = 20,
    heat_nx: int = 151,
    heat_ny: int = 151,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    dt: float = 0.0,
    heat_alpha: float = 0.85,
    vehicle_L: float = 0.12,
    vehicle_W: float = 0.07,
) -> None:
    """Render and save a Dubins3D trajectory animation with a value-function heatmap.

    The heatmap shows V sliced over (x, y) at the vehicle's current heading θ,
    updated each frame. Green dashed contour = target set boundary,
    brown dashed = obstacle boundary, black solid = V = 0.

    Args:
        dyn:          Dubins3D dynamics object.
        model:        Trained value network.
        state_traj:   Shape (N, K+1, 3) or (K+1, 3).
        true_score:   Per-trajectory cost, shape (N,).  Negative = success.
        out_path:     Output file path (.mp4 or .gif).
        select_mode:  "best", "worst", "random", "random_fail", or "index".
        select_index: Used when select_mode="index".
        seed:         RNG seed for random selection modes.
        fps:          Frames per second.
        heat_nx/ny:   Heatmap grid resolution.
        xlim / ylim:  Plot bounds. Auto-computed from state bounds if None.
        dt:           Time step; used to compute time-to-go per frame.
        heat_alpha:   Heatmap opacity.
        vehicle_L/W:  Triangle dimensions for the vehicle marker.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    dtype = dyn.dtype

    # ------------------------------------------------------------------
    # Failure scatter
    # ------------------------------------------------------------------
    if state_traj.ndim == 3:
        scatter_path = os.path.splitext(out_path)[0] + "_failure_scatter.png"
        _save_failure_scatter(state_traj, true_score, scatter_path)

    # ------------------------------------------------------------------
    # Select and trim trajectory
    # ------------------------------------------------------------------
    if state_traj.ndim == 3:
        n_idx = choose_traj_index(true_score, mode=select_mode, index=select_index, seed=seed)
        X     = jnp.asarray(state_traj[n_idx], dtype=dtype)
        cost  = true_score[n_idx]
        # Trim at first success
        if float(cost) <= 0:
            success = jnp.nonzero(dyn.target_l(X) < 0)[0]
            if success.size > 0:
                X = X[: int(success[0]) + 1]
    elif state_traj.ndim == 2:
        n_idx = 0
        X     = jnp.asarray(state_traj, dtype=dtype)
        cost  = true_score
    else:
        raise ValueError(
            f"state_traj must be (N, T, 3) or (T, 3), got {tuple(state_traj.shape)}"
        )

    T = X.shape[0]
    log.info("Animating trajectory #%d  cost=%.4f  T=%d", n_idx, float(cost), T)

    xs_traj = np.asarray(X[:, _X])
    ys_traj = np.asarray(X[:, _Y])
    th_traj = np.asarray(X[:, _TH])

    # ------------------------------------------------------------------
    # Plot bounds — default to state box
    # ------------------------------------------------------------------
    if xlim is None:
        xlim = (float(dyn.state_low[_X]), float(dyn.state_high[_X]))
    if ylim is None:
        ylim = (float(dyn.state_low[_Y]), float(dyn.state_high[_Y]))

    extent = [xlim[0], xlim[1], ylim[0], ylim[1]]

    # ------------------------------------------------------------------
    # Initial frame
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))

    V0 = _ValueFn(model, (T - 1) * dt)
    xs, ys, Z, t_lvl, a_lvl = _xy_slice(V0, dyn, float(th_traj[0]), xlim, ylim, heat_nx, heat_ny)

    im = ax.imshow(
        Z, origin="lower", extent=extent,
        interpolation="bilinear", alpha=heat_alpha,
        aspect="auto", cmap="coolwarm_r",
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="V(x, y, θ_cur, t)")
    cs_v, cs_t, cs_a = _draw_contours(ax, Z, t_lvl, a_lvl, extent)

    # Trajectory trail and vehicle
    (trail,) = ax.plot([], [], lw=1.5, color="k", alpha=0.6, label="trajectory")
    poly = Polygon(
        _vehicle_triangle(xs_traj[0], ys_traj[0], th_traj[0], L=vehicle_L, W=vehicle_W),
        closed=True, facecolor="steelblue", edgecolor="k", alpha=0.95, zorder=5,
    )
    ax.add_patch(poly)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=8)
    title = ax.set_title("")

    # ------------------------------------------------------------------
    # Animation callbacks
    # ------------------------------------------------------------------

    def _init():
        trail.set_data([], [])
        return im, trail, poly, title

    def _update(t: int):
        nonlocal cs_v, cs_t, cs_a

        t_scalar = (T - t - 1) * dt
        Vt = _ValueFn(model, t_scalar)
        xs, ys, Z, t_lvl, a_lvl = _xy_slice(
            Vt, dyn, float(th_traj[t]), xlim, ylim, heat_nx, heat_ny
        )
        im.set_data(Z)
        im.set_clim(Z.min(), Z.max())

        _remove_contours(cs_v, cs_t, cs_a)
        cs_v, cs_t, cs_a = _draw_contours(ax, Z, t_lvl, a_lvl, extent)

        trail.set_data(xs_traj[: t + 1], ys_traj[: t + 1])
        poly.set_xy(_vehicle_triangle(
            xs_traj[t], ys_traj[t], th_traj[t], L=vehicle_L, W=vehicle_W
        ))

        tgt  = float(dyn.target_l(X[t][None])[0])
        avd  = float(dyn.avoid_l(X[t][None])[0])
        title.set_text(
            f"traj #{n_idx}  t={t}/{T - 1}  θ={th_traj[t]:.2f} rad"
            f"  cost={float(cost):.3f}  target_l={tgt:.3f}  avoid_l={avd:.3f}"
        )
        return im, trail, poly, title

    anim = FuncAnimation(fig, _update, frames=T, init_func=_init, blit=False)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    try:
        anim.save(out_path, writer=FFMpegWriter(fps=fps, bitrate=4000))
        log.info("Saved animation → %s", out_path)
    except Exception as exc:
        gif_path = os.path.splitext(out_path)[0] + ".gif"
        log.warning("MP4 failed (%s) — falling back to GIF: %s", exc, gif_path)
        anim.save(gif_path, writer=PillowWriter(fps=fps))
        log.info("Saved animation → %s", gif_path)

    plt.close(fig)
