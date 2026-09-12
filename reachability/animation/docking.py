import logging
import math
import os
from typing import Optional, Tuple

import numpy as np
from configs.constants import PROJECT_NAME
import jax
import jax.numpy as jnp
from flax import nnx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Polygon, Rectangle
from matplotlib.lines import Line2D

from reachability.dynamics import Dynamics

log = logging.getLogger(f"{PROJECT_NAME}.{__name__}")

# Docking6D state layout: [px, py, vx, vy, theta, omega]
_PX, _PY, _VX, _VY, _TH, _OM = 0, 1, 2, 3, 4, 5


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _chaser_rect(
    px: float, py: float, theta: float, w: float = 1.0, h: float = 1.0
) -> np.ndarray:
    """Return (4, 2) corners of the chaser rectangle centred at (px, py) at angle θ."""
    hw, hh = w / 2, h / 2
    corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=np.float32)
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return corners @ R.T + np.array([px, py], dtype=np.float32)


def _add_target_geometry(ax, dyn: Dynamics) -> None:
    """Draw the stationary target spacecraft and docking port on ``ax``."""
    # Target body: rectangle centred at origin, width w_t, height h_t
    wt, ht = dyn.w_t, dyn.h_t
    body = Rectangle((-wt / 2, 0), wt, ht,
                     linewidth=1.2, edgecolor="dimgray", facecolor="lightgray",
                     zorder=2, label="target")
    ax.add_patch(body)

    # Docking port: semicircle at bottom centre
    theta_arc = np.linspace(math.pi, 2 * math.pi, 60)
    r = dyn.dock_rad
    ax.fill(r * np.cos(theta_arc), r * np.sin(theta_arc),
            color="steelblue", alpha=0.4, zorder=3)
    ax.plot(r * np.cos(theta_arc), r * np.sin(theta_arc),
            color="steelblue", linewidth=1.0, zorder=3)

    # Docking approach axis
    ax.axvline(0, color="steelblue", linestyle=":", linewidth=0.8, alpha=0.5)


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
        true_score: (N,) cost tensor — negative = docked successfully.
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


def _pxpy_slice(
    V_fn: _ValueFn,
    dyn: Dynamics,
    x_ref: jax.Array,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    nx: int,
    ny: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate V, target_l, avoid_l on a (ny, nx) grid over (px, py).

    All other state dims (vx, vy, θ, ω) are held fixed at ``x_ref``.

    Returns:
        xs:    (nx,) px-axis grid
        ys:    (ny,) py-axis grid
        Z:     (ny, nx) value heatmap
        t_lvl: (ny, nx) target_l
        a_lvl: (ny, nx) avoid_l
    """
    dtype = dyn.dtype
    xs = jnp.linspace(xlim[0], xlim[1], nx, dtype=dtype)
    ys = jnp.linspace(ylim[0], ylim[1], ny, dtype=dtype)
    Xg, Yg = jnp.meshgrid(xs, ys, indexing="xy")

    S = jnp.broadcast_to(jnp.asarray(x_ref, dtype=dtype)[None, :], (nx * ny, dyn.state_dim))
    S = S.at[:, _PX].set(Xg.reshape(-1))
    S = S.at[:, _PY].set(Yg.reshape(-1))

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
    """Save a (px, py) scatter of all failing trajectory poses."""
    fail_mask = true_score > 0
    if not bool(jnp.any(fail_mask)):
        log.info("No failing trajectories — skipping failure scatter.")
        return

    ft = np.asarray(state_traj[fail_mask])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(ft[:, 0, _PX], ft[:, 0, _PY], s=8, c="tab:red", alpha=0.5,
               marker="*", label="start")
    px_all = ft[:, :, _PX].reshape(-1)
    py_all = ft[:, :, _PY].reshape(-1)
    ax.scatter(px_all, py_all, s=0.5, c="tab:red", alpha=0.1, label="traj")
    ax.set_xlabel("px (m)")
    ax.set_ylabel("py (m)")
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
                      colors="black",       linewidths=1.5)
    cs_t = ax.contour(t_lvl, levels=[0.0], origin="lower", extent=extent,
                      colors="green",       linewidths=1.0, linestyles="--")
    cs_a = ax.contour(a_lvl, levels=[0.0], origin="lower", extent=extent,
                      colors="saddlebrown", linewidths=1.0, linestyles="--")
    return cs_v, cs_t, cs_a


def _remove_contours(*contour_sets):
    for cs in contour_sets:
        if hasattr(cs, "collections"):
            for coll in cs.collections:
                coll.remove()
        else:
            cs.remove()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def animate_docking(
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
    heat_alpha: float = 0.80,
) -> None:
    """Render and save a Docking6D trajectory animation with a value-function heatmap.

    Layout: main axes show the (px, py) position plane with the heatmap and
    target/chaser geometry; a narrow side panel shows vx, vy, θ, and ω
    as time-series so you can see the full 6D state evolve.

    Args:
        dyn:          Docking6D dynamics object.
        model:        Trained value network.
        state_traj:   Shape (N, K+1, 6) or (K+1, 6).
        true_score:   Per-trajectory cost, shape (N,).  Negative = docked.
        out_path:     Output file path (.mp4 or .gif).
        select_mode:  "best", "worst", "random", "random_fail", or "index".
        select_index: Used when select_mode="index".
        seed:         RNG seed for random selection modes.
        fps:          Frames per second.
        heat_nx/ny:   Heatmap grid resolution.
        xlim / ylim:  Position-plane bounds. Default: state box.
        dt:           Time step; used to compute time-to-go per frame.
        heat_alpha:   Heatmap opacity.
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
            f"state_traj must be (N, T, 6) or (T, 6), got {tuple(state_traj.shape)}"
        )

    T = X.shape[0]
    log.info("Animating trajectory #%d  cost=%.4f  T=%d", n_idx, float(cost), T)

    px_traj = np.asarray(X[:, _PX])
    py_traj = np.asarray(X[:, _PY])
    vx_traj = np.asarray(X[:, _VX])
    vy_traj = np.asarray(X[:, _VY])
    th_traj = np.asarray(X[:, _TH])
    om_traj = np.asarray(X[:, _OM])
    t_axis  = np.arange(T) * dt

    # ------------------------------------------------------------------
    # Plot bounds
    # ------------------------------------------------------------------
    if xlim is None:
        xlim = (float(dyn.state_low[_PX]), float(dyn.state_high[_PX]))
    if ylim is None:
        ylim = (float(dyn.state_low[_PY]), float(dyn.state_high[_PY]))

    extent = [xlim[0], xlim[1], ylim[0], ylim[1]]

    # ------------------------------------------------------------------
    # Figure layout: main axes (heatmap) + 4 side-panel axes
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(11, 7))
    gs  = fig.add_gridspec(4, 2, width_ratios=[2.2, 1], hspace=0.45, wspace=0.35)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_vx   = fig.add_subplot(gs[0, 1])
    ax_vy   = fig.add_subplot(gs[1, 1])
    ax_th   = fig.add_subplot(gs[2, 1])
    ax_om   = fig.add_subplot(gs[3, 1])

    # Side-panel full time-series (static background)
    for ax_s, data, ylabel, color in [
        (ax_vx, vx_traj, "vx (m/s)",   "steelblue"),
        (ax_vy, vy_traj, "vy (m/s)",   "darkorange"),
        (ax_th, th_traj, "θ (rad)",    "purple"),
        (ax_om, om_traj, "ω (rad/s)",  "green"),
    ]:
        ax_s.plot(t_axis, data, color=color, lw=1.0, alpha=0.3)
        ax_s.set_ylabel(ylabel, fontsize=7)
        ax_s.tick_params(labelsize=6)
        ax_s.set_xlim(t_axis[0], t_axis[-1])

    # Animated dots on side panels
    side_dots = []
    for ax_s, data in [(ax_vx, vx_traj), (ax_vy, vy_traj),
                       (ax_th, th_traj),  (ax_om, om_traj)]:
        (dot,) = ax_s.plot([], [], "o", markersize=4, color="k", zorder=5)
        side_dots.append((ax_s, dot, data))

    # ------------------------------------------------------------------
    # Initial heatmap
    # ------------------------------------------------------------------
    V0 = _ValueFn(model, (T - 1) * dt)
    xs, ys, Z, t_lvl, a_lvl = _pxpy_slice(V0, dyn, X[0], xlim, ylim, heat_nx, heat_ny)

    im = ax_main.imshow(
        Z, origin="lower", extent=extent,
        interpolation="bilinear", alpha=heat_alpha,
        aspect="auto", cmap="coolwarm_r",
    )
    fig.colorbar(im, ax=ax_main, fraction=0.035, pad=0.03, label="V(px,py | vx,vy,θ,ω fixed, t)")
    cs_v, cs_t, cs_a = _draw_contours(ax_main, Z, t_lvl, a_lvl, extent)

    # Target spacecraft geometry (static)
    _add_target_geometry(ax_main, dyn)

    # Chaser trail and body
    (trail,) = ax_main.plot([], [], lw=1.5, color="k", alpha=0.6, label="chaser")
    chaser_poly = Polygon(
        _chaser_rect(px_traj[0], py_traj[0], th_traj[0], w=dyn.w_c, h=dyn.h_c),
        closed=True, facecolor="steelblue", edgecolor="navy", alpha=0.9, zorder=6,
    )
    ax_main.add_patch(chaser_poly)

    # Legend handles
    legend_handles = [
        Line2D([0], [0], color="black",       lw=1.5,              label="V = 0"),
        Line2D([0], [0], color="green",       lw=1.0, ls="--",     label="target set"),
        Line2D([0], [0], color="saddlebrown", lw=1.0, ls="--",     label="collision set"),
        mpatches.Patch(facecolor="steelblue", edgecolor="navy",    label="chaser"),
        mpatches.Patch(facecolor="lightgray", edgecolor="dimgray", label="target"),
    ]
    ax_main.legend(handles=legend_handles, loc="upper right", fontsize=7)

    ax_main.set_xlim(*xlim)
    ax_main.set_ylim(*ylim)
    ax_main.set_aspect("equal", adjustable="box")
    ax_main.set_xlabel("px (m)")
    ax_main.set_ylabel("py (m)")
    title = ax_main.set_title("")

    # ------------------------------------------------------------------
    # Animation callbacks
    # ------------------------------------------------------------------

    def _init():
        trail.set_data([], [])
        for _, dot, _ in side_dots:
            dot.set_data([], [])
        return (im, trail, chaser_poly, title, *[d for _, d, _ in side_dots])

    def _update(t: int):
        nonlocal cs_v, cs_t, cs_a

        t_scalar = (T - t - 1) * dt
        Vt = _ValueFn(model, t_scalar)
        xs, ys, Z, t_lvl, a_lvl = _pxpy_slice(
            Vt, dyn, X[t], xlim, ylim, heat_nx, heat_ny
        )
        im.set_data(Z)
        im.set_clim(Z.min(), Z.max())

        _remove_contours(cs_v, cs_t, cs_a)
        cs_v, cs_t, cs_a = _draw_contours(ax_main, Z, t_lvl, a_lvl, extent)

        trail.set_data(px_traj[: t + 1], py_traj[: t + 1])
        chaser_poly.set_xy(_chaser_rect(
            px_traj[t], py_traj[t], th_traj[t], w=dyn.w_c, h=dyn.h_c
        ))

        # Side panel dots
        for ax_s, dot, data in side_dots:
            dot.set_data([t_axis[t]], [data[t]])

        tgt = float(dyn.target_l(X[t][None])[0])
        avd = float(dyn.avoid_l(X[t][None])[0])
        title.set_text(
            f"traj #{n_idx}  t={t}/{T - 1}  θ={th_traj[t]:.2f} rad"
            f"  cost={float(cost):.3f}  target_l={tgt:.3f}  avoid_l={avd:.3f}"
        )
        return (im, trail, chaser_poly, title, *[d for _, d, _ in side_dots])

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