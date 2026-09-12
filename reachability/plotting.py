import os
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from configs.constants import PROJECT_NAME

import jax
import jax.numpy as jnp
from flax import nnx

from reachability.dynamics import Dynamics

log = logging.getLogger(f"{PROJECT_NAME}.{__name__}")

_VIS_CHUNK_SIZE = 4096

def vis_val_fn(
    net: nnx.Module,
    dyn: Dynamics,
    out_path: str,
    x_res: int,
    y_res: int,
    title_prefix: str = "",
    delta_level=None,
    Tmax: float = 1.0,
    problem_type: Optional[str] = "BRT",
    x_lo: float = -1.0,
    x_hi: float = -1.0,
    y_lo: float = -1.0,
    y_hi: float = -1.0,
    frame: str = "world",
) -> None:
    """Render a grid of V slices.

    frame:
        ``"world"``  – track dynamics are gridded in world x/y and re-projected
                       per point with ``world_to_frenet``. Ignored by dynamics
                       that are not track-based; they always grid in state dims.
        ``"frenet"`` – track dynamics are gridded directly in (s, e), so no
                       re-projection is needed. Track-based dynamics only.
                       ``x_lo/x_hi/y_lo/y_hi`` are world-frame overrides and are
                       not consulted in this frame.
    """
    # 1. Setup Configuration
    cfg = dyn.plot_config()
    x_idx, y_idx, z_idx = cfg["x_axis_idx"], cfg["y_axis_idx"], cfg["z_axis_idx"]
    z_vals = cfg["z_vals"]
    state_labels = cfg.get("state_labels", [f"x{i}" for i in range(dyn.state_dim)])
    t_vals = np.linspace(0.0, Tmax, 5).astype(np.float32)

    assert len(cfg["state_slices"]) == dyn.state_dim, (
        "plot_config['state_slices'] length must equal dyn.state_dim"
    )
    assert len(state_labels) == dyn.state_dim, (
        "plot_config['state_labels'] length must equal dyn.state_dim"
    )
    
    if frame not in ("world", "frenet"):
        raise ValueError(f"frame must be 'world' or 'frenet', got {frame!r}")

    # 2. Prepare Grid
    # We create the spatial grid once
    is_track = hasattr(dyn, "lib") and hasattr(dyn.lib, "tracks")
    if frame == "frenet" and not is_track:
        raise ValueError(
            "frame='frenet' needs track-based dynamics (dyn.lib.tracks); "
            f"{type(dyn).__name__} grids in state dims already."
        )

    if is_track and frame == "world":
        idx = cfg["track_idx"]
        if x_lo < 0.0 or x_hi < 0.0 or y_lo < 0.0 or y_hi < 0.0:
            bounds = dyn.track_bounds[idx]
            x_min, x_max = bounds["x_min"], bounds["x_max"]
            y_min, y_max = bounds["y_min"], bounds["y_max"]
        else:
            x_min, x_max = x_lo, x_hi
            y_min, y_max = y_lo, y_hi
        x_range = jnp.linspace(x_min, x_max, x_res)
        y_range = jnp.linspace(y_min, y_max, y_res)
        x_label, y_label = "x (m)", "y (m)"
    elif is_track:
        x_range = jnp.linspace(0.0, _vis_track_length(dyn, cfg, x_idx), x_res)
        y_range = jnp.linspace(dyn.state_low[y_idx], dyn.state_high[y_idx], y_res)
        x_label, y_label = state_labels[x_idx], state_labels[y_idx]
    else:
        x_range = jnp.linspace(dyn.state_low[x_idx], dyn.state_high[x_idx], x_res)
        y_range = jnp.linspace(dyn.state_low[y_idx], dyn.state_high[y_idx], y_res)
        x_label, y_label = state_labels[x_idx], state_labels[y_idx]
    X_grid, Y_grid = jnp.meshgrid(x_range, y_range, indexing="xy")
    xys = jnp.stack([X_grid.ravel(), Y_grid.ravel()], axis=-1) # (N_grid, 2)
    N_grid = xys.shape[0]

    # Create base state template
    base_state_tmp = jnp.array(cfg["state_slices"], dtype=dyn.dtype)
    base_state = jnp.tile(base_state_tmp, (N_grid, 1))

    # 3. Plotting Setup
    n_rows, n_cols = len(t_vals), len(z_vals)
    panel_w, panel_h = (5.0, 4.5) if frame == "world" else (6.2, 2.9)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(panel_w * n_cols, panel_h * n_rows),
        constrained_layout=True,
        squeeze=False # Ensures axes is always 2D
    )

    extent = [x_range[0].item(), x_range[-1].item(), y_range[0].item(), y_range[-1].item()]
    X_np, Y_np = np.array(X_grid), np.array(Y_grid)

    delta_fn = delta_level if callable(delta_level) else (lambda _t: delta_level)

    # 4. Main Rendering Loop
    for r, t in enumerate(t_vals):
        t_batch = jnp.full((N_grid,), float(t), dtype=dyn.dtype)
        delta_r = delta_fn(float(t))

        for c, z in enumerate(z_vals):
            ax = axes[r, c]
            
            # Prepare batch for this specific subplot
            x_batch = jnp.array(base_state)
            x_batch = x_batch.at[:, x_idx].set(xys[:, 0])
            x_batch = x_batch.at[:, y_idx].set(xys[:, 1])
            x_batch = x_batch.at[:, z_idx].set(float(z))

            if is_track:
                x_chunks, v_chunks = [], []
                for i in range(0, N_grid, _VIS_CHUNK_SIZE):
                    xb = x_batch[i:i + _VIS_CHUNK_SIZE]
                    xb = dyn.world_to_frenet(xb) if frame == "world" else dyn.wrap_state(xb)
                    x_chunks.append(xb)
                    v_chunks.append(net(xb, t_batch[i:i + _VIS_CHUNK_SIZE])["V"])
                x_batch = jnp.concatenate(x_chunks, axis=0)
                v_out = jnp.concatenate(v_chunks, axis=0)
            else:
                x_batch = dyn.wrap_state(x_batch)
                v_out = net(x_batch, t_batch)["V"]
            V_img = np.array(v_out.reshape(y_res, x_res))

            # Heatmap
            im = ax.imshow(
                V_img, extent=extent, origin="lower",
                cmap="coolwarm_r", aspect="auto"
            )
            fig.colorbar(im, ax=ax, shrink=0.8)

            # Overlays
            _draw_boundaries(ax, dyn, x_batch, X_np, Y_np, x_res, y_res, problem_type)
            
            # Raw V=0 boundary.
            ax.contour(X_np, Y_np, V_img, levels=[0.0], colors="black", linewidths=2)

            # Labeling. The certified level goes in the row's y-label rather
            # than the title: it is per-t, and titles get clipped by colorbars.
            ax.set_title(f"{title_prefix} {state_labels[z_idx]}={z:.2f}, t={t:.2f}", fontsize=14)
            if r == n_rows - 1: ax.set_xlabel(x_label)
            if c == 0:
                ylab = y_label
                if delta_r is not None:
                    ylab += f"\ncert @ V={float(delta_r):+.3f}"
                ax.set_ylabel(ylab)
            ax.set_aspect("equal" if frame == "world" else "auto")
            if frame == "frenet":
                _mark_turns(ax, dyn, cfg, x_range)

    # 5. Save
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

def vis_val_fn_bev(
    net: nnx.Module,
    dyn: Dynamics,
    out_path: str,
    x_res: int,
    y_res: int,
    title_prefix: str = "",
    delta_level=None,
    Tmax: float = 1.0,
    problem_type: Optional[str] = "BRT",
    x_lo: float = -1.0,
    x_hi: float = -1.0,
    y_lo: float = -1.0,
    y_hi: float = -1.0,
    vis_chunk_size: int = 2000,
) -> None:
    """BEV-aware value-function visualiser for F1TenthBEV.

    Grids over the active track's world x-y extent (``dyn.track_bounds[track_idx]``)
    or an explicit zoom region if any of ``x_lo/x_hi/y_lo/y_hi`` is given (>= 0),
    then evaluates V in chunks of ``vis_chunk_size``. The ego-BEV is rendered per
    grid state INSIDE the (jitted) value net, so a full grid through the 2D BEV
    conv would OOM -- chunking keeps it bounded. Grid states carry the vis track's
    ``track_idx`` (via ``plot_config['state_slices']``), so each renders that track.
    """
    cfg = dyn.plot_config()
    x_idx, y_idx, z_idx = cfg["x_axis_idx"], cfg["y_axis_idx"], cfg["z_axis_idx"]
    z_vals = cfg["z_vals"]
    state_labels = cfg.get("state_labels", [f"x{i}" for i in range(dyn.state_dim)])
    t_vals = np.linspace(0.0, Tmax, 5).astype(np.float32)

    # Zoom region: explicit VIS.X_LO/X_HI/Y_LO/Y_HI when all are >= 0, else the
    # active track's world bounding box.
    if min(x_lo, x_hi, y_lo, y_hi) < 0.0:
        b = dyn.track_bounds[cfg["track_idx"]]
        x_lo, x_hi, y_lo, y_hi = b["x_min"], b["x_max"], b["y_min"], b["y_max"]

    x_range = jnp.linspace(x_lo, x_hi, x_res)
    y_range = jnp.linspace(y_lo, y_hi, y_res)
    X_grid, Y_grid = jnp.meshgrid(x_range, y_range, indexing="xy")
    xys = jnp.stack([X_grid.ravel(), Y_grid.ravel()], axis=-1)
    N_grid = xys.shape[0]
    base_state = jnp.tile(jnp.array(cfg["state_slices"], dtype=dyn.dtype), (N_grid, 1))
    X_np, Y_np = np.array(X_grid), np.array(Y_grid)
    extent = [float(x_lo), float(x_hi), float(y_lo), float(y_hi)]

    n_rows, n_cols = len(t_vals), len(z_vals)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows),
                             constrained_layout=True, squeeze=False)

    for r, t in enumerate(t_vals):
        t_batch = jnp.full((N_grid,), float(t), dtype=dyn.dtype)
        for c, z in enumerate(z_vals):
            ax = axes[r, c]
            x_batch = (base_state
                       .at[:, x_idx].set(xys[:, 0])
                       .at[:, y_idx].set(xys[:, 1])
                       .at[:, z_idx].set(float(z)))
            x_batch = dyn.wrap_state(x_batch)

            v_chunks = []
            for i in range(0, N_grid, vis_chunk_size):
                v_chunks.append(net(x_batch[i:i + vis_chunk_size],
                                    t_batch[i:i + vis_chunk_size])["V"])
            v_out = jnp.concatenate(v_chunks, axis=0)
            V_img = np.array(v_out.reshape(y_res, x_res))

            im = ax.imshow(V_img, extent=extent, origin="lower",
                           cmap="coolwarm_r", aspect="equal")
            fig.colorbar(im, ax=ax, shrink=0.8)

            _draw_boundaries(ax, dyn, x_batch, X_np, Y_np, x_res, y_res, problem_type)
            ax.contour(X_np, Y_np, V_img, levels=[0.0], colors="black", linewidths=2)

            ax.set_title(f"{title_prefix} {state_labels[z_idx]}={z:.2f}, t={t:.2f}", fontsize=14)
            if r == n_rows - 1:
                ax.set_xlabel(state_labels[x_idx])
            if c == 0:
                ax.set_ylabel(state_labels[y_idx])

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def _vis_track_length(dyn, cfg, x_idx: int) -> float:
    """Arc length of the track being plotted, for the Frenet s axis.

    Falls back to the state box when the LUT is not shaped as expected, which
    only over-extends the axis rather than plotting the wrong thing.
    """
    try:
        idx = dyn._lut.track_id_to_idx[cfg["track_idx"]]
        return float(dyn._lut.total_length[idx])
    except (AttributeError, KeyError, IndexError, TypeError):
        log.warning("Could not read per-track arc length; falling back to state bounds.")
        return float(dyn.state_high[x_idx])


def _mark_turns(ax, dyn, cfg, s_range, kappa_tol: float = 1e-3) -> None:
    """Dotted verticals at turn entry/exit.

    Curvature is invisible once the track is unrolled into (s, e), so without
    these there is no way to tell which part of a panel is a corner.
    """
    try:
        idx = dyn._lut.track_id_to_idx[cfg["track_idx"]]
        ti = jnp.full((s_range.shape[0],), idx, dtype=jnp.int32)
        kappa = np.asarray(dyn._lut.kappa(ti, s_range))
    except (AttributeError, KeyError, IndexError, TypeError):
        return
    is_turn = np.abs(kappa) > kappa_tol
    for i in np.flatnonzero(np.diff(is_turn.astype(np.int8)) != 0):
        ax.axvline(float(s_range[i]), color="0.25", lw=0.7, ls=":", alpha=0.8)


def _draw_boundaries(ax, dyn, x_batch, X_np, Y_np, x_res, y_res, problem_type):
    """Helper to draw target/avoid or safety boundaries."""
    import matplotlib.patches as mpatches
    cfg = dyn.plot_config()
    
    # 1. Draw ALL Goals as Green Dashed Circles
    if "goal_positions" in cfg:
        r_goal = getattr(dyn, "goal_radius", 0.5)
        for goal in cfg["goal_positions"]:
            circle = mpatches.Circle((goal[0], goal[1]), r_goal, 
                                     color='green', fill=False, linestyle='--', linewidth=1.5)
            ax.add_patch(circle)
            
    # 2. Draw ALL Obstacles as Red Dashed Polygons
    if "obstacle_vertices" in cfg and problem_type in ["BRAT"]:
        for verts in cfg["obstacle_vertices"]:
            poly = mpatches.Polygon(verts, closed=True, fill=False, 
                                    edgecolor='red', linestyle='--', linewidth=1.5)
            ax.add_patch(poly)
            
    has_reach_avoid = (
        hasattr(dyn, "target_l") and callable(dyn.target_l)
        and hasattr(dyn, "avoid_l") and callable(dyn.avoid_l)
    )
    # Reach-avoid problems: show target (green) and avoid (red) boundaries
    if has_reach_avoid and problem_type in ["BRAT"]:
        lT = np.array(dyn.target_l(x_batch).reshape(y_res, x_res))
        lA = np.array(dyn.avoid_l(x_batch).reshape(y_res, x_res))
        ax.contour(X_np, Y_np, lT, levels=[0.0], colors="green", linestyles="--", linewidths=1)
        ax.contour(X_np, Y_np, lA, levels=[0.0], colors="red",   linestyles="--", linewidths=1)

    # Safety problems: show safety boundary (brown)
    elif hasattr(dyn, "l") and callable(dyn.l) and problem_type in ["BRS", "BRT"]:
        l = np.array(dyn.l(x_batch).reshape(y_res, x_res))
        ax.contour(X_np, Y_np, l, levels=[0.0], colors="saddlebrown", linestyles="--", linewidths=1)

    else:
        log.warning("Dynamics does not expose target/avoid or safety level sets; skipping boundary overlays.")

# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_history(
    history: Dict[str, list],
    out_dir: str,
    fname: str = "training_curves.png",
) -> str:
    """Plot per-curriculum-step training metrics from the history dict.

    Produces a multi-panel figure with one subplot per metric key.
    Reads directly from the dict returned by ``Trainer.train()`` or loaded
    from ``history.json``.

    Panels:
      - ``loss_v`` (value loss)
      - ``loss_pi`` (policy imitation loss)
      - ``frac_reachable`` (fraction of states predicted inside BRT/BRS)
            - ``ctrl_acc`` and ``dist_acc`` on the same axes (policy accuracy)
      - ``teacher_frac`` (teacher blending schedule)

    Args:
        history:  Dict of lists, one entry per curriculum step.
                  Must contain at least ``"step"``.
        out_dir:  Output directory.
        fname:    Filename for the output image.

    Returns:
        out_path
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, fname)
    steps = history.get("step")
    if not steps:
        first_nonempty = next((v for v in history.values() if isinstance(v, list) and len(v) > 0), None)
        if first_nonempty is None:
            log.warning("Training history is empty; skipping training curve plot.")
            return out_path
        steps = list(range(len(first_nonempty)))

    # Define panel layout: (keys_to_plot, ylabel, title)
    panels: List[Tuple[List[str], str, str]] = [
        (["loss_v"],                 "MSE",        "Value loss"),
        (["loss_pi"],                "BCE",        "Policy loss"),
        (["frac_reachable"],         "fraction",   "Reachable fraction"),
        (["ctrl_acc", "dist_acc"],   "accuracy",   "Policy accuracy"),
        (["teacher_frac"],           "fraction",   "Teacher fraction"),
    ]

    # Only include panels where at least one key has data.
    panels = [(keys, yl, title) for keys, yl, title in panels
              if any(k in history and history[k] for k in keys)]

    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, (keys, ylabel, title) in zip(axes, panels):
        for i, key in enumerate(keys):
            if key in history and history[key]:
                ax.plot(steps, history[key], label=key,
                        color=colors[i % len(colors)], linewidth=1.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Curriculum step")
    fig.suptitle("Training history", fontsize=13, y=1.01)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved training history plot → %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    metrics: Dict[str, float],
    out_dir: str,
    control_role: str = "max",
    fname: str = "confusion_matrix.png",
) -> str:
    """Plot a 2x2 confusion matrix heatmap from ``confusion_metrics`` output.

    Args:
        metrics:      Dict returned by ``confusion_metrics()``.
        out_dir:      Output directory.
        control_role: ``"max"`` (BRT/BRAT) or ``"min"`` (BRS) — controls labels.
        fname:        Filename for the output image.
    Returns:
        out_path
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, fname)

    TP = int(metrics["TP"])
    FP = int(metrics["FP"])
    TN = int(metrics["TN"])
    FN = int(metrics["FN"])
    N  = int(metrics["N"])

    # Row = predicted, Col = true
    mat = np.array([[TP, FP], [FN, TN]], dtype=float)

    if control_role == "max":
        class_labels = ["Predicted safe", "Predicted unsafe"]
        col_labels   = ["True safe",      "True unsafe"]
    else:
        class_labels = ["Predicted reach", "Predicted no-reach"]
        col_labels   = ["True reach",      "True no-reach"]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mat, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(col_labels,   fontsize=9)
    ax.set_yticklabels(class_labels, fontsize=9)

    cell_labels = [
        [f"TP\n{TP}\n({100*TP/max(1,N):.1f}%)", f"FP\n{FP}\n({100*FP/max(1,N):.1f}%)"],
        [f"FN\n{FN}\n({100*FN/max(1,N):.1f}%)", f"TN\n{TN}\n({100*TN/max(1,N):.1f}%)"],
    ]
    thresh = mat.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cell_labels[i][j],
                    ha="center", va="center", fontsize=9,
                    color="white" if mat[i, j] > thresh else "black")

    ax.set_title(
        f"Confusion matrix  |  ACC={metrics['ACC']:.3f}  "
        f"TPR={metrics['TPR']:.3f}  TNR={metrics['TNR']:.3f}",
        fontsize=9,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved confusion matrix → %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Predicted vs true score scatter
# ---------------------------------------------------------------------------

def plot_value_scatter(
    pred_scores: np.ndarray,
    true_scores: np.ndarray,
    out_dir: str,
    max_points: int = 5000,
    fname: str = "value_scatter.png",
) -> str:
    """Scatter plot of predicted vs true scores, coloured by absolute error.

    Useful for diagnosing whether prediction errors are systematic
    (the cloud is shifted) or random (the cloud is symmetric around y=x).
    The diagonal y=x line is the ideal prediction.

    Args:
        pred_scores: Network predictions V(x, T), shape (N,).
        true_scores: Ground-truth scores from rollouts, shape (N,).
        out_dir:    Output directory.
        max_points:  Subsample to this many points if N is large.
        fname:       Filename for the output image.

    Returns:
        out_path
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, fname)

    if len(pred_scores) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(pred_scores), max_points, replace=False)
        pred_scores = pred_scores[idx]
        true_scores = true_scores[idx]

    abs_err = np.abs(pred_scores - true_scores)

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(
        true_scores, pred_scores,
        c=abs_err, cmap="YlOrRd", s=4, alpha=0.5,
        norm=mcolors.LogNorm(vmin=max(abs_err.min(), 1e-4), vmax=abs_err.max()),
    )
    fig.colorbar(sc, ax=ax, label="|pred − true|")

    lims = [
        min(true_scores.min(), pred_scores.min()) - 0.05,
        max(true_scores.max(), pred_scores.max()) + 0.05,
    ]
    ax.plot(lims, lims, "k--", linewidth=1, label="y = x (perfect)")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True score")
    ax.set_ylabel("Predicted V(x, T)")
    ax.set_title("Predicted vs true scores")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    log.info("Saved value scatter → %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Boundary comparison
# ---------------------------------------------------------------------------

def plot_boundary_comparison(
    value_net: nnx.Module,
    dyn: Dynamics,
    true_scores: jax.Array,
    states: jax.Array,
    out_dir: str,
    T: float,
    fname: str = "boundary_comparison.png",
) -> str:
    """Overlay predicted and true zero-level set boundaries on a 2D slice.

    Plots the state points coloured by sign (inside/outside), with the
    predicted V=0 boundary from the network and the empirical boundary
    from rollout scores shown as contours.

    Args:
        value_net:   Trained value network.
        dyn:         Dynamics object.
        true_scores: Ground-truth scores from rollouts, shape (N,).
        states:      Corresponding initial states, shape (N, state_dim).
        out_dir:    Output folder.
        T:           Time horizon for V(x, T) evaluation.
        fname:       Filename for the output image.

    Returns:
        out_path
    """
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, fname)
    plot_cfg = dyn.plot_config()
    xi = int(plot_cfg["x_axis_idx"])
    yi = int(plot_cfg["y_axis_idx"])
    labels = plot_cfg.get("state_labels", [f"x{i}" for i in range(dyn.state_dim)])

    t_full = jnp.full((states.shape[0],), T, dtype=dyn.dtype)
    pred_scores = value_net(states, t_full)["V"]
    pred_scores = np.array(pred_scores)
    true_np = np.array(true_scores)

    states = states.state if hasattr(states, 'state') else states
    xs = np.array(states[:, xi])
    ys = np.array(states[:, yi])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: true classification
    ax = axes[0]
    ax.scatter(xs[true_np < 0],  ys[true_np < 0],  s=2, c="steelblue",  alpha=0.4, label="true inside")
    ax.scatter(xs[true_np >= 0], ys[true_np >= 0], s=2, c="lightsalmon", alpha=0.4, label="true outside")
    ax.set_title("Ground truth (rollout)")
    ax.set_xlabel(labels[xi])
    ax.set_ylabel(labels[yi])
    ax.legend(fontsize=7, markerscale=4)

    # Panel 2: predicted classification
    ax = axes[1]
    ax.scatter(xs[pred_scores < 0],  ys[pred_scores < 0],  s=2, c="steelblue",  alpha=0.4, label="pred inside")
    ax.scatter(xs[pred_scores >= 0], ys[pred_scores >= 0], s=2, c="lightsalmon", alpha=0.4, label="pred outside")
    ax.set_title("Predicted (network)")
    ax.set_xlabel(labels[xi])
    ax.legend(fontsize=7, markerscale=4)

    # Panel 3: errors
    ax = axes[2]
    true_inside = true_np < 0
    pred_inside = pred_scores < 0

    correct = true_inside == pred_inside
    fp = (~true_inside) & pred_inside   # predicted inside but actually outside
    fn = true_inside & (~pred_inside)   # predicted outside but actually inside

    ax.scatter(xs[correct], ys[correct], s=2, c="lightgray",   alpha=0.3, label="correct")
    ax.scatter(xs[fp],      ys[fp],      s=6, c="red",         alpha=0.7, label=f"FP ({fp.sum()})")
    ax.scatter(xs[fn],      ys[fn],      s=6, c="darkorange",  alpha=0.7, label=f"FN ({fn.sum()})")
    ax.set_title("Prediction errors")
    ax.set_xlabel(labels[xi])
    ax.legend(fontsize=7, markerscale=3)

    fig.suptitle(
        f"Boundary comparison  |  N={len(xs)}  "
        f"FP={fp.sum()}  FN={fn.sum()}  "
        f"ACC={correct.mean():.3f}",
        fontsize=10,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    log.info("Saved boundary comparison → %s", out_path)
    return out_path

def plot_value_histograms(
    true_scores: np.ndarray,
    pred_scores: np.ndarray,
    out_dir: str,
    fname: str = "value_hist.png",
    bins: int = 100,
) -> None:
    """Plot overlaid true/predicted score histograms and a false-negative diff plot.

    Saves two files:
      - ``<fname>``             – overlaid histogram of true and predicted scores.
      - ``<fname>_fn_diff.png`` – histogram of (true − pred) in the FN region,
                                  with quantile markers. Only saved if FN > 0.

    Args:
        true_scores: Ground-truth rollout scores, shape (N,).
        pred_scores: Network predictions V(x, T), shape (N,).
        out_dir:     Directory to save plots in.
        fname:       Filename for the main histogram.
        bins:        Number of histogram bins.
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- Main histogram ---
    out_path = os.path.join(out_dir, fname)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(true_scores, bins=bins, alpha=0.6, label="True score",        density=True)
    ax.hist(pred_scores, bins=bins, alpha=0.6, label="Predicted V(x, T)", density=True)
    ax.axvline(0.0, color="k", linestyle="--", linewidth=2, label="V = 0")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title("Value score distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved histogram → %s", out_path)

    # --- False-negative region: pred < 0 but true >= 0 ---
    mask = (pred_scores < 0.0) & (true_scores >= 0.0)
    if mask.sum() == 0:
        return

    fname_stem = os.path.splitext(fname)[0]
    out_diff   = os.path.join(out_dir, f"{fname_stem}_fn_diff.png")
    diff       = true_scores[mask] - pred_scores[mask]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(true_scores[mask], bins=bins, alpha=0.6,
            label="True score (FN region)", density=True)
    ax.hist(diff, bins=bins, alpha=0.6, label="True − Pred", density=True)
    for q, color, label in [
        (0.90, "orange", "90th pct"),
        (0.99, "red",    "99th pct"),
        (0.999, "m",     "99.9th pct"),
    ]:
        ax.axvline(np.quantile(diff, q), color=color,
                   linestyle="-.", linewidth=1.5, label=label)
    ax.axvline(0.0, color="k", linestyle="--", linewidth=2, label="V = 0")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title(f"False-negative region: true − predicted  (N={mask.sum()})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_diff, dpi=150)
    plt.close(fig)
    log.info("Saved FN diff histogram → %s", out_diff)

def _plot_panel(ax, fig, X0, Xi, V, title: str, vmin: float, vmax: float):
    s = ax.contourf(X0, Xi, V, cmap="coolwarm_r", levels=256,
                    vmin=vmin, vmax=vmax)
    ax.contour(X0, Xi, V, levels=[0.0],
               colors="black", linewidths=1.5, linestyles="--")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(s, cax=cax)
    cbar.set_ticks([vmin, 0.0, vmax])
    cbar.set_ticklabels([f"{vmin:.2f}", "0", f"{vmax:.2f}"])
    ax.set_xlabel(r"$x_0$")
    ax.set_ylabel(r"$x_i$")
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)


def plot_paper_slice(
    X0, Xi,
    v_learned: np.ndarray,
    v_gt,        # np.ndarray or None
    out_path: str,
    N: int,
    t_final: float,
    mse: float = None,
    mae: float = None,
):
    has_gt = v_gt is not None
    ncols  = 2 if has_gt else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols + 0.5, 5))
    if ncols == 1:
        axes = [axes]

    v_all = [v_learned] + ([v_gt] if has_gt else [])
    vabs  = max(np.abs(v).max() for v in v_all)
    vmin, vmax = -vabs, vabs

    _plot_panel(axes[0], fig, X0, Xi, v_learned,
                f"Learned  ($T={t_final:.2f}$)", vmin, vmax)
    if has_gt:
        _plot_panel(axes[1], fig, X0, Xi, v_gt, "GT (pairwise BRT)", vmin, vmax)

    err_str = ""
    if mse is not None:
        err_str = f"   MSE={mse:.5f}   MAE={mae:.5f}"
    fig.suptitle(
        rf"$V_{{N={N}}}(x_0,\,y,\ldots,y)$,  BRT boundary (dashed){err_str}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved paper slice → %s", out_path)