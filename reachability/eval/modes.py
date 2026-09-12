"""Top-level evaluation modes, wiring the other eval modules together.

Each mode is a self-contained entry point called from ``eval.py``'s dispatch:

    run_standard_mode  single-pass rollout + confusion metrics + plots
    run_batch_mode     batched MSE-vs-analytical-GT (+ optional rollout)
"""

import json
import logging
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from reachability.dynamics import Dynamics
from reachability.data.sampling import sample_val_states_uniform
from reachability.plotting import (
    plot_value_histograms,
    plot_value_scatter,
    plot_confusion_matrix,
    plot_boundary_comparison,
)
from configs import PROJECT_NAME
from .ground_truth import evaluate_mse, gt_nd_value_array
from .metrics import calculate_inflation_delta, confusion_metrics
from .rollout import (
    evaluate_rollout,
    predicted_value_at_initial_time,
    rollout_trajectories,
)
from .slices import _maybe_plot_paper_slice

log = logging.getLogger(f"{PROJECT_NAME}.{__name__}")


def run_batch_mode(
    *,
    value_net: nnx.Module,
    policy_net,  # may be None when no_rollout=True
    dyn: Dynamics,
    interp,
    t_final,
    dt,
    cfg,
    args,
    key: jax.Array,
    save_path: str,
) -> None:
    """Batch-based evaluation (MSE + optional rollout + paper slice)."""
    mse = mae = None

    # ── MSE / confusion ──────────────────────────────────────────────────
    if interp is not None:
        log.info("Evaluating MSE over %d × %d states …", args.m_batches, args.batch_size)
        key, mse_key = jax.random.split(key)
        mse, mae, conf = evaluate_mse(
            value_net, dyn, interp,
            m_batches=args.m_batches,
            batch_size=args.batch_size,
            t_final=t_final,
            key=mse_key,
            control_role=cfg.GAME.CONTROL_ROLE,
        )
        results_path = os.path.join(save_path, "mse.txt")
        with open(results_path, "w") as f:
            f.write(
                f"t_final: {t_final:.4f}\n"
                f"m_batches: {args.m_batches}\n"
                f"batch_size: {args.batch_size}\n"
                f"MSE: {mse:.8f}\n"
                f"MAE: {mae:.8f}\n"
                f"TPR: {conf['TPR']:.6f}\n"
                f"FPR: {conf['FPR']:.6f}\n"
                f"TNR: {conf['TNR']:.6f}\n"
                f"FNR: {conf['FNR']:.6f}\n"
                f"BRT_vol_GT: {conf['vol_gt']:.6f}\n"
                f"BRT_vol_pred: {conf['vol_pred']:.6f}\n"
                f"GT_min: {conf['gt_min']:.6f}\n"
                f"GT_max: {conf['gt_max']:.6f}\n"
            )
        log.info("Saved metrics → %s", results_path)

    # ── Rollout ───────────────────────────────────────────────────────────
    if interp is not None and not args.no_rollout:
        if policy_net is None:
            log.warning("Skipping rollout: policy_net was not loaded (--no_rollout set at load time).")
        else:
            log.info("Rolling out learned policy over %d × %d states …",
                     args.m_batches, args.batch_size)
            key, rollout_key = jax.random.split(key)
            rollout_stats = evaluate_rollout(
                value_net, policy_net, dyn, interp,
                m_batches=args.m_batches,
                batch_size=args.batch_size,
                t_final=t_final,
                dt=dt,
                num_substeps=cfg.GAME.TIME.NUM_SUBSTEPS,
                key=rollout_key,
            )
            rollout_path = os.path.join(save_path, "rollout.txt")
            with open(rollout_path, "w") as f:
                f.write(
                    f"t_final: {t_final:.4f}\n"
                    f"m_batches: {args.m_batches}\n"
                    f"batch_size: {args.batch_size}\n"
                    f"MSE_rollout_vs_GT: {rollout_stats['mse_rollout_vs_gt']:.8f}\n"
                    f"MAE_rollout_vs_GT: {rollout_stats['mae_rollout_vs_gt']:.8f}\n"
                    f"MSE_rollout_vs_net: {rollout_stats['mse_rollout_vs_net']:.8f}\n"
                    f"MAE_rollout_vs_net: {rollout_stats['mae_rollout_vs_net']:.8f}\n"
                    f"mean_gap: {rollout_stats['mean_gap']:.8f}\n"
                    f"BRT_vol_GT: {rollout_stats['vol_gt']:.6f}\n"
                    f"BRT_vol_rollout: {rollout_stats['vol_rollout']:.6f}\n"
                    f"BRT_vol_net: {rollout_stats['vol_net']:.6f}\n"
                )
            log.info("Saved rollout metrics → %s", rollout_path)

    # ── Paper slice ───────────────────────────────────────────────────────
    _maybe_plot_paper_slice(value_net, dyn, interp, save_path,
                            args.grid_res, t_final, mse=mse, mae=mae)


def run_standard_mode(
    *,
    value_net: nnx.Module,
    policy_net: nnx.Module,
    dyn: Dynamics,
    interp,
    t_final,
    dt,
    cfg,
    args,
    key: jax.Array,
    save_path: str,
) -> None:
    """Single-pass rollout evaluation with plots."""
    # ── Sample and predict ────────────────────────────────────────────────
    key, sample_key = jax.random.split(key)
    x0         = sample_val_states_uniform(sample_key, dyn, args.num_states)
    pred_score = predicted_value_at_initial_time(dyn=dyn, value_net=value_net, t_final=t_final, cfg=cfg, x=x0)

    log.info("Sampled %d initial states", args.num_states)

    # ── Rollout ───────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    true_score, Xtraj = rollout_trajectories(
        dyn=dyn, value_net=value_net, policy_net=policy_net,
        t_final=t_final, dt=dt, cfg=cfg, x0=x0, policy_steps_per_dt=args.policy_steps_per_dt,
    )
    elapsed = time.perf_counter() - t0
    log.info("Rollout completed in %.3f s (%.6f s/sample).", elapsed, elapsed / args.num_states)

    # ── Analytical comparison ─────────────────────────────────────────────
    metrics_analytical: dict = {}
    if interp is not None:
        analytical_gt = gt_nd_value_array(interp, x0)
        mse_vs_gt = float(jnp.mean((pred_score - analytical_gt) ** 2))
        mae_vs_gt = float(jnp.mean(jnp.abs(pred_score - analytical_gt)))
        metrics_analytical = {"mse_vs_analytical": mse_vs_gt, "mae_vs_analytical": mae_vs_gt}
        log.info("Analytical GT  MSE=%.6f  MAE=%.6f", mse_vs_gt, mae_vs_gt)

    # ── Save trajectory ───────────────────────────────────────────────────
    traj_path = os.path.join(save_path, "state_traj.npy")
    np.save(traj_path, np.asarray(Xtraj))
    log.info("Saved trajectory → %s", traj_path)

    # ── Metrics ───────────────────────────────────────────────────────────
    metrics = confusion_metrics(
        pred_score=pred_score,
        true_score=true_score,
        control_role=cfg.GAME.CONTROL_ROLE,
    )
    metrics.update(metrics_analytical)

    safety_stats = calculate_inflation_delta(true_score, pred_score, cfg.GAME.CONTROL_ROLE)
    if safety_stats:
        metrics.update({f"inflation_{k}": v for k, v in safety_stats.items()})

    log.info("=== Evaluation ===")
    log.info("run_dir:  %s", args.run_dir)
    log.info("dynamics: %s", cfg.DYNAMICS.CLASS)
    log.info("problem:  %s  T=%.3f  dt=%.4f", cfg.GAME.PROBLEM_TYPE, t_final, dt)
    log.info(
        "N=%d  TP=%d  FP=%d  TN=%d  FN=%d  success=%d",
        int(metrics["N"]), int(metrics["TP"]), int(metrics["FP"]),
        int(metrics["TN"]), int(metrics["FN"]), metrics["n_success"],
    )
    log.info(
        "TPR=%.4f  FPR=%.4f  TNR=%.4f  FNR=%.4f  ACC=%.4f",
        metrics["TPR"], metrics["FPR"], metrics["TNR"], metrics["FNR"], metrics["ACC"],
    )

    metrics_path = os.path.join(save_path, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    log.info("Saved metrics → %s", metrics_path)

    # ── Plots ─────────────────────────────────────────────────────────────
    true_np = np.asarray(true_score)
    pred_np = np.asarray(pred_score)

    plot_value_histograms(true_scores=true_np, pred_scores=pred_np, out_dir=save_path)
    plot_value_scatter(pred_scores=pred_np, true_scores=true_np, out_dir=save_path)
    plot_confusion_matrix(metrics=metrics, out_dir=save_path, control_role=cfg.GAME.CONTROL_ROLE)
    plot_boundary_comparison(
        value_net=value_net, dyn=dyn,
        true_scores=true_score, states=x0,
        out_dir=save_path, T=t_final,
    )

    _maybe_plot_paper_slice(
        value_net, dyn, interp, save_path,
        args.grid_res, t_final,
        mse=metrics.get("mse_vs_analytical"),
        mae=metrics.get("mae_vs_analytical"),
    )

    # ── Animation (NarrowPassage only) ─────────────────────────────────────
    if getattr(args, "animate", False):
        if cfg.DYNAMICS.CLASS == "NarrowPassage":
            try:
                from reachability.animation.narrow_passage import animate_two_car_with_heatmap
                anim_path = os.path.join(save_path, "animation.mp4")
                animate_two_car_with_heatmap(
                    dyn=dyn,
                    model=value_net,
                    state_traj=Xtraj,
                    true_score=true_score,
                    out_path=anim_path,
                    select_mode=args.animate_select,
                    seed=args.seed,
                    fps=args.animate_fps,
                    dt=dt,
                )
                log.info("Saved animation → %s", anim_path)
            except Exception as exc:
                log.warning("animate_two_car_with_heatmap failed: %s", exc)
        else:
            log.warning("--animate is only supported for NarrowPassage; got %s. Skipping.",
                        cfg.DYNAMICS.CLASS)
