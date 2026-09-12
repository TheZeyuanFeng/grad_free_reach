"""
eval_lesslinear.py  —  Evaluate a trained LessLinearND value function (JAX/nnx).

Reuses the shared eval building blocks in reachability/eval/ (the same ones
eval.py --m_batches uses), so the numeric logic lives in one place:

  1. MSE / MAE + confusion vs the ground-truth pairwise-BRT decomposition
       V_ND(x) = Σᵢ₌₁^{N-1} V_2d(x₀, xᵢ)                     (evaluate_mse)
  2. Rollout BRT cost vs GT / vs net                          (evaluate_rollout)
  3. "Paper slice" heatmap  V_ND(x₀, y, …, y) = (N-1)·V_2d(x₀, y),
     learned vs GT side-by-side                               (_maybe_plot_paper_slice)

Single run:
    python eval_lesslinear.py --run_dir <run_dir> \\
        [--pairwise_path data/lesslinear_pairwise_values.npy] \\
        [--m_batches 10] [--batch_size 1000] [--grid_res 256]

Multi-seed aggregation (looks for {run_dir}_seed{i}[_<timestamp>]):
    python eval_lesslinear.py --run_dir <prefix> --num_seeds 5 ...
"""

import argparse
import glob
import json
import logging
import os

import numpy as np
import jax
jax.config.update("jax_default_matmul_precision", "high")
from flax import nnx

from configs import get_cfg_defaults, PROJECT_NAME
from utils import (
    attach_file_handler,
    find_latest_step,
    load_policy_net,
    load_value_net,
    make_dynamics,
    resolve_device,
    setup_logging,
)
from reachability.eval.ground_truth import _load_pairwise, evaluate_mse
from reachability.eval.rollout import evaluate_rollout
from reachability.eval.slices import _maybe_plot_paper_slice

setup_logging()
log = logging.getLogger(f"{PROJECT_NAME}.{__name__}")


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def _load_cfg(run_dir: str):
    cfg = get_cfg_defaults()
    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"config.yaml not found in {run_dir}")
    cfg.merge_from_file(cfg_path)
    cfg.freeze()
    return cfg


# ---------------------------------------------------------------------------
# Single-run evaluation
# ---------------------------------------------------------------------------

def _eval_single(run_dir: str, args, interp) -> dict:
    """Full LessLinear evaluation for one run directory; returns a metrics dict."""
    cfg = _load_cfg(run_dir)
    dyn = make_dynamics(cfg, override_class=args.dynamics_class)
    if not hasattr(dyn, "N"):
        raise ValueError(
            f"{type(dyn).__name__} has no `.N`; eval_lesslinear is for LessLinearND only."
        )
    N = dyn.N

    rngs = nnx.Rngs(args.seed)
    key = jax.random.PRNGKey(args.seed)

    ckpt_dir = os.path.join(run_dir, cfg.IO.CKPT_DIRNAME)
    step = args.ckpt_step or find_latest_step(ckpt_dir)
    dt = args.dt or cfg.GAME.TIME.DT
    # Same convention as eval.py: a checkpoint's own trained horizon is DT*step
    # under time marching, unless overridden by --eval_T.
    t_final = args.eval_T if args.eval_T is not None else step * cfg.GAME.TIME.DT
    if t_final > cfg.GAME.TIME.T + 1e-9:
        log.warning("t_final %.4f exceeds GAME.TIME.T %.4f; clamping.", t_final, cfg.GAME.TIME.T)
        t_final = cfg.GAME.TIME.T
    log.info("Loading checkpoint step %d (N=%d, t_final=%.4f, dt=%.4f)", step, N, t_final, dt)

    value_net = load_value_net(cfg, dyn, rngs, ckpt_dir, step)

    save_path = os.path.join(run_dir, cfg.IO.EVAL_DIRNAME)
    os.makedirs(save_path, exist_ok=True)

    result = {"run_dir": run_dir, "N": N, "t_final": t_final}
    mse = mae = None

    # ── MSE / MAE + confusion ──────────────────────────────────────────────
    if interp is not None:
        log.info("Evaluating MSE over %d × %d states …", args.m_batches, args.batch_size)
        key, mse_key = jax.random.split(key)
        mse, mae, conf = evaluate_mse(
            value_net, dyn, interp,
            m_batches=args.m_batches, batch_size=args.batch_size,
            t_final=t_final, key=mse_key, control_role=cfg.GAME.CONTROL_ROLE,
        )
        with open(os.path.join(save_path, "mse.txt"), "w") as f:
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
        log.info("Saved metrics → %s", os.path.join(save_path, "mse.txt"))
        result.update({"mse": mse, "mae": mae,
                       "TPR": conf["TPR"], "FPR": conf["FPR"],
                       "TNR": conf["TNR"], "FNR": conf["FNR"],
                       "vol_gt": conf["vol_gt"], "vol_pred": conf["vol_pred"]})

    # ── Rollout ────────────────────────────────────────────────────────────
    if interp is not None and not args.no_rollout:
        log.info("Rolling out learned policy over %d × %d states …", args.m_batches, args.batch_size)
        try:
            policy_net = load_policy_net(cfg, dyn, rngs, ckpt_dir, step)
            key, rollout_key = jax.random.split(key)
            rs = evaluate_rollout(
                value_net, policy_net, dyn, interp,
                m_batches=args.m_batches, batch_size=args.batch_size,
                t_final=t_final, dt=dt, num_substeps=cfg.GAME.TIME.NUM_SUBSTEPS,
                key=rollout_key,
            )
            with open(os.path.join(save_path, "rollout.txt"), "w") as f:
                f.write(
                    f"t_final: {t_final:.4f}\n"
                    f"m_batches: {args.m_batches}\n"
                    f"batch_size: {args.batch_size}\n"
                    f"MSE_rollout_vs_GT: {rs['mse_rollout_vs_gt']:.8f}\n"
                    f"MAE_rollout_vs_GT: {rs['mae_rollout_vs_gt']:.8f}\n"
                    f"MSE_rollout_vs_net: {rs['mse_rollout_vs_net']:.8f}\n"
                    f"MAE_rollout_vs_net: {rs['mae_rollout_vs_net']:.8f}\n"
                    f"mean_gap: {rs['mean_gap']:.8f}\n"
                    f"BRT_vol_GT: {rs['vol_gt']:.6f}\n"
                    f"BRT_vol_rollout: {rs['vol_rollout']:.6f}\n"
                    f"BRT_vol_net: {rs['vol_net']:.6f}\n"
                )
            log.info("Saved rollout metrics → %s", os.path.join(save_path, "rollout.txt"))
            result.update({k: rs[k] for k in (
                "mse_rollout_vs_gt", "mae_rollout_vs_gt",
                "mse_rollout_vs_net", "mae_rollout_vs_net",
                "mean_gap", "vol_rollout", "vol_net")})
        except FileNotFoundError as e:
            log.warning("Skipping rollout evaluation: %s", e)

    # ── Paper slice (learned vs GT) ────────────────────────────────────────
    _maybe_plot_paper_slice(value_net, dyn, interp, save_path, args.grid_res, t_final,
                            mse=mse, mae=mae)

    with open(os.path.join(save_path, "eval_lesslinear.json"), "w") as f:
        json.dump(result, f, indent=2, sort_keys=True, default=float)
    return result


# ---------------------------------------------------------------------------
# Multi-seed helpers
# ---------------------------------------------------------------------------

def _find_seed_dir(run_dir_prefix: str, seed: int) -> str:
    """Resolve the directory for a seed, ignoring any trailing timestamp.

    Accepts {prefix}_seed{seed} and {prefix}_seed{seed}_{timestamp}; when several
    timestamped matches exist the lexicographically latest is used."""
    exact = f"{run_dir_prefix}_seed{seed}"
    if os.path.isdir(exact):
        return exact
    candidates = sorted(d for d in glob.glob(f"{exact}_*") if os.path.isdir(d))
    if not candidates:
        raise FileNotFoundError(
            f"No directory for seed {seed} matching '{exact}' or '{exact}_*'")
    if len(candidates) > 1:
        log.warning("Multiple dirs for seed %d: %s — using latest", seed, candidates)
    return candidates[-1]


def _write_multiseed_summary(all_results: list, out_path: str) -> None:
    scalar_keys = [
        "mse", "mae", "TPR", "FPR", "TNR", "FNR", "vol_gt", "vol_pred",
        "mse_rollout_vs_gt", "mae_rollout_vs_gt",
        "mse_rollout_vs_net", "mae_rollout_vs_net",
        "mean_gap", "vol_rollout", "vol_net",
    ]
    lines = ["Multi-seed evaluation summary", "=" * 60,
             f"Seeds evaluated: {len(all_results)}", ""]
    for key in scalar_keys:
        vals = [r[key] for r in all_results if key in r]
        if vals:
            lines.append(f"{key:<28s}  {float(np.mean(vals)):.6f} ± {float(np.std(vals)):.6f}")
    lines += ["", "Per-seed details", "-" * 60]
    for i, r in enumerate(all_results):
        lines.append(f"  seed {i}  ({os.path.basename(r['run_dir'])})")
        for key in scalar_keys:
            if key in r:
                lines.append(f"    {key:<26s}  {r[key]:.6f}")
    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(text)
    log.info("Multi-seed summary → %s", out_path)
    print(text)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a trained LessLinearND value function.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run_dir", type=str, required=True,
                   help="Run directory (or prefix when --num_seeds is set).")
    p.add_argument("--num_seeds", type=int, default=None,
                   help="If set, evaluate seeds 0..num_seeds-1 by looking for "
                        "{run_dir}_seed{i}[_<timestamp>] directories and write an "
                        "aggregated summary.")
    p.add_argument("--dynamics_class", type=str, default=None,
                   help="Override dynamics class from config.")
    p.add_argument("--ckpt_step", type=int, default=None,
                   help="Checkpoint step to load. Defaults to the highest step found.")
    p.add_argument("--pairwise_path", type=str,
                   default="data/lesslinear_pairwise_values.npy",
                   help="Path to pairwise BRT .npy for GT comparison.")
    p.add_argument("--m_batches", type=int, default=10,
                   help="Number of batches for MSE / rollout evaluation.")
    p.add_argument("--batch_size", type=int, default=1000,
                   help="States per batch.")
    p.add_argument("--grid_res", type=int, default=256,
                   help="Paper-slice grid resolution.")
    p.add_argument("--eval_T", type=float, default=None,
                   help="Override the evaluation time-to-go. Default: DT * ckpt_step.")
    p.add_argument("--dt", type=float, default=None,
                   help="Rollout time step. Default: GAME.TIME.DT from config.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_rollout", action="store_true",
                   help="Skip rollout evaluation (faster; skips policy loading).")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    interp = None
    if os.path.isfile(args.pairwise_path):
        interp = _load_pairwise(args.pairwise_path)
    else:
        log.warning("Pairwise BRT not found at '%s' — skipping MSE / rollout / GT overlay.",
                    args.pairwise_path)

    if args.num_seeds is None:
        attach_file_handler(os.path.join(args.run_dir, "eval_lesslinear.log"))
        cfg = _load_cfg(args.run_dir)
        resolve_device(cfg)
        _eval_single(args.run_dir, args, interp)
        return

    all_results = []
    for seed in range(args.num_seeds):
        try:
            seed_dir = _find_seed_dir(args.run_dir, seed)
        except FileNotFoundError as e:
            log.warning("Skipping seed %d: %s", seed, e)
            continue
        log.info("=== Evaluating seed %d: %s ===", seed, seed_dir)
        r = _eval_single(seed_dir, args, interp)
        r["seed"] = seed
        all_results.append(r)

    if not all_results:
        log.error("No seed directories found — nothing to summarize.")
        return
    _write_multiseed_summary(all_results,
                             out_path=os.path.join(f"{args.run_dir}_eval_seeds", "summary.txt"))


if __name__ == "__main__":
    main()
