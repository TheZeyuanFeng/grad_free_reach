import argparse
import json
import logging
import os

import jax
jax.config.update("jax_default_matmul_precision", "high")
from flax import nnx

from reachability.eval.ground_truth import _load_pairwise
from reachability.eval.modes import run_batch_mode, run_standard_mode
from reachability.eval.verification import run_verification
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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

setup_logging()
log = logging.getLogger(f"{PROJECT_NAME}.{__name__}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a trained HJ reachability model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run_dir", type=str, required=True,
                   help="Run directory containing config.yaml and checkpoints/.")
    p.add_argument("--ckpt_step", type=int, default=None,
                   help="Checkpoint step to load. Defaults to the highest step found.")
    p.add_argument("--dynamics_class", type=str, default=None,
                   help="Override dynamics class from config.")
    p.add_argument("--num_states", type=int, default=20000,
                   help="Number of initial states to evaluate.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dt", type=float, default=None,
                   help="Time step for evaluation. If not specified, uses the value from config.")
    p.add_argument("--policy_steps_per_dt", type=int, default=10,
                   help="Number of policy sub-steps per dt during rollout.")
    p.add_argument("--pairwise_path", type=str, default=None,
                   help="Path to pairwise BRT .npy for LessLinear analytical GT.")
    p.add_argument("--grid_res", type=int, default=256,
                   help="Resolution for the paper slice heatmap.")
    p.add_argument("--m_batches", type=int, default=None,
                   help="Number of batches for batch-based MSE evaluation. If set, enables batch mode.")
    p.add_argument("--batch_size", type=int, default=1000,
                   help="States per batch for batch-based evaluation.")
    p.add_argument("--no_rollout", action="store_true",
                   help="Skip rollout evaluation (faster; skips policy loading).")
    p.add_argument("--animate", action="store_true",
                   help="Render a trajectory animation after evaluation. "
                        "Only NarrowPassage is supported (two-car heatmap); ignored otherwise.")
    p.add_argument("--animate_select", type=str, default="best",
                   help="Which trajectory to animate: best | worst | random | random_fail | index.")
    p.add_argument("--animate_fps", type=int, default=20,
                   help="Frames per second for the animation.")
    p.add_argument("--eval_T", type=float, default=None,
                   help="Override the evaluation time-to-go. Must be > 0 and <= GAME.TIME.T. "
                        "Default: None, meaning the CHECKPOINT'S OWN trained horizon, "
                        "GAME.TIME.DT * ckpt_step -- under time marching each step extends "
                        "the horizon by one DT, so a mid-training checkpoint was never "
                        "trained out to the full GAME.TIME.T.")

    v = p.add_argument_group(
        "verification (--verify)",
        "PAC-style certification: bound the true violation rate of the claimed-safe "
        "set at confidence 1-beta. See reachability/eval/verification.py.")
    v.add_argument("--verify", action="store_true",
                   help="Run probabilistic verification instead of the usual eval modes.")
    v.add_argument("--beta", type=float, default=1e-6,
                   help="Confidence parameter; the bound holds w.p. >= 1-beta.")
    v.add_argument("--target_epsilon", type=float, default=0.01,
                   help="Violation rate to certify against.")
    v.add_argument("--verify_N", type=int, default=2000,
                   help="Scenario draws per test.")
    v.add_argument("--no_adversarial", action="store_true",
                   help="Drive disturbance with the policy net's learned d_pred instead "
                        "of worst-case probing. Weakens the guarantee -- see module docs.")
    v.add_argument("--no_search", action="store_true",
                   help="Skip the threshold search and certify --initial_delta_level directly.")
    v.add_argument("--initial_delta_level", type=float, default=0.0,
                   help="Starting (or, with --no_search, the only) safety threshold.")
    v.add_argument("--delta_grid_min", type=float, default=0.0)
    v.add_argument("--delta_grid_max", type=float, default=1.0)
    v.add_argument("--max_search_iters", type=int, default=12)
    v.add_argument("--convergence_tolerance", type=float, default=1e-3)
    v.add_argument("--max_reject_batches", type=int, default=200,
                   help="Cap on rejection-sampling batches when filling a draw.")
    v.add_argument("--sample_batch_size", type=int, default=4096)
    v.add_argument("--rollout_batch_size", type=int, default=512)
    v.add_argument("--N_volume_estimation", type=int, default=100000)
    v.add_argument("--volume_batch_size", type=int, default=8192)
    v.add_argument("--N_policy_success_rate_estimation", type=int, default=20000,
                   help="States rolled out UNIFORMLY (no claimed-safe filter, no threshold) to "
                        "measure the policy's overall success rate. Unlike epsilon this is "
                        "defined even when no threshold certifies, so it is what makes a "
                        "failing run comparable to a passing one. 0 disables.")
    v.add_argument("--curve_points", type=int, default=25,
                   help="Thresholds in the post-hoc epsilon-vs-delta curve (free: swept "
                        "from the certification sample). Comparison aid, not a certificate.")
    v.add_argument("--pin_state_dims", type=float, nargs="+", default=None,
                   metavar="IDX VALUE",
                   help="Flat IDX VALUE pairs fixing state dimensions in every draw, so "
                        "the certificate covers only that slice instead of the full "
                        "validation range. Intended for parameter-like dimensions the "
                        "dynamics holds constant -- e.g. '--pin_state_dims 9 0.92 10 0.0' "
                        "certifies the exact operating point a filter runs at.")
    return p


def parse_state_pins(raw, dyn) -> dict:
    """``[9, 0.92, 10, 0.0]`` -> ``{9: 0.92, 10: 0.0}``.

    Indices arrive as floats because argparse applies one type to the whole
    list, so they are checked for integrality here rather than silently
    truncating 9.5 to dim 9.
    """
    if not raw:
        return {}
    if len(raw) % 2 != 0:
        raise ValueError(
            f"--pin_state_dims takes IDX VALUE pairs, got {len(raw)} values: {raw}")
    pins = {}
    for idx_f, value in zip(raw[::2], raw[1::2]):
        idx = int(idx_f)
        if idx != idx_f:
            raise ValueError(f"--pin_state_dims index {idx_f} is not an integer.")
        if not 0 <= idx < dyn.state_dim:
            raise ValueError(
                f"--pin_state_dims index {idx} out of range for "
                f"{type(dyn).__name__} (state_dim={dyn.state_dim}).")
        if idx in pins:
            raise ValueError(f"--pin_state_dims pins dim {idx} twice.")
        pins[idx] = float(value)
    return pins

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_cfg(run_dir: str):
    """Load and freeze the yacs config saved by train.py."""
    config_path = os.path.join(run_dir, "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"config.yaml not found in {run_dir}. "
            "Make sure this run was produced by the current train.py."
        )
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_path)
    cfg.freeze()
    log.info("Loaded config: %s", config_path)
    return cfg

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_argparser().parse_args()
    key  = jax.random.PRNGKey(args.seed)
    rngs = nnx.Rngs(args.seed)

    cfg = load_cfg(args.run_dir)
    attach_file_handler(os.path.join(args.run_dir, "eval.log"))
    resolve_device(cfg)

    dyn = make_dynamics(cfg, override_class=args.dynamics_class)

    ckpt_dir = os.path.join(cfg.IO.LOG_DIR, cfg.IO.CKPT_DIRNAME)
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    step = args.ckpt_step or find_latest_step(ckpt_dir)
    dt = args.dt or cfg.GAME.TIME.DT
    if args.eval_T is not None:
        if not (0.0 < args.eval_T <= cfg.GAME.TIME.T):
            raise ValueError(
                f"--eval_T ({args.eval_T}) must be in (0, GAME.TIME.T={cfg.GAME.TIME.T}]"
            )
        t_final = args.eval_T
        log.info("Evaluating at t=%.3f (--eval_T override; GAME.TIME.T=%.3f)",
                 t_final, cfg.GAME.TIME.T)
    else:
        # The checkpoint's OWN horizon: time marching extends the trained
        # horizon by one DT per step, so step 150 of a T=8.0/DT=0.05 run was
        # only ever trained out to 7.5. Evaluating it at 8.0 asks the value net
        # for a horizon it never saw.
        t_final = cfg.GAME.TIME.DT * step
        if t_final > cfg.GAME.TIME.T + 1e-9:
            log.warning("DT * step = %.4f exceeds GAME.TIME.T = %.4f; clamping to T. "
                        "(step %d is past the configured horizon.)",
                        t_final, cfg.GAME.TIME.T, step)
            t_final = cfg.GAME.TIME.T
        log.info("Evaluating at t=%.4f = GAME.TIME.DT(%.4g) * step(%d)   [GAME.TIME.T=%.4f]",
                 t_final, cfg.GAME.TIME.DT, step, cfg.GAME.TIME.T)
    log.info("Evaluating checkpoint at step %d", step)

    value_net = load_value_net(cfg, dyn, rngs, ckpt_dir, step)

    save_path = os.path.join(cfg.IO.LOG_DIR, cfg.IO.EVAL_DIRNAME)
    os.makedirs(save_path, exist_ok=True)
    log.info("Saving evaluation results to %s", save_path)

    # Policy is only needed when rollout is requested.
    policy_net = None
    if not args.no_rollout:
        policy_net = load_policy_net(cfg, dyn, rngs, ckpt_dir, step)

    if args.verify:
        if policy_net is None:
            raise ValueError("Verification requires a policy net; remove --no_rollout.")
        va = {
            "beta": args.beta, "target_epsilon": args.target_epsilon, "N": args.verify_N,
            "adversarial": not args.no_adversarial, "search": not args.no_search,
            "initial_delta_level": args.initial_delta_level,
            "delta_grid_min": args.delta_grid_min, "delta_grid_max": args.delta_grid_max,
            "max_search_iters": args.max_search_iters,
            "convergence_tolerance": args.convergence_tolerance,
            "max_reject_batches": args.max_reject_batches,
            "sample_batch_size": args.sample_batch_size,
            "rollout_batch_size": args.rollout_batch_size,
            "policy_steps_per_dt": args.policy_steps_per_dt,
            "N_volume_estimation": args.N_volume_estimation,
            "volume_batch_size": args.volume_batch_size,
            "N_policy_success_rate_estimation": args.N_policy_success_rate_estimation,
            "curve_points": args.curve_points,
            "pin_state_dims": parse_state_pins(args.pin_state_dims, dyn),
        }
        logs = run_verification(key, dyn=dyn, value_net=value_net, policy_net=policy_net,
                                cfg=cfg, t_final=t_final, dt=dt, va=va)
        out_path = os.path.join(save_path, "verification.json")
        with open(out_path, "w") as f:
            json.dump(logs, f, indent=2, sort_keys=True, default=float)
        log.info("Saved verification → %s", out_path)
        return

    interp = None
    if args.pairwise_path and os.path.isfile(args.pairwise_path):
        log.info("Loading pairwise GT from %s", args.pairwise_path)
        interp = _load_pairwise(args.pairwise_path)

    if args.m_batches is not None:
        log.info("Running batch-based evaluation (m_batches=%d, batch_size=%d)",
                 args.m_batches, args.batch_size)
        run_batch_mode(
            value_net=value_net, policy_net=policy_net,
            dyn=dyn, interp=interp, t_final=t_final, dt=dt, cfg=cfg, args=args,
            key=key, save_path=save_path,
        )
    else:
        if policy_net is None:
            raise ValueError(
                "Standard evaluation requires a policy net. "
                "Remove --no_rollout or switch to --m_batches mode."
            )
        run_standard_mode(
            value_net=value_net, policy_net=policy_net,
            dyn=dyn, interp=interp, t_final=t_final, dt=dt, cfg=cfg, args=args,
            key=key, save_path=save_path,
        )


if __name__ == "__main__":
    main()
