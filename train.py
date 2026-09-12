import argparse
from datetime import datetime, timezone
import logging
import os
import secrets
import sys

from reachability.training import Trainer
from configs import get_cfg_defaults
from utils import (
    resolve_device, 
    make_dynamics, 
    setup_logging, 
    install_exception_hook, 
    attach_file_handler
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

setup_logging()
log = logging.getLogger("train")

# ---------------------------------------------------------------------------
# Argument parser — minimal; all hyperparams live in the YAML
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Time-marching HJ reachability trainer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", type=str, default=None, metavar="YAML",
        help="Path to a YAML config file. Values override default_config.py defaults.",
    )
    p.add_argument(
        "--opt", nargs=argparse.REMAINDER, default=None,
        metavar="KEY VALUE",
        help=(
            "Override any config key inline, e.g. "
            "--opt TRAIN.LR 5e-4 NET.VALUE.ARCH transformer. "
            "Must be the last argument on the command line."
        ),
    )
    p.add_argument(
        "--resume", type=str, default=None,
        help=(
            "Resume from <OUTDIR>/checkpoints/<STEP> if it exists. "
        ),
    )
    p.add_argument(
        "--print_cfg", action="store_true",
        help="Print the fully-resolved config and exit without training.",
    )
    return p


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_cfg(args):
    """Resolve config: defaults → YAML → --opt flags. Returns a CfgNode."""
    cfg = get_cfg_defaults()

    if args.config is not None:
        if not os.path.isfile(args.config):
            raise FileNotFoundError(
                f"config.yaml not found in {args.config}. "
                "Make sure this run was produced by the current train.py."
            )
        cfg.merge_from_file(args.config)
        log.info("Loaded config: %s", args.config)
    return cfg

def load_brt_cfg(path: str):
    """Load the BRT config from the same directory as the checkpoint."""
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"BRT checkpoint not found at {path}. "
            "Cannot load BRT-based filter without a valid checkpoint."
        )
    brt_dir = os.path.dirname(os.path.dirname(path))
    brt_cfg_path = os.path.join(brt_dir, "config.yaml")
    if not os.path.isfile(brt_cfg_path):
        raise FileNotFoundError(
            f"BRT config.yaml not found at {brt_cfg_path}. "
            "Expected to find config.yaml in the same directory as the BRT checkpoint."
        )
    return load_cfg(argparse.Namespace(config=brt_cfg_path, opt=None, resume=None, print_cfg=False))

def create_path(directory: str, exp_name: str, prefix: str = "checkpoints", timestamp: bool = True) -> str:
    """Create a run directory with a time-specific name.

    If ``exp_name`` is empty, use ``<token_hex>_<timestamp>``.
    Otherwise, use ``<exp_name>_<timestamp>``.
    """
    os.makedirs(directory, exist_ok=True)
    if timestamp:
        while True:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if exp_name and exp_name.strip():
                run_id = f"{exp_name}_{timestamp}"
            else:
                run_id = f"{secrets.token_hex(4)}_{timestamp}"

            path = os.path.join(directory, run_id, prefix)
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                return path
    else:
        if exp_name and exp_name.strip():
            run_id = f"{exp_name}"
        else:
            run_id = f"{secrets.token_hex(4)}"
        path = os.path.join(directory, run_id, prefix)
        os.makedirs(path, exist_ok=True)
        return path


def save_cfg(cfg, outdir: str, args) -> None:
    """Write the resolved config to ``<outdir>/config.yaml``.

    Embeds the CLI invocation as a comment so the run is exactly
    reproducible by passing the saved file back as --config.
    """
    config_path = os.path.join(outdir, "config.yaml")
    timestamp_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    header = (
        "# ------------------------------------------------------------\n"
        "# Run metadata\n"
        f"# Generated (UTC): {timestamp_utc}\n"
        f"# Working dir: {os.getcwd()}\n"
        f"# Source YAML: {args.config or '(none — defaults only)'}\n"
        "# Reproduce command:\n"
        f"#   python {' '.join(sys.argv)}\n"
        "# ------------------------------------------------------------\n\n"
    )
    with open(config_path, "w") as f:
        f.write(header)
        f.write(cfg.dump())
    log.info("Config saved → %s", config_path)

# ---------------------------------------------------------------------------
# Auto-resume
# ---------------------------------------------------------------------------

def resolve_resume(cfg, args):
    """Determine whether to resume and from which path.

    Returns:
        (snapshot_path: str | None)
    """
    if args.resume is None:
        log.info("--resume not set; starting from scratch.")
        return None

    if not os.path.isdir(args.resume):
        log.error("--resume requested but no checkpoint dir found at %s", args.resume)
        sys.exit(1)

    log.info("Checkpoint detected at %s — auto-resuming.", args.resume)
    # Directory layout: outdir/<hex>/checkpoints/step_NNN
    auto_config = os.path.join(
        os.path.dirname(os.path.dirname(args.resume)), "config.yaml"
    )
    cfg.merge_from_file(auto_config)
    return args.resume

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    install_exception_hook()

    args = build_argparser().parse_args()
    cfg  = load_cfg(args)

    # --- Resume ---
    resume_path = resolve_resume(cfg, args)

    if args.opt:
        cfg.merge_from_list(args.opt)
        log.info("Applied --opt overrides: %s", args.opt)

    # --- Directories ---
    outdir = cfg.IO.OUTDIR
    if resume_path is not None:
        ckpt_dir = os.path.dirname(resume_path)   # outdir/<hex>/checkpoints        
    else:
        ckpt_dir = create_path(outdir, cfg.IO.EXP_NAME, timestamp=cfg.IO.TIMESTAMP_RUNS)
    
    cfg.IO.LOG_DIR = os.path.dirname(ckpt_dir)

    # --- File logging ---
    attach_file_handler(os.path.join(cfg.IO.LOG_DIR, "train.log"))
    log.info("Output directory: %s", os.path.abspath(outdir))

    cfg.freeze()  # Finalize config

    # --- Save resolved config immediately ---
    save_cfg(cfg, cfg.IO.LOG_DIR, args)

    if args.print_cfg:
        print(cfg.dump())
        sys.exit(0)

    # --- Device ---
    resolve_device(cfg)

    # --- Dynamics ---
    dyn = make_dynamics(cfg)

    log.info(
        "Starting | problem=%s | T=%.3f | dt=%.4f | slices=%d | net=%s",
        cfg.GAME.PROBLEM_TYPE,
        cfg.GAME.TIME.T,
        cfg.GAME.TIME.DT,
        round(cfg.GAME.TIME.T / cfg.GAME.TIME.DT),
        cfg.NET.VALUE.ARCH,
    )

    # --- Train ---
    trainer = Trainer(dyn, cfg, resume_path)
    trainer.train()

if __name__ == "__main__":
    main()