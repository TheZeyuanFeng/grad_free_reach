import inspect
import logging
import math
import os
import sys
import traceback
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import orbax.checkpoint as ocp

import reachability.dynamics as dyn_module
from reachability.dynamics import Dynamics
from reachability.modules import *
from configs import PROJECT_NAME

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_FMT  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

def setup_logging(level: int = logging.DEBUG, console_level: int = logging.INFO) -> None:
    """Configure root logger with console output and prepare for file handler.
    """
    project_logger = logging.getLogger(PROJECT_NAME)
    project_logger.setLevel(level)  # Project logger allows all messages through
    project_logger.propagate = False 
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)  # Console filters to INFO+
    ch.setFormatter(logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT))
    project_logger.addHandler(ch)


def attach_file_handler(log_path: str, level: int = logging.DEBUG) -> None:
    """Add a rotating file handler to the root logger.

    Safe to call multiple times; each call adds exactly one handler for the
    given *log_path*.
    """
    project_logger = logging.getLogger(PROJECT_NAME)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT))
    project_logger.addHandler(fh)
    project_logger.info("Log file: %s", log_path)


def install_exception_hook() -> None:
    """Route uncaught exceptions through the logger so they land in log files.

    KeyboardInterrupt is passed through to Python's default handler so
    Ctrl-C exits cleanly without a traceback wall.
    """
    def _hook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logging.getLogger(PROJECT_NAME).critical(
            "Uncaught exception - aborting:\n%s",
            "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
        )
    sys.excepthook = _hook


# ---------------------------------------------------------------------------
# Network registry
# ---------------------------------------------------------------------------

NET_REGISTRY: dict = {
    "valmultinet":    TimeValueMultiNet,
    "vqpolmultinet":  VQPolicyMultiNet,
}

# Single-window building block behind each multi-window ARCH, keyed the same
# way as NET_REGISTRY -- used by load_*_net_last_window to construct just one
# window's net instead of the full multi-net container.
_INNER_NET_FOR_MULTI: dict = {
    "valmultinet":    TimeValueNet,
    "vqpolmultinet":  VQPolicyNet,
}

# ---------------------------------------------------------------------------
# Dynamics factory
# ---------------------------------------------------------------------------

def make_dynamics(cfg, override_class: str = None):
    """Instantiate the Dynamics subclass named in *cfg* (with optional overrides).

    Parameters
    ----------
    cfg:
        Resolved yacs CfgNode.  Reads ``cfg.DYNAMICS.CLASS``,
        ``cfg.DYNAMICS.KWARGS``, and ``cfg.IO.DEVICE``.
    override_class:
        If given, use this class name instead of ``cfg.DYNAMICS.CLASS``.
        Useful in eval / animate where ``--dynamics_class`` can be passed.
    override_device:
        If given, use this device string instead of ``cfg.IO.DEVICE``.

    Returns
    -------
    Dynamics instance.

    Raises
    ------
    ValueError
        If the resolved class name is not found in ``reachability.dynamics``.
    """
    log = logging.getLogger(PROJECT_NAME)

    class_name = override_class or cfg.DYNAMICS.CLASS

    if not hasattr(dyn_module, class_name):
        available = [k for k in dir(dyn_module) if k[0].isupper()]
        raise ValueError(
            f"Dynamics class '{class_name}' not found in reachability/dynamics.\n"
            f"Available: {available}"
        )

    cls    = getattr(dyn_module, class_name)

    all_params = dict(cfg.DYNAMICS.KWARGS)
    valid = inspect.signature(cls.__init__).parameters
    filtered = {k: v for k, v in all_params.items() if k in valid}
    dyn = cls(**filtered)

    log.info(
        "Dynamics: %s  state_dim=%d  control_dim=%d  disturb_dim=%d  obs_dim=%s  obs_ch_dim=%s  obs_embed_dim=%s",
        class_name, dyn.state_dim, dyn.control_dim, dyn.disturb_dim, dyn.obs_dim, dyn.obs_ch_dim, dyn.obs_embed_dim
    )
    return dyn


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def resolve_device(cfg) -> str:
    """Resolve ``cfg.IO.DEVICE``; falls back to CPU with a warning if CUDA is unavailable."""
    log = logging.getLogger(PROJECT_NAME)
    requested = cfg.IO.DEVICE
    
    devices = jax.devices()
    log.info("Available JAX devices: %s", [f"{d.platform}:{d.device_kind}" for d in devices])
    for device in devices:
        if requested.startswith("cuda") and device.platform == "gpu":
            log.info("JAX Platform: %s  Device: %s", device.platform, device)
            return device.platform
    log.warning(
        "Requested device '%s' is unavailable. Falling back to CPU (JAX backend: %s).",
        requested, device.platform
    )
    return devices[0].platform

# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def find_latest_step(ckpt_dir: str) -> int:
    """Return the highest step number found under ``<ckpt_dir>/step_NNN/``.

    Raises
    ------
    FileNotFoundError
        If no ``step_NNN`` subdirectories are found.
    """
    steps = []
    for name in os.listdir(ckpt_dir):
        if name.startswith("step_") and os.path.isdir(os.path.join(ckpt_dir, name)):
            try:
                steps.append(int(name.split("_")[-1]))
            except ValueError:
                pass
    if not steps:
        raise FileNotFoundError(f"No step_NNN directories found in {ckpt_dir}")
    return max(steps)


# ---------------------------------------------------------------------------
# Network loading (construct + restore weights)
# ---------------------------------------------------------------------------

def _count_params(model: nnx.Module) -> str:
    """Filter for Params and count elements."""
    params = nnx.state(model, nnx.Param)    
    n = sum(x.size for x in jax.tree_util.tree_leaves(params))
    return f"{n:,}"

def build_net(
    arch: str,
    cfg_kwargs: dict,
    context_params: dict,
    rngs: nnx.Rngs,
    wrapper_fn=None,
    label: str = "Net",
    registry: dict = NET_REGISTRY,
) -> nnx.Module:
    """
    Shared factory for value/policy nets.

    Priority: cfg_kwargs < context_params  (structured params win over config)
    Warns if cfg_kwargs tries to override a structured param.
    """
    if arch not in registry:
        raise ValueError(
            f"Unknown ARCH '{arch}'. Available: {sorted(registry)}"
        )
    cls = registry[arch]

    # Detect and warn about collisions before merging
    collisions = set(cfg_kwargs) & set(context_params)
    if collisions:
        logging.warning(
            "%s: cfg KWARGS %s shadowed by structured params — cfg values ignored",
            arch, collisions
        )

    all_params = {**cfg_kwargs, **context_params}

    # Filter to what the constructor actually accepts
    valid = inspect.signature(cls.__init__).parameters
    filtered = {k: v for k, v in all_params.items() if k in valid}

    dropped = set(all_params) - set(filtered)
    if dropped:
        logging.debug("%s: dropping incompatible params: %s", arch, dropped)

    net = cls(rngs=rngs, **filtered)
    if wrapper_fn is not None:
        net = wrapper_fn(net)

    logging.info("%s '%s': %s trainable params", label, arch, _count_params(net))
    return net

def load_value_net(cfg, dyn: Dynamics, rngs: nnx.Rngs, ckpt_dir: str, step: int) -> nnx.Module:
    """Construct the value network and load weights from a step checkpoint."""
    log = logging.getLogger(PROJECT_NAME)

    # Build the value network
    net = build_net(
        arch=cfg.NET.VALUE.ARCH,
        cfg_kwargs=dict(cfg.NET.VALUE.KWARGS),
        context_params={
            "window_time": cfg.GAME.TIME.WINDOW_TIME,
            "T_max":       cfg.GAME.TIME.T,
            "phi_dim":     dyn.input_dim,
            "control_dim": dyn.control_dim,
            "disturb_dim": dyn.disturb_dim,
            "obs_dim":       dyn.obs_dim,
            "obs_ch_dim":    dyn.obs_ch_dim,
            "obs_embed_dim": dyn.obs_embed_dim,
            "obs_kind":      getattr(dyn, "obs_kind", "seq"),
        },
        rngs=rngs,
        wrapper_fn=lambda net: BoundaryAwareNet(net, dyn, cfg.GAME.PROBLEM_TYPE, cfg.GAME.EXACT_BC),
        label="Value net",
    )

    # Restore weights from the orbax checkpoint written by Trainer._save_checkpoint,
    # which stores {"value_net", "value_opt", "policy_net", "policy_opt", "step"}
    # in a step_NNN directory. We partial-restore only the value_net subtree.
    ckpt_path = os.path.abspath(os.path.join(ckpt_dir, f"step_{step:03d}"))
    if not os.path.isdir(ckpt_path):
        raise FileNotFoundError(f"Value checkpoint not found: {ckpt_path}")

    item = {"value_net": nnx.state(net)}
    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(
        ckpt_path,
        args=ocp.args.PyTreeRestore(
            item=item,
            restore_args=ocp.checkpoint_utils.construct_restore_args(item),
            partial_restore=True,
        ),
    )
    nnx.update(net, restored["value_net"])
    net.eval()

    log.info("Loaded value net from %s", ckpt_path)
    return net

def load_policy_net(cfg, dyn: Dynamics, rngs: nnx.Rngs, ckpt_dir: str, step: int) -> nnx.Module:
    """Construct the policy network and load weights from a step checkpoint."""
    log = logging.getLogger(PROJECT_NAME)

    # Build the policy network
    net = build_net(
        arch=cfg.NET.POLICY.ARCH,
        cfg_kwargs=dict(cfg.NET.POLICY.KWARGS),
        context_params={
            "window_time": cfg.GAME.TIME.WINDOW_TIME,
            "T_max":       cfg.GAME.TIME.T,
            "phi_dim":     dyn.input_dim,
            "control_dim": dyn.control_dim,
            "disturb_dim": dyn.disturb_dim,
            "obs_dim":       dyn.obs_dim,
            "obs_ch_dim":    dyn.obs_ch_dim,
            "obs_embed_dim": dyn.obs_embed_dim,
            "obs_kind":      getattr(dyn, "obs_kind", "seq"),
            # Ternary VQ (neutral codeword) is implied by triple-sided probing;
            # build_net drops it for archs whose __init__ doesn't accept it.
            "ternary":       cfg.GAME.TRIPLE_SIDED_PROBING,
        },
        rngs=rngs,
        label="Policy net",
    )

    # Restore weights from the orbax checkpoint (see load_value_net); partial-restore
    # only the policy_net subtree.
    ckpt_path = os.path.abspath(os.path.join(ckpt_dir, f"step_{step:03d}"))
    if not os.path.isdir(ckpt_path):
        raise FileNotFoundError(f"Policy checkpoint not found: {ckpt_path}")

    item = {"policy_net": nnx.state(net)}
    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(
        ckpt_path,
        args=ocp.args.PyTreeRestore(
            item=item,
            restore_args=ocp.checkpoint_utils.construct_restore_args(item),
            partial_restore=True,
        ),
    )
    nnx.update(net, restored["policy_net"])
    net.eval()

    log.info("Loaded policy net from %s", ckpt_path)
    return net


def _num_windows(cfg) -> int:
    return math.ceil(cfg.GAME.TIME.T / cfg.GAME.TIME.WINDOW_TIME)


def _assert_window_present(
    checkpointer: ocp.PyTreeCheckpointer, ckpt_path: str, path: list, target_key: str, label: str,
) -> None:
    """Raise a clear error if `target_key` isn't actually stored in the checkpoint
    at `path` (a sequence of dict keys into the checkpoint's item tree).

    Without this check, Orbax's partial_restore silently leaves the freshly-built
    template -- an untrained, randomly-initialized network -- unchanged for a
    missing key instead of erroring, which would let a bad window_idx (e.g. one
    that isn't actually present in an exported single-window bundle) run
    silently on untrained weights instead of failing loudly.
    """
    node = checkpointer.metadata(ckpt_path).item_metadata
    for p in path:
        node = node[p]
    if target_key not in node:
        available = sorted(node.keys(), key=int)
        raise KeyError(
            f"{label}: window {target_key} not found in checkpoint {ckpt_path} "
            f"(windows actually present: {available})."
        )


def load_value_net_last_window(
    cfg, dyn: Dynamics, rngs: nnx.Rngs, ckpt_dir: str, step: int,
    window_idx: Optional[int] = None,
) -> nnx.Module:
    """Like load_value_net, but constructs and restores only a single time
    window's sub-network (plus its scalar inflation) instead of every window
    in the multi-net checkpoint. Defaults to the final window.

    A caller that only ever queries the BRT at one fixed t (e.g. the deployed
    safety filter, tri_sia_filter.py, normally queried at t=T) never reaches
    any other window, so materializing all of them just wastes memory/disk on
    target hardware -- or, when `window_idx` is passed explicitly, evaluates
    the checkpoint as though it had only been trained out to some earlier
    time-to-go than its actual T (e.g. to see how the filter would have
    behaved with a shorter horizon, without retraining). Orbax's
    partial_restore lets `item` name a single window's subtree by its
    checkpoint index -- the other windows are never read off disk.
    """
    log = logging.getLogger(PROJECT_NAME)

    arch = cfg.NET.VALUE.ARCH
    if arch not in _INNER_NET_FOR_MULTI:
        raise ValueError(
            f"load_value_net_last_window requires a multi-window ARCH, got '{arch}'"
        )

    num_windows = _num_windows(cfg)
    last_idx = num_windows - 1 if window_idx is None else window_idx
    if not (0 <= last_idx < num_windows):
        raise ValueError(f"window_idx must be in [0, {num_windows - 1}], got {last_idx}")

    inner = build_net(
        arch=arch,
        cfg_kwargs=dict(cfg.NET.VALUE.KWARGS),
        context_params={
            "T_max":         cfg.GAME.TIME.T,
            "phi_dim":       dyn.input_dim,
            "obs_dim":       dyn.obs_dim,
            "obs_ch_dim":    dyn.obs_ch_dim,
            "obs_embed_dim": dyn.obs_embed_dim,
            "obs_kind":      getattr(dyn, "obs_kind", "seq"),
        },
        rngs=rngs,
        registry=_INNER_NET_FOR_MULTI,
        label="Value net (last window)",
    )

    ckpt_path = os.path.abspath(os.path.join(ckpt_dir, f"step_{step:03d}"))
    if not os.path.isdir(ckpt_path):
        raise FileNotFoundError(f"Value checkpoint not found: {ckpt_path}")

    checkpointer = ocp.PyTreeCheckpointer()
    _assert_window_present(
        checkpointer, ckpt_path, ["value_net", "net", "nets"], str(last_idx), "Value net")

    item = {
        "value_net": {
            "net": {
                "nets": {str(last_idx): nnx.state(inner)},
                "inflations": {"value": jax.ShapeDtypeStruct((num_windows,), jnp.float32)},
            }
        }
    }
    restored = checkpointer.restore(
        ckpt_path,
        args=ocp.args.PyTreeRestore(
            item=item,
            restore_args=ocp.checkpoint_utils.construct_restore_args(item),
            partial_restore=True,
        ),
    )
    nnx.update(inner, restored["value_net"]["net"]["nets"][str(last_idx)])
    last_inflation = restored["value_net"]["net"]["inflations"]["value"][last_idx:last_idx + 1]

    windowed = WindowSlice(
        [inner], last_idx, cfg.GAME.TIME.WINDOW_TIME, num_windows,
        inflations=last_inflation,
    )
    net = BoundaryAwareNet(windowed, dyn, cfg.GAME.PROBLEM_TYPE, cfg.GAME.EXACT_BC)
    net.eval()

    log.info("Loaded value net window %d/%d ONLY from %s", last_idx, num_windows, ckpt_path)
    return net


def load_policy_net_last_window(
    cfg, dyn: Dynamics, rngs: nnx.Rngs, ckpt_dir: str, step: int,
    window_idx: Optional[int] = None,
) -> nnx.Module:
    """Policy-net counterpart of load_value_net_last_window (see there for
    why, including the meaning of `window_idx`); policy nets carry no
    inflation term."""
    log = logging.getLogger(PROJECT_NAME)

    arch = cfg.NET.POLICY.ARCH
    if arch not in _INNER_NET_FOR_MULTI:
        raise ValueError(
            f"load_policy_net_last_window requires a multi-window ARCH, got '{arch}'"
        )

    num_windows = _num_windows(cfg)
    last_idx = num_windows - 1 if window_idx is None else window_idx
    if not (0 <= last_idx < num_windows):
        raise ValueError(f"window_idx must be in [0, {num_windows - 1}], got {last_idx}")

    inner = build_net(
        arch=arch,
        cfg_kwargs=dict(cfg.NET.POLICY.KWARGS),
        context_params={
            "T_max":         cfg.GAME.TIME.T,
            "phi_dim":       dyn.input_dim,
            "control_dim":   dyn.control_dim,
            "disturb_dim":   dyn.disturb_dim,
            "obs_dim":       dyn.obs_dim,
            "obs_ch_dim":    dyn.obs_ch_dim,
            "obs_embed_dim": dyn.obs_embed_dim,
            "obs_kind":      getattr(dyn, "obs_kind", "seq"),
            "ternary":       cfg.GAME.TRIPLE_SIDED_PROBING,
        },
        rngs=rngs,
        registry=_INNER_NET_FOR_MULTI,
        label="Policy net (last window)",
    )

    ckpt_path = os.path.abspath(os.path.join(ckpt_dir, f"step_{step:03d}"))
    if not os.path.isdir(ckpt_path):
        raise FileNotFoundError(f"Policy checkpoint not found: {ckpt_path}")

    checkpointer = ocp.PyTreeCheckpointer()
    _assert_window_present(checkpointer, ckpt_path, ["policy_net", "nets"], str(last_idx), "Policy net")

    item = {"policy_net": {"nets": {str(last_idx): nnx.state(inner)}}}
    restored = checkpointer.restore(
        ckpt_path,
        args=ocp.args.PyTreeRestore(
            item=item,
            restore_args=ocp.checkpoint_utils.construct_restore_args(item),
            partial_restore=True,
        ),
    )
    nnx.update(inner, restored["policy_net"]["nets"][str(last_idx)])

    net = WindowSlice([inner], last_idx, cfg.GAME.TIME.WINDOW_TIME, num_windows)
    net.eval()

    log.info("Loaded policy net window %d/%d ONLY from %s", last_idx, num_windows, ckpt_path)
    return net


def export_last_window_checkpoint(
    cfg, dyn: Dynamics, rngs: nnx.Rngs, ckpt_dir: str, step: int, out_ckpt_dir: str,
    window_idx: Optional[int] = None,
) -> str:
    """Write a standalone checkpoint containing ONLY one time window's
    value_net + policy_net (+ inflation) -- for shipping to disk-constrained
    deployment hardware instead of the full multi-window training checkpoint
    (which stays on disk in its entirety even though load_*_net_last_window
    only ever *reads* one window's worth of it). Defaults to the final
    window; pass `window_idx` to export an earlier one instead (e.g. to test
    hardware behavior as though the model had only been trained to a shorter
    horizon, without retraining) -- see export_hardware_bundle.py, which also
    overrides GAME.TIME.T in the exported config.yaml to match, since
    load_*_net_last_window (and tri_sia_filter.py on the consuming end) both
    derive "the last/target window" from cfg.GAME.TIME.T rather than from a
    window index baked into the checkpoint itself.

    `cfg` here must still reflect the ORIGINAL training horizon -- it's used
    to size the restore against the on-disk checkpoint's actual (num_windows,)
    structure, not the horizon this export is meant to represent.

    Goes through load_value_net_last_window/load_policy_net_last_window
    internally, so exporting itself never reads the other windows off disk
    either. Saves under the exact same key layout those loaders expect, so
    they can point straight at the export directory unmodified -- ship this
    directory + a copy of the run's config.yaml to the target and load as
    usual.
    """
    log = logging.getLogger(PROJECT_NAME)

    value_net = load_value_net_last_window(cfg, dyn, rngs, ckpt_dir, step, window_idx=window_idx)
    policy_net = load_policy_net_last_window(cfg, dyn, rngs, ckpt_dir, step, window_idx=window_idx)

    num_windows = _num_windows(cfg)
    target_idx = num_windows - 1 if window_idx is None else window_idx

    # load_*_net_last_window always requests a full (num_windows,)-length
    # inflations array (whatever the checkpoint it points at actually
    # contains) -- keep that shape here, zero-filled except this export's one
    # window, so the exact same restore item/shape works against either the
    # full training checkpoint or this export.
    inflations_full = jnp.zeros((num_windows,), dtype=jnp.float32)
    inflations_full = inflations_full.at[target_idx].set(value_net.net.inflations[0])

    item = {
        "value_net": {"net": {
            "nets": {str(target_idx): nnx.state(value_net.net.nets[0])},
            "inflations": {"value": inflations_full},
        }},
        "policy_net": {"nets": {str(target_idx): nnx.state(policy_net.nets[0])}},
        "step": step,
    }

    out_path = os.path.abspath(os.path.join(out_ckpt_dir, f"step_{step:03d}"))
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(out_path, item, force=True)

    log.info("Exported window-only checkpoint (window %d/%d) to %s", target_idx, num_windows, out_path)
    return out_path


def decode_actions(
    out: dict,
    dyn: Dynamics,
) -> Tuple[jax.Array | None, jax.Array | None]:
    """Quantise policy outputs to a box vertex (the exact single-step argmax).

    3-way mapping with a neutral dead-zone so a ternary VQ codeword 0 decodes to
    the zero action: logits > 1/3 -> u_max, logits < -1/3 -> u_min, else -> 0.
    For a binary VQ the codeword is ±1, so |logits| = 1 > 1/3 and this matches the
    old ``where(logits>=0, hi, lo)`` everywhere the binary policy operates.
    """
    _DEADZONE = 1.0 / 3.0
    actions = []
    for logit_key, hi_attr, lo_attr in (("u_pred", "u_max", "u_min"),
                                        ("d_pred", "d_max", "d_min")):
        logits = out.get(logit_key)
        if logits is None:
            actions.append(None)
            continue
        hi = getattr(dyn, hi_attr).reshape(1, -1)
        lo = getattr(dyn, lo_attr).reshape(1, -1) if hasattr(dyn, lo_attr) else -hi
        actions.append(jnp.where(logits > _DEADZONE, hi,
                                 jnp.where(logits < -_DEADZONE, lo, 0.0)))
    return actions[0], actions[1]

def set_optimizer_lr(optimizer: nnx.Optimizer, target_lr: float):
    """Set the learning rate of an NNX/optax optimizer in place.

    Requires the optimizer to have been built with ``optax.inject_hyperparams``
    (see Trainer.__init__), which exposes ``learning_rate`` as a live state leaf
    at ``state['opt_state']['hyperparams']['learning_rate']``.
    """
    state = nnx.state(optimizer)
    state["opt_state"]["hyperparams"]["learning_rate"].value = jnp.asarray(
        target_lr, dtype=jnp.float32
    )
    nnx.update(optimizer, state)

class SummaryWriter:
    """Elegant WandB wrapper that embraces dictionary-based logging."""
    def __init__(self, cfg):
        import wandb
        self.wandb = wandb
        
        self.run = self.wandb.init(
            project=PROJECT_NAME,
            name=cfg.IO.EXP_NAME,
            config=cfg,
            entity=cfg.IO.WANDB_ENTITY,
        )
        self.set_metrics()

    def watch_model(self, model, log="gradients", log_freq=100):
        self.wandb.watch(model, log=log, log_freq=log_freq)

    def log(self, metrics: dict):
        """Native dictionary-based logging for all scalars."""
        self.wandb.log(metrics)

    def log_image(self, tag: str, img_data, step_dict: dict) -> None:
        """Handles images seamlessly, updating with the custom step dictionary."""
        if not isinstance(img_data, str):
            img_data = np.array(img_data)

        img = self.wandb.Image(img_data)            
        payload = {tag: img}
        payload.update(step_dict)
        self.wandb.log(payload)

    def set_metrics(self) -> None:
        # Define the custom x-axes
        self.wandb.define_metric("value_iter")
        self.wandb.define_metric("policy_iter")
        self.wandb.define_metric("curriculum_step")

        # Bind metrics to their respective x-axes
        self.wandb.define_metric("train/value_*", step_metric="value_iter")
        self.wandb.define_metric("train/teacher_frac", step_metric="value_iter")
        self.wandb.define_metric("train/policy_*", step_metric="policy_iter")
        self.wandb.define_metric("train/ctrl_acc", step_metric="policy_iter")
        self.wandb.define_metric("train/dist_acc", step_metric="policy_iter")
        
        # FIX: Bind the window and visualization metrics as well!
        self.wandb.define_metric("train/window_*", step_metric="curriculum_step")
        self.wandb.define_metric("visuals/*", step_metric="curriculum_step")

    def close(self):
        self.wandb.finish()

    def __enter__(self): return self
    def __exit__(self, *args): self.close()