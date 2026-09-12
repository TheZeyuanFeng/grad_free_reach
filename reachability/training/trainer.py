import itertools
import time
import json
import math
import os
import gc

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import nnx
from tqdm import tqdm
import logging
import numpy as np
from typing import Callable, Dict, Optional

from configs.constants import PROJECT_NAME
from reachability.dynamics import Dynamics
from reachability.data.anchor import (
    AnchorDataset,
    sample_anchor_minibatch,
    build_anchor_dataset_window,
)
from reachability.data.sampling import (
    sample_uniform_states,
    BoundaryAwareSampler
)
from .losses import (
    fp_weighted_anchor_loss,
    vq_policy_loss,
)
from .engine import HJSolver
from .strategy import ProbingStrategy
from reachability.eval.metrics import confusion_metrics, _unsafe_mask
from .functional import make_rollout_length_schedule
from utils import (
    set_optimizer_lr,
    build_net,
    SummaryWriter,
)
from reachability.plotting import plot_training_history, vis_val_fn, vis_val_fn_bev
from reachability.modules import *

log = logging.getLogger(f"{PROJECT_NAME}.{__name__}")


class _PlateauTracker:
    """Running-min plateau detector over fixed-size loss blocks.

    Feed one loss per iteration to :meth:`update`. It averages ``check_every``
    consecutive losses into a block and reports convergence once ``patience``
    consecutive blocks each fail to beat the running-min block mean (over all
    strictly-prior blocks) by at least ``rel_tol``. Purely relative, so it is
    scale-free across systems. No convergence before ``min_iters`` iterations.
    """

    def __init__(self, check_every: int, patience: int, rel_tol: float,
                 min_iters: int = 0) -> None:
        self.check_every = max(1, int(check_every))
        self.patience = int(patience)
        self.rel_tol = float(rel_tol)
        self.min_iters = int(min_iters)
        self.block_sum = 0.0
        self.block_n = 0
        self.best = None
        self.stalled = 0
        self.last_block_mean = None

    def update(self, loss: float, step: int) -> bool:
        """Record ``loss`` at 1-based ``step``; True iff just converged."""
        self.block_sum += float(loss)
        self.block_n += 1
        if step % self.check_every != 0:
            return False
        block_mean = self.block_sum / max(1, self.block_n)
        self.block_sum, self.block_n = 0.0, 0
        self.last_block_mean = block_mean
        converged = False
        if step >= self.min_iters and self.best is not None:
            improved = block_mean < (1.0 - self.rel_tol) * self.best
            self.stalled = 0 if improved else self.stalled + 1
            converged = self.stalled >= self.patience
        self.best = block_mean if self.best is None else min(self.best, block_mean)
        return converged


# ---------------------------------------------------------------------------
# Network registry
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(
        self,
        dyn: Dynamics,
        cfg,
        resume_path: str = None,
    ) -> None:
        self.dyn = dyn
        self.cfg = cfg
        self.key = jax.random.PRNGKey(cfg.IO.SEED)
        rngs = nnx.Rngs(cfg.IO.SEED)
        self.value_net = self._build_value_net(rngs, cfg, dyn)
        self.policy_net = self._build_policy_net(rngs, cfg, dyn)

        # EMA/Polyak-averaged copy of the value net
        self.use_ema_target = cfg.TRAIN.USE_EMA_TARGET
        self.ema_tau = cfg.TRAIN.EMA_TAU
        if self.use_ema_target:
            self.value_target_net = nnx.clone(self.value_net)
        else:
            self.value_target_net = None

        self.checkpointer = ocp.PyTreeCheckpointer()

        if cfg.IO.USE_WANDB:
            self.summary_writer = SummaryWriter(cfg)
            self.summary_writer.watch_model(self.value_net, log="gradients", log_freq=cfg.IO.LOG_EVERY)
            self.summary_writer.watch_model(self.policy_net, log="gradients", log_freq=cfg.IO.LOG_EVERY)
        else:
            self.summary_writer = None

        self.start_step = -1 if cfg.PRETRAIN.ENABLED else 0
        self.value_counter = 0
        self.policy_counter = 0
        # Per-step metric history — accumulated throughout training.
        self.history = self._default_history()

        self.finetune_lr = 0.3 * cfg.TRAIN.LR  # Finetune last layers with smaller LR

        def _clipped_adamw(learning_rate, b1, b2, weight_decay, eps=1e-8):
            return optax.chain(
                optax.clip_by_global_norm(cfg.TRAIN.GRAD_CLIP_NORM),
                optax.adamw(learning_rate=learning_rate, b1=b1, b2=b2,
                            weight_decay=weight_decay, eps=eps),
            )

        _make_adamw = optax.inject_hyperparams(_clipped_adamw)
        tx_value = _make_adamw(
            learning_rate=cfg.TRAIN.LR,
            b1=0.9,
            b2=0.95,
            weight_decay=1e-5
        )
        tx_policy = _make_adamw(
            learning_rate=cfg.TRAIN.POLICY_LR,
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=1e-4,
        )

        self.value_opt = nnx.Optimizer(self.value_net, tx_value, wrt=nnx.Param)
        self.policy_opt = nnx.Optimizer(self.policy_net, tx_policy, wrt=nnx.Param)

        self._tx_value = tx_value
        self._tx_policy = tx_policy
        self._value_opt_window: Optional[int] = None
        self._policy_opt_window: Optional[int] = None

        self.current_window: int = 0
        self._update_step_cache = {}
        self._probe_jit_cache = {}
        self._window_net_cache = {}
        self._persistent_window_net_cache = {}
        self._finetune_scratch = None

        if resume_path is not None:
            self.start_step = self._load_checkpoint(
                resume_path,
                value_net=self.value_net,
                policy_net=self.policy_net,
                value_target_net=self.value_target_net,
            )
            self.history = self._load_history()

        # HJSolver owns the TD-target rollout logic and the two strategies.
        self.solver = HJSolver(
            dyn, self.value_net, self.policy_net, cfg,
        )

        self.sampler = BoundaryAwareSampler(
            dyn,
            value_net=self.value_net,
            w_uniform=cfg.SAMPLING.FRAC_UNIFORM,
            w_target=cfg.SAMPLING.FRAC_TARGET,
            w_avoid=cfg.SAMPLING.FRAC_AVOID,
            w_user_target=cfg.SAMPLING.FRAC_USER_TARGET,
            w_value_boundary=cfg.SAMPLING.FRAC_BOUNDARY,
            w_safe=cfg.SAMPLING.FRAC_SAFE,
            w_unsafe=cfg.SAMPLING.FRAC_UNSAFE,
            unsafe_is_positive=(cfg.GAME.CONTROL_ROLE != "max"),
            boundary_band=cfg.SAMPLING.BOUNDARY_BAND,
            buffer_size=cfg.SAMPLING.BUFFER_SIZE,
            pool_size=cfg.SAMPLING.POOL_SIZE,
            pool_max_draws=cfg.SAMPLING.POOL_MAX_DRAWS,
            pool_chunk=cfg.SAMPLING.POOL_CHUNK,
            w_nominal=cfg.SAMPLING.FRAC_NOMINAL,
            nominal_steps=cfg.SAMPLING.NOMINAL_STEPS,
            nominal_burn_in_steps=cfg.SAMPLING.NOMINAL_BURN_IN_STEPS,
            nominal_samples_per_rollout=cfg.SAMPLING.NOMINAL_SAMPLES_PER_ROLLOUT,
        )

        self.teacher = ProbingStrategy(self.value_net, cfg)

        self.lam: float = cfg.TRAIN.LAMBDA
        self.M: int = cfg.TRAIN.M
        self.log_every: int = cfg.IO.LOG_EVERY
        self.total_steps: int = int(cfg.GAME.TIME.T / cfg.GAME.TIME.DT)
        self.save_every: int = cfg.IO.SAVE_EVERY
        self.vis_every: int = cfg.VIS.EVERY
        self._pg_loss_weight: float = 1.0  # dynamically adjusted in combined mode

        # Fixed validation batch — used every step to track V coverage.
        self.key, val_key = jax.random.split(self.key)
        self._val_x = sample_uniform_states(val_key, dyn, cfg.TRAIN.BATCH_SIZE)

    # ------------------------------------------------------------------
    # Shared training skeleton
    # ------------------------------------------------------------------

    def _training_loop(
        self,
        desc: str,
        net: nnx.Module,
        min_steps: int,
        max_steps: int,
        check_every: int,
        patience: int,
        rel_tol: float,
        iter_fn: Callable[[int], dict],
        trainable=None,
    ) -> dict:
        """Inner loop with running-min block plateau convergence.

        Stops when ``patience`` consecutive ``check_every``-iter blocks fail to
        beat the running-min block-mean loss by ``rel_tol`` (never before
        ``min_steps``), or at ``max_steps`` (restoring the best-loss state).
        """
        best_loss = math.inf
        best_state = None
        result: dict = {}
        tracker = _PlateauTracker(check_every, patience, rel_tol, min_iters=min_steps)

        with tqdm(desc=desc, unit="it", dynamic_ncols=True) as pbar:
            for counter in itertools.count():
                result = iter_fn(counter)

                # Convert JAX 0D scalar array to Python float for logging/tracking
                loss_val = float(result["loss"])

                if loss_val < best_loss:
                    best_loss = loss_val
                    best_state = nnx.state(net, trainable) if trainable is not None else nnx.state(net)

                self.value_counter += 1 if "Value" in desc else 0
                self.policy_counter += 1 if "Policy" in desc else 0

                converged = tracker.update(loss_val, counter + 1)
                timed_out = counter >= max_steps
                should_stop = converged or timed_out
                if counter % 50 == 0 or should_stop:
                    postfix = {k: f"{float(v):.6f}" for k, v in result.items() if v is not None}
                    postfix["conv"] = f"{tracker.stalled}/{patience}"
                    pbar.set_postfix(postfix)
                if should_stop:
                    if timed_out and best_state is not None:
                        nnx.update(net, best_state)  # Restore best state
                    break

                if counter % self.log_every == 0:
                    if self.summary_writer:
                        sub_task = "value" if "Value" in desc else "policy"
                        _step = self.value_counter if sub_task == "value" else self.policy_counter
                        
                        metrics = {
                            f"train/{sub_task}_loss": loss_val,
                            f"{sub_task}_iter": _step
                        }
                        if sub_task == "policy":
                            metrics["train/ctrl_acc"] = float(result["ctrl_acc"])
                            metrics["train/dist_acc"] = float(result["dist_acc"])
                        if sub_task == "value":
                            metrics["train/teacher_frac"] = float(result["teacher_frac"])
                            
                        self.summary_writer.log(metrics)
                    log.debug("Iter %d | loss=%.6f | converged=%s", counter, loss_val, converged)
                pbar.update(1)

        return result

    # ------------------------------------------------------------------
    # Per-curriculum-step training
    # ------------------------------------------------------------------
    
    def _train_policy_step(self, step: int) -> dict:
        dyn, cfg = self.dyn, self.cfg
        phase_cfg = self._phase_cfg(step)
        is_pretrain = self._is_pretrain_step(step)
        batch_size = cfg.TRAIN.BATCH_SIZE
        dt, T = cfg.GAME.TIME.DT, cfg.GAME.TIME.T
        snap_time_to_dt = cfg.TRAIN.SNAP_TIME_TO_DT

        policy_net_w = self._get_windowed_net("policy", self.current_window, self.current_window)
        self._ensure_policy_optimizer(policy_net_w)  # (re)build only when the window changes
        update_step = self._get_policy_update_step()

        probe_kind = "value_target" if self.use_ema_target else "value"
        pwr = self._probe_window_range()
        if pwr is not None:
            value_net_probe = self._get_windowed_net(probe_kind, pwr[0], pwr[1])
        else:
            value_net_probe = self.value_target_net if self.use_ema_target else self.value_net

        # 2. Data generation loop
        def iter_fn(counter: int) -> dict:
            t_min = self.current_window * self._window_time(self.policy_net) + dt if self._window_time(self.policy_net) is not None else 0.0
            self.key, sample_key = jax.random.split(self.key)

            x, t = self.sampler.sample(
                key=sample_key, batch_size=batch_size, T=T, dt=dt, step=step, t_min=t_min, snap_time_to_dt=snap_time_to_dt
            )
            # Evaluate weights and teacher targets
            if cfg.TRAIN.BOUNDARY_FOCAL_LOSS:
                boundary_w = self._get_boundary_weights_jit()(value_net_probe, x, t)
            else:
                boundary_w = jnp.ones(batch_size, dtype=dyn.dtype)

            u_t, d_t, u_gap, d_gap = self._get_teacher_conf_jit()(value_net_probe, x, t)
            u_conf = u_gap * boundary_w[:, None]
            d_conf = d_gap * boundary_w[:, None] if (d_t is not None and d_t.size > 0) else None

            loss, ctrl_acc, dist_acc, sat, frozen = update_step(
                policy_net_w, self.policy_opt, x, t, u_t, d_t, u_conf, d_conf
            )

            return {
                "loss": loss,
                "ctrl_acc": ctrl_acc,
                "dist_acc": dist_acc,
                "sat": sat,
                "frozen": frozen,
            }

        return self._training_loop(
            "Pretraining Policy Net" if is_pretrain else "Policy Training",
            policy_net_w,
            phase_cfg.STEPS.MIN_POLICY,
            phase_cfg.STEPS.MAX_POLICY,
            phase_cfg.CONVERGE.CHECK_EVERY,
            phase_cfg.CONVERGE.PATIENCE,
            phase_cfg.CONVERGE.REL_TOL,
            iter_fn,
        )

    def _maybe_seed_boundary_buffer(self, step: int, value_net=None, force: bool = False) -> None:
        """Top the value-boundary ring up by projection when it is too thin.

        Self-limiting: skips entirely once ordinary training batches keep the
        ring full, so the cost is confined to cold starts and window changes.

        Times are spread across the ACTIVE WINDOW rather than pinned to one
        value. The zero level set moves with t, so seeding at a single t both
        under-covers the window and risks projecting against a slice where V
        does not cross zero at all -- which yields nothing and looks like the
        seed silently failing.
        """
        cfg = self.cfg
        if not cfg.SAMPLING.SEED_BOUNDARY_BUFFER or self.sampler.w["value_boundary"] <= 0.0:
            return
        need = cfg.SAMPLING.SEED_MIN_RATIO * cfg.TRAIN.BATCH_SIZE
        if not force and self.sampler.buffer_occupancy >= need:
            return

        # SEED_SIZE <= 0 => auto: 8 * TRAIN.BATCH_SIZE (keeps the seed projection's
        # grad-through-obs-encoder memory tied to the training batch the user already
        # sized to fit; matters for image observations like F1TenthBEV).
        n = int(cfg.SAMPLING.SEED_SIZE) if cfg.SAMPLING.SEED_SIZE > 0 else 8 * int(cfg.TRAIN.BATCH_SIZE)
        dt, T = cfg.GAME.TIME.DT, cfg.GAME.TIME.T
        wt = self._window_time(self.value_net)
        if wt is None:
            t_lo, t_hi = dt, max(dt, min(T, step * dt))
        else:
            t_lo = max(dt, self.current_window * wt + dt)
            t_hi = max(t_lo, min(T, (self.current_window + 1) * wt))
        if value_net is None:
            value_net = self._get_windowed_net("value", self.current_window, self.current_window)

        self.key, kt, ks = jax.random.split(self.key, 3)
        t_vec = jax.random.uniform(kt, (n,), dtype=self.dyn.dtype, minval=t_lo, maxval=t_hi)
        n_new = self.sampler.seed_boundary_buffer(ks, value_net, t_vec, n)
        log.debug("seeded value-boundary buffer over t in [%.3f, %.3f]: +%d/%d valid "
                 "(occupancy %d)", t_lo, t_hi, n_new, n, int(self.sampler.buffer_occupancy))

    def _log_boundary_occupancy(self) -> None:
        """Report boundary-bucket supply against the batch size.

        The RATIO is the thing to watch, for both the value-boundary ring
        buffer and the static target/avoid pools: below ~10x the batch, draws
        collide and a "boundary" batch is mostly repeats of a few states.
        Raise SAMPLING.BUFFER_SIZE / SAMPLING.POOL_SIZE respectively -- see the
        sizing note in the config for why BUFFER_SIZE has to account for the
        in-band acceptance rate.
        """
        B = max(1, self.cfg.TRAIN.BATCH_SIZE)
        sup = []
        if self.sampler.w["value_boundary"] > 0.0:
            sup.append(("value buffer", int(self.sampler.buffer_occupancy), self.sampler.buffer_size))
        for label, pool, w in (("target pool", self.sampler._target_pool, self.sampler.w["target"]),
                               ("avoid pool", self.sampler._avoid_pool, self.sampler.w["avoid"])):
            if w > 0.0:
                sup.append((label, 0 if pool is None else int(pool.shape[0]), self.sampler.pool_size))
        for label, have, cap in sup:
            ratio = have / B
            (log.warning if ratio < 10.0 else log.debug)(
                "%s: %d / %d states (%.1fx batch)%s", label, have, cap, ratio,
                "  <-- below 10x, boundary batches will contain duplicates" if ratio < 10.0 else "")

    def _train_value_step(self, step: int) -> dict:
        dyn, cfg = self.dyn, self.cfg
        phase_cfg = self._phase_cfg(step)
        is_pretrain = self._is_pretrain_step(step)
        batch_size = cfg.TRAIN.BATCH_SIZE
        dt, T = cfg.GAME.TIME.DT, cfg.GAME.TIME.T
        snap_time_to_dt = cfg.TRAIN.SNAP_TIME_TO_DT

        value_net_w = self._get_windowed_net("value", self.current_window, self.current_window)
        self._ensure_value_optimizer(value_net_w)  # (re)build only when the window changes
        self._maybe_seed_boundary_buffer(step, value_net_w)
        update_step = self._get_value_update_step()

        value_target_kind = "value_target" if self.use_ema_target else "value"
        target_wr = self._target_window_range()
        if target_wr is not None:
            value_net_targets = self._get_windowed_net(value_target_kind, target_wr[0], target_wr[1])
            policy_net_targets = self._get_windowed_net("policy", target_wr[0], target_wr[1])
        else:
            value_net_targets = self.value_target_net if self.use_ema_target else self.value_net
            policy_net_targets = self.policy_net

        if self.use_ema_target:
            target_net_w = self._get_windowed_net("value_target", self.current_window, self.current_window)
            ema_update = self._get_ema_update_jit()

        # 2. Data generation loop
        def iter_fn(counter: int) -> dict:
            t_min = self.current_window * self._window_time(self.value_net) + dt if self._window_time(self.value_net) is not None else 0.0
            self.key, sample_key, rollout_key, targets_key = jax.random.split(self.key, 4)

            x, t = self.sampler.sample(
                key=sample_key, batch_size=batch_size, T=T, dt=dt, step=step, t_min=t_min, snap_time_to_dt=snap_time_to_dt
            )

            n_bc = int(cfg.TRAIN.BC_PROB * batch_size)
            if n_bc > 0:
                t = t.at[:n_bc].set(t_min if t_min > 0 else float(dt))  # JAX array in-place update syntax

            n_steps = make_rollout_length_schedule(rollout_key, batch_size, self.lam, self.M)
            teacher_frac = self._compute_teacher_frac(step)

            y_td = self.solver.compute_targets(targets_key, x, t, n_steps, teacher_frac=teacher_frac,
                                               window_range=target_wr, v_net=value_net_targets, p_net=policy_net_targets)

            # Execute the compiled JIT step
            loss = update_step(
                value_net_w, self.value_opt, x, t, y_td
            )

            if self.use_ema_target:
                ema_update(value_net_w, target_net_w)

            # Update boundary buffer (kept outside JIT if buffer mutates host-side size/state)
            self.sampler.update_boundary_buffer(x, t, value_net=value_net_w)

            return {
                "loss": loss,
                "teacher_frac": teacher_frac,
            }

        return self._training_loop(
            "Pretraining Value Net" if is_pretrain else f"Value Training (step {step})",
            value_net_w,
            phase_cfg.STEPS.MIN_VALUE,
            phase_cfg.STEPS.MAX_VALUE,
            phase_cfg.CONVERGE.CHECK_EVERY,
            phase_cfg.CONVERGE.PATIENCE,
            phase_cfg.CONVERGE.REL_TOL,
            iter_fn,
        )

    # ------------------------------------------------------------------
    # Window management
    # ------------------------------------------------------------------

    def _finalize_window(self, step: int) -> None:
        """Finalize current window at boundary: build anchor dataset, finetune, compute inflation, advance window.
        
        This method encapsulates the full window lifecycle: boundary detection, anchor supervision,
        finetuning, FP/FPR evaluation, inflation threshold computation, and window advancement.
        """
        self._log_boundary_occupancy()
        window_time = self._window_time(self.value_net)
        if window_time is None:
            return

        dyn, cfg = self.dyn, self.cfg
        dt = float(cfg.GAME.TIME.DT)
        t_now = step * dt

        # Compute terminal time for this window (end of current window phase).
        terminal_time = self.current_window * window_time
        next_boundary = (self.current_window + 1) * window_time
        max_window = self._window_count() - 1

        # Check boundary conditions.
        if t_now + 1e-12 < next_boundary:
            return

        log.debug(
            "Window finalization at step %d (window %d, t=%.3f)",
            step, self.current_window, t_now,
        )

        self.key, anchor_build_key = jax.random.split(self.key)
        value_net_scoped = self._get_persistent_window_net("value", self.current_window)
        policy_net_scoped = self._get_persistent_window_net("policy", self.current_window)

        # Full-horizon MC anchors (default) roll the student to t=0 and bootstrap
        # only the exact boundary, so policy/value must span windows [0, current]
        # (the rollout crosses earlier windows). The legacy one-window target uses
        # just the current window's net.
        full_mc = bool(cfg.FINETUNE.FULL_HORIZON_MC)
        if full_mc:
            anchor_policy = self._get_windowed_net("policy", 0, self.current_window)
            anchor_value  = self._get_windowed_net("value", 0, self.current_window)
        else:
            anchor_policy, anchor_value = policy_net_scoped, value_net_scoped

        # Build anchor dataset for this window.
        num_batches = cfg.FINETUNE.ANCHOR_BATCHES
        anchor_t0 = float(terminal_time) + float(window_time)
        anchor_x0 = self._sample_anchor_states(num_batches, anchor_t0, step)
        _t_anchor = time.time()
        anchor_ds, _ = build_anchor_dataset_window(
            key=anchor_build_key,
            num_data_batches=num_batches,
            dyn=dyn,
            policy=anchor_policy,
            value_fn=anchor_value,
            cfg=cfg,
            terminal_time=terminal_time,
            x0=anchor_x0,
            full_horizon=full_mc,
        )
        log.info("Window %d anchor build: %.1fs", self.current_window, time.time() - _t_anchor)

        # Finetune value network on anchor supervision + TD targets.
        _t_ft = time.time()
        self._finetune_window_last_layers(step, anchor_ds)
        log.info("Window %d finetune: %.1fs", self.current_window, time.time() - _t_ft)
        value_net_scoped = self._get_persistent_window_net("value", self.current_window)
        policy_net_scoped = self._get_persistent_window_net("policy", self.current_window)
        # Re-fetch spanning views so calibration reflects the just-finetuned window.
        if full_mc:
            calib_policy = self._get_windowed_net("policy", 0, self.current_window)
            calib_value  = self._get_windowed_net("value", 0, self.current_window)
        else:
            calib_policy, calib_value = policy_net_scoped, value_net_scoped

        self.key, calib_key = jax.random.split(self.key)
        _t_calib = time.time()
        _, calib_snapshots = build_anchor_dataset_window(
            key=calib_key,
            num_data_batches=cfg.FINETUNE.CALIB_BATCHES,
            dyn=dyn,
            policy=calib_policy,
            value_fn=calib_value,
            cfg=cfg,
            terminal_time=terminal_time,
            full_horizon=full_mc,
        )
        log.info("Window %d calibration build: %.1fs", self.current_window, time.time() - _t_calib)

        x_vals, y_vals, t_vals = calib_snapshots["x0"], calib_snapshots["y0"], calib_snapshots["t0"]

        chunk_size = cfg.TRAIN.BATCH_SIZE
        V_vals_pred = jnp.concatenate([
            value_net_scoped(x_vals[i:i + chunk_size], t_vals[i:i + chunk_size])["V"]
            for i in range(0, x_vals.shape[0], chunk_size)
        ], axis=0)

        control_role = cfg.GAME.CONTROL_ROLE
        cm = confusion_metrics(V_vals_pred, y_vals, control_role)
        FP, TN, FPR = int(cm["FP"]), int(cm["TN"]), cm["FPR"]
        # False-positive set (predicted safe, actually unsafe) sizes the inflation.
        mask = (~_unsafe_mask(V_vals_pred, control_role)) & _unsafe_mask(y_vals, control_role)
        log.info(
            "Window %d post-finetune (%d calib states): ACC=%.4f TPR=%.4f FPR=%.4f "
            "TNR=%.4f FNR=%.4f | TP=%d FP=%d TN=%d FN=%d",
            self.current_window, int(cm["N"]), cm["ACC"], cm["TPR"], FPR,
            cm["TNR"], cm["FNR"], int(cm["TP"]), FP, TN, int(cm["FN"]),
        )

        # Compute inflation threshold.
        fp_threshold = cfg.FINETUNE.FP_THRESHOLD
        if FPR < fp_threshold:
            inflation_threshold = 0.0
        else:
            quantile = 1.0 - fp_threshold / FPR if control_role == "max" else fp_threshold / FPR
            inflation_threshold = -float(np.quantile(
                np.asarray(V_vals_pred[mask]), quantile
            ))

        # Store inflation threshold in the value net module.
        target = Trainer._get_module(self.value_net)
        target.inflations.value = target.inflations.value.at[self.current_window].set(inflation_threshold)

        # Safe-set volume: fraction of uniformly-sampled states classified SAFE by
        # the (finetuned) value net, before vs after applying the inflation. The
        # sampler is dyn.sample when defined (e.g. F1TenthBEV draws poses only
        # near the track, so the ratio is over that near-track measure, not the
        # full x-y box); control_role='max' => safe is V>0 and the (<=0) inflation
        # shrinks the certified safe set.
        self.key, vol_key = jax.random.split(self.key)
        n_vol = 8 * int(cfg.TRAIN.BATCH_SIZE)
        x_vol = sample_uniform_states(vol_key, dyn, n_vol)
        t_vol = jnp.full((n_vol,), min(float(next_boundary), float(cfg.GAME.TIME.T)), dtype=dyn.dtype)
        cs = int(cfg.TRAIN.BATCH_SIZE)
        V_vol = jnp.concatenate([
            value_net_scoped(x_vol[i:i + cs], t_vol[i:i + cs])["V"]
            for i in range(0, n_vol, cs)
        ], axis=0)
        safe_before = ~_unsafe_mask(V_vol, control_role)
        safe_after = ~_unsafe_mask(V_vol + inflation_threshold, control_role)
        vol_before = float(jnp.mean(safe_before.astype(jnp.float32)))
        vol_after = float(jnp.mean(safe_after.astype(jnp.float32)))

        log.info(
            "Window %d finalized: inflation_threshold=%.4f (FP_THRESHOLD=%.4g) | "
            "safe-set volume (uniform sample): %.4f -> %.4f (post-inflation)",
            self.current_window, inflation_threshold, fp_threshold, vol_before, vol_after,
        )

        # Log window-specific metrics
        if self.summary_writer is not None:
            self.summary_writer.log({
                "train/ACC": cm["ACC"],
                "train/TPR": cm["TPR"],
                "train/FPR": FPR,
                "train/TNR": cm["TNR"],
                "train/FNR": cm["FNR"],
                "train/inflation_threshold": inflation_threshold,
                "train/safe_vol_before": vol_before,
                "train/safe_vol_after": vol_after,
                "curriculum_step": step,
            })

        # Advance to next window and warm-start.
        if self.current_window < max_window:
            self.sampler.clear_buffer()  # clear buffer to resample for new window
            self.current_window += 1
            self._maybe_seed_boundary_buffer(step, force=True)
            modules_to_warm_start = [self.value_net, self.policy_net]
            if self.use_ema_target:
                modules_to_warm_start.append(self.value_target_net)
            for module in modules_to_warm_start:
                window_module = Trainer._get_module(module)
                if hasattr(window_module, "warm_start_from_previous"):
                    window_module.warm_start_from_previous(self.current_window)
            self._clear_stale_window_jit_caches()
            log.debug("Advanced to window %d", self.current_window)

    def _finetune_window_last_layers(
        self,
        step: int,
        anchor_ds: AnchorDataset,
    ) -> dict:
        """Run window-end value finetuning with TD targets plus anchor supervision."""
        dyn, cfg = self.dyn, self.cfg

        batch_size = cfg.TRAIN.BATCH_SIZE
        fp_lambda = cfg.FINETUNE.FP_LAMBDA
        dt, T = cfg.GAME.TIME.DT, cfg.GAME.TIME.T
        snap_time_to_dt = cfg.TRAIN.SNAP_TIME_TO_DT

        log.debug(
            "Value finetune step %d | (max_steps=%d)",
            step,
            cfg.FINETUNE.MAX_STEPS,
        )
        
        scratch_net, scratch_core, update_step = self._get_finetune_scratch()
        value_net_core = Trainer._get_module(self.value_net)
        nnx.update(scratch_core, nnx.state(value_net_core.nets[self.current_window]))

        scratch_opt = nnx.Optimizer(scratch_net, self._tx_value, wrt=nnx.Param)
        nnx.update(scratch_opt, nnx.state(self.value_opt))
        set_optimizer_lr(scratch_opt, self.finetune_lr)

        value_target_kind = "value_target" if self.use_ema_target else "value"
        target_wr = self._target_window_range()
        if target_wr is not None:
            value_net_targets = self._get_windowed_net(value_target_kind, target_wr[0], target_wr[1])
            policy_net_targets = self._get_windowed_net("policy", target_wr[0], target_wr[1])
        else:
            value_net_targets = self.value_target_net if self.use_ema_target else self.value_net
            policy_net_targets = self.policy_net

        if self.use_ema_target:
            target_net_w = self._get_windowed_net("value_target", self.current_window, self.current_window)
            ema_update = self._get_ema_update_jit()

        try:
            def iter_fn(counter: int) -> dict:
                t_min = self.current_window * self._window_time(self.value_net) + dt if self._window_time(self.value_net) is not None else 0.0
                self.key, sample_key, rollout_key, anchor_key, targets_key = jax.random.split(self.key, 5)

                x, t = self.sampler.sample(
                    key=sample_key,
                    batch_size=batch_size,
                    T=T,
                    dt=dt,
                    step=step,
                    t_min=t_min,
                    snap_time_to_dt=snap_time_to_dt,
                )

                # BC samples pin t to the window bottom (last_end+dt), where the TD
                # target bootstraps on the PREVIOUS window's value -- the one place
                # the finetune bootstrap conflicts with the full-horizon MC anchor.
                # Drop them during finetune (keep the spatial buckets, which sample t
                # across the window and don't cause that misalignment).
                n_bc = 0 if cfg.FINETUNE.DROP_BC_SAMPLES else int(cfg.TRAIN.BC_PROB * batch_size)
                if n_bc > 0:
                    t = t.at[:n_bc].set(t_min if t_min > 0 else float(dt))  # JAX array in-place update syntax

                n_steps = make_rollout_length_schedule(rollout_key, batch_size, self.lam, self.M)
                
                x_a, y_a, t_a = sample_anchor_minibatch(anchor_key, anchor_ds, batch_size=batch_size)
                y_td = self.solver.compute_targets(targets_key, x, t, n_steps, teacher_frac=0.0,
                                                    window_range=target_wr, v_net=value_net_targets, p_net=policy_net_targets)

                loss, loss_td, loss_sup = update_step(
                    scratch_net, scratch_opt, x, t, x_a, y_a, t_a, y_td
                )

                if self.use_ema_target:
                    ema_update(scratch_net, target_net_w)

                return {
                    "loss": loss,
                    "loss_td": loss_td,
                    "loss_sup": loss_sup,
                }

            return self._training_loop(
                "Value Finetuning",
                scratch_net,
                cfg.FINETUNE.MIN_STEPS,
                cfg.FINETUNE.MAX_STEPS,
                cfg.FINETUNE.CONVERGE.CHECK_EVERY,
                cfg.FINETUNE.CONVERGE.PATIENCE,
                cfg.FINETUNE.CONVERGE.REL_TOL,
                iter_fn,
            )
        finally:
            nnx.update(value_net_core.nets[self.current_window], nnx.state(scratch_core))

    # ---------------------------------------------------------------------------
    # Networks
    # ---------------------------------------------------------------------------

    def _build_value_net(self, rngs, cfg, dyn) -> nnx.Module:
        return build_net(
            arch=cfg.NET.VALUE.ARCH,
            cfg_kwargs=dict(cfg.NET.VALUE.KWARGS),
            rngs=rngs,
            context_params={
                "window_time":   cfg.GAME.TIME.WINDOW_TIME,
                "T_max":         cfg.GAME.TIME.T,
                "phi_dim":       dyn.input_dim,
                "obs_dim":       dyn.obs_dim,
                "obs_ch_dim":    dyn.obs_ch_dim,
                "obs_embed_dim": dyn.obs_embed_dim,
                "obs_kind":      getattr(dyn, "obs_kind", "seq"),
            },
            wrapper_fn=lambda net: BoundaryAwareNet(net, dyn, cfg.GAME.PROBLEM_TYPE, cfg.GAME.EXACT_BC),
            label="Value net",
        )

    def _build_policy_net(self, rngs, cfg, dyn) -> nnx.Module:
        return build_net(
            arch=cfg.NET.POLICY.ARCH,
            cfg_kwargs=dict(cfg.NET.POLICY.KWARGS),
            rngs=rngs,
            context_params={
                "window_time":   cfg.GAME.TIME.WINDOW_TIME,
                "T_max":         cfg.GAME.TIME.T,
                "phi_dim":       dyn.input_dim,
                "control_dim":   dyn.control_dim,
                "disturb_dim":   dyn.disturb_dim,
                "obs_dim":       dyn.obs_dim,
                "obs_ch_dim":    dyn.obs_ch_dim,
                "obs_embed_dim": dyn.obs_embed_dim,
                "obs_kind":      getattr(dyn, "obs_kind", "seq"),
                # Ternary VQ (neutral codeword) is implied by triple-sided probing.
                "ternary":       cfg.GAME.TRIPLE_SIDED_PROBING,
            },
            label="Policy net",
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _ensure_value_optimizer(self, value_net_w) -> None:
        if self._value_opt_window != self.current_window:
            self.value_opt = nnx.Optimizer(value_net_w, self._tx_value, wrt=nnx.Param)
            self._value_opt_window = self.current_window

    def _ensure_policy_optimizer(self, policy_net_w) -> None:
        """Scope the policy optimizer to the current window's net view (see above)."""
        if self._policy_opt_window != self.current_window:
            self.policy_opt = nnx.Optimizer(policy_net_w, self._tx_policy, wrt=nnx.Param)
            self._policy_opt_window = self.current_window

    # ------------------------------------------------------------------
    # Jitted value-net evaluations
    # ------------------------------------------------------------------

    def _sample_anchor_states(self, num_batches: int, t0: float, step: int) -> jax.Array:
        """Anchor start states from the training sampler, all pinned to ``t0``.

        ``_sample_t`` draws in ``[t_min, clip(step*dt, t_min, T)]``, so passing
        ``T == t_min == t0`` collapses it to the single time the anchor rollout
        actually starts from. Batches are drawn one at a time at TRAIN.BATCH_SIZE
        so they reuse the sampler kernel already compiled for training rather
        than triggering a fresh trace at ANCHOR_BATCHES*BATCH_SIZE.
        """
        cfg = self.cfg
        xs = []
        for _ in range(num_batches):
            self.key, k = jax.random.split(self.key)
            x, _ = self.sampler.sample(
                key=k, batch_size=cfg.TRAIN.BATCH_SIZE, T=t0, dt=cfg.GAME.TIME.DT,
                step=step, t_min=t0, snap_time_to_dt=cfg.TRAIN.SNAP_TIME_TO_DT,
            )
            xs.append(x)
        return jnp.concatenate(xs, axis=0)

    def _probe_window_range(self):
        """Static [w_lo, w_hi] window range touched by a single-step lookback
        probe (t and t-dt, per _teacher_conf_jit/_boundary_weights_jit). Same
        soundness argument as _target_window_range, just with horizon=1 step
        instead of an M-step rollout. None => full range (non-windowed nets)."""
        wt = self._window_time(self.value_net)
        if wt is None:
            return None
        n_win = int(Trainer._get_module(self.value_net).num_windows)
        dt = self.cfg.GAME.TIME.DT
        horizon = math.ceil(dt / wt)
        w_hi = min(n_win - 1, self.current_window)
        w_lo = max(0, self.current_window - horizon)
        return (w_lo, w_hi)

    def _build_teacher_conf_jit(self, window_range: tuple):
        """Jitted probing-teacher eval. The value net is passed as an explicit
        jit arg and used to build V_next_fn, so the CURRENT params are traced
        each call (a closure-captured net would be frozen at trace time)."""
        from reachability.training.functional import (
            infer_u_d_double_sided, probe_horizon, substeps_for,
        )
        cfg, dyn = self.cfg, self.dyn
        h, h_substeps = probe_horizon(cfg), substeps_for(probe_horizon(cfg), cfg)

        @nnx.jit
        def teacher_conf(v_net, x, t):
            t_next = jnp.clip(t - h, min=0.0)

            def v_next_fn(x_query):
                qB, refB = x_query.shape[0], t_next.shape[0]
                t_exp = t_next if qB == refB else jnp.repeat(t_next, qB // refB, axis=0)
                return v_net(x_query, t_exp[:qB], window_range=window_range)["V"]

            return infer_u_d_double_sided(
                x=x, V_next_fn=v_next_fn, dyn=dyn, dt=h,
                control_role=cfg.GAME.CONTROL_ROLE, disturb_role=cfg.GAME.DISTURB_ROLE,
                problem_type=cfg.GAME.PROBLEM_TYPE, num_substeps=h_substeps,
                return_conf=True,
            )

        return teacher_conf

    def _build_boundary_weights_jit(self, window_range: tuple):
        cfg = self.cfg

        @nnx.jit
        def boundary_weights(v_net, x, t):
            alpha = cfg.TRAIN.BOUNDARY_FOCUS_ALPHA
            tau = cfg.TRAIN.BOUNDARY_FOCUS_TAU
            V = v_net(x, t, window_range=window_range)["V"]
            w = 1.0 + alpha / (1.0 + jnp.exp(jnp.abs(V) / tau))
            return w / jnp.clip(jnp.mean(w), min=1e-6)

        return boundary_weights

    def _get_teacher_conf_jit(self):
        """Cached per window_range: compiles once/window, like the update steps."""
        wr = self._probe_window_range()
        key = ("teacher_conf", wr)
        fn = self._probe_jit_cache.get(key)
        if fn is None:
            fn = self._build_teacher_conf_jit(wr)
            self._probe_jit_cache[key] = fn
        return fn

    def _get_boundary_weights_jit(self):
        wr = self._probe_window_range()
        key = ("boundary_weights", wr)
        fn = self._probe_jit_cache.get(key)
        if fn is None:
            fn = self._build_boundary_weights_jit(wr)
            self._probe_jit_cache[key] = fn
        return fn

    def _get_value_update_step(self):
        key = ("value", self.current_window)
        if key in self._update_step_cache:
            return self._update_step_cache[key]
        cfg, dt = self.cfg, self.cfg.GAME.TIME.DT

        @nnx.jit
        def update_step(value_net, optimizer, x, t, y_td):
            diff = nnx.DiffState(0, nnx.Param)

            def loss_fn(model):
                out = model(x, t)
                if cfg.GAME.EXACT_BC:
                    loss_elements = (out["V"] - y_td) ** 2
                    t_weights = 1.0 / jnp.clip(jnp.reshape(t, -1), min=dt) ** 2
                    return jnp.mean(loss_elements * t_weights)
                return jnp.mean((out["V"] - y_td) ** 2)

            loss, grads = nnx.value_and_grad(loss_fn, argnums=diff)(value_net)
            optimizer.update(value_net, grads)
            return loss

        self._update_step_cache[key] = update_step
        return update_step

    def _get_ema_update_jit(self):
        """Polyak-update value_target_net towards value_net's current Params:
        target <- tau*live + (1-tau)*target. Both nets are passed in already
        scoped to the current window (see _get_windowed_net), so split/merge
        cost is bounded to that window's params, not the full multi-window net."""
        key = ("ema_value", self.current_window)
        if key in self._update_step_cache:
            return self._update_step_cache[key]
        tau = self.ema_tau

        @nnx.jit
        def ema_update(live_net, target_net):
            live_state = nnx.state(live_net, nnx.Param)
            target_state = nnx.state(target_net, nnx.Param)
            new_target_state = jax.tree_util.tree_map(
                lambda l, t: tau * l + (1.0 - tau) * t, live_state, target_state)
            nnx.update(target_net, new_target_state)

        self._update_step_cache[key] = ema_update
        return ema_update

    def _get_policy_update_step(self):
        key = ("policy", self.current_window)
        if key in self._update_step_cache:
            return self._update_step_cache[key]
        dyn = self.dyn

        @nnx.jit
        def update_step(policy_net, optimizer, x, t, u_t, d_t, u_conf, d_conf):
            diff = nnx.DiffState(0, nnx.Param)

            def loss_fn(model):
                out = model(dyn.nn_inputs(x), t)
                return vq_policy_loss(out, u_t, d_t, u_conf, d_conf), out

            (loss, out), grads = nnx.value_and_grad(loss_fn, argnums=diff, has_aux=True)(policy_net)
            optimizer.update(policy_net, grads)
            u_pred = out.get("u_pred")
            d_pred = out.get("d_pred") if dyn.disturb_dim > 0 else None
            def _acc(pred, target, conf):
                if pred is None:
                    return jnp.array(jnp.inf)
                correct = ((pred >= 0) == (target >= 0)).astype(pred.dtype)
                if conf is None:
                    return jnp.mean(correct)
                w = jnp.clip(jnp.broadcast_to(conf, correct.shape), min=0)
                return jnp.sum(correct * w) / jnp.clip(jnp.sum(w), min=1e-6)

            ctrl_acc = _acc(u_pred, u_t, u_conf)
            dist_acc = _acc(d_pred, d_t, d_conf)

            z_raw_u = out.get("z_raw_u")
            if z_raw_u is not None:
                past = jnp.abs(z_raw_u) >= 1.0
                sat = jnp.mean(past)
                frozen = jnp.mean(past & (jnp.sign(z_raw_u) != jnp.sign(u_t)))
            else:
                sat = jnp.array(0.0)
                frozen = jnp.array(0.0)
            return loss, ctrl_acc, dist_acc, sat, frozen

        self._update_step_cache[key] = update_step
        return update_step

    def _compute_teacher_frac(self, step: int) -> float:
        frac_init  = self.cfg.TRAIN.TEACHER_FRAC_INIT
        frac_final = self.cfg.TRAIN.TEACHER_FRAC_FINAL
        progress   = (step - 1) / max(1, self.total_steps - 1)
        return frac_init + (frac_final - frac_init) * progress

    def _compute_sampling_weight_updates(self, step: int) -> dict:
        """Curriculum kwargs for self.sampler.set_weights() -- ALWAYS
        resupplies all five raw weights, every call.

        set_weights() falls back to self.w[key] (already-normalized) for any
        key not passed in. That's fine for a one-off manual tweak, but fatal
        for a per-step scheduler: mixing a freshly-scheduled raw value (e.g.
        w_nominal) with stale already-normalized values for the rest creates
        compounding drift call over call, converging to a different (wrong)
        answer than a clean one-shot normalization.
        """
        s = self.cfg.SAMPLING
        warmup_t = s.WARMUP_T if s.WARMUP_T > 0 else self.cfg.GAME.TIME.T
        progress = min(1.0, (step * self.cfg.GAME.TIME.DT) / max(warmup_t, 1e-8))

        return BoundaryAwareSampler.curriculum_weights(
            progress,
            w_uniform=(s.FRAC_UNIFORM, s.FRAC_UNIFORM_END),
            w_target=s.FRAC_TARGET,
            w_avoid=s.FRAC_AVOID,
            w_user_target=s.FRAC_USER_TARGET,
            w_value_boundary=(s.FRAC_BOUNDARY, s.FRAC_BOUNDARY_END),
            w_safe=s.FRAC_SAFE,
            w_unsafe=s.FRAC_UNSAFE,
            w_nominal=(s.FRAC_NOMINAL, s.FRAC_NOMINAL_END),
        )

    def _target_window_range(self):
        """Static [w_lo, w_hi] a target rollout can touch this curriculum window:
        samples are in window `current_window`; the rollout rolls time back by up to
        M*dt -> ceil(M*dt/window_time) earlier windows. None => full range (non-windowed
        nets). Changes only at window boundaries, so the engine jit compiles once/window."""
        wt = self._window_time(self.value_net)
        if wt is None:
            return None
        n_win = int(Trainer._get_module(self.value_net).num_windows)
        dt = self.cfg.GAME.TIME.DT
        M  = self.cfg.TRAIN.M
        horizon = math.ceil((M * dt) / wt)
        w_hi = min(n_win - 1, self.current_window)
        w_lo = max(0, self.current_window - horizon)
        return (w_lo, w_hi)

    @staticmethod
    def _get_module(module: nnx.Module) -> nnx.Module:
        """Return the trainable core module, handling optional BoundaryAware wrapping."""
        current = module
        while isinstance(current, BoundaryAwareNet):
            current = current.net
        return current

    def _window_count(self) -> int:
        """Return the available number of windows across active windowed modules."""
        counts = []
        for module in (self.value_net, self.policy_net):
            window_module = Trainer._get_module(module)
            if hasattr(window_module, "num_windows"):
                counts.append(int(window_module.num_windows))
        return min(counts) if counts else 1

    def _window_time(self, module) -> Optional[float]:
        """Return configured window duration if any windowed module is active."""
        window_module = Trainer._get_module(module)
        if hasattr(window_module, "window_time"):
            return float(window_module.window_time)
        return None

    def _get_windowed_net(self, kind: str, w_lo: int, w_hi: int):
        """Cheap ``nets[w_lo:w_hi+1]`` view of value_net/policy_net/
        value_target_net, cached per (kind, w_lo, w_hi). Passing the full
        multi-window container into nnx.jit costs split/merge time
        proportional to *every* window's params; this scopes hot call sites
        (update steps, TD-target rollouts, teacher/boundary probes) down to
        just the windows they can reach. Non-windowed nets (no ``.nets``) are
        returned unchanged."""
        key = (kind, w_lo, w_hi)
        cached = self._window_net_cache.get(key)
        if cached is not None:
            return cached

        full_net = {
            "value": self.value_net,
            "policy": self.policy_net,
            "value_target": self.value_target_net,
        }[kind]
        core = Trainer._get_module(full_net)
        if not hasattr(core, "nets"):
            self._window_net_cache[key] = full_net
            return full_net

        window_time = self._window_time(full_net)
        windowed = WindowSlice(list(core.nets[w_lo:w_hi + 1]), w_lo, window_time, core.num_windows)
        if isinstance(full_net, BoundaryAwareNet):
            windowed = BoundaryAwareNet(windowed, full_net.dyn, full_net.problem_type, full_net.exact_bc)

        self._window_net_cache[key] = windowed
        return windowed

    def _clear_stale_window_jit_caches(self) -> None:
        """Drop every previously-compiled per-window jit variant.
        """
        self._update_step_cache.clear()
        self._probe_jit_cache.clear()
        self._window_net_cache.clear()
        self.solver._targets_jit_cache.clear()
        gc.collect()

    def _get_persistent_window_net(self, kind: str, w: int):
        full_net = {"value": self.value_net, "policy": self.policy_net}[kind]
        core = Trainer._get_module(full_net)
        if not hasattr(core, "nets"):
            return full_net

        cached = self._persistent_window_net_cache.get(kind)
        if cached is None:
            scratch_core = nnx.clone(core.nets[w])
            window_time = self._window_time(full_net)
            windowed = WindowSlice([scratch_core], w, window_time, core.num_windows)
            scratch = (
                BoundaryAwareNet(windowed, full_net.dyn, full_net.problem_type, full_net.exact_bc)
                if isinstance(full_net, BoundaryAwareNet) else windowed
            )
            self._persistent_window_net_cache[kind] = (scratch, scratch_core)
            return scratch

        scratch, scratch_core = cached
        nnx.update(scratch_core, nnx.state(core.nets[w]))
        return scratch

    def _get_finetune_scratch(self):
        if self._finetune_scratch is not None:
            return self._finetune_scratch

        cfg, dt = self.cfg, self.cfg.GAME.TIME.DT
        fp_lambda = cfg.FINETUNE.FP_LAMBDA

        core = Trainer._get_module(self.value_net)
        scratch_core = nnx.clone(core.nets[0])
        window_time = self._window_time(self.value_net)
        windowed = WindowSlice([scratch_core], 0, window_time, core.num_windows)
        scratch_net = BoundaryAwareNet(windowed, self.dyn, cfg.GAME.PROBLEM_TYPE, cfg.GAME.EXACT_BC)

        @nnx.jit
        def update_step(value_net, optimizer, x, t, x_a, y_a, t_a, y_td):
            diff = nnx.DiffState(0, nnx.Param)

            def loss_fn(model):
                out = model(x, t)
                if cfg.GAME.EXACT_BC:
                    loss_elements = (out["V"] - y_td) ** 2
                    t_weights = 1.0 / jnp.clip(jnp.reshape(t, -1), min=dt) ** 2
                    loss_td = jnp.mean(loss_elements * t_weights)
                else:
                    loss_td = jnp.mean((out["V"] - y_td) ** 2)

                pred_a = model(x_a, t_a)["V"]

                if cfg.GAME.EXACT_BC:
                    loss_sup_elements = fp_weighted_anchor_loss(pred_a, y_a, cfg.GAME.CONTROL_ROLE, fp_lambda, reduction='none')
                    t_weights = 1.0 / jnp.clip(jnp.reshape(t_a, -1), min=dt) ** 2
                    loss_sup = jnp.mean(loss_sup_elements * t_weights)
                else:
                    loss_sup = fp_weighted_anchor_loss(pred_a, y_a, cfg.GAME.CONTROL_ROLE, fp_lambda)

                loss = loss_td + cfg.FINETUNE.SUPERVISION_LAMBDA * loss_sup
                return loss, (loss_td, loss_sup)

            (loss, (loss_td, loss_sup)), grads = nnx.value_and_grad(loss_fn, argnums=diff, has_aux=True)(value_net)
            optimizer.update(value_net, grads)
            return loss, loss_td, loss_sup

        self._finetune_scratch = (scratch_net, scratch_core, update_step)
        return self._finetune_scratch

    def _is_pretrain_step(self, step: int) -> bool:
        """Return True only for the dedicated pretraining curriculum step."""
        return step == 0 and self.cfg.PRETRAIN.ENABLED

    def _phase_cfg(self, step: int):
        """Return the active phase config (PRETRAIN for step-1, else TRAIN)."""
        return self.cfg.PRETRAIN if self._is_pretrain_step(step) else self.cfg.TRAIN

    def _boundary_focus_weights(self, value_net: nnx.Module, x: jax.Array, t: jax.Array) -> jax.Array:
        """Return per-sample loss weights that emphasize states near the boundary.

        The weight is smooth and bounded below by 1:
            w = 1 + alpha / (1 + exp(|margin| / tau))
        where margin is the signed boundary value for the current problem type.
        """
        dyn, cfg = self.dyn, self.cfg
        alpha = cfg.TRAIN.BOUNDARY_FOCUS_ALPHA  # Maximum additional weight for boundary states
        tau = cfg.TRAIN.BOUNDARY_FOCUS_TAU   # Controls the steepness of the transition

        V = value_net(x, t)["V"]

        w = 1.0 + alpha / (1.0 + jnp.exp(jnp.abs(V) / tau))
        return w / jnp.clip(jnp.mean(w), min=1e-6)  # Normalize to mean weight of 1

    def _step_diagnostics(
        self,
        step: int,
        loss_v: float,
        loss_pi: float,
        u_acc: float,
        d_acc: float,
        teacher_frac: float = 0.0,
    ) -> None:
        """Compute and log per-step diagnostics. Called once per curriculum step."""
        dyn, cfg = self.dyn, self.cfg

        dt, T = cfg.GAME.TIME.DT, cfg.GAME.TIME.T
        t_now = min(step * dt, T)
        t_query = jnp.full((self._val_x.shape[0],), t_now, dtype=dyn.dtype)

        window_time = self._window_time(self.value_net)
        if window_time is not None:
            n_win = int(Trainer._get_module(self.value_net).num_windows)
            w = min(n_win - 1, max(0, math.floor((t_now - 1e-6) / window_time)))
            value_net_probe = self._get_windowed_net("value", w, w)
        else:
            value_net_probe = self.value_net
        V_pred = value_net_probe(self._val_x, t_query)["V"]
        frac_reachable = float(jnp.mean(V_pred < 0).astype(jnp.float32))

        # Accumulate history.
        self.history["step"].append(step)
        self.history["loss_v"].append(loss_v)
        self.history["loss_pi"].append(loss_pi)
        self.history["frac_reachable"].append(frac_reachable)
        self.history["ctrl_acc"].append(u_acc)
        self.history["dist_acc"].append(d_acc)
        self.history["teacher_frac"].append(teacher_frac)

    def _save_checkpoint(self, step: int) -> None:
        save_path = os.path.join(self.cfg.IO.LOG_DIR, self.cfg.IO.CKPT_DIRNAME, f"step_{step:03d}")
        checkpoint_data = {
            "value_net": nnx.state(self.value_net),
            "value_opt": nnx.state(self.value_opt),
            "policy_net": nnx.state(self.policy_net),
            "policy_opt": nnx.state(self.policy_opt),
            "step": step,
            "current_window": self.current_window,
        }
        if self.use_ema_target:
            checkpoint_data["value_target_net"] = nnx.state(self.value_target_net)

        self.checkpointer.save(os.path.abspath(save_path), checkpoint_data, force=True)
        log.debug("Saved JAX checkpoint to %s", save_path)

    def _save_history(self) -> None:
        history_path = os.path.join(self.cfg.IO.LOG_DIR, "history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        log.info("Saved training history → %s", history_path)

    def _load_checkpoint(self, resume_path: str, value_net: nnx.Module, policy_net: nnx.Module, value_target_net: Optional[nnx.Module] = None) -> int:
        """Restore a checkpoint written by _save_checkpoint.

        value_opt/policy_opt are rebuilt via _ensure_value_optimizer/
        _ensure_policy_optimizer *before* being restored into: they're scoped
        to whichever window was active at save time (see _save_checkpoint),
        so a freshly-constructed Trainer's full-net-scoped optimizers don't
        match the saved structure until current_window is set correctly and
        the optimizers are rebuilt against that window's net view.
        """
        if not os.path.isdir(resume_path):
            log.warning("Resume directory not found: %s. Starting from scratch.", resume_path)
            return 0

        try:
            abspath = os.path.abspath(resume_path)
            meta_item = {"step": 0, "current_window": 0}
            try:
                meta = self.checkpointer.restore(
                    abspath,
                    args=ocp.args.PyTreeRestore(
                        item=meta_item,
                        restore_args=ocp.checkpoint_utils.construct_restore_args(meta_item),
                        partial_restore=True,
                    ),
                )
                current_window = int(meta["current_window"])
            except Exception:
                current_window = 0

            self.current_window = current_window
            value_net_w = self._get_windowed_net("value", current_window, current_window)
            policy_net_w = self._get_windowed_net("policy", current_window, current_window)
            self._ensure_value_optimizer(value_net_w)
            self._ensure_policy_optimizer(policy_net_w)

            abstract_state = {
                "value_net": nnx.state(value_net),
                "value_opt": nnx.state(self.value_opt),
                "policy_net": nnx.state(policy_net),
                "policy_opt": nnx.state(self.policy_opt),
                "step": 0,
                "current_window": 0,
            }
            if value_target_net is not None:
                abstract_state["value_target_net"] = nnx.state(value_target_net)

            try:
                restored = self.checkpointer.restore(abspath, item=abstract_state)
            except Exception:
                if value_target_net is None:
                    raise
                # Older checkpoint predates the EMA target net
                abstract_state.pop("value_target_net")
                restored = self.checkpointer.restore(abspath, item=abstract_state)

            nnx.update(value_net, restored["value_net"])
            nnx.update(policy_net, restored["policy_net"])

            nnx.update(self.value_opt, restored["value_opt"])
            nnx.update(self.policy_opt, restored["policy_opt"])

            if value_target_net is not None:
                if "value_target_net" in restored:
                    nnx.update(value_target_net, restored["value_target_net"])
                else:
                    nnx.update(value_target_net, nnx.state(value_net))

            step = int(restored["step"])
            self.current_window = int(restored.get("current_window", current_window))
            log.info("Loaded JAX checkpoint step=%d window=%d from %s", step, self.current_window, resume_path)
            return step

        except Exception as exc:
            log.warning("Failed to load checkpoint in %s (%s). Starting from scratch.", resume_path, exc)
            return 0
    
    def _load_history(self) -> Dict[str, list]:
        """Load existing training history if resuming from a checkpoint."""
        history_path = os.path.join(self.cfg.IO.LOG_DIR, "history.json")
        if os.path.isfile(history_path):
            with open(history_path, "r") as f:
                history = json.load(f)
            log.info("Loaded existing training history from %s", history_path)
            return history
        else:
            log.info("No existing history found. Starting fresh.")
            return self._default_history()
        
    def _default_history(self) -> Dict[str, list]:
        """Return the default structure for training history."""
        return {
            "step": [], "loss_v": [], "loss_pi": [],
            "frac_reachable": [], "ctrl_acc": [], "dist_acc": [], "teacher_frac": [],
        }

    def _certified_level_fn(self):
        """Maps a plot row's ``t`` to the V-level of the CERTIFIED boundary.

        Certified-safe is ``V + inflation[w] > 0``, so the contour to draw is
        ``V = -inflation[w]``. Returns None when no window has been calibrated
        yet, so early plots just show the raw zero level set.
        """
        module = Trainer._get_module(self.value_net)
        infl = np.asarray(module.inflations[...])
        if not np.any(infl):
            return None
        wt = self._window_time(self.value_net)
        n_win = int(module.num_windows)

        def level_for_t(t: float):
            w = 0 if wt is None else int(np.clip(np.floor((t - 1e-6) / wt), 0, n_win - 1))
            return -float(infl[w])

        return level_for_t

    def _visualize(self, step: int) -> None:
        vis_dir = os.path.join(self.cfg.IO.LOG_DIR, self.cfg.VIS.DIRNAME)
        os.makedirs(vis_dir, exist_ok=True)
        delta_level = self._certified_level_fn()

        # Image-observation dynamics (F1TenthBEV) use the dedicated BEV visualiser:
        # it zooms to the active track (or an explicit VIS.X_LO/.. region) and
        # chunks the grid eval through the 2D conv. No Frenet frame for BEV.
        if getattr(self.dyn, "obs_kind", None) == "bev":
            out_path = os.path.join(vis_dir, f"step_{step:03d}.png")
            vis_val_fn_bev(
                net=self.value_net, dyn=self.dyn, out_path=out_path,
                x_res=self.cfg.VIS.X_RES, y_res=self.cfg.VIS.Y_RES,
                title_prefix=f"step={step} |", delta_level=delta_level,
                Tmax=self.cfg.GAME.TIME.T, problem_type=self.cfg.GAME.PROBLEM_TYPE,
                x_lo=self.cfg.VIS.X_LO, x_hi=self.cfg.VIS.X_HI,
                y_lo=self.cfg.VIS.Y_LO, y_hi=self.cfg.VIS.Y_HI,
            )
            if self.summary_writer is not None:
                self.summary_writer.log_image(
                    "visuals/value_function", out_path, step_dict={"curriculum_step": step})
            log.info("Saved visualization %s", out_path)
            return

        frames = [("world", f"step_{step:03d}.png", "visuals/value_function")]
        if hasattr(self.dyn, "lib") and hasattr(self.dyn.lib, "tracks"):
            frames.append(("frenet", f"step_se_{step:03d}.png",
                           "visuals/value_function_se"))

        for frame, fname, tag in frames:
            out_path = os.path.join(vis_dir, fname)
            vis_val_fn(
                net=self.value_net, dyn=self.dyn, out_path=out_path,
                x_res=self.cfg.VIS.X_RES, y_res=self.cfg.VIS.Y_RES,
                title_prefix=f"step={step} |",
                delta_level=delta_level,
                Tmax=self.cfg.GAME.TIME.T,
                problem_type=self.cfg.GAME.PROBLEM_TYPE,
                x_lo=self.cfg.VIS.X_LO,
                x_hi=self.cfg.VIS.X_HI,
                y_lo=self.cfg.VIS.Y_LO,
                y_hi=self.cfg.VIS.Y_HI,
                frame=frame,
            )
            if self.summary_writer is not None:
                self.summary_writer.log_image(
                    tag,
                    out_path,
                    step_dict={"curriculum_step": step}
                )
            log.info("Saved visualization %s", out_path)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def train(self) -> None:
        log.info("Starting curriculum training...")
        start_time = time.time()

        try:
            for step in range(self.start_step + 1, self.total_steps + 1):
                self.sampler.set_weights(**self._compute_sampling_weight_updates(step))
                pi = self._train_policy_step(step)
                val = self._train_value_step(step)

                # Window finalization (if at boundary): anchor + finetune + inflation + advance.
                self._finalize_window(step)

                self._step_diagnostics(
                    step=step,
                    loss_v=float(val["loss"]),
                    loss_pi=float(pi["loss"]),
                    u_acc=float(pi["ctrl_acc"]),
                    d_acc=float(pi["dist_acc"]),
                    teacher_frac=val["teacher_frac"],
                )

                if self.save_every > 0 and (step % self.save_every == 0 or step == self.total_steps):
                    self._save_checkpoint(step)

                if self.vis_every and (step % self.vis_every == 0):
                    self._visualize(step)
        finally:
            end_time = time.time()

            if self.summary_writer is not None:
                self.summary_writer.close()

            # Calculate elapsed time
            elapsed_time = end_time - start_time
            elapsed_hours = int(elapsed_time // 3600)
            elapsed_minutes = int((elapsed_time % 3600) // 60)
            elapsed_seconds = int(elapsed_time % 60)

            log.info(
                "Training completed in %d hours, %d minutes, and %d seconds.",
                elapsed_hours, elapsed_minutes, elapsed_seconds
            )

            self._save_history()

            plot_training_history(
                self.history,
                out_dir=self.cfg.IO.LOG_DIR, fname="training_curves.png",
            )