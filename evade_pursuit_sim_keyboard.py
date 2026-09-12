"""
evade_pursuit_sim_keyboard.py  —  Human vs AI pursuit-evasion (JAX, exact-time control).

Two live figures start from the same initial state:
  Figure 1 (left) : learned pursuer vs learned evader  (AI baseline)
  Figure 2 (right): learned pursuer vs HUMAN evader    (keyboard control)

Control is EXACT-TIME: each step queries the learned policy once at the exact
remaining time-to-go t = clip((K - step)*dt, dt, T) and applies its bang-bang
(pursuer u, and the evader d when not human-controlled). No adaptive-window
voting.

Controls — focus either window to capture keys:
  Left / A   turn left    Right / D  turn right
  Up / W     accelerate   Down / S   decelerate
  R          reset         Q / Escape quit

Usage:
    python evade_pursuit_sim_keyboard.py --run_dir <run_dir> [--brt_dir <brt_dir>]

Requires a GUI matplotlib backend (TkAgg, Qt5Agg, ...). If the window is blank:
    MPLBACKEND=TkAgg python evade_pursuit_sim_keyboard.py ...
"""

import argparse
import logging
import os
import time

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from utils import decode_actions, make_dynamics, setup_logging
from reachability.training.functional import rollout_with_intermediate_checks
from reachability.data.sampling import sample_uniform_states
from evade_pursuit_sim import (
    load_cfg,
    load_final_nets,
    load_brt_nets,
    clamp_state,
)

setup_logging()
log = logging.getLogger("keyboard_sim")

# Jit is REQUIRED for real time: eager JAX dispatches every op separately, so a
# single m-substep rollout is ~165 ms (measured) -> ~0.1x. Jitting the policy
# query, the rollout, and the l-evals drops each to <1 ms. Cached per dynamics.
_SIM_JIT_CACHE = {}


def _get_sim_jit(dyn, dt, m):
    key = (id(dyn), float(dt), int(m))
    fns = _SIM_JIT_CACHE.get(key)
    if fns is None:
        @nnx.jit
        def policy_ud(net, x, t):          # (u, d) bang-bang from one policy query
            return decode_actions(net(dyn.nn_inputs(x), t), dyn)

        @nnx.jit
        def vquery(net, x, t):
            return net(x, t)["V"]

        @jax.jit
        def step_fn(x, u, d):              # one-step rollout + next-state costs
            xn = rollout_with_intermediate_checks(
                x=x, u=u, d=d, dyn=dyn, dt=dt, m=m, problem_type="BRAT")[0]
            return xn, dyn.target_l(xn), dyn.avoid_l(xn)

        fns = (policy_ud, vquery, step_fn)
        _SIM_JIT_CACHE[key] = fns
    return fns


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Human-in-the-loop pursuit-evasion keyboard game (exact-time control).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run_dir",  type=str, required=True)
    p.add_argument("--sim_time", type=float, default=20.0, help="Episode duration (s).")
    p.add_argument("--seed",     type=int, default=0)
    p.add_argument("--brt_dir",  type=str, default=None,
                   help="Optional BRT run directory for a least-restrictive safety filter.")
    p.add_argument("--show_ai", action="store_true",
                   help="Also show the AI-vs-AI baseline panel (2nd figure). Off by default "
                        "to halve compute/rendering — only the AI-vs-HUMAN panel is shown.")
    p.add_argument("--human", type=str, default="evader", choices=["evader", "pursuer"],
                   help="Which side the human controls. 'evader': A/D turn, W/S (or up/down) "
                        "accelerate. 'pursuer': A/D turn pursuer 1, left/right turn pursuer 2 "
                        "(the evader then uses the learned policy).")
    return p


# ---------------------------------------------------------------------------
# Human controller
# ---------------------------------------------------------------------------

class HumanController:
    """Tracks held keys and produces evader disturbance [(omega_e, a_e)]."""

    def __init__(self, dyn):
        self._dyn = dyn
        self._keys = set()
        self.omega = 0.0
        self.accel = 0.0

    def on_key_press(self, event):
        self._keys.add(event.key); self._refresh()

    def on_key_release(self, event):
        self._keys.discard(event.key); self._refresh()

    def _refresh(self):
        turn_left = "left" in self._keys or "a" in self._keys
        turn_right = "right" in self._keys or "d" in self._keys
        acc_fwd = "up" in self._keys or "w" in self._keys
        acc_back = "down" in self._keys or "s" in self._keys
        self.omega = (self._dyn.omega_max_e if (turn_left and not turn_right)
                      else -self._dyn.omega_max_e if (turn_right and not turn_left) else 0.0)
        self.accel = (self._dyn.a_max if (acc_fwd and not acc_back)
                      else -self._dyn.a_max if (acc_back and not acc_fwd) else 0.0)

    def get_action(self, x_full):
        return jnp.asarray([[self.omega, self.accel]], dtype=x_full.dtype)   # (1, 2)

    def hud_text(self, s):
        steer = "L" if self.omega > 0 else ("R" if self.omega < 0 else " ")
        acc = "+" if self.accel > 0 else ("-" if self.accel < 0 else " ")
        return f"steer {steer}  accel {acc}  v={s[3]:.2f}"


class HumanPursuerController:
    """Tracks held keys and produces pursuer control [omega_p1, omega_p2].

    A/D turn pursuer 1; Left/Right arrows turn pursuer 2 (pursuers are constant
    speed, so there is no throttle)."""

    def __init__(self, dyn):
        self._dyn = dyn
        self._keys = set()
        self.omega1 = 0.0
        self.omega2 = 0.0

    def on_key_press(self, event):
        self._keys.add(event.key); self._refresh()

    def on_key_release(self, event):
        self._keys.discard(event.key); self._refresh()

    def _refresh(self):
        om = self._dyn.omega_max_p
        p1_l, p1_r = "a" in self._keys, "d" in self._keys
        p2_l, p2_r = "left" in self._keys, "right" in self._keys
        self.omega1 = om if (p1_l and not p1_r) else (-om if (p1_r and not p1_l) else 0.0)
        self.omega2 = om if (p2_l and not p2_r) else (-om if (p2_r and not p2_l) else 0.0)

    def get_action(self, x_full):
        return jnp.asarray([[self.omega1, self.omega2]], dtype=x_full.dtype)   # (1, 2)

    def hud_text(self, s):
        b1 = "L" if self.omega1 > 0 else ("R" if self.omega1 < 0 else " ")
        b2 = "L" if self.omega2 > 0 else ("R" if self.omega2 < 0 else " ")
        return f"pursuer1 {b1}   pursuer2 {b2}"


# ---------------------------------------------------------------------------
# Live simulation (one-step-at-a-time, exact-time control)
# ---------------------------------------------------------------------------

class LiveSimulation:
    """Holds game state and advances one DT step at a time (exact-time policy)."""

    def __init__(self, x0, dyn, value_net, policy_net, all_steps, cfg, sim_time,
                 brt_value_net=None, brt_policy_net=None, brt_t_max=None,
                 human_controller=None, human_pursuer=None):
        self.dyn = dyn
        self.value_net = value_net
        self.policy_net = policy_net
        self.all_steps = all_steps
        self.cfg = cfg
        self.dt = cfg.GAME.TIME.DT
        self.T = all_steps[-1] * self.dt
        self.m = cfg.GAME.TIME.NUM_SUBSTEPS
        self.K = int(round(sim_time / self.dt))
        self.brt_value_net = brt_value_net
        self.brt_policy_net = brt_policy_net
        self.brt_t_max = brt_t_max
        self.human_controller = human_controller
        self.human_pursuer = human_pursuer
        self._pj, self._vq, self._sf = _get_sim_jit(dyn, self.dt, self.m)
        self.reset(x0)

    def reset(self, x0):
        dyn = self.dyn
        self.x_cur = x0
        self.xs = [np.asarray(x0)[0]]
        self.l_target_hist = [float(dyn.target_l(x0)[0])]
        self.l_avoid_hist = [float(dyn.avoid_l(x0)[0])]
        self.step_n = 0
        self.done = False
        self.brt_override = False

    @property
    def caught(self) -> bool:
        return self.l_target_hist[-1] <= 0.0

    def step(self):
        if self.done:
            return
        dyn = self.dyn
        x_cur = self.x_cur
        dt, T = self.dt, self.T

        # Exact remaining time-to-go for this step.
        t_togo = min(max((self.K - self.step_n) * dt, dt), T)
        t_arr = jnp.full((1,), t_togo, dtype=dyn.dtype)

        # One (jitted) policy query -> bang-bang pursuer control u and evader d.
        u, d_pred = self._pj(self.policy_net, x_cur, t_arr)

        # Pursuer control: human or learned policy.
        if self.human_pursuer is not None:
            u = self.human_pursuer.get_action(x_cur)
        elif u is None:
            u = jnp.zeros((1, dyn.control_dim), dtype=dyn.dtype)

        # Evader disturbance: human or learned policy.
        if self.human_controller is not None:
            d = self.human_controller.get_action(x_cur)
        elif d_pred is not None:
            d = d_pred
        else:
            d = jnp.zeros((1, dyn.disturb_dim), dtype=dyn.dtype)

        # Least-restrictive BRT safety filter (pursuer override) — skip when the
        # human is driving the pursuers (don't fight the human's control).
        self.brt_override = False
        if self.brt_value_net is not None and self.human_pursuer is None:
            t_brt = jnp.full((1,), self.brt_t_max, dtype=dyn.dtype)
            t_brat = jnp.full((1,), T, dtype=dyn.dtype)
            V_brt = float(self._vq(self.brt_value_net, clamp_state(dyn, x_cur), t_brt)[0])
            V_brat = float(self._vq(self.value_net, x_cur, t_brat)[0])
            if (V_brt <= 0.3) and (V_brat > 0.0):
                u, _ = self._pj(self.brt_policy_net, x_cur, t_brt)
                self.brt_override = True

        x_next, tl, al = self._sf(x_cur, u, d)
        self.x_cur = x_next
        self.xs.append(np.asarray(x_next)[0])
        self.l_target_hist.append(float(tl[0]))
        self.l_avoid_hist.append(float(al[0]))
        self.step_n += 1


# ---------------------------------------------------------------------------
# Live display panel (one matplotlib figure)
# ---------------------------------------------------------------------------

class SimPanel:
    """Field + cost subplot for one simulation. Call update() each frame."""

    _E_COL = "royalblue"
    _P_COL = "firebrick"
    # Per-pursuer colors so the human can tell which one each key group drives
    # (pursuer 1 = A/D, pursuer 2 = arrows).
    _P_COLS = ["firebrick", "darkorange", "purple", "teal"]

    def __init__(self, fig_num, title, dyn, sim_time, is_human=False, control_help=None):
        self.dyn = dyn
        self.sim_time = sim_time
        self.fig, (self.ax_f, self.ax_c) = plt.subplots(
            2, 1, num=fig_num, figsize=(6, 8), gridspec_kw={"height_ratios": [3, 1]})
        self.fig.suptitle(title, fontsize=10, fontweight="bold")
        self.fig.tight_layout(pad=2.0)

        half = dyn.half
        obs_cx, obs_cy, obs_r = dyn.circle_obstacles[0]
        ax = self.ax_f
        ax.set_xlim(-half - 0.5, half + 0.5); ax.set_ylim(-half - 0.5, half + 0.5)
        ax.set_aspect("equal"); ax.set_xlabel("x (m)", fontsize=8); ax.set_ylabel("y (m)", fontsize=8)
        ax.add_patch(mpatches.Rectangle((-half, -half), dyn.field_size, dyn.field_size,
                                        linewidth=2, edgecolor="black", facecolor="none"))
        ax.add_patch(mpatches.Circle((obs_cx, obs_cy), obs_r,
                                     edgecolor="saddlebrown", facecolor="saddlebrown", alpha=0.45))

        self.ghost_e, = ax.plot([], [], color=self._E_COL, alpha=0.2, lw=1)
        self.ghost_p = [ax.plot([], [], color=self._P_COLS[j % len(self._P_COLS)], alpha=0.2, lw=1)[0]
                        for j in range(dyn.num_pursuers)]

        self.evader_body = mpatches.Circle((0, 0), dyn.robot_radius, color=self._E_COL, alpha=0.9, zorder=4)
        ax.add_patch(self.evader_body)
        self.evader_head, = ax.plot([], [], color=self._E_COL, lw=2.5, zorder=5)

        self.pursuer_bodies, self.pursuer_catches, self.pursuer_heads, self.pursuer_labels = [], [], [], []
        for j in range(dyn.num_pursuers):
            col = self._P_COLS[j % len(self._P_COLS)]
            body = mpatches.Circle((0, 0), dyn.robot_radius, color=col, alpha=0.9, zorder=4)
            catch = mpatches.Circle((0, 0), dyn.catch_radius, edgecolor=col, facecolor="none",
                                    linestyle="--", alpha=0.4, zorder=3)
            head, = ax.plot([], [], color=col, lw=2.5, zorder=5)
            label = ax.text(0, 0, str(j + 1), color="white", fontsize=9, fontweight="bold",
                            ha="center", va="center", zorder=6)
            ax.add_patch(body); ax.add_patch(catch)
            self.pursuer_bodies.append(body); self.pursuer_catches.append(catch)
            self.pursuer_heads.append(head); self.pursuer_labels.append(label)

        self.time_text = ax.text(0.02, 0.97, "", transform=ax.transAxes, fontsize=8, va="top",
                                 family="monospace", bbox=dict(boxstyle="round", fc="white", alpha=0.7))
        if is_human:
            self.ctrl_text = ax.text(0.02, 0.03, "", transform=ax.transAxes, fontsize=8, va="bottom",
                                     family="monospace", bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.85))
            help_str = control_help or "A/D turn  W/S accel\nR reset  Q quit"
            ax.text(0.98, 0.03, help_str,
                    transform=ax.transAxes, fontsize=7, ha="right", va="bottom", family="monospace",
                    bbox=dict(boxstyle="round", fc="white", alpha=0.7))
        else:
            self.ctrl_text = None

        self.outcome_text = ax.text(0.5, 0.5, "", transform=ax.transAxes, fontsize=24, fontweight="bold",
                                    ha="center", va="center", color="white", visible=False,
                                    bbox=dict(boxstyle="round", fc="black", alpha=0.65))

        ac = self.ax_c
        ac.set_xlim(0, sim_time); ac.axhline(0, color="black", lw=0.8, linestyle="--")
        ac.set_xlabel("t (s)", fontsize=8); ac.set_ylabel("value", fontsize=8)
        ac.grid(True, alpha=0.3); ac.tick_params(labelsize=7)
        self.l_tgt_line, = ac.plot([], [], color=self._E_COL, lw=1.5, label="target_l")
        self.l_avd_line, = ac.plot([], [], color=self._P_COL, lw=1.5, label="avoid_l")
        self.cursor = ac.axvline(0.0, color="gray", lw=1.0, linestyle=":")
        ac.legend(fontsize=7, loc="upper right")
        self._cost_ymin, self._cost_ymax = -1.0, 1.0
        ac.set_ylim(self._cost_ymin, self._cost_ymax)

        # Blitting: cache the static background once, redraw only the moving
        # artists each frame (far cheaper than a full canvas redraw).
        self._bg = None
        self._f_artists = [self.ghost_e, *self.ghost_p, self.evader_body, self.evader_head,
                           *self.pursuer_bodies, *self.pursuer_catches, *self.pursuer_heads,
                           *self.pursuer_labels, self.time_text, self.outcome_text]
        if self.ctrl_text is not None:
            self._f_artists.append(self.ctrl_text)
        self._c_artists = [self.l_tgt_line, self.l_avd_line, self.cursor]

    def update(self, sim, human_ctrl=None):
        dyn = self.dyn
        al = dyn.robot_radius * 2.0
        s = sim.xs[-1]
        t_now = sim.step_n * sim.dt

        xe, ye, the = s[0], s[1], s[2]
        self.evader_body.center = (xe, ye)
        self.evader_body.set_facecolor("gold" if sim.caught else self._E_COL)
        self.evader_head.set_xdata([xe, xe + al * np.cos(the)]); self.evader_head.set_ydata([ye, ye + al * np.sin(the)])

        xs_arr = np.array(sim.xs)
        self.ghost_e.set_xdata(xs_arr[:, 0]); self.ghost_e.set_ydata(xs_arr[:, 1])
        for j in range(dyn.num_pursuers):
            base = dyn._pursuer_base(j)
            xp, yp, thp = s[base], s[base + 1], s[base + 2]
            self.pursuer_bodies[j].center = (xp, yp); self.pursuer_catches[j].center = (xp, yp)
            self.pursuer_labels[j].set_position((xp, yp))
            self.pursuer_heads[j].set_xdata([xp, xp + al * np.cos(thp)])
            self.pursuer_heads[j].set_ydata([yp, yp + al * np.sin(thp)])
            self.ghost_p[j].set_xdata(xs_arr[:, base]); self.ghost_p[j].set_ydata(xs_arr[:, base + 1])

        self.time_text.set_text(f"t={t_now:.2f}s" + ("  [BRT]" if sim.brt_override else ""))
        if self.ctrl_text is not None and human_ctrl is not None:
            self.ctrl_text.set_text(human_ctrl.hud_text(s))

        times = np.arange(len(sim.l_target_hist)) * sim.dt
        self.l_tgt_line.set_xdata(times); self.l_tgt_line.set_ydata(sim.l_target_hist)
        self.l_avd_line.set_xdata(times); self.l_avd_line.set_ydata(sim.l_avoid_hist)
        self.cursor.set_xdata([t_now, t_now])

        ylim_changed = False
        all_vals = sim.l_target_hist + sim.l_avoid_hist
        if all_vals:
            new_ymin = min(self._cost_ymin, min(all_vals) - 0.3)
            new_ymax = max(self._cost_ymax, max(all_vals) + 0.3)
            if new_ymin != self._cost_ymin or new_ymax != self._cost_ymax:
                self._cost_ymin, self._cost_ymax = new_ymin, new_ymax
                self.ax_c.set_ylim(new_ymin, new_ymax)
                ylim_changed = True

        if sim.done:
            txt, color = ("CAUGHT!", "darkred") if sim.caught else ("ESCAPED!", "navy")
            self.outcome_text.set_text(txt)
            self.outcome_text.get_bbox_patch().set_facecolor(color)
            self.outcome_text.set_visible(True)
        else:
            self.outcome_text.set_visible(sim.caught)
            if sim.caught:
                self.outcome_text.set_text("CAUGHT!")
                self.outcome_text.get_bbox_patch().set_facecolor("darkred")

        # Blit: full redraw + re-cache only when the background changes (first frame
        # or a cost-axis rescale); otherwise restore the cached background and redraw
        # just the moving artists. Much faster than draw_idle every frame.
        if self._bg is None or ylim_changed:
            self.fig.canvas.draw()
            self._bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        else:
            self.fig.canvas.restore_region(self._bg)
            for a in self._f_artists:
                self.ax_f.draw_artist(a)
            for a in self._c_artists:
                self.ax_c.draw_artist(a)
            self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()


# ---------------------------------------------------------------------------
# Initial state sampler
# ---------------------------------------------------------------------------

def sample_valid_x0(dyn, brt_value_net, brt_t_max, key):
    """Return (x0 (1,sd), key) satisfying validity filters (not-yet-caught, pursuers safe)."""
    for _ in range(500):
        key, sk = jax.random.split(key)
        batch = sample_uniform_states(sk, dyn, 128)
        keep = (np.asarray(dyn.target_l(batch)) > 0) & (np.asarray(dyn.avoid_l(batch)) < 0)
        if brt_value_net is not None:
            V_brt = np.asarray(brt_value_net(batch, jnp.full((batch.shape[0],), brt_t_max, dtype=dyn.dtype))["V"])
            keep = keep & (V_brt >= 1.5)
        idxs = np.nonzero(keep)[0]
        if idxs.size > 0:
            return batch[int(idxs[0]): int(idxs[0]) + 1], key
    raise RuntimeError("Could not sample a valid initial state after 500 attempts.")


def _try_position_window(fig, x, y):
    try:
        fig.canvas.manager.window.wm_geometry(f"+{x}+{y}")   # Tkinter
        return
    except AttributeError:
        pass
    try:
        fig.canvas.manager.window.move(x, y)                  # Qt
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = build_argparser().parse_args()

    # Disable matplotlib's default key shortcuts so our WASD / arrows / R reach the
    # handler. Otherwise 's' opens the Save dialog (and steals focus, breaking
    # later keys), 'r'/'h' reset the view, 'f' toggles fullscreen, 'q' quits, etc.
    for _k in list(plt.rcParams):
        if _k.startswith("keymap."):
            plt.rcParams[_k] = []

    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    rngs = nnx.Rngs(args.seed)

    cfg = load_cfg(args.run_dir)
    dyn = make_dynamics(cfg)
    ckpt_dir = os.path.join(args.run_dir, cfg.IO.CKPT_DIRNAME)
    value_net, policy_net, all_steps, _ = load_final_nets(cfg, dyn, rngs, ckpt_dir)

    brt_value_net = brt_policy_net = brt_t_max = None
    if args.brt_dir:
        brt_value_net, brt_policy_net, brt_t_max = load_brt_nets(args.brt_dir, dyn, rngs)

    dt = cfg.GAME.TIME.DT
    K = int(round(args.sim_time / dt))

    plt.ion()
    if args.human == "pursuer":
        human_ctrl = HumanPursuerController(dyn)
        human_kwargs = {"human_pursuer": human_ctrl}
        human_title = "HUMAN Pursuers  vs  AI Evader   [A/D  <-/->]"
        human_help = "A/D   pursuer 1\n<-/-> pursuer 2\nR reset   Q quit"
    else:
        human_ctrl = HumanController(dyn)
        human_kwargs = {"human_controller": human_ctrl}
        human_title = "AI Pursuer  vs  HUMAN Evader   [A/D  W/S]"
        human_help = "A/D turn   W/S accel\nR reset   Q quit"
    _flags = {"quit": False, "reset": False}

    def _on_press(event):
        if event.key in ("q", "Q", "escape"):
            _flags["quit"] = True
        elif event.key in ("r", "R"):
            _flags["reset"] = True
        else:
            human_ctrl.on_key_press(event)

    def _on_release(event):
        human_ctrl.on_key_release(event)

    human_panel = SimPanel(2, human_title, dyn, args.sim_time, is_human=True, control_help=human_help)
    panels = [human_panel]
    ai_panel = None
    if args.show_ai:
        ai_panel = SimPanel(1, "AI Pursuer  vs  AI Evader", dyn, args.sim_time, is_human=False)
        panels.append(ai_panel)
    for panel in panels:
        panel.fig.canvas.mpl_connect("key_press_event", _on_press)
        panel.fig.canvas.mpl_connect("key_release_event", _on_release)
    _try_position_window(human_panel.fig, 50, 50)
    if ai_panel is not None:
        _try_position_window(ai_panel.fig, 700, 50)
    plt.pause(0.1)   # realize the window(s)

    def new_episode(key):
        x0, key = sample_valid_x0(dyn, brt_value_net, brt_t_max, key)
        log.info("New episode - x0 sampled  (R=reset  Q=quit)")
        common = dict(dyn=dyn, value_net=value_net, policy_net=policy_net, all_steps=all_steps,
                      cfg=cfg, sim_time=args.sim_time, brt_value_net=brt_value_net,
                      brt_policy_net=brt_policy_net, brt_t_max=brt_t_max)
        human_sim = LiveSimulation(x0=x0, **human_kwargs, **common)
        ai_sim = LiveSimulation(x0=x0, **common) if args.show_ai else None
        for panel in panels:
            panel._cost_ymin, panel._cost_ymax = -1.0, 1.0
            panel.ax_c.set_ylim(-1.0, 1.0)
            panel._bg = None   # force background re-cache on the fresh episode
        return ai_sim, human_sim, key

    ai_sim, human_sim, key = new_episode(key)

    while not _flags["quit"]:
        if not plt.fignum_exists(2) or (args.show_ai and not plt.fignum_exists(1)):
            break
        if _flags["reset"]:
            _flags["reset"] = False
            ai_sim, human_sim, key = new_episode(key)

        t0 = time.perf_counter()
        if not human_sim.done:
            human_sim.step()
            if human_sim.step_n >= K:
                human_sim.done = True
        if ai_sim is not None and not ai_sim.done:
            ai_sim.step()
            if ai_sim.step_n >= K:
                ai_sim.done = True

        human_panel.update(human_sim, human_ctrl)     # blits + flush_events
        if ai_panel is not None:
            ai_panel.update(ai_sim)

        # Pace to ~real-time by running the GUI event loop (this is what processes
        # held-key press/release events!). Unlike plt.pause it does NOT force a full
        # redraw, so the blit stays fast.
        pause_t = max(0.001, dt - (time.perf_counter() - t0))
        try:
            human_panel.fig.canvas.start_event_loop(pause_t)
        except Exception:
            break

    plt.close("all")
    log.info("Goodbye.")


if __name__ == "__main__":
    main()
