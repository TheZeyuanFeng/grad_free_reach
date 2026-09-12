"""F1Tenth filtering eval: drive a nominal MPPI controller around a track under a
learned BEV-conditioned HJ safety filter and measure how safe/how far it gets.

  * value/policy nets loaded from a training run (F1TenthBEV);
  * collision + MPPI lane-cost use F1TenthBEV's in-memory SDF (no .npy artifacts);
  * filters: ``lr`` (least-restrictive switch), ``sb`` (sampling CBF), ``none``;
  * metrics: crash rate, distance travelled, survival time, filter-activation frac.

Example
-------
  PYTHONPATH=. python f1tenth_filter_eval.py --run_dir runs/f1bev_t0_21 \
      --filter lr --track_id 001 --num_trials 50 --animate
"""

import argparse
import json
import os

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from tqdm import tqdm

from configs import get_cfg_defaults
from utils import make_dynamics, load_value_net, load_policy_net, find_latest_step
from reachability.eval.f1tenth_mppi import tvd_rk3, forward_euler, mppi, LxGrid
from reachability.eval.f1tenth_filter import (
    LeastRestrictiveF1TenthFilter, SamplingF1TenthFilter,
)

_V_MIN, _V_MAX = 1.0, 10.0
_SV_MAX, _A_MAX = 3.2, 9.51


def build_argparser():
    p = argparse.ArgumentParser(description="F1Tenth BEV safety-filter evaluation.")
    p.add_argument("--run_dir", required=True, help="Training run dir (config.yaml + checkpoints/).")
    p.add_argument("--step", type=int, default=-1, help="Checkpoint step (-1 = latest).")
    p.add_argument("--filter", choices=["lr", "sb", "none"], default="lr",
                   help="lr=least-restrictive, sb=sampling-CBF, none=nominal only.")
    p.add_argument("--track_id", default=None, help="Track id to eval on (default: vis track).")
    p.add_argument("--num_trials", type=int, default=50)
    p.add_argument("--horizon", type=float, default=10.0, help="Episode length (seconds).")
    p.add_argument("--control_dt", type=float, default=0.02,
                   help="Controller + filter step (seconds); one control is held over this interval.")
    p.add_argument("--sim_dt", type=float, default=-1.0,
                   help="Integration step (seconds); -1 = min(0.01, control_dt/5).")
    p.add_argument("--filter_rollout_dt", type=float, default=0.1,
                   help="[sb] internal look-ahead horizon for the CBF rollout (seconds).")
    p.add_argument("--filter_rollout_steps", type=int, default=3,
                   help="[sb] internal Euler substeps over filter_rollout_dt.")
    p.add_argument("--turn_weight", type=float, default=0.25,
                   help="[sb] weight on steering-rate deviation vs accel (1.0) in the "
                        "candidate-selection norm; <1 prioritises turning (cheaper).")
    p.add_argument("--eval_t", type=float, default=-1.0, help="Value query time-to-go (-1 = GAME.TIME.T).")
    p.add_argument("--threshold", type=float, default=0.1, help="Safety-value threshold V-thr.")
    p.add_argument("--v_init_min", type=float, default=0.2,
                   help="Only start episodes from states with value V > this (rejection-sampled).")
    p.add_argument("--target_speed", type=float, default=10.0, help="MPPI target speed (m/s).")
    p.add_argument("--mppi_horizon", type=int, default=10)
    p.add_argument("--mppi_threads", type=int, default=32)
    p.add_argument("--mppi_lambda", type=float, default=1000.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default=None, help="Where to save results (default: <run_dir>/filter_eval).")
    p.add_argument("--animate", action="store_true", help="Save a trajectory plot over the track.")
    return p


def _sample_start(rng, free_xy, cnt, v_of_state, v_init_min, max_tries=200):
    """Rejection-sample a drivable start pose whose value ``V > v_init_min``.

    Falls back to the highest-value candidate found if none clears the bar
    within ``max_tries`` draws.
    """
    best_s, best_v = None, -np.inf
    for _ in range(max_tries):
        pos = free_xy[int(rng.integers(cnt))]
        theta = rng.uniform(-np.pi, np.pi)
        v = rng.uniform(_V_MIN, 0.5 * _V_MAX)
        s = np.array([pos[0], pos[1], 0.0, v, theta, 0.0, 0.0], np.float32)
        Vs = v_of_state(s)
        if Vs > v_init_min:
            return s
        if Vs > best_v:
            best_v, best_s = Vs, s
    return best_s


def main():
    args = build_argparser().parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(os.path.join(args.run_dir, "config.yaml"))
    # The value/policy nets are track-agnostic (they only see BEV+proprio), so an
    # out-of-distribution track can be evaluated by loading its map into the
    # dynamics -- append it to the run's track list before construction.
    run_tracks = list(cfg.DYNAMICS.KWARGS.track_ids)
    if args.track_id and args.track_id not in run_tracks:
        cfg.DYNAMICS.KWARGS.track_ids = run_tracks + [args.track_id]
    dyn = make_dynamics(cfg)
    if getattr(dyn, "obs_kind", None) != "bev":
        raise SystemExit(f"{cfg.DYNAMICS.CLASS} is not a BEV F1Tenth dynamics.")

    track_id = args.track_id or dyn.vis_track_id
    if track_id not in dyn.track_ids:
        raise SystemExit(f"track {track_id} not found under DYNAMICS.KWARGS.tracks_dir")
    tidx = dyn.track_ids.index(track_id)
    mpp = float(dyn._mpps[tidx])
    lx_grid = LxGrid(np.asarray(dyn._lx[tidx]), mpp)
    free_xy = np.asarray(dyn._free_xy[tidx])
    free_cnt = int(dyn._free_cnt[tidx])

    ckpt_dir = os.path.join(args.run_dir, cfg.IO.CKPT_DIRNAME)
    step = find_latest_step(ckpt_dir) if args.step < 0 else args.step
    rngs = nnx.Rngs(0)
    value_net = load_value_net(cfg, dyn, rngs, ckpt_dir, step)
    policy_net = load_policy_net(cfg, dyn, rngs, ckpt_dir, step)
    eval_t = float(cfg.GAME.TIME.T) if args.eval_t < 0 else args.eval_t

    # Raw value V(s) at the eval time (BEV rendered internally), for start-state
    # rejection sampling. Appends the active track_idx to the 7-D physical state.
    @nnx.jit
    def _v_jit(vnet, x, t):
        return vnet(x, t)["V"]

    def v_of_state(s7):
        x = np.concatenate([np.asarray(s7, np.float32), [float(tidx)]])[None]
        return float(_v_jit(value_net, jnp.asarray(x),
                            jnp.full((1,), eval_t, dtype=dyn.dtype))[0])

    if args.filter == "lr":
        filt = LeastRestrictiveF1TenthFilter(dyn, value_net, policy_net, tidx, eval_t,
                                             threshold=args.threshold,
                                             filter_rollout_dt=args.filter_rollout_dt,
                                             filter_rollout_steps=args.filter_rollout_steps)
    elif args.filter == "sb":
        filt = SamplingF1TenthFilter(dyn, value_net, policy_net, tidx, eval_t,
                                     filter_rollout_dt=args.filter_rollout_dt,
                                     filter_rollout_steps=args.filter_rollout_steps,
                                     threshold=args.threshold,
                                     turn_weight=args.turn_weight, seed=args.seed)
    else:
        filt = None

    rng = np.random.default_rng(args.seed)
    control_dt = args.control_dt
    sim_dt = args.sim_dt if args.sim_dt > 0 else min(0.01, control_dt / 5.0)
    substeps = max(1, int(round(control_dt / sim_dt)))   # sim steps per held control
    n_control = int(round(args.horizon / control_dt))
    col_thresh = -0.5 * mpp   # bilinear ~mpp/2 uncertainty; declare crash below this
    results = []
    trajs = []
    n_crash = 0
    pbar = tqdm(range(args.num_trials), desc=f"filter={args.filter} track={track_id}", unit="trial")
    for trial in pbar:
        s = _sample_start(rng, free_xy, free_cnt, v_of_state, args.v_init_min)
        u0_plan = np.zeros(args.mppi_horizon)
        u1_plan = np.zeros(args.mppi_horizon)
        crashed = False
        dist = 0.0
        acts = 0
        n_ctrl = 0
        traj = [s[:2].copy()]
        active_pt = []          # one entry per integration substep (for the animation)
        t_survived = 0.0
        for _ in range(n_control):
            # Nominal MPPI plans at the control resolution; take the first control.
            u0_plan, u1_plan = mppi(args.mppi_horizon, args.mppi_threads, s, control_dt,
                                    lx_grid, u0_plan, u1_plan, _SV_MAX, _A_MAX,
                                    args.target_speed, args.mppi_lambda, rng)
            u_nom = np.array([u0_plan[0], u1_plan[0]], np.float32)
            if filt is not None:
                out = filt.filter_control(u_nom, s)
                u = out["u"]
                a = int(out["active"] > 0.5)
            else:
                u, a = u_nom, 0
            acts += a
            n_ctrl += 1
            # Zero-order hold: integrate the chosen control over control_dt at sim_dt.
            broke = False
            for _ in range(substeps):
                s_next = tvd_rk3(s, u, sim_dt)
                dist += float(np.linalg.norm(s_next[:2] - s[:2]))
                s = s_next
                traj.append(s[:2].copy())
                active_pt.append(a)
                t_survived += sim_dt
                if lx_grid(s[0:2]) < col_thresh:
                    crashed = True
                    broke = True
                    break
            if broke:
                break
        n_crash += int(crashed)
        results.append({
            "crashed": bool(crashed),
            "distance": dist,
            "survival_time": t_survived,
            "filter_active_frac": (acts / max(1, n_ctrl)) if filt is not None else 0.0,
        })
        trajs.append((np.array(traj), np.array(active_pt, dtype=bool), crashed))
        pbar.set_postfix(crash_rate=f"{n_crash/(trial+1):.2f}", dist=f"{dist:.0f}m")

    # ---- aggregate ----
    crash_rate = float(np.mean([r["crashed"] for r in results]))
    dists = np.array([r["distance"] for r in results])
    surv = np.array([r["survival_time"] for r in results])
    act = np.array([r["filter_active_frac"] for r in results])
    summary = {
        "run_dir": args.run_dir, "step": step, "filter": args.filter, "track_id": track_id,
        "num_trials": args.num_trials, "eval_t": eval_t, "threshold": args.threshold,
        "v_init_min": args.v_init_min, "control_dt": control_dt, "sim_dt": sim_dt,
        "filter_rollout_dt": args.filter_rollout_dt, "filter_rollout_steps": args.filter_rollout_steps,
        "crash_rate": crash_rate,
        "distance_mean": float(dists.mean()), "distance_median": float(np.median(dists)),
        "survival_time_mean": float(surv.mean()),
        "filter_active_frac_mean": float(act.mean()),
    }
    print("\n=== F1Tenth filter eval ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    out_dir = args.out_dir or os.path.join(args.run_dir, "filter_eval")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{args.filter}_{track_id}_step{step}"
    with open(os.path.join(out_dir, f"{tag}.json"), "w") as fh:
        json.dump({"summary": summary, "trials": results}, fh, indent=2)
    print(f"  saved -> {os.path.join(out_dir, tag + '.json')}")

    if args.animate:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib.lines import Line2D
        H, W = lx_grid.data.shape
        extent = [0, W * mpp, 0, H * mpp]

        # --- static overview PNG (all trajectories) ---
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(lx_grid.data > 0, extent=extent, origin="lower", cmap="gray", alpha=0.6)
        for tr, _act, crashed in trajs:
            ax.plot(tr[:, 0], tr[:, 1], lw=0.8, alpha=0.7, color="red" if crashed else "lime")
            if crashed:
                ax.plot(tr[-1, 0], tr[-1, 1], "rx", ms=6)
        ax.set_title(f"{tag}  crash_rate={crash_rate:.2f}  dist(mean)={dists.mean():.1f}m")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect("equal")
        fig.tight_layout()
        png = os.path.join(out_dir, f"{tag}.png")
        fig.savefig(png, dpi=140); plt.close(fig)
        print(f"  saved -> {png}")

        # --- animation: all trials' cars moving over time ---
        n_frames = max(len(tr) for tr, _, _ in trajs)
        stride = max(1, n_frames // 200)   # cap ~200 frames
        frames = list(range(0, n_frames, stride))
        figA, axA = plt.subplots(figsize=(8, 8))
        axA.imshow(lx_grid.data > 0, extent=extent, origin="lower", cmap="gray", alpha=0.6)
        axA.set_xlabel("x [m]"); axA.set_ylabel("y [m]"); axA.set_aspect("equal")
        lines = [axA.plot([], [], lw=0.9, alpha=0.6,
                          color="red" if cr else "lime")[0] for _, _, cr in trajs]
        heads = axA.scatter([t[0, 0] for t, _, _ in trajs],
                            [t[0, 1] for t, _, _ in trajs], s=18, zorder=5)
        axA.legend(handles=[Line2D([], [], marker="o", ls="", color="deepskyblue", label="rolling"),
                            Line2D([], [], marker="o", ls="", color="gold", label="filter active"),
                            Line2D([], [], marker="x", ls="", color="red", label="crashed")],
                   loc="upper right", fontsize=8, framealpha=0.8)

        def _update(f):
            pts, cols = [], []
            for (tr, act, cr) in trajs:
                k = min(f, len(tr) - 1)
                pts.append(tr[k])
                if cr and k == len(tr) - 1:
                    cols.append("red")
                elif k > 0 and k - 1 < len(act) and act[k - 1]:
                    cols.append("gold")
                else:
                    cols.append("deepskyblue")
            for ln, (tr, _a, _c) in zip(lines, trajs):
                k = min(f, len(tr) - 1)
                ln.set_data(tr[:k + 1, 0], tr[:k + 1, 1])
            heads.set_offsets(np.array(pts))
            heads.set_color(cols)
            axA.set_title(f"{tag}  t={f*sim_dt:.2f}s  crash_rate={crash_rate:.2f}")
            return (*lines, heads)

        anim = animation.FuncAnimation(figA, _update, frames=frames, blit=False)
        fps = max(1, int(round(1.0 / (sim_dt * stride))))
        vid = os.path.join(out_dir, f"{tag}.mp4")
        try:
            anim.save(vid, writer=animation.FFMpegWriter(fps=fps, bitrate=4000))
        except Exception as e:
            vid = os.path.join(out_dir, f"{tag}.gif")
            anim.save(vid, writer=animation.PillowWriter(fps=fps))
        plt.close(figA)
        print(f"  saved -> {vid}")


if __name__ == "__main__":
    main()
