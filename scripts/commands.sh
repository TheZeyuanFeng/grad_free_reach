# python eval.py --run_dir runs/tm_narrow_passage

# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.2

# ################ Dubins3D (reach-avoid, BRAT) ####################################
python train.py --opt \
  DYNAMICS.CLASS Dubins3D IO.EXP_NAME dubins3d \
  GAME.TIME.DT 0.05 GAME.TIME.T 1.0 GAME.TIME.WINDOW_TIME 1.0 \
  NET.POLICY.ARCH vqpolmultinet PRETRAIN.CONVERGE.CHECK_EVERY 25 \
  PRETRAIN.CONVERGE.PATIENCE 5 FINETUNE.ANCHOR_BATCHES 10\
  FINETUNE.CONVERGE.PATIENCE 5 FINETUNE.CONVERGE.CHECK_EVERY 25\
  TRAIN.CONVERGE.PATIENCE 5
# ################ NarrowPassage ###################################################
python train.py  --opt \
  DYNAMICS.CLASS NarrowPassage IO.EXP_NAME np \
  GAME.TIME.DT 0.05 GAME.TIME.T 8.0 GAME.TIME.WINDOW_TIME 2.0 \
  NET.POLICY.ARCH vqpolmultinet

################ LessLinearND ###################################################

python train.py  --opt  DYNAMICS.CLASS LessLinearND IO.EXP_NAME less_linear40D GAME.PROBLEM_TYPE BRT  GAME.CONTROL_ROLE min GAME.DISTURB_ROLE max  GAME.TIME.DT 0.025  GAME.TIME.T 1.0  GAME.TIME.WINDOW_TIME 1.0  GAME.TIME.NUM_SUBSTEPS 5  FINETUNE.FP_LAMBDA 0.0 DYNAMICS.KWARGS.N 40

python train.py  --opt  DYNAMICS.CLASS LessLinearND IO.EXP_NAME less_linear80D GAME.PROBLEM_TYPE BRT  GAME.CONTROL_ROLE min GAME.DISTURB_ROLE max  GAME.TIME.DT 0.025  GAME.TIME.T 1.0  GAME.TIME.WINDOW_TIME 1.0  GAME.TIME.NUM_SUBSTEPS 5  FINETUNE.FP_LAMBDA 0.0 DYNAMICS.KWARGS.N 80

# python eval_lesslinear.py --run_dir runs/XXX --pairwise_path data/lesslinear_pairwise_values.npy   --m_batches 10 --batch_size 1000 --grid_res 256

# ################ QuadrotorCylinderAvoidance (safety, BRT) ############################
python train.py  --opt \
  DYNAMICS.CLASS QuadrotorCylinderAvoidance IO.EXP_NAME qca \
  GAME.PROBLEM_TYPE BRT GAME.CONTROL_ROLE max GAME.DISTURB_ROLE min \
  GAME.TIME.DT 0.025 GAME.TIME.T 1.0 GAME.TIME.WINDOW_TIME 1.0 GAME.TIME.NUM_SUBSTEPS 5 \
  FINETUNE.FP_LAMBDA 10.0 NET.POLICY.ARCH vqpolmultinet PRETRAIN.CONVERGE.CHECK_EVERY 75
################ QuadrotorGateAvoidance (safety, BRT) ############################
python train.py  --opt \
  DYNAMICS.CLASS QuadrotorGateAvoidance IO.EXP_NAME qga \
  GAME.PROBLEM_TYPE BRT GAME.CONTROL_ROLE max GAME.DISTURB_ROLE min \
  GAME.TIME.DT 0.02 GAME.TIME.T 2.0 GAME.TIME.WINDOW_TIME 0.5 GAME.TIME.NUM_SUBSTEPS 5 \
  FINETUNE.FP_LAMBDA 10.0 NET.POLICY.ARCH vqpolmultinet
# ################ QuadrotorGateTraversal (BRAT) ############################
python train.py  --opt \
  DYNAMICS.CLASS QuadrotorGateTraversal IO.EXP_NAME qgt \
  GAME.PROBLEM_TYPE BRAT GAME.CONTROL_ROLE min GAME.DISTURB_ROLE max \
  GAME.TIME.DT 0.02 GAME.TIME.T 2.0 GAME.TIME.WINDOW_TIME 0.5 GAME.TIME.NUM_SUBSTEPS 5 \
  FINETUNE.FP_LAMBDA 30.0 NET.POLICY.ARCH vqpolmultinet

################ PursuerEvader (BRAT) ############################
python train.py  --opt \
  DYNAMICS.CLASS PursuitEvasion IO.EXP_NAME pursuit_evade \
  GAME.PROBLEM_TYPE BRAT GAME.CONTROL_ROLE min GAME.DISTURB_ROLE max \
  GAME.TIME.DT 0.1 GAME.TIME.T 20.0 GAME.TIME.WINDOW_TIME 5.0 GAME.TIME.NUM_SUBSTEPS 5 \
  FINETUNE.FP_LAMBDA 1.0 NET.POLICY.ARCH vqpolmultinet FINETUNE.ANCHOR_BATCHES 100
MPLBACKEND=TkAgg python evade_pursuit_sim_keyboard.py --run_dir runs/pursuit_evade   # human evader (default)
MPLBACKEND=TkAgg python evade_pursuit_sim_keyboard.py --run_dir runs/pursuit_evade --human pursuer  # human pursuer
MPLBACKEND=TkAgg python evade_pursuit_sim_keyboard.py --run_dir runs/pursuit_evade --show_ai   # + AI-vs-AI

# ################ F1Tenth with BEV inputs ############################
# build sdf 
python -c "
import time, reachability.dynamics as dm
ids=[f'{i:03d}' for i in range(1,23)]          # 001..022  (tracks 0-21)
t0=time.time(); dyn=dm.F1TenthBEV(tracks_dir='tracks', track_ids=ids)
print(f'OK: {dyn.n_tracks} tracks + stacked SDF built in {time.time()-t0:.1f}s, '
      f'maps={tuple(dyn._maps_f.shape)}')"
# launch training
python train.py  --opt \
  DYNAMICS.CLASS F1TenthBEV IO.EXP_NAME f1bev_t0_21 \
  DYNAMICS.KWARGS.tracks_dir tracks \
  "DYNAMICS.KWARGS.track_ids" "['001','002','003','004','005','006','007','008','009','010','011','012','013','014','015','016','017','018','019','020','021']" \
  DYNAMICS.KWARGS.vis_track_idx 001 \
  GAME.PROBLEM_TYPE BRT GAME.CONTROL_ROLE max GAME.DISTURB_ROLE min \
  GAME.TIME.DT 0.05 GAME.TIME.T 2.0 GAME.TIME.WINDOW_TIME 0.5 GAME.TIME.NUM_SUBSTEPS 5 \
  FINETUNE.FP_LAMBDA 10.0 \
  NET.POLICY.ARCH vqpolmultinet \
  TRAIN.BATCH_SIZE 256 SAMPLING.SEED_SIZE 1024 VIS.X_LO 50.0 VIS.X_HI 75.0 VIS.Y_LO 82.0 VIS.Y_HI 107.0 

python f1tenth_filter_eval.py --run_dir runs/f1bev_t0_21  --filter lr --track_id 001 --num_trials 50 --animate
python f1tenth_filter_eval.py --run_dir runs/f1bev_t0_21  --filter sb --track_id 001 --num_trials 50 --animate
python f1tenth_filter_eval.py --run_dir runs/f1bev_t0_21  --filter none --track_id 001 --num_trials 50 --animate
python f1tenth_filter_eval.py --run_dir runs/f1bev_t0_21  --filter lr --track_id 023 --num_trials 50 --animate
python f1tenth_filter_eval.py --run_dir runs/f1bev_t0_21  --filter sb --track_id 023 --num_trials 50 --animate
python f1tenth_filter_eval.py --run_dir runs/f1bev_t0_21  --filter none --track_id 023 --num_trials 50 --animate