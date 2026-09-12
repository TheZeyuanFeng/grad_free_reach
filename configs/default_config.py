from third_party.config import CfgNode as CN
from .constants import *

_C = CN()

# --------------------------------------------------
# PROBLEM / DYNAMICS: The "Math" of the game
# --------------------------------------------------
_C.GAME = CN()
_C.GAME.PROBLEM_TYPE = ProblemType.BRAT.value
_C.GAME.CONTROL_ROLE = Role.MIN.value
_C.GAME.DISTURB_ROLE = Role.MAX.value
_C.GAME.EXACT_BC     = False         # V = l(x) + t*NN(x,t); exact boundary condition

_C.GAME.TIME = CN()
_C.GAME.TIME.T            = 4.0
_C.GAME.TIME.DT           = 0.02
_C.GAME.TIME.WINDOW_TIME  = 1.0
_C.GAME.TIME.PROBE_HORIZON = -1.0
_C.GAME.TIME.MAX_SUBSTEP = 0.01
_C.GAME.TIME.NUM_SUBSTEPS = 3
_C.GAME.TIME.GAMMA        = 1.0   # discount for TD targets (<1 for long-horizon avoid)

# Triple-sided probing: also evaluate u_mid=0 as a candidate control action
# (chosen when within tau of the better bang-bang extreme). Useful for stiff
# systems (e.g. quadrotor) where bang-bang every step chatters and neutral
# control is often correct. Disturbance stays double-sided. False = double-sided.
# Also switches the VQ policy to a ternary codebook {-1,0,+1} (via build_net) so
# the learned policy can actually emit the neutral action the probe selects.
_C.GAME.TRIPLE_SIDED_PROBING = False

# --------------------------------------------------
# EXPORT: Set only in a hardware bundle's config.yaml (export_hardware_bundle.py)
# to record that the checkpoint has been truncated to a single time window --
# GAME.TIME.T/WINDOW_TIME here stay at their ORIGINAL training values (needed
# so T_max-based time-embedding normalization stays correct; see
# TimeEmbedding), so the truncated window/query time can't be inferred from
# them and must be carried separately. Sentinels (-1 / -1.0) mean "not an
# exported bundle -- use the full checkpoint's last window as usual."
# --------------------------------------------------
_C.EXPORT = CN()
_C.EXPORT.WINDOW_IDX = -1
_C.EXPORT.EVAL_T     = -1.0

# --------------------------------------------------
# DYNAMICS: Which system to instantiate
# --------------------------------------------------
_C.DYNAMICS = CN()
_C.DYNAMICS.CLASS  = DynamicsType.DUBINS_3D.value  # name of a Dynamics subclass in reachability/dynamics.py
_C.DYNAMICS.KWARGS = CN()                          # forwarded as **kwargs to the constructor
# make_dynamics() filters these to each class's __init__ signature, so a key
# only reaches dynamics that accept it (others silently ignore it).
_C.DYNAMICS.KWARGS.tracks_dir = "datasets/tracks"  # path to the tracks directory (Racing/F1Tenth)
# pin to one track (empty = all tracks / multi-track mode)
_C.DYNAMICS.KWARGS.track_ids = []
_C.DYNAMICS.KWARGS.vis_track_idx = "001"
_C.DYNAMICS.KWARGS.track_id = "001"                # single track id (F1TenthBEV)
_C.DYNAMICS.KWARGS.N = 2                            # state dimension (LessLinearND)

# --------------------------------------------------
# MODEL: The Architecture
# --------------------------------------------------
_C.NET = CN()

_C.NET.VALUE = CN()
_C.NET.VALUE.ARCH   = VALUE_ARCH.MULTINET.value
_C.NET.VALUE.KWARGS = CN()
_C.NET.VALUE.KWARGS.n_layers = 8
_C.NET.VALUE.KWARGS.width = 512

_C.NET.POLICY = CN()
_C.NET.POLICY.ARCH   = POLICY_ARCH.VQ_MULTINET.value
_C.NET.POLICY.KWARGS = CN()
_C.NET.POLICY.KWARGS.width = 512
_C.NET.POLICY.KWARGS.n_layers = 5
# NOTE: VQ ternary (3-level codebook {-1,0,+1}) is implied by GAME.TRIPLE_SIDED_PROBING
# (wired via build_net), so it is NOT a separate NET.POLICY.KWARGS knob.

# --------------------------------------------------
# PRETRAINING: Optional initial phase to warm-start the value and policy nets
# --------------------------------------------------

_C.PRETRAIN = CN()
_C.PRETRAIN.ENABLED = True

_C.PRETRAIN.STEPS = CN()
_C.PRETRAIN.STEPS.MIN_VALUE     = 100
_C.PRETRAIN.STEPS.MAX_VALUE     = 30000
_C.PRETRAIN.STEPS.MIN_POLICY    = 100
_C.PRETRAIN.STEPS.MAX_POLICY    = 30000
# Running-min block plateau convergence (warm-up phase). Stop when PATIENCE
# consecutive CHECK_EVERY-iter blocks fail to beat the running-min block mean by
# REL_TOL; never before MIN_STEPS.
_C.PRETRAIN.CONVERGE = CN()
_C.PRETRAIN.CONVERGE.CHECK_EVERY = 100
_C.PRETRAIN.CONVERGE.PATIENCE    = 15
_C.PRETRAIN.CONVERGE.REL_TOL     = 0.01

# --------------------------------------------------
# SOLVER / TRAINING: The "How-To" for the optimizer
# --------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE             = 4096
_C.TRAIN.LR                     = 3e-4
_C.TRAIN.POLICY_LR              = 3e-4
_C.TRAIN.GRAD_CLIP_NORM         = 1.0      # Global-norm gradient clip applied to both value/policy optimizers
_C.TRAIN.LAMBDA                 = 0.5      # TD(λ) parameter for rollout length distribution
_C.TRAIN.M                      = 5        # Max rollout length (in steps) for TD(λ) scheduling 
_C.TRAIN.BC_PROB                = 0.2      # Probability of including BC samples in each batch (if BC_LAMBDA > 0)
_C.TRAIN.SNAP_TIME_TO_DT        = True     # Whether to snap rollout lengths to integer multiples of the dynamics dt
_C.TRAIN.BOUNDARY_FOCAL_LOSS    = False    # Whether to use a focal loss to emphasize learning near the reachability boundary
_C.TRAIN.TEACHER_FRAC_INIT      = 0.5      # Initial teacher forcing fraction for rollout target generation
_C.TRAIN.TEACHER_FRAC_FINAL     = 0.5      # Final teacher forcing fraction for rollout target generation
_C.TRAIN.TEACHER_ROLLOUT_STEPS  = 1        # Number of steps to rollout with the teacher policy when generating targets (if TEACHER_FRAC > 0)
_C.TRAIN.BOUNDARY_FOCUS_ALPHA   = 2.0      # Maximum additional weight for boundary states
_C.TRAIN.BOUNDARY_FOCUS_TAU     = 0.02     # Controls the steepness of the transition
# Running-min block plateau convergence (per-curriculum-step / time-stepping phase).
_C.TRAIN.CONVERGE = CN()
_C.TRAIN.CONVERGE.CHECK_EVERY = 10
_C.TRAIN.CONVERGE.PATIENCE    = 10
_C.TRAIN.CONVERGE.REL_TOL     = 0.01
_C.TRAIN.USE_EMA_TARGET         = True     # Maintain a Polyak/EMA-averaged copy of the value net; use it (instead of
                                            # the live, currently-training net) for TD-bootstrap targets and the
                                            # policy's teacher probe, to decouple them from the live net's per-step churn
_C.TRAIN.EMA_TAU                = 0.005    # EMA update rate per value-training iteration: target = tau*live + (1-tau)*target

# Alternating value / policy update step budgets
_C.TRAIN.STEPS = CN()
_C.TRAIN.STEPS.MIN_VALUE  = 100
_C.TRAIN.STEPS.MAX_VALUE  = 1000
_C.TRAIN.STEPS.MIN_POLICY = 100
_C.TRAIN.STEPS.MAX_POLICY = 1000

# --------------------------------------------------
# SAMPLING: Data generation strategy
# --------------------------------------------------
_C.SAMPLING = CN()
_C.SAMPLING.FRAC_UNIFORM      = 0.40  # Fraction for uniform sampling (kept high to limit buffer-induced bias).
_C.SAMPLING.FRAC_TARGET       = 0.05  # Fraction for goal boundary (target_l~=0 band)
_C.SAMPLING.FRAC_AVOID        = 0.05  # Fraction for obstacle boundary (avoid_l~=0 band)
_C.SAMPLING.FRAC_USER_TARGET  = 0.15  # Fraction for user goal region (dyn.sample_target_states, else uniform)
_C.SAMPLING.FRAC_BOUNDARY     = 0.15  # Fraction for value boundary (|V|<band).
_C.SAMPLING.FRAC_SAFE         = 0.10  # Fraction for safe side of V (|V|>=band).
_C.SAMPLING.FRAC_UNSAFE       = 0.10  # Fraction for unsafe side of V (|V|>=band).
_C.SAMPLING.BOUNDARY_BAND     = 0.10  # Half-width of the boundary band
_C.SAMPLING.BUFFER_SIZE       = 500_000

# Static target/avoid boundary pools, banked once at construction.
_C.SAMPLING.POOL_SIZE         = 131072      # states per pool; >= 10x BATCH_SIZE avoids collisions
_C.SAMPLING.POOL_MAX_DRAWS    = 200_000_000 # cap on uniform draws while filling (~20 s at 0.02%)
_C.SAMPLING.POOL_CHUNK        = 65536       # draws per filling iteration

# Warm-start the value-boundary ring by Newton projection onto {V ~ 0}
_C.SAMPLING.SEED_BOUNDARY_BUFFER = True
_C.SAMPLING.SEED_SIZE            = -1     # <=0 => auto: 8 * TRAIN.BATCH_SIZE
_C.SAMPLING.SEED_MIN_RATIO       = 10.0

_C.SAMPLING.FRAC_NOMINAL               = 0.0  # Fraction for nominal-policy rollout states.
_C.SAMPLING.NOMINAL_STEPS              = 50   # Fixed rollout length (macro-steps) after burn-in.
_C.SAMPLING.NOMINAL_BURN_IN_STEPS      = 5    # Steps always run before the snapshot window
_C.SAMPLING.NOMINAL_SAMPLES_PER_ROLLOUT = 10   # DISTINCT snapshots harvested per rollout (<= NOMINAL_STEPS)
_C.SAMPLING.FRAC_NOMINAL_END           = -1.0 # e.g. set < FRAC_NOMINAL for a decay (see below)
_C.SAMPLING.WARMUP_T                   = -1.0
_C.SAMPLING.FRAC_UNIFORM_END           = -1.0
_C.SAMPLING.FRAC_BOUNDARY_END          = -1.0

# --------------------------------------------------
# FINETUNE: Anchoring / fine-tuning on past slices
# --------------------------------------------------
_C.FINETUNE = CN()
_C.FINETUNE.EVERY               = 10      # run a finetune pass every N slices
_C.FINETUNE.MAX_STEPS           = 30000    # max optimization steps per finetune pass
_C.FINETUNE.MIN_STEPS           = 100     # minimum steps per finetune pass (even if the loss converges early)
_C.FINETUNE.LR                  = 5e-5
_C.FINETUNE.ANCHOR_LAMBDA       = 5.0
_C.FINETUNE.ANCHOR_BATCHES      = 300
_C.FINETUNE.CALIB_BATCHES       = 10      # Held-out rollout batches for FP/FPR + inflation calibration (kept separate from ANCHOR_BATCHES to avoid measuring FPR on data the finetune step was just supervised on)
_C.FINETUNE.FULL_HORIZON_MC     = True    # Anchor supervision + calibration use FULL-horizon Monte-Carlo reach-cost targets (roll the student to t=0, bootstrap only the exact boundary) instead of a one-window bootstrap. Removes cross-window bootstrap bias; set False for the legacy one-window target.
_C.FINETUNE.DROP_BC_SAMPLES     = False   # If True, drop the finetune TD term's boundary-condition (BC) samples -- the BC_PROB fraction pinned to the window bottom (last_end+dt), where the TD target bootstraps on the PREVIOUS window's value. Kept ON (False) by default: empirically dropping them made results slightly worse. Set True to drop them.
_C.FINETUNE.FP_LAMBDA           = 3.0
_C.FINETUNE.FP_THRESHOLD        = 0.001
_C.FINETUNE.SUPERVISION_LAMBDA  = 1.0

# Window-finetune running-min plateau (used by the window finalize / finetune).
_C.FINETUNE.CONVERGE = CN()
_C.FINETUNE.CONVERGE.ENABLED     = True
_C.FINETUNE.CONVERGE.CHECK_EVERY = 50
_C.FINETUNE.CONVERGE.REL_TOL     = 0.01
_C.FINETUNE.CONVERGE.PATIENCE    = 10

# --------------------------------------------------
# INFRASTRUCTURE: Logging, Saving, Viz, Hardware
# --------------------------------------------------
_C.IO = CN()
_C.IO.OUTDIR         = "runs"                # root output directory for this run
_C.IO.USE_WANDB      = False                 # whether to log to Weights & Biases
_C.IO.WANDB_ENTITY   = "your-workspace"      # WandB entity (user or team) for logging
_C.IO.EXP_NAME       = "dubins_3d_baseline"  # subdir name for this experiment (if empty, will use timestamp-based default)
_C.IO.LOG_DIR        = ""                    # path for logs (overridden by train.py to include run-specific subdir)
_C.IO.CKPT_DIRNAME   = "checkpoints"         # subdir name for checkpoints within the log dir
_C.IO.EVAL_DIRNAME   = "eval"                # subdir name for evaluation results within the log dir
_C.IO.LOG_EVERY      = 10
_C.IO.SAVE_EVERY     = 50                    # save a checkpoint every N slices
_C.IO.DEVICE         = "cuda"
_C.IO.SEED           = 0
_C.IO.TIMESTAMP_RUNS = False                 # Whether to include a timestamp in the default experiment name for better run organization

_C.VIS = CN()
_C.VIS.EVERY   = 10
_C.VIS.EVERY_ITER = 1000   # (currently unused) plot value fn every N inner value-iters; 0 = off
_C.VIS.X_RES   = 200
_C.VIS.Y_RES   = 200
_C.VIS.DIRNAME = "vis"
_C.VIS.X_LO = -1.0
_C.VIS.X_HI = -1.0
_C.VIS.Y_LO = -1.0
_C.VIS.Y_HI = -1.0