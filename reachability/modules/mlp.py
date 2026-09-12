from typing import Dict, Optional
import functools

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import math

from .base import ReachabilityPolicyNet, ReachabilityValueNet
from .layers import ResidualBlock, ProprioHead, ResBlock1D, ResBlock2D, TimeEmbedding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _soft_bound(z_e: jax.Array) -> jax.Array:
    """Bound encoder outputs to (-1, 1) without creating a zero-gradient region.

    Was ``jnp.clip(z_e, -1, 1)``. The intent -- keep logits bounded so a few
    confidently-wrong samples cannot dominate the MSE gradient -- is right, but
    a hard clip achieves it by zeroing the gradient past the boundary, and that
    boundary sits exactly on the loss target (``sign(teacher_u) = ±1``, see
    ``vq_policy_loss``). So a sample that saturates on the WRONG side sits at
    maximum loss with zero gradient: the loss supplies no restoring force, and
    only weight decay pulls it back -- slowly, and for reasons unrelated to it
    being wrong. That is precisely the confidently-wrong case the bound was
    added to handle, so its polarity was inverted.

    This matters more here than in ordinary supervised learning because the
    labels are non-stationary: the teacher probes the *current* value net, so
    ``u*`` for a fixed state flips as V improves. A sample previously driven
    past the boundary can have its target flip out from under it, landing it in
    the dead region without ever having been mis-trained.

    ``tanh`` keeps the bound, keeps ``sign(tanh(z)) == sign(z)`` so the
    quantiser and STE path are unchanged, and is never exactly flat. The MSE
    floor is nonzero since ±1 is unreachable -- harmless, since the loss exists
    to get the sign right, not to reach zero.
    """
    return jnp.tanh(z_e)


def _nearest_corner(z_e: jax.Array, ternary: bool = False):
    """Quantise ``z_e`` (bounded to (-1,1) by _soft_bound) to a codeword.

    Binary (default): sign -> {-1, +1}. Ternary: round -> {-1, 0, +1}, adding a
    neutral codeword so a triple-sided teacher's u=0 target is representable.
    STE passes gradients straight through to ``z_e``.
    """
    z_q = jnp.round(z_e) if ternary else jnp.sign(z_e)
    z_q_ste = z_e + jax.lax.stop_gradient(z_q - z_e)  # STE
    return z_q_ste, z_q

# ---------------------------------------------------------------------------
# Observation Encoding
# ---------------------------------------------------------------------------

class ObsEncoder(nnx.Module):
    def __init__(self, in_dim, embed_dim, rngs: nnx.Rngs, seq_len: int = 30):
        super().__init__()

        self.initial_conv = nnx.Sequential(
            nnx.Conv(in_dim, 32, kernel_size=5, strides=1, padding="SAME", use_bias=False, rngs=rngs),
            nnx.GroupNorm(32, 8, rngs=rngs),
            nnx.silu
        )

        # Residual stack: gentler downsampling
        self.res_stack = nnx.Sequential(
            # Input: [B, seq_len, 32]  (NWC)
            ResBlock1D(32, 64, rngs=rngs, stride=2),
            ResBlock1D(64, 128, rngs=rngs, stride=2),
            ResBlock1D(128, 128, rngs=rngs, stride=1)
        )

        dummy = jnp.zeros((1, seq_len, in_dim))
        pooled_len = self.res_stack(self.initial_conv(dummy)).shape[-2]

        self.project = nnx.Sequential(
            nnx.Linear(128 * pooled_len, 256, rngs=rngs),
            nnx.silu,
            nnx.Linear(256, embed_dim, rngs=rngs),
            nnx.LayerNorm(embed_dim, rngs=rngs),
        )

    def __call__(self, x):
        h = jnp.swapaxes(x, -1, -2)     # (B, C, L) -> (B, L, C)
        h = self.initial_conv(h)
        h = self.res_stack(h)
        h = jnp.swapaxes(h, -1, -2)       # (..., L', 128) -> (..., 128, L')
        h = h.reshape(*h.shape[:-2], -1)  # (..., 128*L'), ordering [c*L' + w];
                                          # rank-agnostic so it works both batched
                                          # (B, C, L) and per-sample under vmap (C, L)
        return self.project(h)            # (..., embed_dim)


class BEVObsEncoder(nnx.Module):
    """2D-conv encoder for an ego-centric BEV occupancy image.

    Drop-in alternative to ``ObsEncoder`` for image observations: it accepts the
    SAME ``(..., C, L)`` layout the value/policy nets feed their encoders (where
    ``L = img_size * img_size``), internally reshapes to a channels-last image
    ``(..., H, W, C)``, and returns an ``embed_dim`` vector -- so selecting it is
    a one-line swap in the net constructors, with no change to their call path.
    """
    def __init__(self, embed_dim, rngs: nnx.Rngs, img_size: int = 128, in_ch: int = 1):
        super().__init__()
        self.img_size = int(img_size)
        self.in_ch = int(in_ch)

        self.stem = nnx.Sequential(
            nnx.Conv(in_ch, 32, kernel_size=(5, 5), strides=(2, 2), padding="SAME",
                     use_bias=False, rngs=rngs),                      # 128 -> 64
            nnx.GroupNorm(32, 8, rngs=rngs),
            nnx.silu,
        )
        self.res_stack = nnx.Sequential(
            ResBlock2D(32, 64, stride=2, rngs=rngs),                  # 64 -> 32
            ResBlock2D(64, 128, stride=2, rngs=rngs),                 # 32 -> 16
            ResBlock2D(128, 128, stride=2, rngs=rngs),                # 16 -> 8
        )

        dummy = jnp.zeros((1, self.img_size, self.img_size, in_ch))
        pooled = self.res_stack(self.stem(dummy))                     # (1, h, w, 128)
        flat = int(np.prod(pooled.shape[1:]))

        self.project = nnx.Sequential(
            nnx.Linear(flat, 256, rngs=rngs),
            nnx.silu,
            nnx.Linear(256, embed_dim, rngs=rngs),
            nnx.LayerNorm(embed_dim, rngs=rngs),
        )

    def __call__(self, x):
        lead = x.shape[:-2]                                           # (...,) batch dims
        C = x.shape[-2]
        img = x.reshape(-1, C, self.img_size, self.img_size)          # (N, C, H, W)
        img = jnp.transpose(img, (0, 2, 3, 1))                        # (N, H, W, C)
        h = self.res_stack(self.stem(img))
        h = h.reshape(h.shape[0], -1)
        emb = self.project(h)                                         # (N, embed_dim)
        return emb.reshape(*lead, emb.shape[-1])

def _make_obs_encoder(obs_kind, obs_dim, obs_ch_dim, obs_embed_dim, rngs):
    """Pick the observation encoder: 2D-conv for ``"bev"``, 1D-conv otherwise."""
    if obs_kind == "bev":
        img_size = int(round((obs_dim // obs_ch_dim) ** 0.5))
        return BEVObsEncoder(embed_dim=obs_embed_dim, rngs=rngs,
                             img_size=img_size, in_ch=obs_ch_dim)
    return ObsEncoder(in_dim=obs_ch_dim, embed_dim=obs_embed_dim, rngs=rngs,
                      seq_len=obs_dim // obs_ch_dim)


# ---------------------------------------------------------------------------
# Value networks
# ---------------------------------------------------------------------------

class TimeValueNet(ReachabilityValueNet):
    """Post-activation ResNet-style MLP value network.

    Architecture::

        [t, phi_x] -> stem (Linear + act) -> N x ResidualBlock
                   -> Linear -> [V, u_raw?, d_raw?]

    Args:
        phi_dim:      Dimensionality of phi_x (NOT including time).
        width:        Uniform width of all residual blocks.
        n_layers:     Number of ``ResidualBlock`` layers.
        control_dim:  Number of control dimensions.
        disturb_dim:  Number of disturbance dimensions.
        dropout:      Dropout probability inside each block.
        act:          Activation module; defaults to ``SiLU()``.
    """

    def __init__(
        self,
        T_max: float,
        phi_dim: int,
        rngs: nnx.Rngs,
        obs_dim: Optional[int] = None,
        obs_ch_dim: Optional[int] = None,
        obs_embed_dim: int = 32,
        obs_kind: str = "seq",
        embed_dim: int = 32,
        width: int = 256,
        n_layers: int = 6,
        dropout: float = 0.0,
        act: Optional[nnx.Module] = None,
    ) -> None:
        self.obs_dim = obs_dim
        self.obs_ch_dim = obs_ch_dim
        self.obs_embed_dim = obs_embed_dim
        self.obs_kind = obs_kind

        super().__init__()

        if obs_dim and obs_ch_dim is not None:
            self.encoder = _make_obs_encoder(obs_kind, obs_dim, obs_ch_dim, obs_embed_dim, rngs)
            self.proprio_head = ProprioHead(in_dim=phi_dim - obs_embed_dim, embed_dim=embed_dim, rngs=rngs)
        else:
            self.encoder = None
            self.proprio_head = None

        self.t_embed = TimeEmbedding(num_octaves=4, T_max=T_max)

        in_dim = self.t_embed.out_dim + phi_dim
        if obs_dim and obs_ch_dim:
            in_dim += embed_dim - (phi_dim - obs_embed_dim)
        _act = act if act is not None else nnx.silu
        self.stem   = nnx.Sequential(nnx.Linear(in_dim, width, rngs=rngs), _act)
        self.blocks = nnx.Sequential(*[
            ResidualBlock(width, dropout, _act, rngs=rngs)
            for _ in range(n_layers)
        ])
        self.head   = nnx.Linear(width, 1, rngs=rngs)

    def __call__(
        self,
        phi_x: jax.Array,
        t: jax.Array,
    ) -> Dict[str, jax.Array]:
        if self.encoder is not None:
            obs_flat = phi_x[:, :self.obs_dim]
            obs_seq = obs_flat.reshape(phi_x.shape[0], self.obs_ch_dim, -1)
            emb = self.encoder(obs_seq)
            phi_x = self.proprio_head(phi_x[:, self.obs_dim:])
            phi_x = jnp.concatenate([emb, phi_x], axis=-1)
        if t.ndim == 1:
            t = t[:, None]
        t_emb = self.t_embed(t)
        model_outs = self.head(self.blocks(self.stem(jnp.concatenate([t_emb, phi_x], axis=-1))))
        out: Dict[str, jax.Array] = {"V": model_outs[:, 0]}
        return out

class WindowSlice(nnx.Module):
    """Read-only view over a *contiguous slice* of a multi-window net's ``nets``.

    Same stack + per-sample-gather forward pass as ``TimeValueMultiNet``/
    ``VQPolicyMultiNet``, but constructed from an explicit (short) list of
    window submodules instead of holding all ``num_windows`` of them. Passing
    the full multi-window container into ``nnx.jit`` costs ``nnx.split``/
    ``nnx.merge`` time proportional to *every* window's parameters, even when
    ``window_range`` only reads 1-2 of them -- this lets forward-pass-only call
    sites (TD-target rollouts, teacher/boundary probes) hand jit just the
    windows that are actually reachable, in place of the whole model.

    Only ever used for inference (never wrapped in ``value_and_grad``), so the
    fact that its ``nets`` alias the real windows' submodules (not copies) has
    no correctness implications here -- reads are always safe.
    """

    def __init__(self, nets: list, w_lo: int, window_time: float, num_windows: int,
                 inflations: Optional[jax.Array] = None):
        self.nets = nnx.List(nets)
        self.w_lo = w_lo
        self.window_time = window_time
        self.num_windows = num_windows
        # Per-net calibrated safety margin (see TimeValueMultiNet.inflations),
        # sliced to align with `nets` -- inflations[i] corresponds to nets[i].
        self.inflations = None if inflations is None else nnx.Variable(jnp.asarray(inflations))

    def __call__(self, x: jax.Array, t: jax.Array, window_range: tuple = None,
                 apply_inflation: bool = False, shared_time: bool = False) -> dict:
        t_flat = jnp.reshape(t, (-1,))
        if t_flat.shape[0] != x.shape[0]:
            t_flat = jnp.broadcast_to(t_flat, (x.shape[0],))
        window_indices = jnp.floor(
            (t_flat - 1e-6) / self.window_time
        ).astype(jnp.int32).clip(0, self.num_windows - 1)

        w_hi = self.w_lo + len(self.nets) - 1
        B = x.shape[0]
        idx = jnp.clip(window_indices - self.w_lo, 0, w_hi - self.w_lo)

        if shared_time:
            # Every row shares one t -- dispatch to exactly the one net every
            # row needs instead of evaluating every net in this slice.
            local_idx = idx[0]
            result = jax.lax.switch(
                local_idx,
                [functools.partial(lambda net, xx: net(xx, t_flat), net)
                 for net in self.nets],
                x,
            )
        else:
            outs = [net(x, t_flat) for net in self.nets]
            rows = jnp.arange(B)
            result = {}
            for k in outs[0].keys():
                result[k] = None if outs[0][k] is None else jnp.stack([o[k] for o in outs], axis=0)[idx, rows]

        if apply_inflation and self.inflations is not None and result.get("V") is not None:
            result["V"] = result["V"] + self.inflations.value[idx].astype(result["V"].dtype)
        return result


class TimeValueMultiNet(ReachabilityValueNet):
    """Window-partitioned value network for time-marching HJ reachability.

    Maintains one ``TimeValueNet`` per time window. Only the current window's parameters are
    trained at any given curriculum step via ``freeze_all_except``.

    Args:
        window_time:  Duration of each time window (seconds).
        T_max:        Total time horizon.
        phi_dim:      Input feature dimension (from ``dyn.input_dim``).
        control_dim:  Control dimension (from ``dyn.control_dim``).
        disturb_dim:  Disturbance dimension (from ``dyn.disturb_dim``).
        width:        Hidden width of each sub-network.
        n_layers:     Number of ResNet blocks per sub-network.
        dropout:      Dropout probability inside each block.
        act:          Activation module; defaults to ``SiLU()``.
    """

    def __init__(
        self,
        window_time: float,
        T_max: float,
        phi_dim: int,
        rngs: nnx.Rngs,
        obs_dim: Optional[int] = None,
        obs_ch_dim: Optional[int] = None,
        obs_embed_dim: int = 32,
        obs_kind: str = "seq",
        width: int = 256,
        n_layers: int = 6,
        dropout: float = 0.0,
        act: Optional[nnx.Module] = None,
    ) -> None:
        self.obs_dim = obs_dim
        self.obs_ch_dim = obs_ch_dim
        self.obs_embed_dim = obs_embed_dim
        self.obs_kind = obs_kind

        super().__init__()
        self.window_time  = window_time
        self.T_max        = T_max
        self.num_windows  = math.ceil(T_max / window_time)
        self.inflations   = nnx.Variable(jnp.zeros((self.num_windows,)))

        self.nets = nnx.List([
            TimeValueNet(
                T_max=T_max,
                phi_dim=phi_dim,
                rngs=rngs,
                obs_dim=obs_dim,
                obs_ch_dim=obs_ch_dim,
                obs_embed_dim=obs_embed_dim,
                obs_kind=obs_kind,
                width=width,
                n_layers=n_layers,
                dropout=dropout,
                act=act
            )
            for k in range(self.num_windows)
        ])

    def __call__(
        self,
        x: jax.Array,
        t: jax.Array,
        apply_inflation: bool = False,
        window_range: tuple = None,
        shared_time: bool = False,
    ) -> Dict[str, jax.Array]:
        # Map each t to its window index: t ∈ (0, W] → window 0, etc.
        window_indices = jnp.floor(
            (t - 1e-6) / self.window_time
        ).astype(jnp.int32).clip(0, self.num_windows - 1)

        if shared_time:
            window_idx = window_indices[0]
            V_out = jax.lax.switch(
                window_idx,
                [functools.partial(lambda i, xx: self.nets[i](xx, t)["V"], i)
                 for i in range(self.num_windows)],
                x,
            ).astype(x.dtype)
        else:
            w_lo, w_hi = (0, self.num_windows - 1) if window_range is None else window_range
            B = x.shape[0]
            V_all = jnp.stack(
                [self.nets[i](x, t)["V"] for i in range(w_lo, w_hi + 1)], axis=0
            )  # (w_hi - w_lo + 1, B)
            local_idx = jnp.clip(window_indices - w_lo, 0, w_hi - w_lo)
            V_out = V_all[local_idx, jnp.arange(B)].astype(x.dtype)

        if apply_inflation:
            # Per-sample inflation gather (integer indexing is jit-safe).
            V_out = V_out + self.inflations.value[window_indices].astype(V_out.dtype)

        return {"V": V_out}

    # ------------------------------------------------------------------
    # Window lifecycle
    # ------------------------------------------------------------------

    def get_trainable_window(self, k: int):
        # Robust to any module prefix (e.g. BoundaryAwareNet wraps this multinet
        # as `.net`, so paths look like ('net','nets',k,...) rather than
        # ('nets',k,...)). Match a `..., 'nets', k, ...` segment anywhere in the
        # path instead of assuming a fixed position.
        def window_filter(path, val):
            keys = tuple(path)
            return isinstance(val, nnx.Param) and any(
                keys[i] == 'nets' and keys[i + 1] == k
                for i in range(len(keys) - 1)
            )
        return window_filter

    def warm_start_from_previous(self, k: int) -> None:
        """Initialize window k's weights from window k-1."""
        prev_idx, curr_idx = k - 1, k
        if 0 <= prev_idx < len(self.nets) and 0 <= curr_idx < len(self.nets):
            state = nnx.state(self.nets[prev_idx])
            nnx.update(self.nets[curr_idx], state)

# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------

class VQPolicyNet(ReachabilityPolicyNet):
    """Single-window VQ policy network.

    Drop-in replacement for ``TimePolicyNet``.  Constructor is intentionally
    signature-compatible so the existing ``load_policy_net`` machinery works
    unchanged when ``NET.POLICY.ARCH = "vqpolnet"``.

    Parameters
    ----------
    phi_dim     : int   Dimensionality of dyn.nn_inputs(x), excluding time.
    control_dim : int   Number of control dimensions.
    disturb_dim : int   Number of disturbance dimensions (0 ⇒ no d head).
    width       : int   Hidden layer width of the MLP encoder.
    n_layers    : int   Number of hidden Linear+SiLU pairs.
    """

    def __init__(
        self,
        T_max: float,
        phi_dim: int,
        control_dim: int,
        disturb_dim: int,
        rngs: nnx.Rngs,
        obs_dim: Optional[int] = None,
        obs_ch_dim: Optional[int] = None,
        obs_embed_dim: int = 32,
        obs_kind: str = "seq",
        embed_dim: int = 32,
        width: int = 256,
        n_layers: int = 3,
        act: Optional[nnx.Module] = None,
        ternary: bool = False,
    ) -> None:
        self.control_dim = control_dim
        self.disturb_dim = disturb_dim
        self.obs_dim = obs_dim
        self.obs_ch_dim = obs_ch_dim
        self.obs_embed_dim = obs_embed_dim
        self.obs_kind = obs_kind
        self.ternary = ternary   # 3-level codebook {-1,0,+1} for triple-sided probing

        super().__init__()

        if obs_dim and obs_ch_dim:
            self.obs_encoder = _make_obs_encoder(obs_kind, obs_dim, obs_ch_dim, obs_embed_dim, rngs)
            self.proprio_head = ProprioHead(in_dim=phi_dim - obs_embed_dim, embed_dim=embed_dim, rngs=rngs)
        else:
            self.obs_encoder = None
            self.proprio_head = None

        self.t_embed = TimeEmbedding(num_octaves=4, T_max=T_max)
        total_dim = control_dim + disturb_dim

        # Shared MLP encoder — same topology as TimePolicyNet
        layers = []
        in_dim = self.t_embed.out_dim + phi_dim  # time concatenated with phi_x
        if obs_dim and obs_ch_dim:
            in_dim += embed_dim - (phi_dim - obs_embed_dim)
        _act = act if act is not None else nnx.silu
        for _ in range(n_layers):
            layers += [nnx.Linear(in_dim, width, rngs=rngs), _act]
            in_dim = width
        layers.append(nnx.Linear(in_dim, total_dim, rngs=rngs))
        self.encoder = nnx.Sequential(*layers)
        # Codebooks are implicit (computed on-the-fly element-wise)

    def __call__(self, phi_x: jax.Array, t: jax.Array, window_range: tuple = None) -> dict:
        """
        Parameters
        ----------
        phi_x : (B, phi_dim)
        t     : (B,) or (B, 1)

        Returns
        -------
        dict with keys:
            "u_pred" : (B, control_dim) STE output in {-1,+1}^D; compatible
                       with decode_actions classification branch.
            "d_pred" : (B, disturb_dim) or None
            "z_e_u"  : (B, control_dim) raw encoder output (for loss)
            "z_q_u"  : (B, control_dim) hard-quantised (stop-gradient)
            "z_e_d"  : (B, disturb_dim) same for d (if disturb_dim > 0)
            "z_q_d"  : (B, disturb_dim)
        """
        # if t.ndim == phi_x.ndim - 1:
        #     t = jnp.expand_dims(t, axis=-1)

        if self.obs_encoder is not None:
            obs_flat = phi_x[..., :self.obs_dim]
            obs_seq = obs_flat.reshape(*phi_x.shape[:-1], self.obs_ch_dim, -1)
            emb = self.obs_encoder(obs_seq)
            phi_x_prop = self.proprio_head(phi_x[..., self.obs_dim:])
            phi_x = jnp.concatenate([emb, phi_x_prop], axis=-1)

        t_emb = self.t_embed(t)
        z_e = self.encoder(jnp.concatenate([t_emb, phi_x], axis=-1))  # (B, ctrl+dist)

        out: dict = {}

        if self.control_dim > 0:
            z_raw_u = z_e[..., :self.control_dim]
            z_e_u = _soft_bound(z_raw_u)
            z_q_ste_u, z_q_u = _nearest_corner(z_e_u, self.ternary)
            out["u_pred"] = z_q_ste_u  # STE — {-1,+1} (or {-1,0,+1} if ternary) forward, ∇z_e_u backward
            out["z_e_u"] = z_e_u
            out["z_q_u"] = z_q_u       # hard-quantised; no gradient
            out["z_raw_u"] = z_raw_u   # pre-bound; saturation diagnostic only
        else:
            out["u_pred"] = None
            out["z_raw_u"] = None

        if self.disturb_dim > 0:
            z_raw_d = z_e[..., self.control_dim:]
            z_e_d = _soft_bound(z_raw_d)
            z_q_ste_d, z_q_d = _nearest_corner(z_e_d)
            out["d_pred"] = z_q_ste_d
            out["z_e_d"] = z_e_d
            out["z_q_d"] = z_q_d
            out["z_raw_d"] = z_raw_d
        else:
            out["d_pred"] = None
            out["z_raw_d"] = None

        return out


# ---------------------------------------------------------------------------
# VQPolicyMultiNet — window-partitioned version (mirrors TimePolicyMultiNet)
# ---------------------------------------------------------------------------

class VQPolicyMultiNet(ReachabilityPolicyNet):
    """Window-partitioned VQ policy network.

    One ``VQPolicyNet`` per time window; only the active window is trained
    at each curriculum step.  Mirrors the interface of ``TimePolicyMultiNet``.

    Parameters
    ----------
    window_time : float  Duration of each time window (seconds).
    T_max       : float  Total time horizon.
    phi_dim     : int
    control_dim : int
    disturb_dim : int
    width       : int
    n_layers    : int
    """

    def __init__(
        self,
        window_time: float,
        T_max: float,
        phi_dim: int,
        control_dim: int,
        disturb_dim: int,
        rngs: nnx.Rngs,
        obs_dim: Optional[int] = None,
        obs_ch_dim: Optional[int] = None,
        obs_embed_dim: int = 32,
        obs_kind: str = "seq",
        width: int = 256,
        n_layers: int = 3,
        act: Optional[nnx.Module] = None,
        ternary: bool = False,
    ) -> None:
        self.control_dim = control_dim
        self.disturb_dim = disturb_dim
        self.obs_dim = obs_dim
        self.obs_ch_dim = obs_ch_dim
        self.obs_embed_dim = obs_embed_dim
        self.obs_kind = obs_kind
        self.ternary = ternary

        super().__init__()
        self.window_time = window_time
        self.T_max = T_max
        self.num_windows = math.ceil(T_max / window_time)

        self.nets = nnx.List([
            VQPolicyNet(
                T_max=T_max,
                phi_dim=phi_dim,
                control_dim=control_dim,
                disturb_dim=disturb_dim,
                rngs=rngs,
                width=width,
                n_layers=n_layers,
                obs_dim=obs_dim,
                obs_ch_dim=obs_ch_dim,
                obs_embed_dim=obs_embed_dim,
                obs_kind=obs_kind,
                act=act,
                ternary=ternary,
            )
            for k in range(self.num_windows)
        ])
    
    def __call__(self, phi_x: jax.Array, t: jax.Array, window_range: tuple = None,
                 shared_time: bool = False) -> dict:
        t = jnp.reshape(t, (-1,))
        if t.shape[0] != phi_x.shape[0]:
            t = jnp.broadcast_to(t, (phi_x.shape[0],))
        window_indices = (
            jnp.floor((t - 1e-6) / self.window_time)
            .astype(jnp.int32)
            .clip(0, self.num_windows - 1)
        )

        if shared_time:
            window_idx = window_indices[0]
            return jax.lax.switch(
                window_idx,
                [functools.partial(lambda i, xx: self.nets[i](xx, t), i)
                 for i in range(self.num_windows)],
                phi_x,
            )

        # Full-batch per-window eval + per-sample select, restricted to [w_lo, w_hi].
        w_lo, w_hi = (0, self.num_windows - 1) if window_range is None else window_range
        B = phi_x.shape[0]
        idx = jnp.clip(window_indices - w_lo, 0, w_hi - w_lo)
        outs = [self.nets[i](phi_x, t) for i in range(w_lo, w_hi + 1)]
        rows = jnp.arange(B)
        result = {}
        for k in outs[0].keys():
            if outs[0][k] is None:
                result[k] = None
            else:
                result[k] = jnp.stack([o[k] for o in outs], axis=0)[idx, rows]
        return result

    def get_trainable_window(self, k: int):
        def window_filter(path, val):
            keys = tuple(path)
            return isinstance(val, nnx.Param) and any(
                keys[i] == 'nets' and keys[i + 1] == k
                for i in range(len(keys) - 1)
            )
        return window_filter

    def warm_start_from_previous(self, k: int) -> None:
        """Initialize window k's weights from window k-1."""
        prev_idx, curr_idx = k - 1, k
        if 0 <= prev_idx < len(self.nets) and 0 <= curr_idx < len(self.nets):
            state = nnx.state(self.nets[prev_idx])
            nnx.update(self.nets[curr_idx], state)