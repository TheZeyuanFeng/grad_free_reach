"""Ego-centric BEV extraction for F1Tenth tracks (JAX, jitted).

The crop -> rotate(theta+pi/2) -> asymmetric crop -> zoom-to-out_size pipeline is
exactly one affine resample: every output BEV pixel maps to a world pixel, sampled
bilinearly. That is expressible as a single ``jax.scipy.ndimage.map_coordinates``
gather over an affine-transformed output grid -- jittable and batchable over poses,
fully GPU-resident.

Coordinate conventions (match the reference):
    World frame : +x = image columns, +y = image rows; ``global_map`` is stored
                  y-up (row 0 = world y = 0), so world (wx,wy) -> pixel
                  (col=wx/mpp, row=wy/mpp). theta CCW, 0 -> +x.
    Ego frame   : +x forward, +y left.
    BEV layout  : row 0 = farthest ahead (forward = up), col 0 = left of ego.
    Forward range is speed-adaptive: x in [-1, 1 + v^2/8], lateral y = +/- half
    that span, so the BEV covers a square of side (x_max - x_min) metres.
"""

from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

# Reference-only (host, numpy) — used for validation, not in the jitted path.
from PIL import Image
import yaml
from scipy.ndimage import (
    rotate as scipy_rotate,
    zoom as scipy_zoom,
    distance_transform_edt,
)


# ---------------------------------------------------------------------------
# Track loading (host, once)
# ---------------------------------------------------------------------------

def load_track(tracks_dir, track_id: str):
    """Load an occupancy map as a y-up uint8 array (free=255, wall=0) and its mpp."""
    tracks_dir = Path(tracks_dir)
    with open(tracks_dir / "map_info.yaml") as f:
        mpp = float(yaml.safe_load(f)[track_id]["res"])
    img = np.array(Image.open(tracks_dir / track_id / f"{track_id}.png").convert("L"))
    img = img[::-1, :].copy()   # flip so row 0 = world y = 0
    return img.astype(np.uint8), mpp


# ---------------------------------------------------------------------------
# Jitted renderer
# ---------------------------------------------------------------------------

def _bev_ranges(v):
    x_min = -1.0
    x_max = 1.0 + v ** 2 / 8.0
    return x_min, x_max


def ego_bev(global_map, wx, wy, theta, v, mpp, out_size: int = 128):
    """Jitted single-pose ego BEV. global_map: (H,W) uint8/float (y-up). Returns
    (out_size, out_size) float32 in [0,1]."""
    gm = global_map.astype(jnp.float32) / 255.0
    x_min, x_max = _bev_ranges(v)
    mpp_bev = (x_max - x_min) / out_size          # metres per BEV pixel (square)

    ii = jnp.arange(out_size, dtype=jnp.float32)
    jj = jnp.arange(out_size, dtype=jnp.float32)
    I, J = jnp.meshgrid(ii, jj, indexing="ij")    # (out,out)

    # ego-frame metres: row 0 -> x_max ahead; ego at col out/2. The reference's
    # rotate+crop puts increasing column to the ego's LEFT, i.e. l grows with J.
    f = x_max - I * mpp_bev
    l = (J - out_size / 2.0) * mpp_bev

    ct, st = jnp.cos(theta), jnp.sin(theta)
    wxp = wx + f * ct - l * st                    # forward along theta, left along theta+90
    wyp = wy + f * st + l * ct
    cols = wxp / mpp
    rows = wyp / mpp

    bev = map_coordinates(gm, [rows, cols], order=1, mode="constant", cval=0.0)
    return bev


def ego_bev_batch(global_map, wx, wy, theta, v, mpp, out_size: int = 128):
    """Batched over (wx, wy, theta, v) — each shape (B,). Returns (B, out_size, out_size)."""
    f = lambda a, b, c, d: ego_bev(global_map, a, b, c, d, mpp, out_size)
    return jax.vmap(f)(wx, wy, theta, v)


# ---------------------------------------------------------------------------
# Signed-distance field (host, once) + jitted world lookup
# ---------------------------------------------------------------------------

def build_lx(global_map, mpp, free_thresh: float = 127.0):
    """Signed-distance field in metres, y-up (index [row, col]).

    Positive inside the drivable region (safe), negative inside walls
    (``lx = -sdf`` with ``sdf = dist_outside - dist_inside``). Stored in the same
    orientation as ``load_track`` (row = wy/mpp, col = wx/mpp), so it is sampled
    with ``[rows, cols]``.
    """
    free = np.asarray(global_map) > free_thresh          # (H, W) y-up
    dist_inside = distance_transform_edt(free)           # pixels to nearest wall
    dist_outside = distance_transform_edt(~free)         # pixels to nearest free
    sdf = (dist_outside - dist_inside) * mpp             # negative inside free
    return (-sdf).astype(np.float32)                     # positive inside free (safe)


def sample_lx(lx, wx, wy, mpp):
    """Jitted bilinear SDF lookup at world (wx, wy). ``lx`` is y-up [row, col].

    ``mode="nearest"`` clamps queries off the map to the border (a wall, hence
    negative/unsafe), avoiding a hard cval cliff in the value target.
    """
    rows = wy / mpp
    cols = wx / mpp
    return map_coordinates(lx, [rows, cols], order=1, mode="nearest")


def free_world_coords(global_map, mpp, free_thresh: float = 127.0):
    """(N, 2) world-metre coords of every drivable pixel, for state sampling."""
    fr, fc = np.where(np.asarray(global_map) > free_thresh)   # rows (y), cols (x)
    return np.stack([fc * mpp, fr * mpp], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Multi-track (host load + stacked, jitted device ops)
# ---------------------------------------------------------------------------
#
# Tracks differ in both pixel size and metres-per-pixel, so maps/SDFs are
# zero-padded up to a common (Hmax, Wmax) canvas and stacked on a leading track
# axis; the per-track ``mpp`` is carried separately. World (0,0) stays at pixel
# (0,0) because padding is appended at the high-row/high-col side (all walls),
# so the world<->pixel map is unchanged and the padding never affects in-track
# queries. Selection is by an integer track index threaded as a coordinate into
# ``map_coordinates`` (exact, since an integer coordinate puts zero interpolation
# weight on the neighbouring track).

def load_tracks(tracks_dir, track_ids):
    """Stack N tracks into padded arrays. Returns (maps (N,Hmax,Wmax) uint8,
    mpps (N,) float32, sizes (N,2) int32 = per-track (H, W))."""
    maps, mpps, sizes = [], [], []
    for tid in track_ids:
        m, mpp = load_track(tracks_dir, tid)
        maps.append(m); mpps.append(mpp); sizes.append(m.shape)
    Hmax = max(s[0] for s in sizes)
    Wmax = max(s[1] for s in sizes)
    padded = np.zeros((len(maps), Hmax, Wmax), dtype=np.uint8)
    for i, m in enumerate(maps):
        padded[i, :m.shape[0], :m.shape[1]] = m
    return padded, np.asarray(mpps, np.float32), np.asarray(sizes, np.int32)


def build_lx_stack(maps_padded, mpps, sizes, free_thresh: float = 127.0):
    """Per-track SDF (positive inside drivable) on each track's UNPADDED region,
    placed into the padded canvas. Padding is left unsafe (negative)."""
    n, H, W = maps_padded.shape
    lx = np.full((n, H, W), -1.0, dtype=np.float32)   # padding reads as wall
    for i in range(n):
        h, w = int(sizes[i][0]), int(sizes[i][1])
        lx[i, :h, :w] = build_lx(maps_padded[i, :h, :w], float(mpps[i]), free_thresh)
    return lx


def free_world_coords_stack(maps_padded, mpps, sizes, free_thresh: float = 127.0):
    """Padded per-track drivable coords: (N, Nmax, 2) float32 + counts (N,) int32."""
    n = maps_padded.shape[0]
    coords = []
    for i in range(n):
        h, w = int(sizes[i][0]), int(sizes[i][1])
        coords.append(free_world_coords(maps_padded[i, :h, :w], float(mpps[i]), free_thresh))
    Nmax = max(len(c) for c in coords)
    padded = np.zeros((n, Nmax, 2), dtype=np.float32)
    counts = np.asarray([len(c) for c in coords], dtype=np.int32)
    for i, c in enumerate(coords):
        padded[i, :len(c)] = c
    return padded, counts


def _bilinear_stack(arr3d, track_idx, rows, cols, mode: str = "constant", cval: float = 0.0):
    """Bilinear sample of a stacked (N, H, W) array at per-point (track, row, col).

    A manual 4-corner gather on the FLATTENED 1D map (``jnp.take``), rather than
    ``map_coordinates`` on the 3D array: inside the TD-rollout ``lax.scan`` the
    latter's 3D gather over the closed-over stack blows up to N*H*W*batch
    intermediates and OOMs; a flat take lowers to an output-sized gather.

    ``track_idx``/``rows``/``cols`` broadcast to a common shape S; returns S.
    ``mode='constant'`` returns ``cval`` off-map; ``'nearest'`` clamps to edge.
    """
    N, H, W = arr3d.shape
    flat = arr3d.reshape(-1)                     # view, no copy
    r0f = jnp.floor(rows); c0f = jnp.floor(cols)
    fr = rows - r0f; fc = cols - c0f
    r0 = r0f.astype(jnp.int32); c0 = c0f.astype(jnp.int32)
    ti = jnp.broadcast_to(track_idx.astype(jnp.int32), jnp.broadcast_shapes(
        jnp.shape(track_idx), jnp.shape(rows), jnp.shape(cols)))

    def corner(rr, cc):
        rc = jnp.clip(rr, 0, H - 1)
        cc2 = jnp.clip(cc, 0, W - 1)
        val = jnp.take(flat, (ti * H + rc) * W + cc2)
        if mode == "constant":
            inb = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            val = jnp.where(inb, val, cval)
        return val

    v00 = corner(r0, c0);      v01 = corner(r0, c0 + 1)
    v10 = corner(r0 + 1, c0);  v11 = corner(r0 + 1, c0 + 1)
    return (v00 * (1 - fr) * (1 - fc) + v01 * (1 - fr) * fc
            + v10 * fr * (1 - fc) + v11 * fr * fc)


def ego_bev_multi(maps_f, track_idx, wx, wy, theta, v, mpps, out_size: int = 128):
    """Multi-track ego BEV. ``maps_f``: (N, H, W) float in [0,1]; ``mpps``: (N,);
    ``track_idx`` (B,) int; ``wx/wy/theta/v`` (B,). Returns (B, out_size, out_size).

    Uses a manual flattened bilinear gather (``_bilinear_stack``) rather than
    ``map_coordinates`` -- see that function for why (rollout-scan OOM).
    """
    mpp = mpps[track_idx]                        # (B,)
    x_min = -1.0
    x_max = 1.0 + v ** 2 / 8.0                   # (B,)
    mpp_bev = (x_max - x_min) / out_size         # (B,)

    ii = jnp.arange(out_size, dtype=jnp.float32)
    jj = jnp.arange(out_size, dtype=jnp.float32)
    I, J = jnp.meshgrid(ii, jj, indexing="ij")   # (out, out)
    I = I[None]; J = J[None]                      # (1, out, out) -> broadcast over B

    b = lambda a: a[:, None, None]               # (B,) -> (B, 1, 1)
    f = b(x_max) - I * b(mpp_bev)
    l = (J - out_size / 2.0) * b(mpp_bev)
    ct, st = jnp.cos(b(theta)), jnp.sin(b(theta))
    wxp = b(wx) + f * ct - l * st
    wyp = b(wy) + f * st + l * ct
    cols = wxp / b(mpp)                           # (B, out, out)
    rows = wyp / b(mpp)
    t = b(track_idx)                              # (B, 1, 1) -> broadcasts

    return _bilinear_stack(maps_f, t, rows, cols, mode="constant", cval=0.0)


def sample_lx_multi(lx, track_idx, wx, wy, mpps):
    """Multi-track SDF lookup. ``lx``: (N, H, W); ``track_idx/wx/wy`` broadcastable.
    Off-map queries clamp to the edge (a wall) via ``mode='nearest'``."""
    mpp = mpps[track_idx]                         # per-element metres/pixel
    rows = wy / mpp
    cols = wx / mpp
    return _bilinear_stack(lx, track_idx, rows, cols, mode="nearest")


# ---------------------------------------------------------------------------
# scipy reference (host, numpy) — for validation only
# ---------------------------------------------------------------------------

def _crop_with_padding(img, cx, cy, half_size, pad_value=0):
    H, W = img.shape[:2]
    r0, r1 = cy - half_size, cy + half_size
    c0, c1 = cx - half_size, cx + half_size
    pt, pb = max(0, -r0), max(0, r1 - H)
    pl, pr = max(0, -c0), max(0, c1 - W)
    crop = img[max(0, r0): min(H, r1), max(0, c0): min(W, c1)]
    if any((pt, pb, pl, pr)):
        crop = np.pad(crop, ((pt, pb), (pl, pr)), constant_values=pad_value)
    return crop


def get_egocentric_bev_scipy(global_map, wx, wy, theta, v, mpp, out_size=128):
    """Numpy/scipy reference crop-rotate-zoom renderer (validates the jitted path), in [0,1]."""
    gm = np.asarray(global_map, dtype=float) / 255.0
    x_min, x_max = -1.0, 1.0 + v ** 2 / 8.0
    half_y = (x_max - x_min) / 2.0
    px_ahead = int(round(x_max / mpp)); px_behind = int(round(abs(x_min) / mpp))
    px_left = int(round(half_y / mpp)); px_right = int(round(half_y / mpp))
    half_big = int(np.ceil(np.sqrt(max(px_left, px_right) ** 2 +
                                   max(px_ahead, px_behind) ** 2))) + 2
    col, row = int(round(wx / mpp)), int(round(wy / mpp))
    big = _crop_with_padding(gm, col, row, half_big, pad_value=0.0)
    big_rot = scipy_rotate(big, np.degrees(theta + np.pi / 2), reshape=False,
                           mode="constant", cval=0.0)
    mid_r, mid_c = big_rot.shape[0] // 2, big_rot.shape[1] // 2
    crop = big_rot[max(0, mid_r - px_ahead): min(big_rot.shape[0], mid_r + px_behind),
                   max(0, mid_c - px_left): min(big_rot.shape[1], mid_c + px_right)]
    h, w = crop.shape
    return scipy_zoom(crop, (out_size / h, out_size / w), order=1)
