"""Numpy F1Tenth simulator dynamics + MPPI nominal controller for filter eval.

The simulator uses the same hybrid kinematic/dynamic bicycle model as training
(7-D physical state ``[x, y, delta, v, theta, omega, slip]``), integrated with a
TVD-RK3 step. The MPPI planner rolls candidate control sequences forward and
weights them by a lane-keeping cost that reads a signed-distance grid (``lx``);
here that grid comes from ``F1TenthBEV``'s in-memory SDF, so no ``.npy`` track
artifacts are needed.
"""

import numpy as np

# Physical constants -- match reachability/dynamics/f1tenth_bev.py
_MU, _C_SF, _C_SR = 1.0489, 4.718, 5.4562
_LF, _LR, _H_CG, _M, _I_ZZ, _G = 0.15875, 0.17145, 0.074, 3.74, 0.04712, 9.81
_LWB = _LF + _LR
_V_SW, _A_MAX = 7.319, 9.51


def _clamp_accel(v, a):
    a_max = _A_MAX * _V_SW / abs(v) if abs(v) > _V_SW else _A_MAX
    return float(np.clip(a, -a_max, a_max))


def f1_dyn(state, control):
    """dx/dt for the hybrid kinematic(|v|<1.5)/dynamic bicycle model (7-D)."""
    delta, v, theta, omega, slip = state[2], state[3], state[4], state[5], state[6]
    sv = control[0]
    a = _clamp_accel(v, control[1])
    x_dot = v * np.cos(slip + theta)
    y_dot = v * np.sin(slip + theta)
    if abs(v) < 1.5:
        theta_dot = v / _LWB * np.tan(delta)
        omega_dot = a / _LWB * np.tan(delta) + v / (_LWB * np.cos(delta) ** 2) * sv
        slip_dot = 0.0
    else:
        Ff = _G * _LR - a * _H_CG
        Fr = _G * _LF + a * _H_CG
        theta_dot = omega
        omega_dot = (
            -_MU * _M / (v * _I_ZZ * _LWB) * (_LF**2 * _C_SF * Ff + _LR**2 * _C_SR * Fr) * omega
            + _MU * _M / (_I_ZZ * _LWB) * (_LR * _C_SR * Fr - _LF * _C_SF * Ff) * slip
            + _MU * _M / (_I_ZZ * _LWB) * _LF * _C_SF * Ff * delta
        )
        slip_dot = (
            (_MU / (v**2 * _LWB) * (_C_SR * Fr * _LR - _C_SF * Ff * _LF) - 1) * omega
            - _MU / (v * _LWB) * (_C_SR * Fr + _C_SF * Ff) * slip
            + _MU / (v * _LWB) * _C_SF * Ff * delta
        )
    return np.array([x_dot, y_dot, sv, a, theta_dot, omega_dot, slip_dot])


def _wrap_theta(s):
    if s[4] > np.pi:
        s[4] -= 2 * np.pi
    elif s[4] < -np.pi:
        s[4] += 2 * np.pi
    return s


def tvd_rk3(s0, control, dt, v_min=1.0, v_max=10.0, omega_max=4.5,
            slip_max=0.8, steer_max=0.4189):
    """TVD Runge-Kutta 3rd-order step with physical clamps (matches training)."""
    s1 = s0 + f1_dyn(s0, control) * dt
    s2 = s1 + f1_dyn(s1, control) * dt
    s_half = 0.75 * s0 + 0.25 * s2
    s_three_half = s_half + f1_dyn(s_half, control) * dt
    s = (1 / 3) * s0 + (2 / 3) * s_three_half
    s = _wrap_theta(s)
    s[2] = np.clip(s[2], -steer_max, steer_max)
    s[3] = np.clip(s[3], v_min, v_max)
    s[5] = np.clip(s[5], -omega_max, omega_max)
    s[6] = np.clip(s[6], -slip_max, slip_max)
    return s


def forward_euler(s0, control, dt):
    return _wrap_theta(s0 + f1_dyn(s0, control) * dt)


class LxGrid:
    """Bilinear signed-distance lookup on a y-up (row=wy/mpp, col=wx/mpp) grid.

    ``data`` is the (H, W) SDF (positive inside drivable), ``mpp`` metres/pixel.
    Off-grid queries clamp to the border (a wall).
    """

    def __init__(self, data: np.ndarray, mpp: float):
        self.data = np.asarray(data, np.float32)
        self.H, self.W = self.data.shape
        self.mpp = float(mpp)
        self.lx_max = float(self.data.max())

    def __call__(self, pos_xy) -> float:
        wx, wy = float(pos_xy[0]), float(pos_xy[1])
        c = np.clip(wx / self.mpp, 0, self.W - 1.001)
        r = np.clip(wy / self.mpp, 0, self.H - 1.001)
        c0, r0 = int(np.floor(c)), int(np.floor(r))
        fc, fr = c - c0, r - r0
        d = self.data
        top = d[r0, c0] * (1 - fc) + d[r0, c0 + 1] * fc
        bot = d[r0 + 1, c0] * (1 - fc) + d[r0 + 1, c0 + 1] * fc
        return float(top * (1 - fr) + bot * fr)


def _cost(x, v_max, lx_grid):
    cost = np.linalg.norm(x[3] - v_max) * 0.5      # penalize being slow
    cost += 0.5 * x[5] ** 2                          # penalize yaw rate
    lx = lx_grid(x[0:2])
    cost += 0.1 * (lx_grid.lx_max - max(lx, 0.2))    # penalize staying centered
    if lx < 0:
        cost += -1000.0 * lx                         # heavy collision penalty
    return cost


def mppi(horizon, threads, x0, dt, lx_grid, u0_plan, u1_plan,
         u0_range, u1_range, v_max, lamda, rng):
    """One MPPI step. Returns (u0_plan, u1_plan) shifted plans; take [0] as control."""
    u0_plan = np.asarray(u0_plan, np.float64).copy()
    u1_plan = np.asarray(u1_plan, np.float64).copy()
    u0_plan[-1] = 0.0
    u1_plan[-1] = 0.0
    costs = np.zeros(threads)
    U0 = np.zeros((threads, horizon))
    U1 = np.zeros((threads, horizon))
    for i in range(threads):
        u0 = np.clip(rng.normal(0, u0_range, horizon) + u0_plan, -u0_range, u0_range)
        u1 = np.clip(rng.normal(0, u1_range, horizon) + u1_plan, -u1_range, u1_range)
        x = x0.copy()
        c = 0.0
        for j in range(horizon):
            c += _cost(x, v_max, lx_grid)
            x = forward_euler(x, [u0[j], u1[j]], dt)
        costs[i] = c
        U0[i] = u0
        U1[i] = u1
    w = np.exp(-lamda * (costs - costs.min()))
    w_sum = w.sum()
    u0_plan = (w[:, None] * U0).sum(0) / w_sum
    u1_plan = (w[:, None] * U1).sum(0) / w_sum
    return u0_plan, u1_plan
