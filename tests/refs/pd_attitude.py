"""The quaternion-error PD attitude control law, coded from ch:gnc-builtin.

Part of the Phase 6 independent-reference set (``tests/refs/manifest.toml``).
This module is the single Python controller Phase 6 exit criterion 2 names:
one implementation evaluated against BOTH the committed mpmath golden vectors
of ``tests/golden/gnc/pd_attitude.toml`` (``test_gnc_pd_golden.py``) and the
torques the compiled C++ ``pd_attitude`` component logs on a mission
(``test_gnc_missions.py``). The criterion's conjunction -- a Python controller
reproducing the built-in controller on a golden scenario -- is only satisfied
by one controller meeting both, which is why the law lives here rather than
inline in either test.

The normative arithmetic is equations ``eq:gnc:deltaq`` through
``eq:gnc:sat`` of Chapter ch:gnc-builtin, restated as the cross-workstream
contract in the ``cpp/include/star/gnc/builtin.hpp`` header::

    dq    = q_cmd^* (x) q_est                       (eq:gnc:deltaq)
    s     = (dq_0 >= 0) ? +1 : -1                   (eq:gnc:sign)
    w_err = w_est - C(dq) w_cmd                     (eq:gnc:werr)
    tau_i = -kp_i s dq_vec_i - kd_i w_err_i         (eq:gnc:pd)
    tau_i = clamp(tau_i, -tau_max_i, +tau_max_i)    (eq:gnc:sat)

with NO renormalization of ``dq`` -- inputs are used as received.

INDEPENDENCE, STATED PRECISELY. The other modules in this directory are blind
references written from the math-library chapters without reading the C++ they
check. This one cannot claim quite that, and the difference is recorded rather
than glossed: the PD law's normative statement is a specification comment that
happens to sit in the implementation's own header, so the specification and the
implementation share a file. What this module does not do is read the C++
function body: the Hamilton product, the quaternion-to-DCM of
``eq:notation:quat2dcm``, the sign branch, and the clamp are written out from
the equations above and from Chapter ch:notation, in NumPy, with different
associativity and different loop structure than the C++. Its independent
anchor is the golden set, whose expected torques are a 60-digit mpmath
evaluation of those same equations and were not produced by either
implementation.

NumPy only; no ``star_reacher`` import, so this module runs on a checkout with
no compiled core.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "error_dcm",
    "error_quaternion",
    "pd_torque",
    "resolve_commanded_rate",
]


def _as_stack(values, width: int, name: str) -> tuple[np.ndarray, bool]:
    """Return ``values`` as a (N, width) stack plus whether it was scalar.

    A single sample and a run of samples share one code path, so the golden
    consumer and the mission consumer exercise identical arithmetic.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        if array.shape[0] != width:
            raise ValueError(f"{name} must have {width} components, got {array.shape}")
        return array.reshape(1, width), True
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape (N, {width}), got {array.shape}")
    return array, False


def error_quaternion(q_cmd, q_est) -> np.ndarray:
    """``dq = q_cmd^* (x) q_est`` (``eq:gnc:deltaq``), Hamilton, scalar-first.

    ``dq`` is the cmd-to-body error rotation. Accepts single quaternions or
    (N, 4) stacks and returns the matching shape. The conjugate keeps the
    scalar part and negates the vector part; no renormalization is applied,
    per the normative statement.
    """
    cmd, scalar = _as_stack(q_cmd, 4, "q_cmd")
    est, est_scalar = _as_stack(q_est, 4, "q_est")
    if scalar != est_scalar:
        raise ValueError("q_cmd and q_est must both be single or both be stacks")
    pw, px, py, pz = cmd[:, 0], -cmd[:, 1], -cmd[:, 2], -cmd[:, 3]
    qw, qx, qy, qz = est[:, 0], est[:, 1], est[:, 2], est[:, 3]
    dq = np.empty((cmd.shape[0], 4))
    dq[:, 0] = pw * qw - px * qx - py * qy - pz * qz
    dq[:, 1] = pw * qx + px * qw + py * qz - pz * qy
    dq[:, 2] = pw * qy - px * qz + py * qw + pz * qx
    dq[:, 3] = pw * qz + px * qy - py * qx + pz * qw
    return dq[0] if scalar else dq


def error_dcm(dq) -> np.ndarray:
    """``C(dq)`` per ``eq:notation:quat2dcm``, the frame-transformation DCM.

    Built element by element from the quaternion components rather than by
    composing an outer product with a skew matrix, so this reference and the
    ``quaternions.quat_to_dcm`` of the same directory reach the same matrix by
    visibly different arithmetic. Accepts a single quaternion or an (N, 4)
    stack; returns (3, 3) or (N, 3, 3).
    """
    stack, scalar = _as_stack(dq, 4, "dq")
    w, x, y, z = stack[:, 0], stack[:, 1], stack[:, 2], stack[:, 3]
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    c = np.empty((stack.shape[0], 3, 3))
    c[:, 0, 0] = ww + xx - yy - zz
    c[:, 0, 1] = 2.0 * (x * y + w * z)
    c[:, 0, 2] = 2.0 * (x * z - w * y)
    c[:, 1, 0] = 2.0 * (x * y - w * z)
    c[:, 1, 1] = ww - xx + yy - zz
    c[:, 1, 2] = 2.0 * (y * z + w * x)
    c[:, 2, 0] = 2.0 * (x * z + w * y)
    c[:, 2, 1] = 2.0 * (y * z - w * x)
    c[:, 2, 2] = ww - xx - yy + zz
    return c[0] if scalar else c


def resolve_commanded_rate(dq, w_cmd) -> np.ndarray:
    """``C(dq) w_cmd`` -- the commanded rate in the estimated body frame.

    The first half of ``eq:gnc:werr``. Split out from :func:`pd_torque` so a
    test can measure how far the error DCM is from the identity, which is what
    decides whether this term is exercised at all on a given scenario.
    """
    rate, scalar = _as_stack(w_cmd, 3, "w_cmd")
    c = np.atleast_3d(error_dcm(dq)).reshape(-1, 3, 3)
    out = np.einsum("kij,kj->ki", c, rate)
    return out[0] if scalar else out


def pd_torque(q_cmd, q_est, w_cmd, w_est, kp, kd, tau_max=None) -> np.ndarray:
    """Commanded body torque, ``eq:gnc:deltaq``--``eq:gnc:sat``.

    ``q_cmd``/``q_est`` are (4,) or (N, 4) scalar-first quaternions and
    ``w_cmd``/``w_est`` the matching (3,) or (N, 3) body rates [rad/s].
    ``kp``, ``kd`` and ``tau_max`` are the three per-axis gain vectors.

    ``tau_max=None`` returns the UNSATURATED torque of ``eq:gnc:pd`` alone.
    Callers use it to count how many samples the clamp of ``eq:gnc:sat``
    actually caught: the saturation branch is only exercised where the two
    results differ, and a gate that never separates them is not testing it.
    """
    dq = error_quaternion(q_cmd, q_est)
    rate, scalar = _as_stack(w_est, 3, "w_est")
    dq_stack = np.atleast_2d(dq)
    s = np.where(dq_stack[:, 0] >= 0.0, 1.0, -1.0)  # sign(0) = +1
    w_cmd_b = np.atleast_2d(resolve_commanded_rate(dq, w_cmd))
    kp = np.asarray(kp, dtype=np.float64)
    kd = np.asarray(kd, dtype=np.float64)
    tau = -kp * s[:, None] * dq_stack[:, 1:] - kd * (rate - w_cmd_b)
    if tau_max is not None:
        limit = np.asarray(tau_max, dtype=np.float64)
        tau = np.clip(tau, -limit, limit)
    return tau[0] if scalar else tau
