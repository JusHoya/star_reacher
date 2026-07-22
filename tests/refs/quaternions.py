"""Quaternion algebra in the project convention, coded from Chapter ch:notation.

Part of the Phase 6 independent-reference set (``tests/refs/manifest.toml``).
The conventions are taken from the notation chapter alone and are restated here
because every downstream reference (pinhole projection, star-tracker statistics)
is only as independent as its attitude algebra:

* Hamilton quaternions, equation ``eq:notation:hamiltonproduct``:
  ``p (x) q = [p_w q_w - p_v . q_v, p_w q_v + q_w p_v + p_v x q_v]``.
  The JPL sign convention is not used anywhere in this project.
* Scalar-first component ordering, equation ``eq:notation:scalarfirst``:
  ``q = [q_w, q_x, q_y, q_z]``, identity ``[1, 0, 0, 0]``.
* The attitude quaternion is ``q_i2b``, the FRAME transformation from inertial
  to body, with the DCM of equation ``eq:notation:quat2dcm``:
  ``C_I2B(q) = (q_w**2 - q_v . q_v) I + 2 q_v q_v^T - 2 q_w [q_v x]``.
* Frame transformations chain left to right, equation
  ``eq:notation:quatcomp``: ``q_i2c = q_i2b (x) q_b2c`` and
  ``C_I2B(p (x) q) = C_I2B(q) C_I2B(p)``. Note the reversal in the DCM
  product -- it is the composition rule that distinguishes this convention
  from the Markley--Crassidis one, and getting it backwards is the single
  most likely defect the notation chapter warns about.

NumPy only; no ``star_reacher`` import, so this module runs on a checkout with
no compiled core.
"""

from __future__ import annotations

import numpy as np


def quat_mul(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton product ``p (x) q``, equation ``eq:notation:hamiltonproduct``."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != (4,) or q.shape != (4,):
        raise ValueError(f"quaternions must be shape (4,), got {p.shape} and {q.shape}")
    pw, pv = p[0], p[1:]
    qw, qv = q[0], q[1:]
    out = np.empty(4)
    out[0] = pw * qw - float(np.dot(pv, qv))
    out[1:] = pw * qv + qw * pv + np.cross(pv, qv)
    return out


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate, which for a unit quaternion is the inverse (ch:notation)."""
    q = np.asarray(q, dtype=float)
    if q.shape != (4,):
        raise ValueError(f"quaternion must be shape (4,), got {q.shape}")
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Return the unit quaternion parallel to ``q``."""
    q = np.asarray(q, dtype=float)
    norm = float(np.linalg.norm(q))
    if norm == 0.0:
        raise ValueError("cannot normalize a zero quaternion")
    return q / norm


def skew(v: np.ndarray) -> np.ndarray:
    """Cross-product matrix ``[v x]`` with ``[v x] a == cross(v, a)``."""
    v = np.asarray(v, dtype=float)
    if v.shape != (3,):
        raise ValueError(f"vector must be shape (3,), got {v.shape}")
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """Frame-transformation DCM, equation ``eq:notation:quat2dcm``.

    For ``q = q_i2b`` this returns ``C_I2B``: it maps the components of a vector
    resolved in the inertial frame to its components in the body frame. It is
    the TRANSPOSE of the active rotation matrix Eigen's ``toRotationMatrix``
    would return for the same quaternion (notation chapter, rule 2), which is
    the convention trap this reference must not fall into.
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (4,):
        raise ValueError(f"quaternion must be shape (4,), got {q.shape}")
    qw, qv = q[0], q[1:]
    return (
        (qw * qw - float(np.dot(qv, qv))) * np.eye(3)
        + 2.0 * np.outer(qv, qv)
        - 2.0 * qw * skew(qv)
    )


def quat_from_rotation_vector(phi: np.ndarray) -> np.ndarray:
    """Exact exponential map of a rotation vector to a unit quaternion.

    ``q = [cos(|phi|/2), sin(|phi|/2) phi/|phi|]``, the construction used by
    equations ``eq:optical:qab`` and ``eq:optical:noiseq``. The zero-rotation
    limit returns the identity exactly rather than dividing by zero.
    """
    phi = np.asarray(phi, dtype=float)
    if phi.shape != (3,):
        raise ValueError(f"rotation vector must be shape (3,), got {phi.shape}")
    angle = float(np.linalg.norm(phi))
    if angle == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    half = 0.5 * angle
    out = np.empty(4)
    out[0] = np.cos(half)
    out[1:] = np.sin(half) * phi / angle
    return out


def rotation_vector_from_quat(q: np.ndarray) -> np.ndarray:
    """Exact logarithmic map, equation ``eq:optical:extract``.

    ``theta = 2 atan2(|q_v|, |q_w|)`` with the axis sign taken from
    ``sgn(q_w)``, ``sgn(0) = +1``, and the zero-vector-part case returning zero.
    Using ``atan2`` on the vector-part norm rather than ``arccos(q_w)`` keeps
    full precision for the small rotations these sensors emit, where
    ``arccos`` loses half its digits.
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (4,):
        raise ValueError(f"quaternion must be shape (4,), got {q.shape}")
    qw, qv = q[0], q[1:]
    qv_norm = float(np.linalg.norm(qv))
    if qv_norm == 0.0:
        return np.zeros(3)
    theta = 2.0 * np.arctan2(qv_norm, abs(qw))
    sign = 1.0 if qw >= 0.0 else -1.0
    return theta * sign * qv / qv_norm
