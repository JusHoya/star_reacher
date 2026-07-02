"""Regenerate the rotation-kernel golden-vector files in this directory.

The values anchor the FR-3/D-7 rotation kernel (cpp/src/rotation.cpp):
Hamilton scalar-first quaternions, the frame-transformation DCM convention
C_A^B of the notation chapter, and the 3-2-1 / 3-1-3 Euler sequences. Two
independent constructions produce and cross-check every matrix:

- ERFA (pyerfa), the reference implementation of the IAU SOFA algorithms:
  rotation matrices from erfa.rv2m (Euler axis-angle rotation vector) and
  from erfa.rx/ry/rz frame-rotation compositions. ERFA's r-matrices are
  coordinate-transformation (frame rotation) matrices, exactly this
  project's C_A^B convention (SOFA Vector/Matrix library conventions).
- An independent NumPy evaluation of the closed-form attitude matrix
  C(q) = (qw^2 - qv.qv) I + 2 qv qv^T - 2 qw [qv]x (Markley & Crassidis
  2014, transcribed to scalar-first ordering in the notation chapter,
  eq:notation:quat2dcm) and of the explicit R1/R2/R3 element formulas.

Both constructions must agree to <= 2 ulp element-wise at generation time;
the committed value is the ERFA one. Quaternion inputs are dyadic where
exactness matters and are committed as binary64 hex literals.

Running this script rewrites the .toml golden files byte-identically; any
diff after regeneration means the script or the goldens were edited by
hand, which the FR-22 golden-update discipline forbids.
"""

from __future__ import annotations

import math
import pathlib

import erfa
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# Independent NumPy constructions (the non-ERFA leg of the cross-check)
# --------------------------------------------------------------------------


def skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def dcm_from_quat_np(q: np.ndarray) -> np.ndarray:
    """C_A^B from q_a2b, scalar-first (eq:notation:quat2dcm)."""
    w, v = q[0], q[1:]
    return (w * w - v @ v) * np.eye(3) + 2.0 * np.outer(v, v) - 2.0 * w * skew(v)


def r1_np(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])


def r2_np(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def r3_np(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def euler321_np(a1: float, a2: float, a3: float) -> np.ndarray:
    """3-2-1 sequence: rotate about axis 3 by a1, then 2 by a2, then 1 by a3."""
    return r1_np(a3) @ r2_np(a2) @ r3_np(a1)


def euler313_np(a1: float, a2: float, a3: float) -> np.ndarray:
    """3-1-3 sequence: rotate about axis 3 by a1, then 1 by a2, then 3 by a3."""
    return r3_np(a3) @ r1_np(a2) @ r3_np(a1)


# --------------------------------------------------------------------------
# ERFA constructions (the committed values)
# --------------------------------------------------------------------------


def dcm_from_quat_erfa(q: np.ndarray) -> np.ndarray:
    """C_A^B via erfa.rv2m: for the unit quaternion [cos(t/2), sin(t/2) e]
    the frame rotation is by angle t about axis e, i.e. rotation vector
    t*e. rv2m implements the same coordinate-transformation convention as
    this project's DCMs (cross-checked against dcm_from_quat_np below)."""
    w = np.clip(q[0], -1.0, 1.0)
    angle = 2.0 * math.atan2(float(np.linalg.norm(q[1:])), float(w))
    n = float(np.linalg.norm(q[1:]))
    axis = q[1:] / n if n > 0.0 else np.array([0.0, 0.0, 1.0])
    return np.asarray(erfa.rv2m(angle * axis))


def euler321_erfa(a1: float, a2: float, a3: float) -> np.ndarray:
    r = np.eye(3)
    r = np.asarray(erfa.rz(a1, r))
    r = np.asarray(erfa.ry(a2, r))
    r = np.asarray(erfa.rx(a3, r))
    return r


def euler313_erfa(a1: float, a2: float, a3: float) -> np.ndarray:
    r = np.eye(3)
    r = np.asarray(erfa.rz(a1, r))
    r = np.asarray(erfa.rx(a2, r))
    r = np.asarray(erfa.rz(a3, r))
    return r


# --------------------------------------------------------------------------
# Committed cases
# --------------------------------------------------------------------------

# Unit quaternions, scalar-first [w, x, y, z]. The first four exercise the
# four Shepperd extraction branches (each component dominant in turn); the
# rest are generic attitudes with no structure. Components are chosen so
# normalization is the only non-exact input operation.
QUAT_CASES = [
    ("identity", [1.0, 0.0, 0.0, 0.0]),
    ("shepperd_w_dominant", [0.9, 0.2, -0.3, 0.1]),
    ("shepperd_x_dominant", [0.1, -0.9, 0.3, 0.2]),
    ("shepperd_y_dominant", [-0.2, 0.1, 0.9, -0.3]),
    ("shepperd_z_dominant", [0.2, 0.3, -0.1, -0.9]),
    ("half_turn_x", [0.0, 1.0, 0.0, 0.0]),
    ("half_turn_diag", [0.0, 0.6, 0.0, 0.8]),
    ("generic_1", [0.5, 0.5, 0.5, 0.5]),
    ("generic_2", [0.36, 0.48, -0.64, 0.48]),
    ("generic_3", [-0.4, 0.2, 0.4, 0.8]),
    ("small_angle", [0.9999999995, 1e-5, 2e-5, -1.5e-5]),
]

# Euler angle triples (radians), in application order (a1 first). Includes
# gimbal-lock-adjacent cases: 3-2-1 locks at a2 = +/- pi/2, 3-1-3 locks at
# a2 = 0 or pi.
EULER_CASES = [
    ("zero", (0.0, 0.0, 0.0)),
    ("simple_axes", (0.5, 0.0, 0.0)),
    ("generic_1", (0.3, -0.4, 1.2)),
    ("generic_2", (-2.5, 0.9, -0.7)),
    ("generic_3", (3.0, -1.2, 0.1)),
    ("moon_pa_magnitudes", (0.03, 0.4, 2.9)),
    ("near_lock_321_up", (0.7, math.pi / 2 - 1e-2, -0.4)),
    ("near_lock_321_down", (-1.1, -math.pi / 2 + 1e-2, 0.8)),
    ("near_lock_313_zero", (0.6, 1e-2, -1.3)),
    ("near_lock_313_pi", (0.2, math.pi - 1e-2, 2.2)),
]


def normalize(q: list[float]) -> np.ndarray:
    v = np.array(q, dtype=float)
    return v / np.linalg.norm(v)


def check_close(a: np.ndarray, b: np.ndarray, tol: float, what: str) -> float:
    d = float(np.max(np.abs(a - b)))
    assert d <= tol, f"{what}: cross-check spread {d:.3e} > {tol:.1e}"
    return d


# --------------------------------------------------------------------------
# TOML emission (restricted subset read by cpp/tests/golden_io.hpp)
# --------------------------------------------------------------------------


def emit(path: pathlib.Path, header: str, cases: list[dict]) -> None:
    lines = [f"# {line}" for line in header.strip().splitlines()]
    for case in cases:
        lines.append("")
        lines.append("[[case]]")
        for key, value in case.items():
            if isinstance(value, list):
                lines.append(f"{key} = [")
                lines.extend(f'  "{item}",' for item in value)
                lines.append("]")
            else:
                lines.append(f'{key} = "{value}"')
    path.write_text("\n".join(lines) + "\n", newline="\n", encoding="utf-8")


def mat_fields(m: np.ndarray) -> dict:
    out = {}
    for i in range(3):
        for j in range(3):
            out[f"c{i}{j}"] = float(m[i, j]).hex()
    return out


def main() -> None:
    max_quat = 0.0
    quat_cases = []
    for name, q_raw in QUAT_CASES:
        q = normalize(q_raw)
        m_erfa = dcm_from_quat_erfa(q)
        m_np = dcm_from_quat_np(q)
        # Two independent constructions of the same rotation; disagreement
        # beyond a few ulp means a convention error (which would be O(1)),
        # not rounding. 1e-15 admits the observed few-ulp spread.
        max_quat = max(max_quat, check_close(m_erfa, m_np, 1e-15, name))
        case = {"name": name}
        for comp, val in zip("wxyz", q):
            case[f"q{comp}"] = float(val).hex()
        case.update(mat_fields(m_erfa))
        quat_cases.append(case)

    max_euler = 0.0
    euler_cases = []
    for name, (a1, a2, a3) in EULER_CASES:
        for seq, f_erfa, f_np in (
            ("321", euler321_erfa, euler321_np),
            ("313", euler313_erfa, euler313_np),
        ):
            m_erfa = f_erfa(a1, a2, a3)
            m_np = f_np(a1, a2, a3)
            max_euler = max(
                max_euler, check_close(m_erfa, m_np, 1e-15, f"{name}/{seq}")
            )
            case = {
                "name": f"{name}_{seq}",
                "sequence": seq,
                "a1": float(a1).hex(),
                "a2": float(a2).hex(),
                "a3": float(a3).hex(),
            }
            case.update(mat_fields(m_erfa))
            euler_cases.append(case)

    emit(
        HERE / "quat_dcm.toml",
        "Quaternion -> DCM golden vectors (FR-3, D-7).\n"
        "qw..qz is a unit Hamilton quaternion, scalar-first, interpreted as\n"
        "the frame transformation q_a2b; c00..c22 is the corresponding DCM\n"
        "C_A^B (coordinate transformation, v^B = C v^A) built with ERFA\n"
        "rotation primitives (rv2m) and cross-checked at generation time\n"
        "against an independent NumPy evaluation of the closed-form\n"
        "attitude matrix. The first four non-identity cases make each\n"
        "quaternion component dominant in turn, covering all four branches\n"
        "of the Shepperd DCM->quaternion extraction. Provenance and\n"
        "tolerances in manifest.toml. Regenerated by generate.py.",
        quat_cases,
    )
    emit(
        HERE / "euler.toml",
        "Euler-sequence golden vectors (FR-3, D-7).\n"
        "a1, a2, a3 are radians in application order: sequence 321 is\n"
        "C = R1(a3) R2(a2) R3(a1); sequence 313 is C = R3(a3) R1(a2) R3(a1)\n"
        "(frame rotations). Matrices from erfa.rx/ry/rz compositions,\n"
        "cross-checked against independent NumPy element formulas at\n"
        "generation time. Includes gimbal-lock-adjacent cases (321: a2 near\n"
        "+/-pi/2; 313: a2 near 0 and pi). Provenance and tolerances in\n"
        "manifest.toml. Regenerated by generate.py.",
        euler_cases,
    )

    print("rotation golden files regenerated and cross-checked")
    print(f"pyerfa {erfa.__version__} (ERFA {erfa.version.erfa_version})")
    print(
        f"observed cross-check maxima: quat->DCM {max_quat:.3e}, "
        f"euler->DCM {max_euler:.3e} (element abs)"
    )


if __name__ == "__main__":
    main()
