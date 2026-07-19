"""Generate the GNC built-in component golden vectors (FR-22 layer 1).

Produces pd_attitude.toml and dead_reckoning.toml. Inputs are first snapped
to binary64 (recorded as float.hex strings, so the consuming doctest uses
bit-identical inputs), then the reference outputs are evaluated with mpmath
at 60 significant digits from those snapped inputs and rounded once to
binary64 for recording. The laws mirror gnc/builtin.hpp exactly:

  pd_attitude (ch:gnc-builtin, eq:gnc:deltaq / eq:gnc:werr / eq:gnc:pd /
  eq:gnc:sat, no renormalization of dq):
    dq    = conj(q_cmd) (x) q_est          (Hamilton, scalar-first)
    s     = +1 if dq_0 >= 0 else -1        (sign(0) = +1)
    w_err = w_est - C(dq) * w_cmd          (C: quaternion-to-DCM,
                                           eq:notation:quat2dcm, resolving
                                           the commanded rate into the
                                           estimated body frame)
    tau_i = -kp_i * s * dq_vec_i - kd_i * w_err_i
    tau_i = clamp(tau_i, -tau_max_i, +tau_max_i)

  dead_reckoning attitude update, per IMU increment dtheta:
    angle = |dtheta|; axis = dtheta/angle (identity rotation when angle == 0)
    dq    = [cos(angle/2), sin(angle/2) * axis]
    q     <- (q (x) dq) / |q (x) dq|

Run from the repository root:  python tests/golden/gnc/generate.py
"""

from __future__ import annotations

import math
from pathlib import Path

import mpmath as mp

mp.mp.dps = 60

OUT_DIR = Path(__file__).resolve().parent


def snap(x) -> float:
    """Round an mpmath value (or float) once to binary64."""
    return float(mp.mpf(x))


def to_mp(values):
    return [mp.mpf(v) for v in values]


def q_conj(q):
    return [q[0], -q[1], -q[2], -q[3]]


def q_mul(p, q):
    """Hamilton product, scalar-first (eq:rotations:hamprod)."""
    return [
        p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
        p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
        p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
        p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0],
    ]


def q_normalize(q):
    n = mp.sqrt(sum(c * c for c in q))
    return [c / n for c in q]


def snap_quat(values):
    """Normalize in extended precision, then snap each component to double."""
    return [snap(c) for c in q_normalize(to_mp(values))]


def hexlist(values):
    return [float(v).hex() for v in values]


def q_to_dcm(q):
    """Quaternion (a-to-b, scalar-first) to DCM C_a2b, mirroring the
    project's rotation::dcm_from_quat elementwise (eq:notation:quat2dcm)."""
    w, x, y, z = q
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    return [
        [ww + xx - yy - zz, 2 * (x * y + w * z), 2 * (x * z - w * y)],
        [2 * (x * y - w * z), ww - xx + yy - zz, 2 * (y * z + w * x)],
        [2 * (x * z + w * y), 2 * (y * z - w * x), ww - xx - yy + zz],
    ]


def pd_reference(q_cmd, q_est, w_est, w_cmd, kp, kd, tau_max):
    """Extended-precision evaluation of the pd_attitude law on snapped inputs."""
    dq = q_mul(q_conj(to_mp(q_cmd)), to_mp(q_est))
    s = mp.mpf(1) if dq[0] >= 0 else mp.mpf(-1)
    # eq:gnc:werr: resolve the commanded rate (commanded frame) into the
    # estimated body frame through the error DCM (dq is cmd-to-body).
    c = q_to_dcm(dq)
    wc = to_mp(w_cmd)
    w_cmd_b = [
        c[i][0] * wc[0] + c[i][1] * wc[1] + c[i][2] * wc[2] for i in range(3)
    ]
    tau = []
    for i in range(3):
        t = -mp.mpf(kp[i]) * s * dq[i + 1] - mp.mpf(kd[i]) * (
            mp.mpf(w_est[i]) - w_cmd_b[i]
        )
        limit = mp.mpf(tau_max[i])
        if t > limit:
            t = limit
        if t < -limit:
            t = -limit
        tau.append(snap(t))
    return tau, snap(dq[0])


def write_pd_cases(path: Path) -> None:
    cases = []

    # Case 1: generic tracking error, dq0 > 0, no axis saturated.
    cases.append(
        dict(
            name="generic_unsaturated",
            q_cmd=snap_quat([0.98, 0.05, -0.11, 0.17]),
            q_est=snap_quat([0.93, -0.21, 0.34, 0.08]),
            w_est=[0.011, -0.032, 0.047],
            w_cmd=[0.002, 0.001, -0.003],
            kp=[2.5, 3.0, 1.75],
            kd=[8.0, 6.5, 7.25],
            tau_max=[50.0, 50.0, 50.0],
        )
    )

    # Case 2: dq0 < 0 exercises the unwinding sign branch (s = -1).
    # dq0 equals the 4-vector dot product of q_cmd and q_est, so the pair is
    # chosen with a negative dot (verified below).
    cases.append(
        dict(
            name="sign_branch_negative_dq0",
            q_cmd=snap_quat([0.5, 0.5, 0.5, 0.5]),
            q_est=snap_quat([-0.8, 0.3, 0.2, 0.1]),
            w_est=[0.0, 0.0, 0.0],
            w_cmd=[0.0, 0.0, 0.0],
            kp=[1.0, 2.0, 4.0],
            kd=[0.5, 0.5, 0.5],
            tau_max=[100.0, 100.0, 100.0],
        )
    )

    # Case 3: dq0 == 0 exactly (90-degree-pair geometry); sign(0) = +1 is
    # the branch under test. q_cmd identity, q_est a pure vector quaternion.
    cases.append(
        dict(
            name="sign_zero_is_plus_one",
            q_cmd=[1.0, 0.0, 0.0, 0.0],
            q_est=[0.0, 1.0, 0.0, 0.0],
            w_est=[0.0, 0.0, 0.0],
            w_cmd=[0.0, 0.0, 0.0],
            kp=[3.0, 5.0, 7.0],
            kd=[1.0, 1.0, 1.0],
            tau_max=[10.0, 10.0, 10.0],
        )
    )

    # Case 4: mixed per-axis saturation (+rail, -rail, unsaturated).
    cases.append(
        dict(
            name="mixed_saturation",
            q_cmd=snap_quat([0.9, 0.3, -0.25, 0.2]),
            q_est=snap_quat([0.9, -0.3, 0.25, -0.2]),
            w_est=[0.4, -0.6, 0.001],
            w_cmd=[0.0, 0.0, 0.0],
            kp=[40.0, 40.0, 0.5],
            kd=[30.0, 30.0, 0.25],
            tau_max=[5.0, 5.0, 5.0],
        )
    )

    # Case 5: attitude aligned (dq identity), pure rate damping.
    cases.append(
        dict(
            name="rate_damping_only",
            q_cmd=snap_quat([0.7, -0.1, 0.5, 0.4]),
            q_est=snap_quat([0.7, -0.1, 0.5, 0.4]),
            w_est=[0.02, -0.015, 0.03],
            w_cmd=[0.005, 0.005, 0.005],
            kp=[2.0, 2.0, 2.0],
            kd=[9.0, 10.0, 11.0],
            tau_max=[1.0, 1.0, 1.0],
        )
    )

    lines = [
        "# pd_attitude control-law golden vectors. GENERATED by generate.py;",
        "# hand-editing is forbidden (tests/golden/README.md update policy).",
        "# Inputs are binary64 hex literals (exact); expected_tau_nm is the",
        "# 60-digit mpmath evaluation of the gnc/builtin.hpp law on those",
        "# exact inputs, rounded once to binary64. dq0 is recorded so the",
        "# consuming test can assert which sign branch each case exercises.",
        "",
    ]
    for c in cases:
        tau, dq0 = pd_reference(
            c["q_cmd"], c["q_est"], c["w_est"], c["w_cmd"], c["kp"], c["kd"],
            c["tau_max"]
        )
        # Generation-time guards: the branch-coverage cases must actually
        # exercise their branches (a silent sign flip here would hollow out
        # the consuming test).
        if c["name"] == "sign_branch_negative_dq0":
            assert dq0 < 0.0, f"case expects dq0 < 0, got {dq0}"
        if c["name"] == "sign_zero_is_plus_one":
            assert dq0 == 0.0, f"case expects dq0 == 0, got {dq0}"
        if c["name"] == "mixed_saturation":
            assert any(abs(t) == m for t, m in zip(tau, c["tau_max"])), (
                "case expects at least one saturated axis"
            )
            assert any(abs(t) < m for t, m in zip(tau, c["tau_max"])), (
                "case expects at least one unsaturated axis"
            )
        lines.append("[[case]]")
        lines.append(f'name = "{c["name"]}"')
        for key in ("q_cmd", "q_est", "w_est", "w_cmd", "kp", "kd", "tau_max"):
            lines.append(f"{key} = [")
            for h in hexlist(c[key]):
                lines.append(f'  "{h}",')
            lines.append("]")
        lines.append(f'dq0 = "{float(dq0).hex()}"')
        lines.append("expected_tau_nm = [")
        for h in hexlist(tau):
            lines.append(f'  "{h}",')
        lines.append("]")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_dead_reckoning_case(path: Path) -> None:
    q0 = snap_quat([0.96, 0.1, -0.2, 0.15])
    increments = [
        [0.02, -0.01, 0.005],
        [0.0, 0.0, 0.0],  # exactly zero: the identity-rotation branch
        [-0.004, 0.012, 0.03],
        [1e-9, 0.0, -1e-9],  # tiny but nonzero: still the axis-angle branch
        [0.25, 0.2, -0.15],  # large single-step rotation
    ]
    q = to_mp(q0)
    states = []
    for dth in increments:
        d = to_mp(dth)
        angle = mp.sqrt(sum(c * c for c in d))
        if angle == 0:
            dq = [mp.mpf(1), mp.mpf(0), mp.mpf(0), mp.mpf(0)]
        else:
            axis = [c / angle for c in d]
            s = mp.sin(angle / 2)
            dq = [mp.cos(angle / 2), s * axis[0], s * axis[1], s * axis[2]]
        q = q_normalize(q_mul(q, dq))
        states.append([snap(c) for c in q])

    lines = [
        "# dead_reckoning attitude-composition golden vector. GENERATED by",
        "# generate.py; hand-editing is forbidden (tests/golden/README.md",
        "# update policy). q0 and each increment are binary64 hex literals",
        "# (exact); q_after_k is the 60-digit mpmath composition",
        "# q <- normalize(q (x) dq(dtheta_k)) rounded once to binary64.",
        "",
        "[[case]]",
        'name = "compose_five_increments"',
        "q0 = [",
    ]
    lines.extend(f'  "{h}",' for h in hexlist(q0))
    lines.append("]")
    for k, dth in enumerate(increments):
        lines.append(f"dtheta_{k} = [")
        lines.extend(f'  "{h}",' for h in hexlist(dth))
        lines.append("]")
    for k, state in enumerate(states):
        lines.append(f"q_after_{k} = [")
        lines.extend(f'  "{h}",' for h in hexlist(state))
        lines.append("]")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    write_pd_cases(OUT_DIR / "pd_attitude.toml")
    write_dead_reckoning_case(OUT_DIR / "dead_reckoning.toml")
    print("wrote pd_attitude.toml and dead_reckoning.toml")


if __name__ == "__main__":
    main()
