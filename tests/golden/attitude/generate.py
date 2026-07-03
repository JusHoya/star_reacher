"""Regenerate the rigid-body attitude and gravity-gradient golden vectors.

The values anchor the FR-1 attitude-dynamics kernel (cpp/src/models/
rigidbody.cpp) and the gravity-gradient torque model (cpp/src/models/
gravgrad.cpp). Five files are produced:

- rhs.toml: pointwise right-hand-side vectors (qdot, omega_dot) for the
  quaternion kinematics qdot = (1/2) q (x) [0, omega] (chapter ch:rigidbody,
  eq:rigidbody:qdot) and Euler's equation with time-varying inertia
  I omega_dot = tau - omega x (I omega) - Idot omega (eq:rigidbody:euler),
  including exact-zero cases (principal-axis spin, spherical inertia, rest
  state) whose zeros are structural, not roundoff.
- gravgrad.toml: gravity-gradient torques tau = (3 mu / r^3) rhat_b x
  (I rhat_b) (chapter ch:gravgrad, eq:gravgrad:torque), including the
  exact-zero geometries (spherical inertia, r along a principal axis) and a
  planar pitch case cross-checked here against the closed-form
  -3 n^2 (I_x - I_z) sin(theta) cos(theta) pitch torque (eq:gravgrad:pitch).
- coning.toml: the closed-form torque-free axisymmetric (coning) solution
  (ch:rigidbody, eq:rigidbody:coning): attitude and rate checkpoints over
  five precession periods for the Phase 4 exit-criterion-4 test. The closed
  form is verified below, before writing, against the ratified kinematics
  and dynamics by central finite differences in extended precision, so a
  sign or convention error in the reference construction cannot land.
- dzhanibekov.toml: intermediate-axis (Dzhanibekov) flip references: the
  exact initial angular momentum and kinetic energy (the conserved gates of
  exit criterion 4) and early-time omega checkpoints from an independent
  mpmath Taylor-series integration (mp.odefun) of Euler's equations.
  Checkpoints stop at 15 s because near-separatrix motion amplifies state
  error as exp(lambda t) (lambda = 0.2598 1/s here); beyond that horizon
  only the conserved quantities are testable, which is exactly what the
  exit criterion gates.
- libration.toml: the analytic small-angle pitch libration frequency
  n sqrt(3 (I_x - I_z) / I_y) of a gravity-gradient-stabilized body on a
  prescribed circular orbit (ch:gravgrad, eq:gravgrad:libfreq) for exit
  criterion 9, plus the finite-amplitude pendulum correction
  omega_lib * pi / (2 K(sin^2 theta_0)) (eq:gravgrad:pendulum) used as a
  tight secondary gate.

References are evaluated with mpmath at 60 significant decimal digits (the
Dzhanibekov ODE reference at 50) from the exact binary64 inputs and rounded
once to binary64. The quaternion algebra mirrors the project convention
exactly: Hamilton product, scalar-first components, q_i2b frame
transformation with the DCM of eq:notation:quat2dcm (docs/mathlib chapter
ch:notation, decision D-7). Inertia values, rates, and geometries are
representative test inputs - the models take all of them from the caller.
mu = 3.986004418e14 m^3/s^2 in the gravity-gradient cases is the IERS
Conventions (2010) value carried by star/constants.hpp; the consuming test
asserts the equality so the golden and the core cannot drift apart.

Running this script rewrites all five .toml files byte-identically; any
diff after regeneration means the script or the goldens were edited by
hand, which the FR-22 golden-update discipline forbids. Regenerating is
`python tests/golden/attitude/generate.py` (requires mpmath).
"""

from __future__ import annotations

import pathlib

import mpmath as mp

mp.mp.dps = 60

HERE = pathlib.Path(__file__).resolve().parent

MU_EARTH = 3.986004418e14  # [m^3/s^2] IERS Conventions (2010), TN 36
MU_MOON = 4.9028e12        # [m^3/s^2] representative Moon-like test value


# ---------------------------------------------------------------------------
# Vector / matrix helpers on mpmath scalars
# ---------------------------------------------------------------------------


def v3(x):
    return [mp.mpf(c) for c in x]


def vnorm(a):
    return mp.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def vsub(a, b):
    return [a[i] - b[i] for i in range(3)]


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def mat3(rows):
    return [[mp.mpf(x) for x in row] for row in rows]


def matvec(m, v):
    return [sum(m[i][j] * v[j] for j in range(3)) for i in range(3)]


def solve3(m, b):
    """Solve m x = b exactly (to working precision) via mpmath LU."""
    a = mp.matrix(m)
    rhs = mp.matrix(b)
    x = mp.lu_solve(a, rhs)
    return [x[0], x[1], x[2]]


# ---------------------------------------------------------------------------
# Quaternion algebra: Hamilton product, scalar-first, per ch:notation (D-7)
# ---------------------------------------------------------------------------


def qmul(p, q):
    """Hamilton product p (x) q (eq:notation:hamiltonproduct)."""
    pw, pv = p[0], p[1:]
    qw, qv = q[0], q[1:]
    w = pw * qw - sum(pv[i] * qv[i] for i in range(3))
    cx = cross(pv, qv)
    v = [pw * qv[i] + qw * pv[i] + cx[i] for i in range(3)]
    return [w] + v


def qconj(q):
    return [q[0], -q[1], -q[2], -q[3]]


def qnorm(q):
    return mp.sqrt(sum(c * c for c in q))


def qnormalized(q):
    n = qnorm(q)
    return [c / n for c in q]


def dcm_from_quat(q):
    """Frame-transformation DCM C_I^B (eq:notation:quat2dcm):
    C = (w^2 - v.v) I + 2 v v^T - 2 w [v]x."""
    w, x, y, z = q
    s = w * w - (x * x + y * y + z * z)
    c = [[s, 0, 0], [0, s, 0], [0, 0, s]]
    v = [x, y, z]
    for i in range(3):
        for j in range(3):
            c[i][j] += 2 * v[i] * v[j]
    sk = [[0, -z, y], [z, 0, -x], [-y, x, 0]]
    for i in range(3):
        for j in range(3):
            c[i][j] -= 2 * w * sk[i][j]
    return c


def qz(a):
    """Elementary frame-rotation quaternion about +z: C(qz(a)) = R3(a)."""
    return [mp.cos(a / 2), mp.mpf(0), mp.mpf(0), mp.sin(a / 2)]


def qx(a):
    """Elementary frame-rotation quaternion about +x: C(qx(a)) = R1(a)."""
    return [mp.cos(a / 2), mp.sin(a / 2), mp.mpf(0), mp.mpf(0)]


def q313(a1, a2, a3):
    """3-1-3 Euler sequence as a frame-transformation quaternion:
    C = R3(a3) R1(a2) R3(a1), composed left-to-right per
    eq:notation:quatcomp: q = qz(a1) (x) qx(a2) (x) qz(a3)."""
    return qmul(qmul(qz(a1), qx(a2)), qz(a3))


# ---------------------------------------------------------------------------
# mpmath mirrors of the models (chapters ch:rigidbody, ch:gravgrad)
# ---------------------------------------------------------------------------


def qdot_mp(q, w):
    """Quaternion kinematics (eq:rigidbody:qdot): qdot = 1/2 q (x) [0, w]."""
    p = qmul(q, [mp.mpf(0)] + list(w))
    return [c / 2 for c in p]


def wdot_mp(w, imat, idot, tau):
    """Euler's equation with time-varying inertia (eq:rigidbody:euler):
    I wdot = tau - w x (I w) - Idot w."""
    iw = matvec(imat, w)
    gyro = cross(w, iw)
    idw = matvec(idot, w)
    rhs = [tau[i] - gyro[i] - idw[i] for i in range(3)]
    return solve3(imat, rhs)


def gravgrad_mp(mu, r_i, q, imat):
    """Gravity-gradient torque (eq:gravgrad:torque):
    tau = (3 mu / r^3) rhat_b x (I rhat_b)."""
    r = v3(r_i)
    rn = vnorm(r)
    rhat_i = [c / rn for c in r]
    c_i2b = dcm_from_quat(q)
    rb = matvec(c_i2b, rhat_i)
    irb = matvec(imat, rb)
    k = 3 * mp.mpf(mu) / (rn * rn * rn)
    return [k * c for c in cross(rb, irb)]


# ---------------------------------------------------------------------------
# Torque-free axisymmetric (coning) closed form (ch:rigidbody,
# eq:rigidbody:coning): I = diag(I_T, I_T, I_A). With the inertially fixed
# angular momentum H defining the auxiliary frame N (z_N = H/|H|), the
# solution is the 3-1-3 Euler history q_n2b(t) = q313(phidot*t, theta,
# psi0 + psidot*t) with constant nutation theta and constant rates
# phidot = |H|/I_T, psidot = w3 (1 - I_A/I_T). The committed q0/w0 doubles
# ARE the motion: the constants below are derived from them exactly.
# ---------------------------------------------------------------------------


class ConingSolution:
    def __init__(self, q0, w0, it, ia):
        self.it = mp.mpf(it)
        self.ia = mp.mpf(ia)
        self.q0 = [mp.mpf(c) for c in q0]
        self.w0 = v3(w0)
        hb = [self.it * self.w0[0], self.it * self.w0[1],
              self.ia * self.w0[2]]
        self.h = vnorm(hb)
        self.theta = mp.acos(hb[2] / self.h)
        # H1 = H sin(theta) sin(psi), H2 = H sin(theta) cos(psi)
        self.psi0 = mp.atan2(hb[0], hb[1])
        self.phidot = self.h / self.it
        self.psidot = self.w0[2] * (1 - self.ia / self.it)
        # q_i2b(t) = q_i2n (x) q_n2b(t); anchor q_i2n on the committed q0
        # (phi0 = 0 fixes the free choice of x_N).
        qn2b0 = q313(mp.mpf(0), self.theta, self.psi0)
        self.qi2n = qmul(self.q0, qconj(qn2b0))

    def q_ref(self, t):
        t = mp.mpf(t)
        return qmul(self.qi2n,
                    q313(self.phidot * t, self.theta,
                         self.psi0 + self.psidot * t))

    def w_ref(self, t):
        """Body rates from the 3-1-3 rates (eq:rigidbody:eulerrates) with
        thetadot = 0: w = phidot*(sin th sin psi, sin th cos psi, cos th)
        + (0, 0, psidot)."""
        t = mp.mpf(t)
        psi = self.psi0 + self.psidot * t
        st = mp.sin(self.theta)
        return [self.phidot * st * mp.sin(psi),
                self.phidot * st * mp.cos(psi),
                self.phidot * mp.cos(self.theta) + self.psidot]


def coning_self_check(sol, imat, checkpoints):
    """Verify the closed form against the ratified kinematics and dynamics
    by central finite differences before committing anything. A solution
    passing these checks IS the unique solution of the initial-value
    problem, so the reference cannot carry a convention error."""
    idot = mat3([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    tau = v3([0, 0, 0])
    # Initial conditions reproduced exactly.
    dq0 = max(abs(a - b) for a, b in zip(sol.q_ref(0), sol.q0))
    dw0 = max(abs(a - b) for a, b in zip(sol.w_ref(0), sol.w0))
    assert dq0 < mp.mpf(10) ** -50, dq0
    assert dw0 < mp.mpf(10) ** -50, dw0
    h = mp.mpf(10) ** -15
    for t in [mp.mpf("31.7"), mp.mpf("87.3")] + [mp.mpf(c) for c in
                                                 checkpoints[:2]]:
        # d(q_ref)/dt against eq:rigidbody:qdot.
        qp, qm = sol.q_ref(t + h), sol.q_ref(t - h)
        fd_q = [(a - b) / (2 * h) for a, b in zip(qp, qm)]
        an_q = qdot_mp(sol.q_ref(t), sol.w_ref(t))
        assert max(abs(a - b) for a, b in zip(fd_q, an_q)) < mp.mpf(10) ** -25
        # d(w_ref)/dt against eq:rigidbody:euler (torque-free, Idot = 0).
        wp, wm = sol.w_ref(t + h), sol.w_ref(t - h)
        fd_w = [(a - b) / (2 * h) for a, b in zip(wp, wm)]
        an_w = wdot_mp(sol.w_ref(t), imat, idot, tau)
        assert max(abs(a - b) for a, b in zip(fd_w, an_w)) < mp.mpf(10) ** -25
        # Unit norm preserved (up to the committed q0's own rounding).
        assert abs(qnorm(sol.q_ref(t)) - qnorm(sol.q0)) < mp.mpf(10) ** -50
        # Inertial angular momentum constant: H^I = C_I^B(q)^T (I w).
        c = dcm_from_quat(qnormalized(sol.q_ref(t)))
        hb = matvec(imat, sol.w_ref(t))
        hi = [sum(c[j][i] * hb[j] for j in range(3)) for i in range(3)]
        c0 = dcm_from_quat(qnormalized(sol.q0))
        hb0 = matvec(imat, sol.w0)
        hi0 = [sum(c0[j][i] * hb0[j] for j in range(3)) for i in range(3)]
        assert max(abs(a - b) for a, b in zip(hi, hi0)) < mp.mpf(10) ** -45


# ---------------------------------------------------------------------------
# TOML emission (restricted subset read by cpp/tests/golden_io.hpp)
# ---------------------------------------------------------------------------


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


def hx(x) -> str:
    return float(x).hex()


def hxv(v) -> list[str]:
    return [float(c).hex() for c in v]


def hxm(m) -> list[str]:
    return [float(m[i][j]).hex() for i in range(3) for j in range(3)]


# ---------------------------------------------------------------------------
# File 1: pointwise RHS vectors
# ---------------------------------------------------------------------------


def gen_rhs() -> None:
    zero3 = [[0.0] * 3] * 3
    # (name, q_raw (normalized here), w, I, Idot, tau, qdot_exact, wdot_exact)
    cases = [
        ("generic_full_tensor",
         (0.4, -0.3, 0.5, 0.7), (0.11, -0.23, 0.31),
         [[110.0, -5.0, 3.0], [-5.0, 95.0, -4.0], [3.0, -4.0, 82.0]],
         [[-0.4, 0.02, -0.01], [0.02, -0.35, 0.03], [-0.01, 0.03, -0.5]],
         (0.7, -1.2, 0.4), False, False),
        ("generic_diag_inertia",
         (1.0, 0.2, -0.1, 0.3), (0.05, 0.4, -0.02),
         [[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]],
         zero3, (0.01, 0.0, -0.02), False, False),
        # Spin about a principal axis of a diagonal inertia, torque-free:
        # w x (I w) has every product carrying a zero factor, so omega_dot
        # is exactly zero and qdot involves only exact products (identity
        # attitude, 0.5 * 0.7 exact).
        ("principal_axis_spin",
         (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.7),
         [[4.0, 0.0, 0.0], [0.0, 2.5, 0.0], [0.0, 0.0, 1.5]],
         zero3, (0.0, 0.0, 0.0), True, True),
        # Spherical inertia with a power-of-two scale (I = 2 Id): I w = 2 w
        # exactly, and each cross-product difference subtracts two
        # identically rounded copies of the same real product, so
        # omega_dot is exactly zero for ANY rate vector.
        ("spherical_inertia",
         (0.9, 0.1, -0.3, 0.2), (0.3, -0.5, 0.7),
         [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
         zero3, (0.0, 0.0, 0.0), False, True),
        # Isolates the Idot omega term (tau = 0, diagonal I).
        ("idot_term",
         (0.6, 0.5, -0.4, 0.2), (0.2, 0.1, -0.3),
         [[10.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 6.0]],
         [[-0.2, 0.0, 0.0], [0.0, -0.15, 0.0], [0.0, 0.0, -0.1]],
         (0.0, 0.0, 0.0), False, False),
        # Rest state: omega = 0 makes qdot = 1/2 q (x) 0 exactly zero and
        # reduces Euler's equation to omega_dot = I^{-1} tau.
        ("rest_state",
         (0.7, -0.2, 0.4, -0.5), (0.0, 0.0, 0.0),
         [[50.0, 2.0, -1.0], [2.0, 40.0, 3.0], [-1.0, 3.0, 30.0]],
         [[-0.1, 0.0, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, -0.1]],
         (1.5, -2.0, 0.8), True, False),
    ]

    out = []
    for name, q_raw, w, imat, idot, tau, qde, wde in cases:
        q = [float(c) for c in qnormalized([mp.mpf(c) for c in q_raw])]
        qm = [mp.mpf(c) for c in q]
        wm = v3(w)
        im = mat3(imat)
        idm = mat3(idot)
        tm = v3(tau)
        qd = qdot_mp(qm, wm)
        wd = wdot_mp(wm, im, idm, tm)
        if qde:
            assert all(float(c) == c for c in qd), (name, qd)
        if wde:
            assert all(c == 0 for c in wd), (name, wd)
        out.append({
            "name": name,
            "q_i2b_wxyz": hxv(q),
            "w_b_radps": hxv(w),
            "i_kgm2": hxm(imat),
            "idot_kgm2ps": hxm(idot),
            "tau_b_nm": hxv(tau),
            "qdot_ref": hxv(qd),
            "wdot_ref_radps2": hxv(wd),
            "qdot_exact": "true" if qde else "false",
            "wdot_exact": "true" if wde else "false",
        })
        print(f"rhs {name:24s} |wdot|={mp.nstr(vnorm(wd), 8)}")

    emit(
        HERE / "rhs.toml",
        "Rigid-body attitude RHS golden vectors (FR-1).\n"
        "qdot_ref is the quaternion kinematics 1/2 q (x) [0, w] (chapter\n"
        "ch:rigidbody, eq:rigidbody:qdot; Hamilton product, scalar-first,\n"
        "q_i2b per ch:notation) and wdot_ref_radps2 is Euler's equation\n"
        "with time-varying inertia I wdot = tau - w x (I w) - Idot w\n"
        "(eq:rigidbody:euler), both evaluated with mpmath at 60\n"
        "significant decimal digits from the exact binary64 inputs and\n"
        "rounded once to binary64. Matrices are row-major 9-vectors.\n"
        "qdot_exact / wdot_exact mark outputs whose values are exact in\n"
        "binary64 (structural zeros and exact products); the consuming\n"
        "test requires bit equality there. Provenance and tolerances in\n"
        "manifest.toml. Regenerated by generate.py.",
        out,
    )


# ---------------------------------------------------------------------------
# File 2: gravity-gradient torques
# ---------------------------------------------------------------------------


def gen_gravgrad() -> None:
    ifull = [[110.0, -5.0, 3.0], [-5.0, 95.0, -4.0], [3.0, -4.0, 82.0]]
    ilib = [[120.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 80.0]]

    # The planar pitch case: vehicle on inertial +x at radius r, body built
    # from the C0 base orientation (x_b = +y_I, y_b = +z_I, z_b = +x_I)
    # pitched by theta about y_b. Committed q is the binary64 rounding of
    # the exact construction; the closed-form cross-check below therefore
    # holds to the rounding of q, not to working precision.
    theta_pitch = 0.15
    r_lib = 7.0e6
    th = mp.mpf(theta_pitch)
    # q_i2b for C_I^B = C0 R3(theta): C0 = R3(+90 deg about z) then
    # R1(+90 deg)? Constructed directly as a quaternion product of
    # elementary frame rotations: C0 maps (x,y,z)_I to (y,z,x)_I-aligned
    # body axes; its quaternion (w >= 0) is 0.5*(1,1,1,1) exactly, verified
    # below against the row construction.
    q_c0 = [mp.mpf("0.5"), mp.mpf("0.5"), mp.mpf("0.5"), mp.mpf("0.5")]
    c0 = dcm_from_quat(q_c0)
    c0_expect = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    assert all(abs(c0[i][j] - c0_expect[i][j]) < mp.mpf(10) ** -55
               for i in range(3) for j in range(3))
    # C_I^B = C0 R3(theta) composes as q_i2b = qz(theta) (x) q_c0: the
    # rotation applied first (R3 about inertial z) is the leftmost factor
    # per eq:notation:quatcomp.
    q_pitch = qmul(qz(th), q_c0)
    q_pitch64 = [float(c) for c in q_pitch]

    cases = [
        ("leo_generic_full_tensor", MU_EARTH, (5.1e6, -3.2e6, 2.4e6),
         (0.8, -0.3, 0.4, 0.1), ifull, False),
        ("lunar_orbit_diag", MU_MOON, (1.9e6, 0.4e6, -0.7e6),
         (0.2, 0.9, -0.1, 0.35), ilib, False),
        ("geo_far_field", MU_EARTH, (2.9e7, 2.1e7, 1.9e7),
         (0.5, -0.6, 0.3, -0.4), ifull, False),
        # Spherical inertia with power-of-two scale: I rhat = 2 rhat
        # exactly, and the cross product subtracts identically rounded
        # copies of the same real products - exactly zero for ANY
        # attitude and position.
        ("spherical_inertia_zero", MU_EARTH, (4.0e6, 3.0e6, 5.0e6),
         (0.7, 0.1, -0.6, 0.2),
         [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]], True),
        # Identity attitude with r along +z_I: rhat_b = (0, 0, 1) exactly,
        # I rhat_b = (0, 0, I_z), and every cross-product term carries a
        # zero factor - exactly zero.
        ("principal_axis_zero", MU_EARTH, (0.0, 0.0, 7.2e6),
         (1.0, 0.0, 0.0, 0.0),
         [[9.0, 0.0, 0.0], [0.0, 7.0, 0.0], [0.0, 0.0, 5.0]], True),
        ("pitch_offset_planar", MU_EARTH, (r_lib, 0.0, 0.0),
         tuple(q_pitch64), ilib, False),
    ]

    out = []
    for name, mu, r_i, q_raw, imat, exact in cases:
        if name == "pitch_offset_planar":
            q = list(q_raw)  # already the rounded exact construction
        else:
            q = [float(c) for c in qnormalized([mp.mpf(c) for c in q_raw])]
        tau = gravgrad_mp(mu, r_i, [mp.mpf(c) for c in q], mat3(imat))
        if exact:
            assert all(c == 0 for c in tau), (name, tau)
        out.append({
            "name": name,
            "mu_m3ps2": hx(mu),
            "r_i_m": hxv(r_i),
            "q_i2b_wxyz": hxv(q),
            "i_kgm2": hxm(imat),
            "tau_ref_nm": hxv(tau),
            "exact_zero": "true" if exact else "false",
        })
        print(f"gravgrad {name:26s} |tau|={mp.nstr(vnorm(tau), 8)} N m")

    # Cross-check the planar case against the closed-form pitch torque
    # (eq:gravgrad:pitch): tau = -3 n^2 (I_x - I_z) sin(th) cos(th) y_b,
    # evaluated from the same committed binary64 q (hence the 1e-13
    # tolerance: the committed q carries its own rounding).
    n2 = mp.mpf(MU_EARTH) / mp.mpf(r_lib) ** 3
    tau_planar = gravgrad_mp(MU_EARTH, (r_lib, 0.0, 0.0),
                             [mp.mpf(c) for c in q_pitch64], mat3(ilib))
    tau_closed = -3 * n2 * (mp.mpf(120) - mp.mpf(80)) * mp.sin(th) * \
        mp.cos(th)
    assert abs(tau_planar[0]) < abs(tau_closed) * mp.mpf(10) ** -13
    assert abs(tau_planar[2]) < abs(tau_closed) * mp.mpf(10) ** -13
    assert abs(tau_planar[1] - tau_closed) < abs(tau_closed) * \
        mp.mpf(10) ** -13

    emit(
        HERE / "gravgrad.toml",
        "Gravity-gradient torque golden vectors (FR-1).\n"
        "tau_ref_nm is (3 mu / r^3) rhat_b x (I rhat_b) with rhat_b the\n"
        "unit central-body-to-vehicle vector expressed in the body frame\n"
        "through q_i2b (chapter ch:gravgrad, eq:gravgrad:torque),\n"
        "evaluated with mpmath at 60 significant decimal digits from the\n"
        "exact binary64 inputs and rounded once to binary64. Matrices are\n"
        "row-major 9-vectors. exact_zero cases are structural zeros the\n"
        "consuming test requires bit-exactly. Provenance and tolerances\n"
        "in manifest.toml. Regenerated by generate.py.",
        out,
    )


# ---------------------------------------------------------------------------
# File 3: torque-free axisymmetric coning checkpoints
# ---------------------------------------------------------------------------


def gen_coning() -> None:
    it, ia = 100.0, 60.0
    w0 = (0.15, 0.05, 0.25)
    q0_raw = (0.9, -0.2, 0.15, 0.35)
    q0 = [float(c) for c in qnormalized([mp.mpf(c) for c in q0_raw])]
    checkpoints = [18.0, 36.0, 54.0, 72.0, 90.0, 108.0, 126.0, 144.0]

    sol = ConingSolution(q0, w0, it, ia)
    imat = mat3([[it, 0, 0], [0, it, 0], [0, 0, ia]])
    coning_self_check(sol, imat, checkpoints)

    period_prec = 2 * mp.pi / sol.phidot
    print(f"coning: |H|={mp.nstr(sol.h, 12)} kg m^2/s, "
          f"theta={mp.nstr(sol.theta, 12)} rad, "
          f"precession period={mp.nstr(period_prec, 12)} s, "
          f"span/period={mp.nstr(mp.mpf(checkpoints[-1]) / period_prec, 6)}")

    out = [{
        "name": "definition",
        "it_kgm2": hx(it),
        "ia_kgm2": hx(ia),
        "q0_i2b_wxyz": hxv(q0),
        "w0_b_radps": hxv(w0),
        "h_kgm2ps": hx(sol.h),
        "nutation_theta_rad": hx(sol.theta),
        "precession_rate_radps": hx(sol.phidot),
        "relative_spin_rate_radps": hx(sol.psidot),
        "precession_period_s": hx(period_prec),
        "precession_period_decimal": mp.nstr(period_prec, 17),
    }]
    for t in checkpoints:
        q = sol.q_ref(t)
        w = sol.w_ref(t)
        out.append({
            "name": f"checkpoint_{int(t):03d}",
            "t_s": hx(t),
            "q_ref_i2b_wxyz": hxv(q),
            "w_ref_b_radps": hxv(w),
        })

    emit(
        HERE / "coning.toml",
        "Torque-free axisymmetric (coning) closed-form checkpoints (FR-1,\n"
        "Phase 4 exit criterion 4). The committed q0/w0 doubles ARE the\n"
        "motion: every reference value is the closed-form solution of\n"
        "chapter ch:rigidbody (eq:rigidbody:coning) evaluated from them\n"
        "with mpmath at 60 significant decimal digits and rounded once to\n"
        "binary64. The generator verifies the closed form against the\n"
        "ratified kinematics (eq:rigidbody:qdot) and dynamics\n"
        "(eq:rigidbody:euler) by extended-precision finite differences\n"
        "before writing. The span covers five body-precession periods.\n"
        "Provenance and tolerances in manifest.toml. Regenerated by\n"
        "generate.py.",
        out,
    )


# ---------------------------------------------------------------------------
# File 4: intermediate-axis (Dzhanibekov) flip references
# ---------------------------------------------------------------------------


def gen_dzhanibekov() -> None:
    i_diag = (3.0, 2.0, 1.0)
    w0 = (1.0e-4, 0.45, 0.0)
    t_span = 120.0
    checkpoints = [5.0, 10.0, 15.0]

    i1, i2, i3 = [mp.mpf(c) for c in i_diag]
    w0m = v3(w0)
    imat = mat3([[i_diag[0], 0, 0], [0, i_diag[1], 0], [0, 0, i_diag[2]]])

    # Conserved references from the exact binary64 initial state with the
    # identity initial attitude: H^I(0) = I w0 (exact products of the
    # committed doubles, rounded once), T(0) = 1/2 w0 . I w0.
    h0 = matvec(imat, w0m)
    h0_mag = vnorm(h0)
    t0 = sum(w0m[i] * h0[i] for i in range(3)) / 2

    # Linearized growth rate about the intermediate-axis spin
    # (eq:rigidbody:lambda): lambda = w2 sqrt((I2-I3)(I1-I2)/(I1 I3)).
    lam = w0m[1] * mp.sqrt((i2 - i3) * (i1 - i2) / (i1 * i3))

    # Independent trajectory reference: mpmath Taylor-series integration
    # (mp.odefun) of Euler's equations at 50 dps. Early times only: state
    # error grows as exp(lambda t), so binary64 checkpoints stay
    # referenceable to ~1e-12 relative for t <= 15 s but not much beyond.
    mp.mp.dps = 50

    def euler_rhs(t, w):
        return [(i2 - i3) / i1 * w[1] * w[2],
                (i3 - i1) / i2 * w[2] * w[0],
                (i1 - i2) / i3 * w[0] * w[1]]

    w_fn = mp.odefun(euler_rhs, 0, [mp.mpf(c) for c in w0], tol=mp.mpf(10)
                     ** -40)
    refs = [w_fn(t) for t in checkpoints]
    # Generation self-checks: the reference trajectory conserves H and T,
    # and the flip has occurred by t = 45 s (w2 fully reversed).
    for t, w in zip(checkpoints, refs):
        hw = matvec(imat, w)
        assert abs(vnorm(hw) - h0_mag) / h0_mag < mp.mpf(10) ** -35
        tk = sum(w[i] * hw[i] for i in range(3)) / 2
        assert abs(tk - t0) / t0 < mp.mpf(10) ** -35
    w45 = w_fn(45.0)
    assert w45[1] < mp.mpf("-0.4"), w45
    print(f"dzhanibekov: lambda={mp.nstr(lam, 10)} 1/s, "
          f"w2(45 s)={mp.nstr(w45[1], 10)} rad/s (flip confirmed)")
    mp.mp.dps = 60

    out = [{
        "name": "definition",
        "i_diag_kgm2": hxv(i_diag),
        "w0_b_radps": hxv(w0),
        "t_span_s": hx(t_span),
        "h0_i_kgm2ps": hxv(h0),
        "h0_mag_kgm2ps": hx(h0_mag),
        "t0_j": hx(t0),
        "growth_rate_1ps": hx(lam),
        "growth_rate_decimal": mp.nstr(lam, 17),
    }]
    for t, w in zip(checkpoints, refs):
        out.append({
            "name": f"checkpoint_{int(t):03d}",
            "t_s": hx(t),
            "w_ref_b_radps": hxv(w),
        })

    emit(
        HERE / "dzhanibekov.toml",
        "Intermediate-axis (Dzhanibekov) flip references (FR-1, Phase 4\n"
        "exit criterion 4). The body I = diag(3, 2, 1) kg m^2 spins about\n"
        "its intermediate axis at 0.45 rad/s with a 1e-4 rad/s\n"
        "perturbation on axis 1 and the identity initial attitude.\n"
        "h0/t0 are the conserved angular momentum and kinetic energy from\n"
        "the exact binary64 initial state (mpmath, 60 digits, rounded\n"
        "once); w_ref checkpoints are an independent mpmath Taylor-series\n"
        "(mp.odefun, 50 dps, tol 1e-40) integration of Euler's equations\n"
        "(eq:rigidbody:euler). Checkpoints stop at 15 s because\n"
        "near-separatrix motion amplifies state differences as\n"
        "exp(lambda t) (chapter ch:rigidbody, eq:rigidbody:lambda);\n"
        "beyond that horizon the conserved quantities are the reference.\n"
        "Provenance and tolerances in manifest.toml. Regenerated by\n"
        "generate.py.",
        out,
    )


# ---------------------------------------------------------------------------
# File 5: gravity-gradient libration frequency
# ---------------------------------------------------------------------------


def gen_libration() -> None:
    mu = MU_EARTH
    r = 7.0e6
    i_diag = (120.0, 100.0, 80.0)  # (I_x along-track, I_y pitch, I_z nadir)
    theta0 = 0.01
    t_span = 65000.0

    mum = mp.mpf(mu)
    rm = mp.mpf(r)
    ix, iy, iz = [mp.mpf(c) for c in i_diag]
    th0 = mp.mpf(theta0)

    n = mp.sqrt(mum / rm ** 3)
    w_lib = n * mp.sqrt(3 * (ix - iz) / iy)  # eq:gravgrad:libfreq
    # Finite-amplitude pendulum correction (eq:gravgrad:pendulum): the
    # planar pitch equation is u'' + w_lib^2 sin u = 0 in u = 2 theta, so
    # the oscillation frequency at amplitude u0 = 2 theta0 is
    # w_lib * pi / (2 K(m)) with m = sin^2(u0/2) = sin^2(theta0)
    # (mpmath ellipk takes the parameter m = k^2).
    assert abs(mp.ellipk(0) - mp.pi / 2) < mp.mpf(10) ** -55
    w_pend = w_lib * mp.pi / (2 * mp.ellipk(mp.sin(th0) ** 2))
    assert abs(w_pend / w_lib - 1) < mp.mpf(10) ** -4  # tiny at 0.01 rad

    print(f"libration: n={mp.nstr(n, 12)} rad/s, "
          f"w_lib={mp.nstr(w_lib, 12)} rad/s, "
          f"pendulum correction={mp.nstr(w_pend / w_lib - 1, 6)}, "
          f"period={mp.nstr(2 * mp.pi / w_lib, 10)} s")

    out = [{
        "name": "definition",
        "mu_m3ps2": hx(mu),
        "r_m": hx(r),
        "i_diag_kgm2": hxv(i_diag),
        "theta0_rad": hx(theta0),
        "t_span_s": hx(t_span),
        "n_radps": hx(n),
        "omega_lib_radps": hx(w_lib),
        "omega_lib_decimal": mp.nstr(w_lib, 17),
        "omega_lib_pendulum_radps": hx(w_pend),
        "omega_lib_pendulum_decimal": mp.nstr(w_pend, 17),
    }]

    emit(
        HERE / "libration.toml",
        "Gravity-gradient pitch libration references (FR-1, Phase 4 exit\n"
        "criterion 9). omega_lib_radps is the analytic small-angle pitch\n"
        "libration frequency n sqrt(3 (I_x - I_z) / I_y) of a rigid body\n"
        "on a prescribed circular orbit of radius r_m about a point-mass\n"
        "mu (chapter ch:gravgrad, eq:gravgrad:libfreq), with n the mean\n"
        "motion sqrt(mu/r^3); omega_lib_pendulum_radps applies the exact\n"
        "finite-amplitude pendulum correction pi/(2 K(sin^2 theta0))\n"
        "(eq:gravgrad:pendulum) for the committed initial pitch offset\n"
        "theta0_rad. All values evaluated with mpmath at 60 significant\n"
        "decimal digits from the exact binary64 inputs and rounded once\n"
        "to binary64. mu is the IERS Conventions (2010) GM_earth carried\n"
        "by star/constants.hpp; the consuming test asserts the equality.\n"
        "Provenance and tolerances in manifest.toml. Regenerated by\n"
        "generate.py.",
        out,
    )


def main() -> None:
    gen_rhs()
    gen_gravgrad()
    gen_coning()
    gen_dzhanibekov()
    gen_libration()
    print(f"attitude goldens regenerated; mpmath {mp.__version__} at "
          f"{mp.mp.dps} dps")


if __name__ == "__main__":
    main()
