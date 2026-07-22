"""Evaluate each candidate mechanism's predicted magnitude on the real run.

Attribution needs every candidate to be costed, not just the favoured one. Each
routine here computes what one mechanism actually contributes on the reference
scenario's own trajectory, so a candidate that predicts a thousandth of the
measured excess can be discarded on arithmetic rather than on argument.

The candidates, and what is computed for each:

``mechanization``
    The filter integrates the central-body gravity it does not measure with
    its own quadrature (``ekf.cpp``, eq:ekf:mech), while the truth trajectory
    is an RK4 solution of the same dynamics. Any difference is a deterministic
    truncation error that no term of the filter's process noise describes.
    Measured directly by running the filter's mechanization on a noise-free
    input and differencing against truth.

    Both quadratures are reported: the ``euler`` scheme is the first-order
    step this diagnosis attributed the NEES excess to, and ``heun`` is the
    second-order predictor-corrector that replaced it
    (ch:ekf, sec:ekf:gravityorder). Keeping the superseded scheme here is
    what makes the row a measurement of the improvement rather than an
    assertion about it.

``transition``
    ``Phi = I + F dt`` (eq:ekf:disc) truncates the matrix exponential. Measured
    as the covariance the truncated and exact transitions produce from the same
    starting covariance, expressed in the units the NEES gate sees.

``process_noise``
    ``Q_k = Gamma Q_c Gamma' dt`` (eq:ekf:G) truncates the van Loan integral.
    Measured against the exact ``integral_0^dt Phi(s) G Q_c G' Phi(s)' ds``
    evaluated by the van Loan matrix-exponential construction.

``reset`` and ``theta_reduction``
    Bounded from the run's own realized angles rather than assumed small.

All five are reported in a common unit: the change they induce in
``trace(P_reported^-1 P_true) - 15``, which is what an ensemble NEES measures.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
# star_reacher is imported from the INSTALLED package, never from
# REPO_ROOT/python: the source tree carries no compiled _core, so putting
# it on sys.path shadows the wheel and makes every core-backed call in
# these diagnostics fail with CoreMissingError.

MU_EARTH = 3.986004418e14  # constants.hpp GM_EARTH_M3_PER_S2


def skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
    )


def dcm_from_quat(q: np.ndarray) -> np.ndarray:
    """Inertial-to-body DCM from a Hamilton scalar-first quaternion (D-7)."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)],
            [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)],
            [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def gravity(p: np.ndarray, mu: float = MU_EARTH) -> np.ndarray:
    r = float(np.linalg.norm(p))
    return -mu * p / (r ** 3)


def gravity_gradient(p: np.ndarray, mu: float = MU_EARTH) -> np.ndarray:
    r = float(np.linalg.norm(p))
    u = p / r
    return (mu / r ** 3) * (3.0 * np.outer(u, u) - np.eye(3))


def mechanization_truncation(
    r0: np.ndarray,
    v0: np.ndarray,
    dt: float,
    n_steps: int,
    mu: float = MU_EARTH,
    scheme: str = "heun",
) -> tuple[np.ndarray, np.ndarray]:
    """Run the filter's gravity mechanization noise-free; return its states.

    Exactly the arithmetic of ``ErrorStateEkf::propagate`` with the IMU
    increments set to zero, which is what the reference scenario's coasting
    vehicle measures up to sensor noise.

    ``heun`` is the shipped second-order predictor-corrector of eq:ekf:mech.
    ``euler`` is the superseded first-order velocity step, retained so the
    two can be measured side by side on the same trajectory.
    """
    p = np.array(r0, dtype=float)
    v = np.array(v0, dtype=float)
    ps = np.empty((n_steps + 1, 3))
    vs = np.empty((n_steps + 1, 3))
    ps[0], vs[0] = p, v
    for k in range(n_steps):
        if scheme == "euler":
            v_new = v + gravity(p, mu) * dt
        elif scheme == "heun":
            g0 = gravity(p, mu)
            v_pred = v + g0 * dt
            p_pred = p + 0.5 * (v + v_pred) * dt
            v_new = v + 0.5 * (g0 + gravity(p_pred, mu)) * dt
        else:
            raise ValueError("unknown scheme %r" % scheme)
        p = p + 0.5 * (v + v_new) * dt
        v = v_new
        ps[k + 1], vs[k + 1] = p, v
    return ps, vs


def rk4_reference(
    r0: np.ndarray, v0: np.ndarray, dt: float, n_steps: int, mu: float = MU_EARTH
) -> tuple[np.ndarray, np.ndarray]:
    """The truth propagator's scheme: classical RK4 on the same point mass."""

    def deriv(state):
        return np.concatenate([state[3:], gravity(state[:3], mu)])

    y = np.concatenate([np.asarray(r0, float), np.asarray(v0, float)])
    ps = np.empty((n_steps + 1, 3))
    vs = np.empty((n_steps + 1, 3))
    ps[0], vs[0] = y[:3], y[3:]
    for k in range(n_steps):
        k1 = deriv(y)
        k2 = deriv(y + 0.5 * dt * k1)
        k3 = deriv(y + 0.5 * dt * k2)
        k4 = deriv(y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        ps[k + 1], vs[k + 1] = y[:3], y[3:]
    return ps, vs


def expm(a: np.ndarray, terms: int = 30) -> np.ndarray:
    """Matrix exponential by scaling-and-squaring with a Taylor core.

    The project is scipy-free, and the matrices here are small and of modest
    norm once scaled, so a Taylor series on ``a / 2^s`` followed by ``s``
    squarings is accurate to machine precision without an external dependency.
    """
    norm = float(np.max(np.sum(np.abs(a), axis=1)))
    squarings = max(0, int(np.ceil(np.log2(norm))) + 1) if norm > 0.5 else 0
    scaled = a / (2.0 ** squarings)
    result = np.eye(a.shape[0])
    term = np.eye(a.shape[0])
    for k in range(1, terms + 1):
        term = term @ scaled / k
        result = result + term
    for _ in range(squarings):
        result = result @ result
    return result


def van_loan(f: np.ndarray, qc: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact discrete (Phi, Q_d) by the van Loan matrix-exponential identity.

    Builds the 2n x 2n block matrix [[-F, Q_c], [0, F']] * dt; its exponential
    carries Phi in the lower-right block and Phi^-1 Q_d in the upper-right, so
    Q_d = Phi * (upper-right). Van Loan, "Computing integrals involving the
    matrix exponential", IEEE Trans. Automat. Contr. 23(3), 1978.
    """
    n = f.shape[0]
    block = np.zeros((2 * n, 2 * n))
    block[:n, :n] = -f * dt
    block[:n, n:] = qc * dt
    block[n:, n:] = f.T * dt
    e = expm(block)
    phi = e[n:, n:].T
    return phi, phi @ e[:n, n:]


def build_f_and_qc(
    q_hat: np.ndarray,
    p_pre: np.ndarray,
    omega_avg: np.ndarray,
    f_avg: np.ndarray,
    sensors: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """The reference filter's F (eq:ekf:F) and the continuous drive Q_c."""
    c_hat = dcm_from_quat(q_hat).T  # body -> inertial
    f = np.zeros((15, 15))
    i3 = np.eye(3)
    f[0:3, 0:3] = -skew(omega_avg)
    f[0:3, 9:12] = -i3
    f[3:6, 0:3] = -c_hat @ skew(f_avg)
    f[3:6, 6:9] = gravity_gradient(p_pre)
    f[3:6, 12:15] = -c_hat
    f[6:9, 3:6] = i3
    if sensors["gyro_tau_s"] > 0.0:
        f[9:12, 9:12] = -i3 / sensors["gyro_tau_s"]
    if sensors["accel_tau_s"] > 0.0:
        f[12:15, 12:15] = -i3 / sensors["accel_tau_s"]

    qc = np.zeros((15, 15))
    qc[0:3, 0:3] = sensors["gyro_arw"] ** 2 * i3
    qc[3:6, 3:6] = sensors["accel_vrw"] ** 2 * (c_hat @ c_hat.T)
    if sensors["gyro_tau_s"] > 0.0:
        qc[9:12, 9:12] = (
            2.0 * sensors["gyro_gm_sigma"] ** 2 / sensors["gyro_tau_s"] * i3
        )
    if sensors["accel_tau_s"] > 0.0:
        qc[12:15, 12:15] = (
            2.0 * sensors["accel_gm_sigma"] ** 2 / sensors["accel_tau_s"] * i3
        )
    return f, qc


REFERENCE_SENSORS = {
    "gyro_arw": 1.0e-5,
    "accel_vrw": 1.0e-4,
    "gyro_gm_sigma": 1.0759973046695306e-7,
    "accel_gm_sigma": 1.0759973046695306e-5,
    "gyro_tau_s": 100.0,
    "accel_tau_s": 100.0,
}


def load_raw(path: Path, index: int = 0) -> dict:
    data = np.load(path)
    prefix = "raw%02d_" % index
    out = {}
    for key in data.files:
        if key.startswith(prefix):
            out[key[len(prefix):]] = data[key]
    out["t_s"] = data["t_s"]
    return out


def report_mechanization(raw: dict, dt: float) -> None:
    truth_r = raw["truth_r_m"]
    truth_v = raw["truth_v_mps"]
    n = truth_r.shape[0] - 1
    sigma = REFERENCE_SENSORS["accel_gm_sigma"]
    print("  filter mechanization vs truth, noise-free, from the true state:")
    finals: dict[str, float] = {}
    for scheme in ("euler", "heun"):
        ps, vs = mechanization_truncation(
            truth_r[0], truth_v[0], dt, n, scheme=scheme
        )
        dv = vs - truth_v
        dp = ps - truth_r
        label = "%s (superseded)" % scheme if scheme == "euler" else scheme
        print("    %s:" % label)
        for frac in (0.25, 0.5, 1.0):
            k = int(frac * n)
            print(
                "      t = %6.1f s   |dv| = %.4e m/s   |dp| = %.4e m"
                % (raw["t_s"][k], np.linalg.norm(dv[k]), np.linalg.norm(dp[k]))
            )
        # The equivalent constant acceleration the truncation looks like,
        # which is what the accelerometer-bias state would have to absorb.
        final = float(np.linalg.norm(dv[n]))
        finals[scheme] = final
        a_eff = final / raw["t_s"][n]
        print(
            "      equivalent constant acceleration %.4e m/s^2 = %.4g x the "
            "accel bias 1-sigma %.4e m/s^2" % (a_eff, a_eff / sigma, sigma)
        )
    if finals["heun"] > 0.0:
        print(
            "    shipped scheme improves the 60 s velocity truncation by "
            "%.0fx" % (finals["euler"] / finals["heun"])
        )


def report_discretization(raw: dict, dt: float, stride: int = 50) -> None:
    """Per-step covariance error of the truncated Phi and Q, in NEES units."""
    x_hat = raw["nav.est_x_hat"]
    p_log = raw["nav.est_P"]
    imu_dtheta = raw["sensors.imu_dtheta_b_rad"]
    imu_dv = raw["sensors.imu_dv_b_mps"]
    from star_reacher.consistency import unpack_symmetric

    worst_phi = 0.0
    worst_q = 0.0
    for k in range(1, x_hat.shape[0], stride):
        q_hat = x_hat[k - 1, 0:4]
        v_pre = x_hat[k - 1, 4:7]
        p_pre = x_hat[k - 1, 7:10]
        bg = x_hat[k - 1, 10:13]
        ba = x_hat[k - 1, 13:16]
        dtheta = imu_dtheta[k - 1] - bg * dt
        dvec = imu_dv[k - 1] - ba * dt
        f, qc = build_f_and_qc(
            q_hat, p_pre, dtheta / dt, dvec / dt, REFERENCE_SENSORS
        )
        p_prev = unpack_symmetric(p_log[k - 1])

        phi_first = np.eye(15) + f * dt
        q_first = qc * dt
        phi_exact, q_exact = van_loan(f, qc, dt)

        p_first = phi_first @ p_prev @ phi_first.T + q_first
        p_phi_only = phi_exact @ p_prev @ phi_exact.T + q_first
        p_both = phi_exact @ p_prev @ phi_exact.T + q_exact

        # trace(P_approx^-1 P_exact) - 15 is the NEES excess the covariance
        # error would produce if it were the only defect at this epoch.
        def nees_units(p_approx, p_true):
            chol = np.linalg.cholesky(p_approx)
            w = np.linalg.solve(chol, p_true)
            w = np.linalg.solve(chol, w.T)
            return float(np.trace(w)) - 15.0

        worst_phi = max(worst_phi, abs(nees_units(p_first, p_phi_only)))
        worst_q = max(worst_q, abs(nees_units(p_phi_only, p_both)))
        del v_pre
    print("  per-propagation-step covariance error, in NEES units:")
    print("    truncated Phi = I + F dt   : %.3e per step" % worst_phi)
    print("    truncated Q = G Qc G' dt   : %.3e per step" % worst_q)
    print(
        "    over %d steps, if fully coherent: Phi %.3e, Q %.3e"
        % (
            x_hat.shape[0] - 1,
            worst_phi * (x_hat.shape[0] - 1),
            worst_q * (x_hat.shape[0] - 1),
        )
    )


def report_reset_and_reduction(raw: dict) -> None:
    """Bound the two small-angle candidates from the run's realized angles."""
    innov = raw.get("nav.innov_y")
    sensor = raw.get("nav.innov_sensor_id")
    if innov is not None and sensor is not None:
        star = innov[sensor == 1][:, :3]
        max_angle = float(np.max(np.linalg.norm(star, axis=1)))
        print(
            "  star-tracker innovation max |y| = %.3e rad; the omitted reset "
            "transport I - 0.5 [dtheta x] departs from identity by <= %.3e"
            % (max_angle, 0.5 * max_angle)
        )
    e = raw["nav.err_e"]
    dq_w = e[:, 0]
    dq_v = e[:, 1:4]
    angle = 2.0 * np.arcsin(np.clip(np.linalg.norm(dq_v, axis=1), -1.0, 1.0))
    max_theta = float(np.max(angle))
    # dtheta = 2 sgn(dq_w) dq_v reproduces theta with relative error
    # theta^2/24 + O(theta^4), from 2 sin(theta/2) vs theta.
    print(
        "  attitude error max |theta| = %.3e rad; the 2 sgn(dq_w) dq_v "
        "reduction is short by a relative %.3e"
        % (max_theta, max_theta ** 2 / 24.0)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures", default=str(REPO_ROOT / "fixtures" / "nees_diag")
    )
    parser.add_argument("--variant", default="base")
    parser.add_argument("--dt", type=float, default=0.1)
    args = parser.parse_args(argv)

    raw = load_raw(Path(args.fixtures) / ("%s.npz" % args.variant))
    print("=" * 72)
    print("candidate magnitudes on %s (dt = %g s)" % (args.variant, args.dt))
    report_mechanization(raw, args.dt)
    report_discretization(raw, args.dt)
    report_reset_and_reduction(raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
