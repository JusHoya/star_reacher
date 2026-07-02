"""Regenerate the integrator/event golden-vector files in this directory.

The values are produced by an independent pure-Python implementation of
classical elliptic two-body propagation (no numpy, no project code):

- Orbit definition: classical orbital elements are converted to an inertial
  state vector once (Vallado, Fundamentals of Astrodynamics and Applications,
  4th ed., ch. 2, COE-to-RV); the resulting binary64 position and velocity
  are then THE definition of the reference orbit. Every downstream value is
  derived from those committed doubles, never from the elements, so the C++
  reference propagator and this script cannot disagree about the input.
- Kepler propagation: eccentric-anomaly form of Kepler's equation
  M = E - e sin E solved by Newton's method to machine precision, state
  reconstructed in the perifocal frame (Vallado, 4th ed., ch. 2, Kepler's
  equation and Kepler's problem).
- Apsis passage times: periapsis at mean anomaly M = 0 (mod 2 pi) and
  apoapsis at M = pi (mod 2 pi), so t = (2 pi k - M0)/n and
  t = (pi + 2 pi k - M0)/n with n the mean motion and M0 the mean anomaly of
  the committed initial state (Vallado, 4th ed., ch. 2).
- max |d^4 r/dt^4|: sampled over one period from the analytic solution by
  5-point central finite differences; feeds the cubic-Hermite dense-output
  error bound err <= (h^4/384) max|y''''| (derived in the integrators
  chapter of the math library).

Self-checks run on every regeneration: Kepler residuals at machine level,
specific-energy and |h| invariance of every checkpoint state, one-period
closure, and |r| and r.v consistency at every apsis time. A failed check
aborts without writing.

Running this script rewrites the .toml golden files byte-identically on the
platform that generated them; cross-platform reruns may differ at the last
ulp through libm, which is why the consuming tests compare within the
tolerances recorded in manifest.toml instead of bitwise.
"""

from __future__ import annotations

import math
import pathlib

HERE = pathlib.Path(__file__).resolve().parent

# Geocentric gravitational parameter [m^3/s^2], IERS Conventions (2010),
# TN No. 36, Table 1.1 -- same source and value as star/constants.hpp.
MU_EARTH = 3.986004418e14

# Reference orbit (LEO-class eccentric, per the Phase 2 exit criterion 5
# prescription e ~ 0.1-0.3): a = 8000 km, e = 0.15 puts periapsis at
# 6800 km radius (~422 km altitude) and apoapsis at 9200 km. Nonzero
# inclination/RAAN/argument/anomaly exercise all six state components.
SMA_M = 8000.0e3
ECC = 0.15
INC_RAD = math.radians(30.0)
RAAN_RAD = math.radians(40.0)
ARGP_RAD = math.radians(60.0)
NU0_RAD = math.radians(120.0)

# Span for the apsis-event golden: 3.2 periods lands between apsis passages
# (apsis times sit near multiples of T/2 offset by M0/n), so no committed
# event time is close enough to the boundary for step truncation to matter.
APSIS_SPAN_PERIODS = 3.2

N_CHECKPOINTS = 8  # analytic states at t = j*(T/8), j = 1..8


# --------------------------------------------------------------------------
# Small vector helpers (pure Python floats; explicit, fixed evaluation order)
# --------------------------------------------------------------------------


def dot(u: list[float], v: list[float]) -> float:
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def cross(u: list[float], v: list[float]) -> list[float]:
    return [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]


def norm(u: list[float]) -> float:
    return math.sqrt(dot(u, u))


def scale(u: list[float], s: float) -> list[float]:
    return [u[0] * s, u[1] * s, u[2] * s]


def add(u: list[float], v: list[float]) -> list[float]:
    return [u[0] + v[0], u[1] + v[1], u[2] + v[2]]


def sub(u: list[float], v: list[float]) -> list[float]:
    return [u[0] - v[0], u[1] - v[1], u[2] - v[2]]


# --------------------------------------------------------------------------
# Orbit definition: COE -> RV, evaluated exactly once
# --------------------------------------------------------------------------


def coe_to_rv() -> tuple[list[float], list[float]]:
    """Classical elements to GCRF state (Vallado, 4th ed., ch. 2)."""
    p = SMA_M * (1.0 - ECC * ECC)
    r_pf_mag = p / (1.0 + ECC * math.cos(NU0_RAD))
    r_pf = [r_pf_mag * math.cos(NU0_RAD), r_pf_mag * math.sin(NU0_RAD), 0.0]
    v_scale = math.sqrt(MU_EARTH / p)
    v_pf = [-v_scale * math.sin(NU0_RAD), v_scale * (ECC + math.cos(NU0_RAD)), 0.0]

    co, so = math.cos(RAAN_RAD), math.sin(RAAN_RAD)
    ci, si = math.cos(INC_RAD), math.sin(INC_RAD)
    cw, sw = math.cos(ARGP_RAD), math.sin(ARGP_RAD)
    # Perifocal -> inertial rotation R3(-RAAN) R1(-i) R3(-argp), written out.
    m = [
        [co * cw - so * sw * ci, -co * sw - so * cw * ci, so * si],
        [so * cw + co * sw * ci, -so * sw + co * cw * ci, -co * si],
        [sw * si, cw * si, ci],
    ]

    def rot(u: list[float]) -> list[float]:
        return [
            m[0][0] * u[0] + m[0][1] * u[1] + m[0][2] * u[2],
            m[1][0] * u[0] + m[1][1] * u[1] + m[1][2] * u[2],
            m[2][0] * u[0] + m[2][1] * u[1] + m[2][2] * u[2],
        ]

    return rot(r_pf), rot(v_pf)


# --------------------------------------------------------------------------
# Analytic elliptic propagation from the committed doubles
# --------------------------------------------------------------------------


class EllipticOrbit:
    """Elliptic two-body orbit defined by an inertial state (Vallado ch. 2)."""

    def __init__(self, mu: float, r0: list[float], v0: list[float]) -> None:
        self.mu = mu
        r0n = norm(r0)
        v0sq = dot(v0, v0)
        alpha = 2.0 / r0n - v0sq / mu  # 1/a; > 0 iff elliptic
        if alpha <= 0.0:
            raise ValueError("reference orbit must be elliptic")
        self.a = 1.0 / alpha
        h_vec = cross(r0, v0)
        e_vec = sub(scale(cross(v0, h_vec), 1.0 / mu), scale(r0, 1.0 / r0n))
        self.e = norm(e_vec)
        if not (1e-12 < self.e < 1.0):
            raise ValueError("reference orbit must be eccentric and bound")
        self.n = math.sqrt(mu / (self.a * self.a * self.a))
        self.period = 2.0 * math.pi / self.n
        # Perifocal basis from the state itself.
        self.p_hat = scale(e_vec, 1.0 / self.e)
        self.w_hat = scale(h_vec, 1.0 / norm(h_vec))
        self.q_hat = cross(self.w_hat, self.p_hat)
        # Eccentric anomaly of the epoch state: cos E = (1 - r/a)/e,
        # sin E = (r.v)/(e sqrt(mu a)).
        cos_e0 = (1.0 - r0n / self.a) / self.e
        sin_e0 = dot(r0, v0) / (self.e * math.sqrt(mu * self.a))
        self.e0_anom = math.atan2(sin_e0, cos_e0)
        self.m0 = self.e0_anom - self.e * math.sin(self.e0_anom)

    def solve_kepler(self, m_anom: float) -> float:
        """Newton solution of M = E - e sin E; residual checked at exit."""
        e_anom = m_anom
        for _ in range(64):
            f = e_anom - self.e * math.sin(e_anom) - m_anom
            fp = 1.0 - self.e * math.cos(e_anom)
            delta = f / fp
            e_anom -= delta
            if abs(delta) < 5e-16:
                break
        residual = e_anom - self.e * math.sin(e_anom) - m_anom
        if abs(residual) > 1e-13:
            raise AssertionError(f"Kepler residual {residual!r} at M={m_anom!r}")
        return e_anom

    def state_at(self, t: float) -> tuple[list[float], list[float]]:
        m_anom = self.m0 + self.n * t
        e_anom = self.solve_kepler(m_anom)
        ce, se = math.cos(e_anom), math.sin(e_anom)
        b = self.a * math.sqrt(1.0 - self.e * self.e)  # semi-minor axis
        r_vec = add(
            scale(self.p_hat, self.a * (ce - self.e)), scale(self.q_hat, b * se)
        )
        r_mag = self.a * (1.0 - self.e * ce)
        vs = math.sqrt(self.mu * self.a) / r_mag
        v_vec = add(
            scale(self.p_hat, -vs * se),
            scale(self.q_hat, vs * math.sqrt(1.0 - self.e * self.e) * ce),
        )
        return r_vec, v_vec

    def apsis_times(self, t_end: float) -> list[tuple[float, str]]:
        """(time, kind) for every apsis passage in (0, t_end]."""
        out: list[tuple[float, str]] = []
        k = 0
        while True:
            t_peri = (2.0 * math.pi * (k + 1) - self.m0) / self.n
            t_apo = (math.pi + 2.0 * math.pi * k - self.m0) / self.n
            progressed = False
            if 0.0 < t_apo <= t_end:
                out.append((t_apo, "apoapsis"))
                progressed = True
            if 0.0 < t_peri <= t_end:
                out.append((t_peri, "periapsis"))
                progressed = True
            if not progressed and t_peri > t_end and t_apo > t_end:
                break
            k += 1
        out.sort(key=lambda item: item[0])
        return out

    def max_fourth_derivative(self) -> float:
        """max |d^4 r/dt^4| over one period, 5-point central differences.

        Step choice: delta = 8 s balances truncation (O(delta^2 |r^(6)|),
        ~1e-4 relative here) against subtractive roundoff (~16 ulp(a)/delta^4,
        ~1e-7 relative), so the sampled maximum is good to ~4 digits; the
        consuming test applies a 1.5x headroom factor on top.
        """
        delta = 8.0
        best = 0.0
        samples = 4096
        for i in range(samples + 1):
            t = self.period * i / samples
            rm2, _ = self.state_at(t - 2.0 * delta)
            rm1, _ = self.state_at(t - delta)
            r00, _ = self.state_at(t)
            rp1, _ = self.state_at(t + delta)
            rp2, _ = self.state_at(t + 2.0 * delta)
            d4 = [
                (rm2[j] - 4.0 * rm1[j] + 6.0 * r00[j] - 4.0 * rp1[j] + rp2[j])
                / (delta**4)
                for j in range(3)
            ]
            best = max(best, norm(d4))
        return best


# --------------------------------------------------------------------------
# Self-checks
# --------------------------------------------------------------------------


def run_self_checks(
    orbit: EllipticOrbit,
    r0: list[float],
    v0: list[float],
    checkpoints: list[tuple[float, list[float], list[float]]],
    apsides: list[tuple[float, str]],
) -> None:
    eps0 = 0.5 * dot(v0, v0) - orbit.mu / norm(r0)
    h0 = norm(cross(r0, v0))
    for t, r, v in checkpoints:
        eps = 0.5 * dot(v, v) - orbit.mu / norm(r)
        h = norm(cross(r, v))
        assert abs((eps - eps0) / eps0) < 1e-12, f"energy drift at t={t}"
        assert abs((h - h0) / h0) < 1e-12, f"|h| drift at t={t}"
    # One-period closure of the analytic solution itself.
    r_t, v_t = orbit.state_at(orbit.period)
    assert norm(sub(r_t, r0)) < 1e-6, "period closure (position)"
    assert norm(sub(v_t, v0)) < 1e-9, "period closure (velocity)"
    # Apsis geometry: |r| at the expected extreme, r.v through zero. The r.v
    # bound reflects time quantization: |d(r.v)/dt| ~ mu*e/p ~ 8e6 s^-1 times
    # ulp-level time error leaves r.v well under 1e-2 m^2/s^2... loosely 1.0.
    for t, kind in apsides:
        r, v = orbit.state_at(t)
        expected = orbit.a * (1.0 - orbit.e if kind == "periapsis" else 1.0 + orbit.e)
        assert abs(norm(r) - expected) < 1e-5, f"|r| at {kind} t={t}"
        assert abs(dot(r, v)) < 1.0, f"r.v at {kind} t={t}"


# --------------------------------------------------------------------------
# TOML emission (restricted subset readable by cpp/tests/golden_io.hpp:
# [[case]] tables, quoted scalars, multi-line arrays of quoted strings)
# --------------------------------------------------------------------------


def hex_str(x: float) -> str:
    return f'"{float(x).hex()}"'


def emit_vector(lines: list[float]) -> str:
    body = "".join(f"  {hex_str(c)},\n" for c in lines)
    return "[\n" + body + "]"


def write_kepler_orbit(
    orbit: EllipticOrbit,
    r0: list[float],
    v0: list[float],
    checkpoints: list[tuple[float, list[float], list[float]]],
    max_r4: float,
) -> None:
    parts = [
        "# Reference eccentric-orbit golden for the integrator acceptance suite.\n"
        "# Generated by generate.py (provenance in manifest.toml); values are\n"
        "# binary64 hex literals (float.hex()). Decimal equivalents appear in\n"
        "# comments for readability only; the hex strings are authoritative.\n"
        "# Hand-editing is forbidden (tests/golden/README.md update policy).\n"
    ]
    parts.append(
        "\n# Orbit definition. The committed r0/v0 doubles ARE the orbit; the\n"
        f"# generating elements were a = {SMA_M} m, e = {ECC}, i = 30 deg,\n"
        "# RAAN = 40 deg, argp = 60 deg, nu0 = 120 deg (COE->RV per Vallado\n"
        "# ch. 2, evaluated once and discarded).\n"
        f"# mu = {MU_EARTH!r} m^3/s^2 (IERS Conventions 2010, TN 36)\n"
        f"# period = {orbit.period!r} s, sma = {orbit.a!r} m, ecc = {orbit.e!r}\n"
        f"# max |d4r/dt4| over one period = {max_r4!r} m/s^4\n"
    )
    parts.append("\n[[case]]\n")
    parts.append('name = "definition"\n')
    parts.append(f"mu_m3ps2 = {hex_str(MU_EARTH)}\n")
    parts.append(f"r0_m = {emit_vector(r0)}\n")
    parts.append(f"v0_mps = {emit_vector(v0)}\n")
    parts.append(f"period_s = {hex_str(orbit.period)}\n")
    parts.append(f"sma_m = {hex_str(orbit.a)}\n")
    parts.append(f"ecc = {hex_str(orbit.e)}\n")
    parts.append(f"mean_anomaly0_rad = {hex_str(orbit.m0)}\n")
    parts.append(f"max_r4_mps4 = {hex_str(max_r4)}\n")
    for j, (t, r, v) in enumerate(checkpoints, start=1):
        parts.append("\n[[case]]\n")
        parts.append(f'name = "checkpoint_{j}"\n')
        parts.append(f"# t = {t!r} s = {j}*(T/8)\n")
        parts.append(f"t_s = {hex_str(t)}\n")
        parts.append(f"r_m = {emit_vector(r)}\n")
        parts.append(f"v_mps = {emit_vector(v)}\n")
    (HERE / "kepler_orbit.toml").write_text("".join(parts), encoding="utf-8", newline="\n")


def write_apsis_times(orbit: EllipticOrbit, apsides: list[tuple[float, str]]) -> None:
    parts = [
        "# Analytic apsis passage times for the reference orbit defined in\n"
        "# kepler_orbit.toml (case \"definition\"), over (0, 3.2 T]. Generated\n"
        "# by generate.py (provenance in manifest.toml). Periapsis: mean\n"
        "# anomaly M = 0 (mod 2 pi); apoapsis: M = pi (mod 2 pi); times are\n"
        "# t = (target - M0)/n from the committed initial state.\n"
        f"# span = {APSIS_SPAN_PERIODS} periods = "
        f"{APSIS_SPAN_PERIODS * orbit.period!r} s\n"
    ]
    parts.append(f"\n[[case]]\nname = \"span\"\n")
    parts.append(f"t_end_s = {hex_str(APSIS_SPAN_PERIODS * orbit.period)}\n")
    for i, (t, kind) in enumerate(apsides):
        parts.append("\n[[case]]\n")
        parts.append(f'name = "apsis_{i}"\n')
        parts.append(f"# t = {t!r} s\n")
        parts.append(f'kind = "{kind}"\n')
        parts.append(f"t_s = {hex_str(t)}\n")
    (HERE / "apsis_times.toml").write_text("".join(parts), encoding="utf-8", newline="\n")


def main() -> None:
    r0, v0 = coe_to_rv()
    orbit = EllipticOrbit(MU_EARTH, r0, v0)
    checkpoints = []
    for j in range(1, N_CHECKPOINTS + 1):
        t = j * (orbit.period / 8.0)
        r, v = orbit.state_at(t)
        checkpoints.append((t, r, v))
    apsides = orbit.apsis_times(APSIS_SPAN_PERIODS * orbit.period)
    max_r4 = orbit.max_fourth_derivative()
    run_self_checks(orbit, r0, v0, checkpoints, apsides)
    write_kepler_orbit(orbit, r0, v0, checkpoints, max_r4)
    write_apsis_times(orbit, apsides)
    print(
        f"wrote kepler_orbit.toml ({N_CHECKPOINTS} checkpoints) and "
        f"apsis_times.toml ({len(apsides)} apsides); "
        f"T = {orbit.period:.6f} s, e = {orbit.e:.6f}, max|r4| = {max_r4:.6e}"
    )


if __name__ == "__main__":
    main()
