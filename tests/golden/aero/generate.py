"""Regenerate the axisymmetric-aero golden-vector files in this directory.

The values anchor the FR-9 axisymmetric aerodynamics model
(cpp/src/models/aero.cpp) against Phase 4 exit criterion 8. Four files are
produced:

- tables.toml: the Mach-table columns (mach, ca, cnalpha_per_rad, xcp_m)
  used by every case, copied at generation time from the committed FR-13
  starter-fleet CSVs (vehicles/electron_class_aero_full.csv and
  vehicles/electron_class_aero_upper.csv) plus one small synthetic table
  whose Mach grid starts above zero so the below-table clamp is reachable.
  xcp_m is the center-of-pressure station in the structural frame (origin
  at the aft plane, +X toward the nose), exactly as the CSV column
  declares.
- interp.toml: coefficient interpolation checks (chapter ch:aero,
  eq:aero:interp): the exact table readout at every Mach breakpoint, the
  piecewise-linear value at every segment midpoint, and the clamped
  readouts below the first and above the last breakpoint.
- breakpoints.toml: the exit-criterion-8 sweep: force/moment
  reconstruction at every Mach breakpoint of both fleet tables
  (eq:aero:axial, eq:aero:normal, eq:aero:cpmoment). The Mach 0 rows
  cannot produce a nonzero force at exactly Mach 0 (Mach 0 implies zero
  airspeed, which is the structural-zero path), so those two rows are
  exercised through the exact interp readout plus a near-zero-Mach force
  case at Mach 2^-10 inside the first segment.
- forcetorque.toml: the formulation families: total-alpha sweep (0, small,
  moderate, 90 deg, retrograde), crossflow roll orientations, CG fore and
  aft of the center of pressure, pitch damping on/off/roll-rate-only
  (eq:aero:damping), reference-area/diameter scaling, an off-breakpoint
  and an above-table case, and the pad structural zero (all outputs
  exactly zero at zero air-relative velocity).

References are evaluated with mpmath at 60 significant decimal digits from
the exact binary64 inputs and rounded once to binary64. The mpmath
evaluation mirrors the model formulation of math-library chapter ch:aero
branch for branch (structural-zero predicates on exact binary64 values,
piecewise-linear segment selection, end clamping) but shares no binary64
rounding with the C++ path, so the committed values check the
double-precision implementation to its own rounding floor. The
piecewise-linear interpolant is continuous at breakpoints and at the clamp
junctions, so segment selection within an ulp of a junction perturbs the
value by O(ulp x segment slope) -- there is no discontinuous branch to
disagree on. Structural-zero cases use exactly representable inputs whose
branch predicates (zero velocity, zero crossflow components, zero Cmq,
zero transverse rate) are unambiguous in binary64.

Running this script rewrites all four .toml files byte-identically; any
diff after regeneration means the script or the goldens were edited by
hand, which the FR-22 golden-update discipline forbids. The script also
prints a binary64 shadow evaluation (plain Python floats in the
implementation's operation order) of the worst relative error against the
committed references, as an advisory preview of the doctest gate.
"""

from __future__ import annotations

import math
import pathlib

import mpmath as mp

mp.mp.dps = 60

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]


# ---------------------------------------------------------------------------
# Committed fleet tables (single source of truth: the vehicles/ CSVs)
# ---------------------------------------------------------------------------


def read_table_csv(path: pathlib.Path) -> dict[str, list[float]]:
    cols: dict[str, list[float]] = {"mach": [], "ca": [], "cn": [], "xcp": []}
    header_seen = False
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not header_seen:
            if line != "mach,ca,cnalpha_per_rad,xcp_m":
                raise ValueError(f"{path}: unexpected header {line!r}")
            header_seen = True
            continue
        m, ca, cn, xcp = (float(c) for c in line.split(","))
        cols["mach"].append(m)
        cols["ca"].append(ca)
        cols["cn"].append(cn)
        cols["xcp"].append(xcp)
    return cols


# ---------------------------------------------------------------------------
# mpmath mirror of the aero model (chapter ch:aero)
# ---------------------------------------------------------------------------


def table_mp(table: dict[str, list[float]]) -> dict[str, list[mp.mpf]]:
    return {k: [mp.mpf(x) for x in v] for k, v in table.items()}


def interp_mp(tbl: dict[str, list[mp.mpf]], mach):
    """Piecewise-linear Mach lookup with end clamping (eq:aero:interp)."""
    m = tbl["mach"]
    n = len(m)
    if mach <= m[0]:
        return tbl["ca"][0], tbl["cn"][0], tbl["xcp"][0]
    if mach >= m[-1]:
        return tbl["ca"][-1], tbl["cn"][-1], tbl["xcp"][-1]
    i = 0
    while i + 2 < n and m[i + 1] <= mach:
        i += 1
    u = (mach - m[i]) / (m[i + 1] - m[i])

    def lerp(col):
        c = tbl[col]
        return c[i] + u * (c[i + 1] - c[i])

    return lerp("ca"), lerp("cn"), lerp("xcp")


def aero_mp(tbl, s_ref, d_ref, cmq, v, rho, a, xcg, omega):
    """Total-angle-of-attack force/moment about the CG (eq:aero:axial,
    eq:aero:normal, eq:aero:cpmoment, eq:aero:damping)."""
    s_ref, d_ref, cmq = mp.mpf(s_ref), mp.mpf(d_ref), mp.mpf(cmq)
    rho, a, xcg = mp.mpf(rho), mp.mpf(a), mp.mpf(xcg)
    vx, vy, vz = (mp.mpf(c) for c in v)
    wy, wz = mp.mpf(omega[1]), mp.mpf(omega[2])
    speed = mp.sqrt(vx * vx + vy * vy + vz * vz)
    zero = mp.mpf(0)
    if speed == 0:
        return zero, zero, zero, [zero] * 3, [zero] * 3
    mach = speed / a  # eq:aero:mach
    qbar = rho * speed * speed / 2  # eq:aero:qbar
    ca, cn, xcp = interp_mp(tbl, mach)
    force = [-(qbar * s_ref * ca), zero, zero]  # eq:aero:axial
    torque = [zero, zero, zero]
    if vy != 0 or vz != 0:
        k_n = qbar * s_ref * cn / speed  # eq:aero:normal
        force[1] = -k_n * vy
        force[2] = -k_n * vz
        lever = xcp - xcg  # eq:aero:cpmoment
        torque[1] = -lever * force[2]
        torque[2] = lever * force[1]
    if cmq != 0 and (wy != 0 or wz != 0):
        k_q = qbar * s_ref * d_ref * d_ref * cmq / (2 * speed)  # eq:aero:damping
        torque[1] += k_q * wy
        torque[2] += k_q * wz
    alpha = mp.atan2(mp.sqrt(vy * vy + vz * vz), vx)  # eq:aero:alpha
    return mach, qbar, alpha, force, torque


# ---------------------------------------------------------------------------
# binary64 shadow of the C++ operation order (advisory error preview only)
# ---------------------------------------------------------------------------


def interp_f64(table, mach):
    m = table["mach"]
    n = len(m)
    if mach <= m[0]:
        return table["ca"][0], table["cn"][0], table["xcp"][0]
    if mach >= m[-1]:
        return table["ca"][-1], table["cn"][-1], table["xcp"][-1]
    i = 0
    while i + 2 < n and m[i + 1] <= mach:
        i += 1
    u = (mach - m[i]) / (m[i + 1] - m[i])

    def lerp(col):
        c = table[col]
        return c[i] + u * (c[i + 1] - c[i])

    return lerp("ca"), lerp("cn"), lerp("xcp")


def aero_f64(table, s_ref, d_ref, cmq, v, rho, a, xcg, omega):
    vx, vy, vz = v
    speed = math.sqrt((vx * vx + vy * vy) + vz * vz)
    if speed == 0.0:
        return 0.0, 0.0, 0.0, [0.0] * 3, [0.0] * 3
    mach = speed / a
    qbar = 0.5 * rho * speed * speed
    ca, cn, xcp = interp_f64(table, mach)
    force = [-(qbar * s_ref * ca), 0.0, 0.0]
    torque = [0.0, 0.0, 0.0]
    if vy != 0.0 or vz != 0.0:
        k_n = qbar * s_ref * cn / speed
        force[1] = -k_n * vy
        force[2] = -k_n * vz
        lever = xcp - xcg
        torque[1] = -lever * force[2]
        torque[2] = lever * force[1]
    if cmq != 0.0 and (omega[1] != 0.0 or omega[2] != 0.0):
        k_q = qbar * s_ref * d_ref * d_ref * cmq / (2.0 * speed)
        torque[1] += k_q * omega[1]
        torque[2] += k_q * omega[2]
    alpha = math.atan2(math.sqrt(vy * vy + vz * vz), vx)
    return mach, qbar, alpha, force, torque


def rel_err(value: float, ref: float) -> float:
    if ref == 0.0:
        return 0.0 if value == 0.0 else float("inf")
    return abs(value - ref) / abs(ref)


def rel_err_vec(value, ref) -> float:
    nref = math.sqrt(sum(float(r) * float(r) for r in ref))
    if nref == 0.0:
        return 0.0 if all(float(x) == 0.0 for x in value) else float("inf")
    diff = math.sqrt(sum((float(x) - float(r)) ** 2 for x, r in zip(value, ref)))
    return diff / nref


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


def hexv(vec):
    return [float(x).hex() for x in vec]


# ---------------------------------------------------------------------------
# case construction
# ---------------------------------------------------------------------------

# Exact (cos, sin) pairs at the cardinal angles so structural-zero velocity
# components are literal binary64 zeros rather than ~1e-61 trigonometric
# residues of mp.pi.
_CARDINAL = {
    0: (mp.mpf(1), mp.mpf(0)),
    90: (mp.mpf(0), mp.mpf(1)),
    180: (mp.mpf(-1), mp.mpf(0)),
    270: (mp.mpf(0), mp.mpf(-1)),
}


def cos_sin_deg(deg: float):
    key = deg % 360
    if key in _CARDINAL:
        return _CARDINAL[key]
    theta = mp.radians(mp.mpf(deg))
    return mp.cos(theta), mp.sin(theta)


def make_v(speed, alpha_deg: float, phi_deg: float):
    """Air-relative velocity at nominal total alpha and crossflow roll,
    rounded once per component to binary64 (the rounded triple is the
    case's exact input; the golden outputs follow from it, not from the
    nominal angles)."""
    ca_, sa_ = cos_sin_deg(alpha_deg)
    cp_, sp_ = cos_sin_deg(phi_deg)
    speed = mp.mpf(speed)
    return (
        float(speed * ca_),
        float(speed * sa_ * cp_),
        float(speed * sa_ * sp_),
    )


def main() -> None:
    full = read_table_csv(ROOT / "vehicles" / "electron_class_aero_full.csv")
    upper = read_table_csv(ROOT / "vehicles" / "electron_class_aero_upper.csv")
    # Synthetic 3-row table whose grid starts above Mach 0: the below-table
    # clamp is unreachable through the fleet tables (their grids start at 0
    # and Mach is non-negative), so this table makes that branch testable.
    synthetic = {
        "mach": [2.0, 4.0, 6.0],
        "ca": [0.5, 0.4, 0.3],
        "cn": [2.0, 2.5, 3.0],
        "xcp": [10.0, 11.0, 12.0],
    }
    tables = {"electron_full": full, "electron_upper": upper, "synthetic_clamp": synthetic}
    sources = {
        "electron_full": "vehicles/electron_class_aero_full.csv",
        "electron_upper": "vehicles/electron_class_aero_upper.csv",
        "synthetic_clamp": "synthetic (this script), grid starting above Mach 0",
    }

    out = []
    for name, tbl in tables.items():
        out.append({
            "name": name,
            "source": sources[name],
            "mach": hexv(tbl["mach"]),
            "ca": hexv(tbl["ca"]),
            "cnalpha_per_rad": hexv(tbl["cn"]),
            "xcp_m": hexv(tbl["xcp"]),
        })
    emit(
        HERE / "tables.toml",
        "Axisymmetric aero Mach tables consumed by the aero golden cases\n"
        "(FR-9, chapter ch:aero). electron_full and electron_upper are the\n"
        "committed FR-13 starter-fleet tables, copied column for column at\n"
        "generation time from the vehicles/ CSVs named in each case's\n"
        "source key (units per the CSV header: mach and ca dimensionless,\n"
        "cnalpha per radian, xcp_m meters in the structural frame -- origin\n"
        "at the aft plane, +X toward the nose). synthetic_clamp exists only\n"
        "to reach the below-table clamp branch. Provenance and tolerances\n"
        "in manifest.toml. Regenerated by generate.py.",
        out,
    )

    # ---- interp.toml -----------------------------------------------------
    interp_cases = []
    shadow_worst = 0.0

    def add_interp(name: str, tname: str, mach: float) -> None:
        nonlocal shadow_worst
        tbl = table_mp(tables[tname])
        ca, cn, xcp = interp_mp(tbl, mp.mpf(mach))
        interp_cases.append({
            "name": name,
            "table": tname,
            "mach": float(mach).hex(),
            "ca": float(ca).hex(),
            "cnalpha_per_rad": float(cn).hex(),
            "xcp_m": float(xcp).hex(),
        })
        ca64, cn64, xcp64 = interp_f64(tables[tname], mach)
        for v64, ref in ((ca64, ca), (cn64, cn), (xcp64, xcp)):
            shadow_worst = max(shadow_worst, rel_err(v64, float(ref)))

    for tname in ("electron_full", "electron_upper", "synthetic_clamp"):
        tbl = tables[tname]
        n = len(tbl["mach"])
        for i, m in enumerate(tbl["mach"]):
            add_interp(f"bp_{tname}_{i:02d}", tname, m)
            # Exact-readout self-check: a breakpoint input must reproduce
            # its own table row bit for bit in the extended mirror.
            row = interp_mp(table_mp(tbl), mp.mpf(m))
            assert [float(x) for x in row] == [tbl["ca"][i], tbl["cn"][i], tbl["xcp"][i]]
        for i in range(n - 1):
            mid = float((mp.mpf(tbl["mach"][i]) + mp.mpf(tbl["mach"][i + 1])) / 2)
            add_interp(f"mid_{tname}_{i:02d}", tname, mid)
        add_interp(f"clamp_{tname}_above", tname, float(tbl["mach"][-1] + 1.5))
    add_interp("clamp_synthetic_clamp_below", "synthetic_clamp", 1.0)

    emit(
        HERE / "interp.toml",
        "Mach-table interpolation golden values (chapter ch:aero,\n"
        "eq:aero:interp). bp_* cases evaluate at a table breakpoint and\n"
        "must reproduce the table row exactly (the interpolation weight is\n"
        "a structural zero there); clamp_* cases evaluate outside the grid\n"
        "and must reproduce the clamped end row exactly; mid_* cases\n"
        "evaluate at binary64 segment midpoints against the 60-digit\n"
        "mpmath piecewise-linear value rounded once to binary64.\n"
        "Provenance and tolerances in manifest.toml. Regenerated by\n"
        "generate.py.",
        interp_cases,
    )

    # ---- force/torque case runner ----------------------------------------
    def run_case(name, tname, s_ref, d_ref, cmq, v, rho, a, xcg, omega):
        nonlocal shadow_worst
        tbl = table_mp(tables[tname])
        mach, qbar, alpha, force, torque = aero_mp(
            tbl, s_ref, d_ref, cmq, v, rho, a, xcg, omega)
        case = {
            "name": name,
            "table": tname,
            "ref_area_m2": float(s_ref).hex(),
            "ref_diameter_m": float(d_ref).hex(),
            "cmq_per_rad": float(cmq).hex(),
            "v_rel_mps": hexv(v),
            "rho_kgpm3": float(rho).hex(),
            "speed_of_sound_mps": float(a).hex(),
            "x_cg_m": float(xcg).hex(),
            "omega_radps": hexv(omega),
            "mach": float(mach).hex(),
            "q_bar_Pa": float(qbar).hex(),
            "alpha_total_rad": float(alpha).hex(),
            "force_N": hexv(force),
            "torque_Nm": hexv(torque),
        }
        m64, q64, al64, f64, t64 = aero_f64(
            tables[tname], s_ref, d_ref, cmq, v, rho, a, xcg, omega)
        shadow_worst = max(
            shadow_worst,
            rel_err(m64, float(mach)),
            rel_err(q64, float(qbar)),
            rel_err(al64, float(alpha)),
            rel_err_vec(f64, [float(x) for x in force]),
            rel_err_vec(t64, [float(x) for x in torque]),
        )
        print(f"{name:28s} M={mp.nstr(mach, 8):>12s} "
              f"|F|={mp.nstr(mp.sqrt(sum(f * f for f in force)), 8):>12s} N")
        return case

    # ---- breakpoints.toml (exit criterion 8 sweep) ------------------------
    bp_cases = []
    sweeps = (
        ("full", "electron_full", 0.75, 8.0, lambda i: 25.0 * i),
        ("upper", "electron_upper", 0.02, 14.0, lambda i: 40.0 * i + 10.0),
    )
    for label, tname, rho, xcg, phi_of in sweeps:
        for i, b in enumerate(tables[tname]["mach"]):
            if b == 0.0:
                # Mach exactly 0 implies zero airspeed (the structural-zero
                # path), so the Mach 0 row is exercised inside its first
                # segment at the dyadic Mach 2^-10 instead.
                name = f"bp_{label}_{i:02d}_nearzero"
                mach_target = 2.0 ** -10
            else:
                name = f"bp_{label}_{i:02d}"
                mach_target = b
            speed = mp.mpf(mach_target) * 256  # exact: dyadic scaling
            v = make_v(speed, 5.0, phi_of(i))
            bp_cases.append(run_case(
                name, tname, 1.13, 1.2, 0.0, v, rho, 256.0, xcg,
                (0.0, 0.0, 0.0)))

    emit(
        HERE / "breakpoints.toml",
        "Force/moment reconstruction at every Mach breakpoint of both\n"
        "committed fleet tables (FR-9, Phase 4 exit criterion 8; chapter\n"
        "ch:aero, eq:aero:axial, eq:aero:normal, eq:aero:cpmoment), at 5\n"
        "deg nominal total alpha with a per-case crossflow roll. The\n"
        "airspeed is the breakpoint Mach times the dyadic speed of sound\n"
        "256 m/s, so the case Mach equals the breakpoint to within one\n"
        "rounding of the velocity components. Evaluated with mpmath at 60\n"
        "significant decimal digits from the exact binary64 inputs and\n"
        "rounded once to binary64. Provenance and tolerances in\n"
        "manifest.toml. Regenerated by generate.py.",
        bp_cases,
    )

    # ---- forcetorque.toml (formulation families) ---------------------------
    m_full = tables["electron_full"]["mach"]
    speed_105 = mp.mpf(m_full[4]) * 256
    speed_150 = mp.mpf(m_full[6]) * 256
    ft_cases = []

    def axial_v(speed, sign=1.0):
        return (float(mp.mpf(sign) * speed), 0.0, 0.0)

    ft = [
        # Total-alpha family at the Mach 1.05 breakpoint.
        ("alpha0_axial_only", "electron_full", 1.13, 1.2, 0.0,
         axial_v(speed_105), 0.75, 256.0, 8.0, (0.01, 0.02, 0.03)),
        ("alpha_small_0p5deg", "electron_full", 1.13, 1.2, 0.0,
         make_v(speed_105, 0.5, 90.0), 0.75, 256.0, 8.0, (0.0, 0.0, 0.0)),
        ("alpha_moderate_10deg", "electron_full", 1.13, 1.2, 0.0,
         make_v(speed_105, 10.0, 30.0), 0.75, 256.0, 8.0, (0.0, 0.0, 0.0)),
        ("alpha_90deg", "electron_full", 1.13, 1.2, 0.0,
         (0.0, float(mp.mpf(0.6) * speed_105), float(mp.mpf(0.8) * speed_105)),
         0.75, 256.0, 8.0, (0.0, 0.0, 0.0)),
        ("retro_axial_alpha_pi", "electron_full", 1.13, 1.2, 0.0,
         axial_v(speed_105, -1.0), 0.75, 256.0, 8.0, (0.0, 0.0, 0.0)),
        # Crossflow roll family at the Mach 2.0 breakpoint, 20 deg alpha.
        ("roll_phi000", "electron_full", 1.13, 1.2, 0.0,
         make_v(mp.mpf(m_full[7]) * 256, 20.0, 0.0), 0.75, 256.0, 8.0,
         (0.0, 0.0, 0.0)),
        ("roll_phi045", "electron_full", 1.13, 1.2, 0.0,
         make_v(mp.mpf(m_full[7]) * 256, 20.0, 45.0), 0.75, 256.0, 8.0,
         (0.0, 0.0, 0.0)),
        ("roll_phi090", "electron_full", 1.13, 1.2, 0.0,
         make_v(mp.mpf(m_full[7]) * 256, 20.0, 90.0), 0.75, 256.0, 8.0,
         (0.0, 0.0, 0.0)),
        ("roll_phi210", "electron_full", 1.13, 1.2, 0.0,
         make_v(mp.mpf(m_full[7]) * 256, 20.0, 210.0), 0.75, 256.0, 8.0,
         (0.0, 0.0, 0.0)),
        # CG on either side of the CP (xcp = 13.2 m at the Mach 0.8 row).
        ("cg_fore_of_cp", "electron_full", 1.13, 1.2, 0.0,
         make_v(mp.mpf(m_full[2]) * 256, 8.0, 60.0), 0.75, 256.0, 8.0,
         (0.0, 0.0, 0.0)),
        ("cg_aft_of_cp", "electron_full", 1.13, 1.2, 0.0,
         make_v(mp.mpf(m_full[2]) * 256, 8.0, 60.0), 0.75, 256.0, 15.0,
         (0.0, 0.0, 0.0)),
        # Pitch damping (eq:aero:damping) at the Mach 1.5 breakpoint.
        ("cmq_with_alpha", "electron_full", 1.13, 1.2, -25.0,
         make_v(speed_150, 6.0, 135.0), 0.75, 256.0, 8.0,
         (0.02, -0.05, 0.03)),
        ("cmq_zero_alpha_pure_damping", "electron_full", 1.13, 1.2, -25.0,
         axial_v(speed_150), 0.75, 256.0, 8.0, (0.0, 0.1, -0.04)),
        ("cmq_roll_rate_only", "electron_full", 1.13, 1.2, -25.0,
         make_v(speed_150, 6.0, 135.0), 0.75, 256.0, 8.0, (0.3, 0.0, 0.0)),
        ("cmq_zero_disables", "electron_full", 1.13, 1.2, 0.0,
         make_v(speed_150, 6.0, 135.0), 0.75, 256.0, 8.0, (0.1, 0.2, 0.3)),
        # Reference-area/diameter scaling on a synthetic configuration.
        ("synthetic_ref_scaling", "electron_full", 2.5, 0.9, -10.0,
         make_v(mp.mpf(m_full[8]) * 256, 12.0, 120.0), 0.3, 256.0, 9.5,
         (0.01, 0.04, -0.02)),
        # Off-breakpoint interpolation inside the force path (upper table,
        # Mach 1.5 mid-segment) and the above-table clamp in the force path.
        ("upper_midsegment", "electron_upper", 1.13, 1.2, 0.0,
         make_v(mp.mpf(384), 4.0, 200.0), 0.02, 256.0, 14.0,
         (0.0, 0.0, 0.0)),
        ("clamp_above_max_mach", "electron_full", 1.13, 1.2, 0.0,
         make_v(mp.mpf(2304), 3.0, 10.0), 0.05, 256.0, 8.0,
         (0.0, 0.0, 0.0)),
        # Pad structural zero (Phase 4 exit criterion 10): zero air-relative
        # velocity returns exactly zero everything, damping configured on.
        ("pad_static_exact_zero", "electron_full", 1.13, 1.2, -25.0,
         (0.0, 0.0, 0.0), 1.225, 340.25, 8.0, (0.001, 0.002, 0.003)),
    ]
    for args in ft:
        ft_cases.append(run_case(*args))

    emit(
        HERE / "forcetorque.toml",
        "Axisymmetric aero formulation families (FR-9; chapter ch:aero):\n"
        "total-alpha sweep (0, 0.5 deg, 10 deg, 90 deg, retrograde),\n"
        "crossflow roll orientations, CG fore/aft of the center of\n"
        "pressure, pitch damping on/off/roll-rate-only, reference\n"
        "area/diameter scaling, off-breakpoint interpolation, above-table\n"
        "clamping, and the pad structural zero (all outputs exactly zero\n"
        "at zero air-relative velocity). Evaluated with mpmath at 60\n"
        "significant decimal digits from the exact binary64 inputs and\n"
        "rounded once to binary64. Structural zeros are literal zeros.\n"
        "Provenance and tolerances in manifest.toml. Regenerated by\n"
        "generate.py.",
        ft_cases,
    )

    print(f"aero goldens regenerated; mpmath {mp.__version__} at "
          f"{mp.mp.dps} dps")
    print(f"binary64 shadow worst relative error vs references: "
          f"{shadow_worst:.3e} (advisory; the doctest gate is 1e-12)")


if __name__ == "__main__":
    main()
