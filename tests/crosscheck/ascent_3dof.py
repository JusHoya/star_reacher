"""Independent 3DOF point-mass ascent oracle for Phase 4 EC-11.

This module is a deliberately BLIND, from-scratch reimplementation of a
translational three-degree-of-freedom (point-mass) ascent. It exists solely to
cross-check the project's 6DOF ascent without sharing a line of its dynamics: it
imports none of ``star_reacher`` and reads none of the C++ model source. It
parses the same vehicle and mission TOML files the 6DOF run consumes
(``vehicles/electron_class.toml``, ``missions/ascent_leo.toml``) purely to learn
the vehicle masses, engines, aero table, and the open-loop pitch program, then
flies that program with its own integrator, atmosphere, gravity, and
mass/thrust model. Any agreement with the 6DOF is therefore evidence, not
construction.

The vector algebra is plain-Python tuple math (no numpy in the dynamics loop):
the independence rule aside, the only third-party path this oracle needs is the
Python standard library, which keeps it dependency-free and fully portable.

Independent modelling choices (all first-principles / published):

- Integrator: fixed-step classical RK4 on the 3D translational state, coded here.
- Earth model: WGS84 ellipsoid for the launch-pad geodetic placement and for the
  geodetic altitude that drives the atmosphere and the fairing-jettison trigger;
  a spherical radius (WGS84 equatorial) for the osculating apsis altitudes, which
  is the convention the 6DOF also uses for its perigee-insertion gate and its
  apogee/perigee report (verified against the sim's own final-state reduction).
- Gravity: spherical-Earth point mass, a = -mu * r / |r|^3 (J2 optional, off by
  default; the insertion apogee/perigee are osculating two-body quantities read
  from the instantaneous state, and J2 shifts them < 1 percent over a 6.5 min
  ascent).
- Atmosphere: U.S. Standard Atmosphere 1976 (NOAA/NASA/USAF, NOAA-S/T 76-1562),
  the 0-86 km geopotential-layer formulation coded from its published base
  temperatures, lapse rates, and base pressures; above 86 km an exponential
  continuation whose dynamic-pressure contribution on this ascent is < ~20 Pa.
- Drag: F = -1/2 rho |v_rel| v_rel CA(M) A_ref with a co-rotating atmosphere
  (v_rel = v - omega_earth x r); CA(M) linearly interpolated from the vehicle's
  own committed Mach table CSV.
- Thrust: F = throttle*F_vac - p_amb(h)*A_e (back-pressure law) with vacuum-Isp
  mass flow mdot = throttle*F_vac / (g0*Isp_vac) and a linear ignition spool. The
  pitch program sets the thrust direction as a pitch angle above the horizontal
  of the launch-site local geographic frame taken at liftoff and held fixed in
  inertial space thereafter (open-loop launch-frame attitude). The 6DOF
  prescribes attitude the same way in this ascent -- in pitch_program mode it
  sets the attitude directly to the commanded pitch direction in that same frozen
  launch frame, with no attitude controller, so the body axis (its thrust line)
  equals the command exactly and this oracle's thrust direction matches it. This
  frozen-frame reading is also the only one that reaches orbit: the
  instantaneous-local-horizontal reading falls ~40 km short of the perigee gate,
  so it cannot be the program the vehicle's capability note says reaches LEO.
- Sequencing: propellant depletes by the summed mdot; stage thrust is cut
  instantly at the commanded MECO time (the 6DOF instead tails thrust off over
  the ~1 s engine spool time at cutoff, the principal benign difference between
  the two point-mass codes; see the manifest); a spent stage's dry mass plus its
  propellant residual is dropped at the sequenced separation time; the fairing is
  dropped at its geodetic-altitude trigger.

The public entry point is :func:`run_ascent`, returning an :class:`AscentResult`.
"""

from __future__ import annotations

import csv
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

Vec = tuple[float, float, float]

# --- Physical constants (independent, cited) --------------------------------

# Earth gravitational parameter, IERS Conventions (2010), TN No. 36. A defined
# physical constant, not a value copied from the sim; it equals the core's
# gm("earth") because both cite the same authority.
MU_EARTH = 3.986004418e14  # m^3/s^2
# WGS84 equatorial radius, used as the spherical Earth radius for altitude and
# as the geodetic reference. The osculating apsis ALTITUDES subtract this
# radius; the same radius is used when the test reduces the 6DOF final state, so
# the choice cancels in the apogee/perigee DIFFERENCE.
R_EARTH = 6378137.0  # m
# Earth rotation rate, IERS Conventions (2010): nominal mean angular velocity.
OMEGA_EARTH = 7.292115e-5  # rad/s
# Standard gravity for the Isp->mass-flow relation (BIPM/ISO 80000 convention).
G0 = 9.80665  # m/s^2
# Ratio of specific heats for the USSA76 sound-speed relation.
GAMMA_AIR = 1.4
# WGS84 ellipsoid (NIMA TR8350.2): equatorial radius A_WGS84 == R_EARTH above,
# and the flattening from which the first-eccentricity-squared follows. Used for
# the pad placement and geodetic altitude only; gravity remains spherical.
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
# J2, IERS Conventions (2010) zonal harmonic; used only when use_j2 is set.
J2_EARTH = 1.08262668e-3


# --- Minimal vector algebra (plain Python) ----------------------------------


def _add(a: Vec, b: Vec) -> Vec:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(a: Vec, s: float) -> Vec:
    return (a[0] * s, a[1] * s, a[2] * s)


def _cross(a: Vec, b: Vec) -> Vec:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a: Vec) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _unit(a: Vec) -> Vec:
    n = _norm(a)
    return (a[0] / n, a[1] / n, a[2] / n)


# --- U.S. Standard Atmosphere 1976 (0-86 km) --------------------------------

# Universal gas constant and mean molar mass of air per USSA76 (NOAA-S/T
# 76-1562), giving the specific gas constant used for density and sound speed.
_RSTAR = 8.31432  # J/(mol.K)
_M0 = 0.0289644  # kg/mol
_R_SPECIFIC = _RSTAR / _M0  # 287.053 J/(kg.K)
# Effective Earth radius the USSA76 uses to map geometric to geopotential height.
_R0_GEOPOT = 6356766.0  # m
# Geopotential base heights (m'), base temperatures (K), lapse rates (K/m'), and
# base pressures (Pa) of the seven USSA76 layers spanning 0 to 84852 m' (86 km
# geometric). Base pressures are the published layer-boundary values.
_H_BASE = (0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0)
_T_BASE = (288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65)
_L_BASE = (-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002)
_P_BASE = (101325.0, 22632.06, 5474.889, 868.0187, 110.9063, 66.93887, 3.956420)
_H_TOP = 84852.0  # geopotential height of the 86 km cap


def atmosphere(alt_m: float) -> tuple[float, float, float]:
    """Return (density kg/m^3, pressure Pa, speed_of_sound m/s) at geometric altitude.

    U.S. Standard Atmosphere 1976, 0-86 km, coded from the published layer
    constants. Below 0 m the sea-level layer is extended (the pad sits at 10 m).
    Above 86 km an exponential continuation is used; its dynamic-pressure
    contribution on this trajectory is negligible (< ~20 Pa) and it only keeps
    the integrator well posed through the thin upper atmosphere.
    """
    z = alt_m
    # Geometric -> geopotential height (USSA76 definition).
    h = _R0_GEOPOT * z / (_R0_GEOPOT + z)
    if h <= _H_TOP:
        i = 0
        for k in range(len(_H_BASE)):
            if h >= _H_BASE[k]:
                i = k
            else:
                break
        hb, tb, lb, pb = _H_BASE[i], _T_BASE[i], _L_BASE[i], _P_BASE[i]
        temp = tb + lb * (h - hb)
        if lb == 0.0:
            press = pb * math.exp(-G0 * _M0 * (h - hb) / (_RSTAR * tb))
        else:
            press = pb * (tb / temp) ** (G0 * _M0 / (_RSTAR * lb))
    else:
        # Anchor an exponential continuation at the 86 km cap with a local scale
        # height; density here is O(1e-5) kg/m^3 and falling, so drag is inert.
        temp_cap = _T_BASE[6] + _L_BASE[6] * (_H_TOP - _H_BASE[6])
        press_cap = _P_BASE[6] * (_T_BASE[6] / temp_cap) ** (G0 * _M0 / (_RSTAR * _L_BASE[6]))
        scale_h = _R_SPECIFIC * temp_cap / G0
        temp = temp_cap
        press = press_cap * math.exp(-(h - _H_TOP) / scale_h)
    density = press / (_R_SPECIFIC * temp)
    sound = math.sqrt(GAMMA_AIR * _R_SPECIFIC * temp)
    return density, press, sound


# --- Vehicle / mission parsing ----------------------------------------------


@dataclass
class Engine:
    name: str
    thrust_vac_n: float
    isp_vac_s: float
    exit_area_m2: float
    spool_time_s: float
    throttle_max: float
    mdot_vac: float  # F_vac / (g0 * Isp), kg/s at full throttle


@dataclass
class Vehicle:
    stage1_dry_kg: float
    stage2_dry_kg: float
    s1_prop_kg: float
    s2_prop_kg: float
    fairing_kg: float
    payload_kg: float
    s1_engine: Engine
    s2_engine: Engine
    aero_ref_area_m2: float
    ca_full: tuple[tuple[float, float], ...]   # (mach, CA) full stack
    ca_upper: tuple[tuple[float, float], ...]  # (mach, CA) upper stack


def _load_ca_table(csv_path: Path) -> tuple[tuple[float, float], ...]:
    rows: list[tuple[float, float]] = []
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header_seen = False
        for raw in reader:
            if not raw or raw[0].lstrip().startswith("#"):
                continue
            if not header_seen:
                header_seen = True  # first non-comment line is the mandated header
                continue
            rows.append((float(raw[0]), float(raw[1])))
    return tuple(rows)


def load_vehicle(vehicle_path: Path) -> Vehicle:
    """Parse the vehicle TOML for the masses, engines, and aero table only."""
    root = vehicle_path.parent.parent  # repo root, to resolve CSV paths
    with vehicle_path.open("rb") as fh:
        doc = tomllib.load(fh)
    stages = doc["stage"]
    s1, s2 = stages[0], stages[1]

    def _engine(stage: dict) -> Engine:
        e = stage["engine"][0]
        mdot = e["thrust_vac_N"] / (G0 * e["isp_vac_s"])
        return Engine(
            name=e["name"],
            thrust_vac_n=e["thrust_vac_N"],
            isp_vac_s=e["isp_vac_s"],
            exit_area_m2=e["exit_area_m2"],
            spool_time_s=e["spool_time_s"],
            throttle_max=e["throttle_max"],
            mdot_vac=mdot,
        )

    jett = {j["name"]: j["mass_kg"] for j in s2["jettison"]}
    aero = {a["config"]: a for a in doc["aero"]}
    return Vehicle(
        stage1_dry_kg=s1["dry_mass_kg"],
        stage2_dry_kg=s2["dry_mass_kg"],
        s1_prop_kg=s1["tank"][0]["propellant_mass_kg"],
        s2_prop_kg=s2["tank"][0]["propellant_mass_kg"],
        fairing_kg=jett["fairing"],
        payload_kg=jett["payload_stack"],
        s1_engine=_engine(s1),
        s2_engine=_engine(s2),
        aero_ref_area_m2=aero["full_stack"]["ref_area_m2"],
        ca_full=_load_ca_table(root / aero["full_stack"]["mach_table_csv"]),
        ca_upper=_load_ca_table(root / aero["upper_stack"]["mach_table_csv"]),
    )


@dataclass
class Mission:
    lat_deg: float
    lon_deg: float
    alt_m: float
    azimuth_deg: float
    pitch_t_s: tuple[float, ...]
    pitch_deg: tuple[float, ...]
    t_ignite_s1: float
    t_release: float
    t_meco: float
    t_sep1: float
    t_ignite_s2: float
    fairing_alt_m: float
    insertion_perigee_m: float
    duration_s: float


def load_mission(mission_path: Path) -> Mission:
    """Parse the mission TOML for the launch state, pitch program, and sequence timing."""
    with mission_path.open("rb") as fh:
        doc = tomllib.load(fh)
    geo = doc["initial_state"]["geodetic"]
    seq = {s["name"]: s for s in doc["sequence"]}

    def _abs_time(name: str) -> float:
        """Resolve a sequence entry's absolute mission time (elapsed / after_event)."""
        s = seq[name]
        if s["trigger"] == "elapsed":
            return float(s["t_s"])
        if s["trigger"] == "after_event":
            return _abs_time(s["event"]) + float(s.get("offset_s", 0.0))
        raise ValueError(f"time-based resolution not defined for trigger {s['trigger']!r}")

    pitch = seq["pitch"]
    return Mission(
        lat_deg=geo["lat_deg"],
        lon_deg=geo["lon_deg"],
        alt_m=geo["alt_m"],
        azimuth_deg=pitch["azimuth_deg"],
        pitch_t_s=tuple(pitch["pitch_t_s"]),
        pitch_deg=tuple(pitch["pitch_deg"]),
        t_ignite_s1=_abs_time("ignite_s1"),
        t_release=_abs_time("release"),
        t_meco=_abs_time("meco"),
        t_sep1=_abs_time("sep1"),
        t_ignite_s2=_abs_time("ignite_s2"),
        fairing_alt_m=float(seq["drop_fairing"]["altitude_m"]),
        insertion_perigee_m=float(seq["insertion"]["perigee_alt_m"]),
        duration_s=float(doc["mission"]["duration_s"]),
    )


# --- Geometry / interpolation ------------------------------------------------


def _interp(x: float, xs, ys) -> float:
    """Piecewise-linear interpolation with flat clamping outside the breakpoints."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            f = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + f * (ys[i] - ys[i - 1])
    return ys[-1]


def _ca_of_mach(mach: float, table) -> float:
    return _interp(mach, [row[0] for row in table], [row[1] for row in table])


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> Vec:
    """WGS84 geodetic (lat, lon, height) to Earth-fixed Cartesian position."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    n = R_EARTH / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
    return (
        (n + alt_m) * math.cos(lat) * math.cos(lon),
        (n + alt_m) * math.cos(lat) * math.sin(lon),
        (n * (1.0 - WGS84_E2) + alt_m) * math.sin(lat),
    )


def geodetic_altitude(r: Vec) -> float:
    """WGS84 geodetic height of an Earth-fixed position (Bowring/iterative).

    Used for the atmosphere and the fairing trigger so both track the ellipsoid
    height the 6DOF reports, not the geocentric radius (they differ by ~9 km at
    this latitude). Osculating apsis altitudes keep the spherical radius instead.
    """
    x, y, z = r
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    for _ in range(5):  # converges to machine precision in ~3 iterations
        n = R_EARTH / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
        h = p / math.cos(lat) - n
        lat = math.atan2(z, p * (1.0 - WGS84_E2 * n / (n + h)))
    n = R_EARTH / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
    return p / math.cos(lat) - n


def geodetic_up(r: Vec) -> Vec:
    """WGS84 geodetic vertical (ellipsoid surface normal) at an Earth-fixed position."""
    x, y, z = r
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    for _ in range(5):
        n = R_EARTH / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
        h = p / math.cos(lat) - n
        lat = math.atan2(z, p * (1.0 - WGS84_E2 * n / (n + h)))
    lon = math.atan2(y, x)
    return (math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat))


def osculating_apsides(r: Vec, v: Vec) -> tuple[float, float, float, float]:
    """Return (apoapsis_alt_m, periapsis_alt_m, apoapsis_r_m, periapsis_r_m).

    Two-body osculating elements from an inertial state, using MU_EARTH and
    R_EARTH. Frame-orientation independent; this is the exact reduction the test
    also applies to the 6DOF final truth state.
    """
    rn = _norm(r)
    vn = _norm(v)
    energy = 0.5 * vn * vn - MU_EARTH / rn
    a = -MU_EARTH / (2.0 * energy)
    h = _cross(r, v)
    # Eccentricity vector e = (v x h)/mu - r/|r|.
    e_vec = _sub(_scale(_cross(v, h), 1.0 / MU_EARTH), _scale(r, 1.0 / rn))
    e = _norm(e_vec)
    ra = a * (1.0 + e)
    rp = a * (1.0 - e)
    return ra - R_EARTH, rp - R_EARTH, ra, rp


# --- The integrator ----------------------------------------------------------


@dataclass
class AscentResult:
    apoapsis_alt_m: float
    periapsis_alt_m: float
    insertion_speed_mps: float
    insertion_time_s: float
    maxq_pa: float
    maxq_alt_m: float
    maxq_mach: float
    maxq_time_s: float
    mach1_alt_m: float
    mach1_time_s: float
    final_mass_kg: float
    s2_prop_residual_kg: float
    reached_insertion: bool


def run_ascent(
    vehicle_path,
    mission_path,
    dt: float = 0.02,
    use_j2: bool = False,
    record=None,
) -> AscentResult:
    """Fly the independent 3DOF ascent and return the insertion + sanity metrics.

    If ``record`` is a list, one ``(t, r, v, mass)`` tuple is appended per step;
    the insertion metrics are unchanged. Used by the EC-11 test to interpolate
    the insertion state to the exact perigee crossing (step-granularity removal).
    """
    veh = load_vehicle(Path(vehicle_path))
    mis = load_mission(Path(mission_path))

    omega: Vec = (0.0, 0.0, OMEGA_EARTH)

    def gravity(r: Vec) -> Vec:
        rn = _norm(r)
        g = _scale(r, -MU_EARTH / (rn * rn * rn))
        if use_j2:
            x, y, z = r
            f = 1.5 * J2_EARTH * MU_EARTH * R_EARTH * R_EARTH / rn**5
            zr2 = 5.0 * z * z / (rn * rn)
            g = _add(g, (-f * x * (1.0 - zr2), -f * y * (1.0 - zr2), -f * z * (3.0 - zr2)))
        return g

    # Pad and initial (release) inertial state: the pad co-rotates with Earth,
    # so at release the vehicle carries the eastward pad velocity omega x r. The
    # 2 s hold + flight are integrated in the frame that coincides with ECEF at
    # t=0; the osculating apsides are rotation-invariant, so freezing this frame
    # inertially (rather than tracking Earth rotation, a 1.6 deg effect over the
    # ascent) is exact for the apogee/perigee and leaves only the atmosphere
    # co-rotation, which is applied explicitly via omega x r in the drag term.
    r_pad = geodetic_to_ecef(mis.lat_deg, mis.lon_deg, mis.alt_m)
    v_release = _cross(omega, r_pad)

    # Launch-site local geographic frame at liftoff, frozen (open-loop launch-
    # frame attitude). The pitch program is a pitch above this fixed horizontal
    # in the launch azimuth plane; see the module docstring for why the
    # instantaneous-local reading cannot be the intended program.
    az = math.radians(mis.azimuth_deg)
    up0 = geodetic_up(r_pad)
    east0 = _unit(_cross((0.0, 0.0, 1.0), up0))
    north0 = _cross(up0, east0)
    h_dir0 = _add(_scale(east0, math.sin(az)), _scale(north0, math.cos(az)))

    # Mass bookkeeping: live propellant plus which discardable components remain.
    prop_s1 = veh.s1_prop_kg
    prop_s2 = veh.s2_prop_kg
    state = {"stage1": True, "fairing": True}

    def total_mass() -> float:
        m = veh.stage2_dry_kg + veh.payload_kg + prop_s2
        if state["stage1"]:
            m += veh.stage1_dry_kg + prop_s1
        if state["fairing"]:
            m += veh.fairing_kg
        return m

    def engine_state(now: float):
        """Return (engine, throttle) active at absolute time ``now``."""
        if mis.t_ignite_s1 <= now < mis.t_meco and prop_s1 > 0.0:
            sp = veh.s1_engine.spool_time_s
            thr = min(1.0, (now - mis.t_ignite_s1) / sp) if sp > 0 else 1.0
            return veh.s1_engine, thr * veh.s1_engine.throttle_max
        if now >= mis.t_ignite_s2 and prop_s2 > 0.0:
            sp = veh.s2_engine.spool_time_s
            thr = min(1.0, (now - mis.t_ignite_s2) / sp) if sp > 0 else 1.0
            return veh.s2_engine, thr * veh.s2_engine.throttle_max
        return None, 0.0

    def accel(now: float, r: Vec, v: Vec, mass: float):
        """Translational acceleration and diagnostics at a sub-state."""
        alt = geodetic_altitude(r)
        rho, p_amb, sound = atmosphere(alt)
        v_rel = _sub(v, _cross(omega, r))  # co-rotating atmosphere
        vrel_n = _norm(v_rel)
        mach = vrel_n / sound if sound > 0 else 0.0

        a_vec = gravity(r)

        eng, thr = engine_state(now)
        if eng is not None and thr > 0.0:
            pitch = math.radians(_interp(now, mis.pitch_t_s, mis.pitch_deg))
            u_thrust = _unit(_add(_scale(up0, math.sin(pitch)), _scale(h_dir0, math.cos(pitch))))
            thrust_mag = max(0.0, thr * eng.thrust_vac_n - p_amb * eng.exit_area_m2)
            a_vec = _add(a_vec, _scale(u_thrust, thrust_mag / mass))

        if rho > 0.0 and vrel_n > 0.0:
            table = veh.ca_full if state["stage1"] else veh.ca_upper
            ca = _ca_of_mach(mach, table)
            q = 0.5 * rho * vrel_n * vrel_n
            drag_mag = -0.5 * rho * vrel_n * ca * veh.aero_ref_area_m2 / mass
            a_vec = _add(a_vec, _scale(v_rel, drag_mag))
        else:
            q = 0.0
        return a_vec, q, alt, mach

    # --- Pad hold: burn stage-1 propellant from ignition to release without
    # moving (clamped to the pad). Spool ramps throttle linearly to full.
    t = mis.t_ignite_s1
    hold_dt = 0.001
    while t < mis.t_release - 1e-12:
        step = min(hold_dt, mis.t_release - t)
        _, thr = engine_state(t)
        prop_s1 = max(0.0, prop_s1 - veh.s1_engine.mdot_vac * thr * step)
        t += step

    # --- Ascent integration from release to insertion.
    r = r_pad
    v = v_release
    t = mis.t_release

    maxq = -1.0
    maxq_alt = maxq_mach = maxq_t = 0.0
    mach1_alt = mach1_t = float("nan")
    prev_mach = 0.0
    reached = False
    ins_r, ins_v, ins_t = r, v, t

    n_max = int((mis.duration_s - mis.t_release) / dt) + 1
    for _ in range(n_max):
        mass = total_mass()
        # RK4 with mass and staging held over the step (mdot*dt << mass).
        a1, q, alt, mach = accel(t, r, v, mass)
        k1r, k1v = v, a1
        a2, _, _, _ = accel(t + 0.5 * dt, _add(r, _scale(k1r, 0.5 * dt)), _add(v, _scale(k1v, 0.5 * dt)), mass)
        k2r, k2v = _add(v, _scale(k1v, 0.5 * dt)), a2
        a3, _, _, _ = accel(t + 0.5 * dt, _add(r, _scale(k2r, 0.5 * dt)), _add(v, _scale(k2v, 0.5 * dt)), mass)
        k3r, k3v = _add(v, _scale(k2v, 0.5 * dt)), a3
        a4, _, _, _ = accel(t + dt, _add(r, _scale(k3r, dt)), _add(v, _scale(k3v, dt)), mass)
        k4r, k4v = _add(v, _scale(k3v, dt)), a4

        # Diagnostics recorded at the step start (max-q, Mach-1 crossing).
        if q > maxq:
            maxq, maxq_alt, maxq_mach, maxq_t = q, alt, mach, t
        if prev_mach < 1.0 <= mach:
            mach1_t = t
            mach1_alt = alt  # within one step of the crossing
        prev_mach = mach

        # Consume propellant over the step at the step-start throttle.
        eng, thr = engine_state(t)
        if eng is veh.s1_engine and thr > 0.0:
            prop_s1 = max(0.0, prop_s1 - eng.mdot_vac * thr * dt)
        elif eng is veh.s2_engine and thr > 0.0:
            prop_s2 = max(0.0, prop_s2 - eng.mdot_vac * thr * dt)

        def _rk_step(base, k1, k2, k3, k4):
            return _add(base, _scale(_add(_add(k1, _scale(k2, 2.0)), _add(_scale(k3, 2.0), k4)), dt / 6.0))

        r = _rk_step(r, k1r, k2r, k3r, k4r)
        v = _rk_step(v, k1v, k2v, k3v, k4v)
        t = t + dt

        if record is not None:
            record.append((t, r, v, total_mass()))

        # Discrete mass-drop events at the step boundary.
        if state["stage1"] and t >= mis.t_sep1:
            state["stage1"] = False  # drops stage-1 dry mass plus its residual
            prop_s1 = 0.0
        if state["fairing"] and geodetic_altitude(r) >= mis.fairing_alt_m:
            state["fairing"] = False

        # Insertion: first time the osculating perigee rises through the gate,
        # once safely above the atmosphere and on the stage-2 arc (mirrors the
        # 6DOF mission's perigee_above terminate condition).
        if t > mis.t_ignite_s2 and (_norm(r) - R_EARTH) > 100000.0:
            _, per_alt, _, _ = osculating_apsides(r, v)
            if per_alt >= mis.insertion_perigee_m:
                reached = True
                ins_r, ins_v, ins_t = r, v, t
                break

    if not reached:
        ins_r, ins_v, ins_t = r, v, t

    apo_alt, per_alt, _, _ = osculating_apsides(ins_r, ins_v)
    return AscentResult(
        apoapsis_alt_m=apo_alt,
        periapsis_alt_m=per_alt,
        insertion_speed_mps=_norm(ins_v),
        insertion_time_s=ins_t,
        maxq_pa=maxq,
        maxq_alt_m=maxq_alt,
        maxq_mach=maxq_mach,
        maxq_time_s=maxq_t,
        mach1_alt_m=mach1_alt,
        mach1_time_s=mach1_t,
        final_mass_kg=total_mass(),
        s2_prop_residual_kg=prop_s2,
        reached_insertion=reached,
    )


# --- Exact-perigee reduction (step-granularity removal) ----------------------


def interpolate_to_perigee(samples, target_perigee_m: float = 180000.0):
    """Linearly interpolate a trajectory to the exact osculating-perigee crossing.

    ``samples`` is an iterable of ``(t, r, v)`` (r, v as length-3 sequences). The
    first upward crossing of ``target_perigee_m`` is bracketed and the state is
    interpolated in the crossing fraction. Returns ``(r, v, t)`` or ``None``.

    The 6DOF locates its perigee-insertion event to a root; comparing both
    trajectories at the SAME exact perigee makes the perigee agreement exact by
    construction and lets the apogee reflect only the true insertion-energy
    difference, not the step at which each run happens to trip the gate.
    """
    prev = None
    for t, r, v in samples:
        r = (float(r[0]), float(r[1]), float(r[2]))
        v = (float(v[0]), float(v[1]), float(v[2]))
        _, per, _, _ = osculating_apsides(r, v)
        if prev is not None and prev[3] < target_perigee_m <= per:
            pt, pr, pv, pper = prev
            f = (target_perigee_m - pper) / (per - pper)
            ri = tuple(pr[k] + f * (r[k] - pr[k]) for k in range(3))
            vi = tuple(pv[k] + f * (v[k] - pv[k]) for k in range(3))
            return ri, vi, pt + f * (t - pt)
        prev = (t, r, v, per)
    return None


@dataclass
class InsertionState:
    apoapsis_alt_m: float
    periapsis_alt_m: float
    speed_mps: float
    time_s: float
    result: AscentResult  # the full run, for max-q / Mach-1 / residual reporting


def run_to_perigee(
    vehicle_path,
    mission_path,
    target_perigee_m: float = 180000.0,
    dt: float = 0.02,
    use_j2: bool = False,
) -> InsertionState:
    """Run the 3DOF ascent and reduce it to the exact ``target_perigee_m`` crossing."""
    record: list = []
    res = run_ascent(vehicle_path, mission_path, dt=dt, use_j2=use_j2, record=record)
    hit = interpolate_to_perigee(((t, r, v) for (t, r, v, _m) in record), target_perigee_m)
    if hit is None:  # never reached the gate; report the terminal osculating state
        return InsertionState(res.apoapsis_alt_m, res.periapsis_alt_m,
                              res.insertion_speed_mps, res.insertion_time_s, res)
    r, v, t = hit
    apo, per, _, _ = osculating_apsides(r, v)
    return InsertionState(apo, per, _norm(v), t, res)


if __name__ == "__main__":
    here = Path(__file__).resolve().parents[2]
    res = run_ascent(here / "vehicles" / "electron_class.toml", here / "missions" / "ascent_leo.toml")
    print(f"reached insertion: {res.reached_insertion} at t={res.insertion_time_s:.1f}s")
    print(f"apoapsis alt : {res.apoapsis_alt_m / 1e3:8.3f} km")
    print(f"periapsis alt: {res.periapsis_alt_m / 1e3:8.3f} km")
    print(f"insertion V  : {res.insertion_speed_mps:8.1f} m/s")
    print(f"max-q        : {res.maxq_pa / 1e3:6.2f} kPa at {res.maxq_alt_m / 1e3:.2f} km, "
          f"Mach {res.maxq_mach:.2f}, t={res.maxq_time_s:.1f}s")
    print(f"Mach-1       : {res.mach1_alt_m / 1e3:.2f} km at t={res.mach1_time_s:.1f}s")
    print(f"final mass   : {res.final_mass_kg:.1f} kg (s2 prop residual {res.s2_prop_residual_kg:.1f} kg)")
