"""``star plot``: quicklook PNG plots from SRLOG files (FR-18).

Two strictly separated layers:

- **Array preparation** (``prep_*`` functions, ``prepare_all``): pure
  reductions from a loaded :class:`~star_reacher.srlog.Run` to the named
  NumPy "plot-feeding arrays" each plot draws. This layer is what the
  committed golden vectors in ``tests/golden/plots/`` gate (Phase 5 exit
  criterion 1: data-level regression); it never imports matplotlib.
- **Rendering** (``render_plots``): a thin matplotlib layer that turns
  prepared arrays into PNGs. matplotlib is imported lazily and forced onto
  the Agg backend so the command works headless (CI runners, a display-less
  Pi 5) and never depends on an interactive backend.

Naming, units, feeding arrays, tolerance policy, and the overlay-labeling
convention are normative in ``docs/formats/plots.md``.

Conventions chosen here (documented in plots.md with the reasoning):

- The groundtrack uses the exact core GCRF->ITRF chain (the same
  ``frames::c_gcrf_to_itrf`` the propagation itself used, dUT1 = 0 per FR-3)
  rather than a display-grade ERA approximation, and geodetic latitude via
  Bowring's closed form with the same fixed two-pass refinement as the
  core's ``geodetic_altitude`` (Bowring 1976; ch:harrispriester,
  eq:hp:geodetic), so the plotted track is the analysis-grade sub-vehicle
  point, not an approximation of it.
- Attitude is plotted as the logged quaternion components (Hamilton,
  scalar-first, per D-7): the logged representation is continuous and free
  of the coordinate singularities an Euler-angle view would inject.
- Per-source forces/torques are plotted as per-source magnitudes on a
  log-scaled axis (the standard perturbation-budget view, cf. Montenbruck &
  Gill, Satellite Orbits, Sect. 3.1): sources oppose in sign, so a stacked
  signed area would misrepresent cancellation, while log magnitudes keep a
  1e-9-relative perturbation and the dominant central force readable on one
  axis - the form that makes model scrutiny easiest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from star_reacher.srlog import Run

_PKG_DIR = Path(__file__).resolve().parent
_COASTLINE_ASSET = _PKG_DIR / "_assets" / "ne_110m_coastline.json"

# The named FR-18 plot set, in rendering order. Output files are
# "<name>.png"; docs/formats/plots.md is normative for the list.
PLOT_NAMES = [
    "groundtrack",
    "altitude_speed",
    "elements",
    "attitude_rates",
    "mass_thrust_throttle",
    "qbar_mach",
    "forces_by_source",
]

# Overlay labels use this many leading hex digits of the header's resolved
# config SHA-256: 48 bits is far beyond collision range for any practical
# set of overlaid runs while staying readable in a legend.
LABEL_HEX_DIGITS = 12

# WGS-84 semi-major axis [m] and inverse flattening (NGA TR8350.2, 2000).
# Deliberate duplicates of cpp/include/star/constants.hpp WGS84_A_M /
# WGS84_INV_F: this module must work from the log alone, and the values are
# defining constants, not tunables.
_WGS84_A_M = 6378137.0
_WGS84_INV_F = 298.257223563

# Display radii for geocentric-altitude fallback on non-Earth bodies [m]:
# Moon = IAU/IAG recommended mean radius, Mars = IAU/IAG equatorial radius
# (Archinal et al., Celest. Mech. Dyn. Astron. 130:22, 2018). Earth uses
# the geodetic path above, never this table.
_BODY_RADIUS_M = {
    "moon": 1737400.0,
    "mars": 3396190.0,
}


@dataclass
class PrepResult:
    """Prepared feeding arrays for one named plot on one run.

    ``arrays`` is None when the log cannot feed the plot at all (the plot is
    skipped); ``note`` explains any skip or partial content in one line.
    ``meta`` carries non-numeric provenance strings (e.g. which channel the
    altitude came from) for the rendering layer and the documentation trail.
    """

    name: str
    arrays: dict[str, np.ndarray] | None
    note: str | None = None
    meta: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Shared reductions
# ---------------------------------------------------------------------------


def run_label(run: Run) -> str:
    """Short resolved-config hash identifying a run in overlays and legends."""
    sha = run.header.get("config_sha256", "")
    return sha[:LABEL_HEX_DIGITS] if sha else "unknown-config"


def _events_of(run: Run) -> tuple[np.ndarray, list[str]]:
    ev = run.events
    return np.asarray(ev["t_s"], dtype=np.float64), [str(d) for d in ev["detail"]]


def _norm_rows(a: np.ndarray) -> np.ndarray:
    return np.sqrt(np.einsum("ij,ij->i", a, a))


def _epoch_tai(header: dict):
    """Two-part TAI epoch (day, sec) of the header's ``epoch_utc``."""
    from star_reacher._corelink import import_core

    core = import_core()
    dt = datetime.fromisoformat(header["epoch_utc"].replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return core, core.utc_to_tai(
        dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond * 1e-6
    )


def _earth_fixed_positions(run: Run) -> np.ndarray:
    """Truth positions rotated GCRF -> ITRF with the exact core chain.

    One matrix per sample at TAI = epoch + t_s (elapsed log time is SI
    seconds, so the addition is exact), dUT1 = 0 - identical to the frame
    the core's own Earth-fixed evaluations used during the run. Cached on
    the Run instance (same pattern as ``Run.elements``): the groundtrack
    and the derived-altitude path share one conversion.
    """
    cache = getattr(run, "_plot_ecef_cache", None)
    if cache is None:
        core, (day, sec) = _epoch_tai(run.header)
        truth = run.groups["truth"]
        t = np.asarray(truth["t_s"], dtype=np.float64)
        r = np.asarray(truth["r_m"], dtype=np.float64)
        cache = np.empty_like(r)
        for i in range(len(t)):
            d_i, s_i = core.tai_add_seconds(day, sec, float(t[i]))
            m = np.array(core.gcrf_to_itrf(d_i, s_i, 0.0), dtype=np.float64)
            cache[i] = m.reshape(3, 3) @ r[i]
        run._plot_ecef_cache = cache
    return cache


def _geodetic_lat_alt(r_ecef: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized WGS-84 geodetic latitude [rad] and altitude [m].

    Bowring's closed form with the same fixed two-pass refinement as the
    core's ``geodetic_altitude`` (cpp/src/models/atmosphere_hp.cpp,
    eq:hp:geodetic; Bowring, Survey Review 23(181), 1976): sub-millimetre
    over the model's altitude domain, and a fixed pass count keeps the
    evaluation deterministic. The transcription is pinned to the core
    binding by tests/python/test_plot_golden.py.
    """
    f = 1.0 / _WGS84_INV_F
    a = _WGS84_A_M
    b = a * (1.0 - f)
    e2 = f * (2.0 - f)
    ep2 = e2 / (1.0 - e2)
    p = np.hypot(r_ecef[:, 0], r_ecef[:, 1])
    z = r_ecef[:, 2]
    u = np.arctan2(z * a, p * b)
    phi = np.zeros_like(p)
    for _ in range(2):
        su = np.sin(u)
        cu = np.cos(u)
        phi = np.arctan2(z + ep2 * b * su**3, p - e2 * a * cu**3)
        u = np.arctan2(b * np.sin(phi), a * np.cos(phi))
    sphi = np.sin(phi)
    n_rad = a / np.sqrt(1.0 - e2 * sphi * sphi)
    # Off the poles the p/cos(phi) form is robust; near them fall back to
    # the z/sin(phi) form (cos(phi) -> 0 would lose all precision).
    with np.errstate(divide="ignore", invalid="ignore"):
        alt = np.where(
            np.abs(sphi) < 0.99,
            p / np.cos(phi) - n_rad,
            z / np.where(sphi == 0.0, 1.0, sphi) - n_rad * (1.0 - e2),
        )
    return phi, alt


def _nearest_indices(t: np.ndarray, ev_t: np.ndarray) -> np.ndarray:
    """Index of the time sample nearest each event time (ties to the earlier)."""
    hi = np.clip(np.searchsorted(t, ev_t, side="left"), 0, len(t) - 1)
    lo = np.maximum(hi - 1, 0)
    pick_lo = np.abs(t[lo] - ev_t) <= np.abs(t[hi] - ev_t)
    return np.where(pick_lo, lo, hi)


# ---------------------------------------------------------------------------
# Array preparation (the golden-gated layer)
# ---------------------------------------------------------------------------


def prep_groundtrack(run: Run) -> PrepResult:
    """Geodetic sub-vehicle track: t_s, lon_deg in [-180, 180), lat_deg.

    Earth only in Phase 5: the coastline asset and the exact ITRF chain are
    Earth's; a lunar principal-axis track would additionally need the
    ephemeris libration angles, which the log alone does not carry.
    ``ev_*`` arrays are the event markers at the nearest truth sample.
    """
    body = str(run.header.get("central_body", ""))
    if body != "earth":
        return PrepResult(
            "groundtrack",
            None,
            note=f"groundtrack is implemented for central_body 'earth' only "
            f"(this log's central body is {body!r})",
        )
    truth = run.groups["truth"]
    t = np.asarray(truth["t_s"], dtype=np.float64)
    r_ecef = _earth_fixed_positions(run)
    lat_rad, _alt = _geodetic_lat_alt(r_ecef)
    lon_deg = np.degrees(np.arctan2(r_ecef[:, 1], r_ecef[:, 0]))
    # Wrap to [-180, 180) so the rendering wrap-split is a plain jump test.
    lon_deg = np.mod(lon_deg + 180.0, 360.0) - 180.0
    ev_t, _details = _events_of(run)
    idx = _nearest_indices(t, ev_t) if len(ev_t) else np.empty(0, dtype=int)
    return PrepResult(
        "groundtrack",
        {
            "t_s": t,
            "lon_deg": lon_deg,
            "lat_deg": np.degrees(lat_rad),
            "ev_t_s": ev_t,
            "ev_lon_deg": lon_deg[idx],
            "ev_lat_deg": np.degrees(lat_rad)[idx],
        },
        meta={"frame": "ITRF via core gcrf_to_itrf, dUT1 = 0; WGS-84 geodetic"},
    )


def prep_altitude_speed(run: Run) -> PrepResult:
    """Altitude and inertial speed: alt_t_s/alt_m and speed_t_s/speed_mps.

    Altitude source, in preference order: the env group's logged geodetic
    ``alt_m`` when the run logged it; otherwise, for Earth, the same
    exact-frame geodetic derivation as the groundtrack; otherwise geocentric
    ``|r| - R_body`` (display convention; cited radius table). Speed is the
    inertial ``|v_mps|`` from the truth group.
    """
    truth = run.groups["truth"]
    t = np.asarray(truth["t_s"], dtype=np.float64)
    speed = _norm_rows(np.asarray(truth["v_mps"], dtype=np.float64))
    arrays: dict[str, np.ndarray] = {"speed_t_s": t, "speed_mps": speed}
    meta: dict[str, str] = {}
    note = None
    body = str(run.header.get("central_body", ""))
    if "env" in run.groups and "alt_m" in run.groups["env"].dtype.names:
        env = run.groups["env"]
        arrays["alt_t_s"] = np.asarray(env["t_s"], dtype=np.float64)
        arrays["alt_m"] = np.asarray(env["alt_m"], dtype=np.float64)
        meta["alt_source"] = "env.alt_m (logged geodetic altitude)"
    elif body == "earth":
        _lat, alt = _geodetic_lat_alt(_earth_fixed_positions(run))
        arrays["alt_t_s"] = t
        arrays["alt_m"] = alt
        meta["alt_source"] = "derived: core GCRF->ITRF + Bowring WGS-84 geodetic"
    elif body in _BODY_RADIUS_M:
        r = np.asarray(truth["r_m"], dtype=np.float64)
        arrays["alt_t_s"] = t
        arrays["alt_m"] = _norm_rows(r) - _BODY_RADIUS_M[body]
        meta["alt_source"] = f"derived: geocentric |r| - R_{body} (display convention)"
    else:
        note = (
            f"altitude panel skipped: no env group and no reference radius "
            f"for central body {body!r}"
        )
    return PrepResult("altitude_speed", arrays, note=note, meta=meta)


def prep_elements(run: Run) -> PrepResult:
    """Osculating elements in display units: a_km, e, and the angles in deg.

    Element definitions, angle ranges, and singular-geometry conventions are
    ``docs/formats/derived_elements.md`` (the loader's ``Run.elements``);
    this prep only rescales to the plotted units.
    """
    try:
        el = run.elements("truth")
    except (ValueError, KeyError) as exc:
        return PrepResult("elements", None, note=f"elements skipped: {exc}")
    t = np.asarray(run.time_s("truth"), dtype=np.float64)
    return PrepResult(
        "elements",
        {
            "t_s": t,
            "a_km": el["a_m"] / 1000.0,
            "e": el["e"],
            "i_deg": np.degrees(el["i_rad"]),
            "raan_deg": np.degrees(el["raan_rad"]),
            "argp_deg": np.degrees(el["argp_rad"]),
            "nu_deg": np.degrees(el["nu_rad"]),
        },
    )


def prep_attitude_rates(run: Run) -> PrepResult:
    """Attitude quaternion components (q_i2b, scalar-first) and body rates.

    The quaternion is plotted as logged (D-7 Hamilton scalar-first): the
    logged representation is continuous and singularity-free, which an
    Euler-angle view is not. Rates are the body-frame w_b_radps in deg/s.
    """
    truth = run.groups["truth"]
    q = np.asarray(truth["q_i2b"], dtype=np.float64)
    w = np.degrees(np.asarray(truth["w_b_radps"], dtype=np.float64))
    return PrepResult(
        "attitude_rates",
        {
            "t_s": np.asarray(truth["t_s"], dtype=np.float64),
            "qw": q[:, 0],
            "qx": q[:, 1],
            "qy": q[:, 2],
            "qz": q[:, 3],
            "wx_dps": w[:, 0],
            "wy_dps": w[:, 1],
            "wz_dps": w[:, 2],
        },
    )


def prep_mass_thrust_throttle(run: Run) -> PrepResult:
    """Mass, thrust magnitude, and throttle: whichever the log carries.

    Mass prefers the ``mass`` group (higher-fidelity composite properties)
    and falls back to the truth group's ``mass_kg``. Thrust is
    ``|f_thrust_b_n|`` from the forces group. SRLOG v1.1 defines no
    throttle channel; the panel renders when a future (additive
    minor-version) log carries a ``throttle`` channel in any group, and is
    otherwise noted as not logged.
    """
    arrays: dict[str, np.ndarray] = {}
    meta: dict[str, str] = {}
    notes: list[str] = []
    if "mass" in run.groups and "mass_kg" in run.groups["mass"].dtype.names:
        grp = run.groups["mass"]
        meta["mass_source"] = "mass.mass_kg (composite mass-properties group)"
    else:
        grp = run.groups["truth"]
        meta["mass_source"] = "truth.mass_kg"
    arrays["mass_t_s"] = np.asarray(grp["t_s"], dtype=np.float64)
    arrays["mass_kg"] = np.asarray(grp["mass_kg"], dtype=np.float64)

    forces = run.groups.get("forces")
    if forces is not None and "f_thrust_b_n" in forces.dtype.names:
        arrays["thrust_t_s"] = np.asarray(forces["t_s"], dtype=np.float64)
        arrays["thrust_n"] = _norm_rows(
            np.asarray(forces["f_thrust_b_n"], dtype=np.float64)
        )
    else:
        notes.append("thrust panel skipped: no forces group with a thrust source")

    throttle_found = False
    for gname, arr in run.groups.items():
        if "throttle" in (arr.dtype.names or ()) and "t_s" in arr.dtype.names:
            arrays["throttle_t_s"] = np.asarray(arr["t_s"], dtype=np.float64)
            arrays["throttle"] = np.asarray(arr["throttle"], dtype=np.float64)
            meta["throttle_source"] = f"{gname}.throttle"
            throttle_found = True
            break
    if not throttle_found:
        notes.append("throttle panel skipped: no throttle channel in this log")
    return PrepResult(
        "mass_thrust_throttle", arrays, note="; ".join(notes) or None, meta=meta
    )


def prep_qbar_mach(run: Run) -> PrepResult:
    """Dynamic pressure and Mach from the env group: t_s, q_pa, mach."""
    env = run.groups.get("env")
    if env is None:
        return PrepResult(
            "qbar_mach", None, note="qbar_mach skipped: this log has no env group"
        )
    return PrepResult(
        "qbar_mach",
        {
            "t_s": np.asarray(env["t_s"], dtype=np.float64),
            "q_pa": np.asarray(env["q_pa"], dtype=np.float64),
            "mach": np.asarray(env["mach"], dtype=np.float64),
        },
    )


def prep_forces_by_source(run: Run) -> PrepResult:
    """Per-source force/torque magnitudes: <src>_force_n, <src>_torque_nm.

    Sources come from the log's own channel dictionary (``f_<src>_b_n`` /
    ``tq_<src>_b_nm`` pairs), in declaration order, so a future vocabulary
    extension needs no change here. Magnitude form: see the module
    docstring (log-scale perturbation-budget view).
    """
    forces = run.groups.get("forces")
    if forces is None:
        return PrepResult(
            "forces_by_source",
            None,
            note="forces_by_source skipped: this log has no forces group",
        )
    sources = [
        nm[2:-4]
        for nm in forces.dtype.names
        if nm.startswith("f_") and nm.endswith("_b_n")
    ]
    if not sources:
        return PrepResult(
            "forces_by_source",
            None,
            note="forces_by_source skipped: forces group declares no source channels",
        )
    arrays: dict[str, np.ndarray] = {
        "t_s": np.asarray(forces["t_s"], dtype=np.float64)
    }
    for src in sources:
        arrays[f"{src}_force_n"] = _norm_rows(
            np.asarray(forces[f"f_{src}_b_n"], dtype=np.float64)
        )
        tq_name = f"tq_{src}_b_nm"
        if tq_name in forces.dtype.names:
            arrays[f"{src}_torque_nm"] = _norm_rows(
                np.asarray(forces[tq_name], dtype=np.float64)
            )
    return PrepResult(
        "forces_by_source", arrays, meta={"sources": ",".join(sources)}
    )


_PREPS = {
    "groundtrack": prep_groundtrack,
    "altitude_speed": prep_altitude_speed,
    "elements": prep_elements,
    "attitude_rates": prep_attitude_rates,
    "mass_thrust_throttle": prep_mass_thrust_throttle,
    "qbar_mach": prep_qbar_mach,
    "forces_by_source": prep_forces_by_source,
}


def prepare_all(run: Run, plots: list[str] | None = None) -> dict[str, PrepResult]:
    """Prepared arrays for the named plots (default: the full FR-18 set)."""
    names = PLOT_NAMES if plots is None else plots
    unknown = [n for n in names if n not in _PREPS]
    if unknown:
        raise ValueError(
            f"unknown plot name(s) {unknown}; valid names: {', '.join(PLOT_NAMES)}"
        )
    return {name: _PREPS[name](run) for name in names}


# ---------------------------------------------------------------------------
# Rendering (thin matplotlib layer; everything above is matplotlib-free)
# ---------------------------------------------------------------------------

# Okabe-Ito colorblind-safe palette (Okabe & Ito 2008; Wong, Nature Methods
# 8:441, 2011), assigned in FIXED order - never cycled by position in a
# changing series list, so an entity keeps its color across runs and plots.
_OKABE_ITO = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#56B4E9",  # sky blue
    "#CC79A7",  # reddish purple
    "#F0E442",  # yellow
    "#000000",  # black
]

# Fixed color per canonical force source (srlog_v1.md section 3.1), so
# "thrust" is the same color in every star_reacher figure; sources outside
# the vocabulary take unclaimed palette entries in declaration order.
_SOURCE_COLORS = {
    "gravity": "#0072B2",
    "thirdbody": "#56B4E9",
    "srp": "#F0E442",
    "drag": "#009E73",
    "aero": "#E69F00",
    "thrust": "#D55E00",
    "rcs": "#CC79A7",
    "gravgrad": "#999999",
    "wheel": "#000000",
}

# Run identity in overlays: linestyle, cycled by run position on the
# command line (color stays with the channel/source entity).
_RUN_LINESTYLES = ["-", "--", "-.", ":"]

_FIG_DPI = 100
_EVENT_COLOR = "0.45"


@dataclass
class RenderReport:
    """What ``render_plots`` wrote and what it skipped (with reasons)."""

    written: list[Path]
    notes: list[str]


def _style_axes(ax) -> None:
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.tick_params(labelsize=8)


def _mark_events(ax, ev_t, details, linestyle, with_labels) -> None:
    """FR-18 event markers: a vertical line per event on a time axis."""
    for t, detail in zip(ev_t, details):
        ax.axvline(t, color=_EVENT_COLOR, linewidth=0.7, linestyle=linestyle, zorder=1)
        if with_labels:
            ax.annotate(
                detail,
                xy=(t, 0.99),
                xycoords=ax.get_xaxis_transform(),
                rotation=90,
                va="top",
                ha="right",
                fontsize=6,
                color="0.3",
                clip_on=True,
            )


def _mark_all_time_axes(axes, per_run_events, top_ax, single_run) -> None:
    """Event lines on every time axis; label text on the top panel only."""
    for ri, (ev_t, details) in enumerate(per_run_events):
        ls = _RUN_LINESTYLES[ri % len(_RUN_LINESTYLES)]
        for ax in axes:
            _mark_events(ax, ev_t, details, ls, with_labels=single_run and ax is top_ax)


def _series_label(base: str, label: str, single_run: bool) -> str:
    return base if single_run else f"{base} ({label})"


def _lon_split(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Insert NaN breaks at antimeridian wraps so the track never smears."""
    jump = np.abs(np.diff(lon)) > 180.0
    if not jump.any():
        return lon, lat
    idx = np.flatnonzero(jump) + 1
    return np.insert(lon, idx, np.nan), np.insert(lat, idx, np.nan)


def _render_groundtrack(fig, entries, single_run) -> None:
    ax = fig.subplots()
    if _COASTLINE_ASSET.is_file():
        # Reuse of the committed Natural Earth asset (provenance in
        # _assets/README.md); one plot call per polyline segment.
        doc = json.loads(_COASTLINE_ASSET.read_text(encoding="utf-8"))
        for seg in doc["segments"]:
            arr = np.asarray(seg, dtype=np.float64)
            ax.plot(arr[:, 0], arr[:, 1], color="0.65", linewidth=0.5, zorder=1)
    for ri, (label, prep, _events) in enumerate(entries):
        a = prep.arrays
        color = _OKABE_ITO[ri % len(_OKABE_ITO)]
        lon, lat = _lon_split(a["lon_deg"], a["lat_deg"])
        ax.plot(
            lon,
            lat,
            color=color,
            linewidth=1.1,
            zorder=3,
            label=None if single_run else label,
        )
        ax.scatter(
            a["ev_lon_deg"],
            a["ev_lat_deg"],
            marker="v",
            s=18,
            color=color,
            zorder=4,
        )
        if single_run:
            _t, details = _events
            for x, y, detail in zip(a["ev_lon_deg"], a["ev_lat_deg"], details):
                ax.annotate(
                    detail, xy=(x, y), fontsize=6, color="0.3",
                    xytext=(2, 2), textcoords="offset points", clip_on=True,
                )
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xticks(np.arange(-180.0, 181.0, 30.0))
    ax.set_yticks(np.arange(-90.0, 91.0, 30.0))
    ax.set_aspect("equal")
    _style_axes(ax)
    ax.set_xlabel("longitude [deg]", fontsize=9)
    ax.set_ylabel("geodetic latitude [deg]", fontsize=9)
    ax.set_title("groundtrack (ITRF, WGS-84 geodetic)", fontsize=10)
    if not single_run:
        ax.legend(fontsize=7, framealpha=0.6, loc="upper right")


def _overlay_panel(ax, entries, tkey, ykey, base_label, color, single_run, yscale=None):
    """One channel overlaid across runs: fixed color, linestyle per run."""
    for ri, (label, prep, _events) in enumerate(entries):
        a = prep.arrays
        if a is None or ykey not in a:
            continue
        ax.plot(
            a[tkey],
            a[ykey],
            color=color,
            linestyle=_RUN_LINESTYLES[ri % len(_RUN_LINESTYLES)],
            linewidth=1.1,
            label=_series_label(base_label, label, single_run),
        )
    if yscale:
        ax.set_yscale(yscale)
    _style_axes(ax)


def _render_altitude_speed(fig, entries, single_run) -> None:
    ax_alt, ax_spd = fig.subplots(2, 1, sharex=True)
    for ri, (label, prep, _events) in enumerate(entries):
        a = prep.arrays
        ls = _RUN_LINESTYLES[ri % len(_RUN_LINESTYLES)]
        if "alt_m" in a:
            ax_alt.plot(
                a["alt_t_s"], a["alt_m"] / 1000.0,
                color=_OKABE_ITO[0], linestyle=ls, linewidth=1.1,
                label=_series_label("altitude", label, single_run),
            )
        ax_spd.plot(
            a["speed_t_s"], a["speed_mps"],
            color=_OKABE_ITO[1], linestyle=ls, linewidth=1.1,
            label=_series_label("speed", label, single_run),
        )
    ax_alt.set_ylabel("altitude [km]", fontsize=9)
    ax_spd.set_ylabel("inertial speed [m/s]", fontsize=9)
    ax_spd.set_xlabel("t [s]", fontsize=9)
    ax_alt.set_title("altitude and speed", fontsize=10)
    for ax in (ax_alt, ax_spd):
        _style_axes(ax)
    if not single_run:
        ax_alt.legend(fontsize=7, framealpha=0.6, loc="best")
        ax_spd.legend(fontsize=7, framealpha=0.6, loc="best")
    _mark_all_time_axes(
        (ax_alt, ax_spd), [e[2] for e in entries], ax_alt, single_run
    )


_ELEMENT_PANELS = [
    ("a_km", "a [km]"),
    ("e", "e [-]"),
    ("i_deg", "i [deg]"),
    ("raan_deg", "RAAN [deg]"),
    ("argp_deg", "arg periapsis [deg]"),
    ("nu_deg", "true anomaly [deg]"),
]


def _render_elements(fig, entries, single_run) -> None:
    axes = fig.subplots(3, 2, sharex=True)
    flat = axes.ravel()
    for ax, (key, ylabel) in zip(flat, _ELEMENT_PANELS):
        _overlay_panel(ax, entries, "t_s", key, ylabel, _OKABE_ITO[0], single_run)
        ax.set_ylabel(ylabel, fontsize=9)
        if not single_run:
            ax.legend(fontsize=6, framealpha=0.6, loc="best")
    for ax in axes[-1]:
        ax.set_xlabel("t [s]", fontsize=9)
    fig.suptitle("osculating elements (truth group)", fontsize=10)
    _mark_all_time_axes(flat, [e[2] for e in entries], flat[0], single_run)


def _render_attitude_rates(fig, entries, single_run) -> None:
    ax_q, ax_w = fig.subplots(2, 1, sharex=True)
    q_keys = [("qw", 0), ("qx", 1), ("qy", 2), ("qz", 3)]
    w_keys = [("wx_dps", 0), ("wy_dps", 1), ("wz_dps", 2)]
    for key, ci in q_keys:
        _overlay_panel(ax_q, entries, "t_s", key, key, _OKABE_ITO[ci], single_run)
    for key, ci in w_keys:
        _overlay_panel(ax_w, entries, "t_s", key, key, _OKABE_ITO[ci], single_run)
    ax_q.set_ylabel("q_i2b components [-]", fontsize=9)
    ax_w.set_ylabel("body rates [deg/s]", fontsize=9)
    ax_w.set_xlabel("t [s]", fontsize=9)
    ax_q.set_title("attitude quaternion and body rates", fontsize=10)
    ax_q.legend(fontsize=7, framealpha=0.6, loc="best")
    ax_w.legend(fontsize=7, framealpha=0.6, loc="best")
    _mark_all_time_axes((ax_q, ax_w), [e[2] for e in entries], ax_q, single_run)


def _render_mass_thrust_throttle(fig, entries, single_run) -> None:
    panels = []
    keysets = [
        ("mass_t_s", "mass_kg", "mass [kg]", None),
        ("thrust_t_s", "thrust_n", "|thrust| [N]", None),
        ("throttle_t_s", "throttle", "throttle [-]", None),
    ]
    present = [
        ks for ks in keysets
        if any(ks[1] in (e[1].arrays or {}) for e in entries)
    ]
    axes = fig.subplots(len(present), 1, sharex=True, squeeze=False)[:, 0]
    for ax, (tkey, ykey, ylabel, yscale) in zip(axes, present):
        _overlay_panel(ax, entries, tkey, ykey, ylabel, _OKABE_ITO[0], single_run, yscale)
        ax.set_ylabel(ylabel, fontsize=9)
        if not single_run:
            ax.legend(fontsize=7, framealpha=0.6, loc="best")
        panels.append(ax)
    axes[-1].set_xlabel("t [s]", fontsize=9)
    axes[0].set_title("mass, thrust, throttle", fontsize=10)
    _mark_all_time_axes(panels, [e[2] for e in entries], axes[0], single_run)


def _render_qbar_mach(fig, entries, single_run) -> None:
    ax_q, ax_m = fig.subplots(2, 1, sharex=True)
    for ri, (label, prep, _events) in enumerate(entries):
        a = prep.arrays
        ls = _RUN_LINESTYLES[ri % len(_RUN_LINESTYLES)]
        ax_q.plot(
            a["t_s"], a["q_pa"] / 1000.0,
            color=_OKABE_ITO[0], linestyle=ls, linewidth=1.1,
            label=_series_label("dynamic pressure", label, single_run),
        )
        ax_m.plot(
            a["t_s"], a["mach"],
            color=_OKABE_ITO[1], linestyle=ls, linewidth=1.1,
            label=_series_label("Mach", label, single_run),
        )
    ax_q.set_ylabel("dynamic pressure [kPa]", fontsize=9)
    ax_m.set_ylabel("Mach [-]", fontsize=9)
    ax_m.set_xlabel("t [s]", fontsize=9)
    ax_q.set_title("dynamic pressure and Mach", fontsize=10)
    for ax in (ax_q, ax_m):
        _style_axes(ax)
    if not single_run:
        ax_q.legend(fontsize=7, framealpha=0.6, loc="best")
        ax_m.legend(fontsize=7, framealpha=0.6, loc="best")
    _mark_all_time_axes((ax_q, ax_m), [e[2] for e in entries], ax_q, single_run)


def _source_color(src: str, taken: dict[str, str]) -> str:
    if src in _SOURCE_COLORS:
        return _SOURCE_COLORS[src]
    if src not in taken:
        # Out-of-vocabulary source: first unclaimed palette entry, stable in
        # declaration order for a given log.
        used = set(_SOURCE_COLORS.values()) | set(taken.values())
        taken[src] = next(
            (c for c in _OKABE_ITO if c not in used), _OKABE_ITO[-1]
        )
    return taken[src]


def _render_forces_by_source(fig, entries, single_run) -> None:
    ax_f, ax_t = fig.subplots(2, 1, sharex=True)
    extra_colors: dict[str, str] = {}
    panel_max = {id(ax_f): 0.0, id(ax_t): 0.0}
    for ri, (label, prep, _events) in enumerate(entries):
        a = prep.arrays
        ls = _RUN_LINESTYLES[ri % len(_RUN_LINESTYLES)]
        sources = prep.meta.get("sources", "").split(",")
        for src in sources:
            color = _source_color(src, extra_colors)
            for ax, key in ((ax_f, f"{src}_force_n"), (ax_t, f"{src}_torque_nm")):
                if key not in a:
                    continue
                mag = a[key]
                if not (mag > 0.0).any():
                    # An all-zero magnitude has no log-scale representation;
                    # the source simply does not appear in this panel.
                    continue
                panel_max[id(ax)] = max(panel_max[id(ax)], float(mag.max()))
                ax.plot(
                    a["t_s"],
                    np.where(mag > 0.0, mag, np.nan),
                    color=color,
                    linestyle=ls,
                    linewidth=1.1,
                    label=_series_label(src, label, single_run),
                )
    ax_f.set_yscale("log")
    ax_t.set_yscale("log")
    for ax in (ax_f, ax_t):
        top = panel_max[id(ax)]
        if top > 0.0:
            # 12 decades below the panel's dominant source: keeps ~1e-9-
            # relative perturbations visible while stopping numerically-zero
            # startup samples from squashing the axis to unreadability.
            ax.set_ylim(top * 1e-12, top * 5.0)
    ax_f.set_ylabel("|force| by source [N]", fontsize=9)
    ax_t.set_ylabel("|torque| by source [N m]", fontsize=9)
    ax_t.set_xlabel("t [s]", fontsize=9)
    ax_f.set_title("per-source force and torque magnitudes (body frame)", fontsize=10)
    for ax in (ax_f, ax_t):
        _style_axes(ax)
        ax.legend(fontsize=7, framealpha=0.6, loc="best")
    _mark_all_time_axes((ax_f, ax_t), [e[2] for e in entries], ax_f, single_run)


_RENDERERS = {
    "groundtrack": (_render_groundtrack, (10.0, 5.6)),
    "altitude_speed": (_render_altitude_speed, (9.0, 6.5)),
    "elements": (_render_elements, (11.0, 7.5)),
    "attitude_rates": (_render_attitude_rates, (9.0, 6.5)),
    "mass_thrust_throttle": (_render_mass_thrust_throttle, (9.0, 7.5)),
    "qbar_mach": (_render_qbar_mach, (9.0, 6.5)),
    "forces_by_source": (_render_forces_by_source, (10.0, 7.5)),
}


def render_plots(
    runs: list[Run],
    outdir,
    plots: list[str] | None = None,
    labels: list[str] | None = None,
) -> RenderReport:
    """Render the named plots for one run, or overlays for several.

    Writes ``<name>.png`` per plot into ``outdir``. A plot is skipped (with
    a note in the report, never an exception) when no given run can feed
    it; with several runs, shared channels overlay on one axes set, each
    series labeled with the run's short resolved-config hash.
    """
    # Agg before pyplot, unconditionally: `star plot` must never touch a
    # display, and a backend another host process selected must not leak in.
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    names = PLOT_NAMES if plots is None else plots
    if labels is None:
        labels = [run_label(run) for run in runs]
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    single_run = len(runs) == 1
    written: list[Path] = []
    notes: list[str] = []
    prepared = [prepare_all(run, names) for run in runs]
    all_events = [_events_of(run) for run in runs]
    for name in names:
        entries = [
            (labels[ri], prepared[ri][name], all_events[ri])
            for ri in range(len(runs))
            if prepared[ri][name].arrays is not None
        ]
        for ri in range(len(runs)):
            if prepared[ri][name].note:
                notes.append(f"{labels[ri]}: {prepared[ri][name].note}")
        if not entries:
            continue
        renderer, figsize = _RENDERERS[name]
        fig = plt.figure(figsize=figsize, dpi=_FIG_DPI, layout="constrained")
        try:
            renderer(fig, entries, single_run)
            path = out / f"{name}.png"
            # Fixed metadata: the PNG bytes stay a pure function of the log
            # bytes and this module (no matplotlib version string, no
            # timestamps), the FR-21 discipline applied to a derived artifact.
            fig.savefig(path, dpi=_FIG_DPI, metadata={"Software": "star_reacher"})
            written.append(path)
        finally:
            plt.close(fig)
    return RenderReport(written=written, notes=notes)
