"""Mission TOML validation, canonicalization, and hashing (D-2, FR-14/FR-15).

All parsing and validation live in Python so the C++ core never touches text
(D-2). Validation follows the four-pass-lite discipline: parse, schema with
unknown-key rejection, field ranges, cross-field checks. Every error in the
file is accumulated and reported together (DX-2) so the user fixes the file
once, not once per rerun; a missing critical input always aborts and is never
silently defaulted. Exit-code policy is enforced by the CLI: 2 for validation
failures, 1 for runtime failures.

Phase 4 additions (FR-14): schema reference
===========================================

Vehicle reference (root key, optional -- point-mass ``[spacecraft]`` missions
need none)::

    vehicle = "vehicles/electron_class.toml"

The referenced file is validated by ``star_reacher.vehicle`` (relative paths
resolve against the working directory, the same rule as ``[environment]``
paths); its errors are accumulated into this mission's report, and its
config SHA-256 is embedded in the mission's resolved config as
``{"vehicle": {"path", "config_sha256"}}`` so the run's config hash -- the
reproducibility anchor -- covers the vehicle definition (FR-15). With
``strict=True`` the vehicle's warning tier is promoted to errors; otherwise
its warnings surface through ``warnings.warn``.

Geodetic initial state (the FR-14 launch-site form; exactly one of
cartesian | keplerian | geodetic is required)::

    [initial_state.geodetic]
    lat_deg = -39.0     # geodetic latitude on the reference ellipsoid
    lon_deg = 177.9     # east longitude
    alt_m = 10.0        # height above the ellipsoid

The vehicle starts on the rotating pad: inertial velocity v = omega_earth x
r, attitude pad-fixed until a ``pad_release`` sequence event. The geodetic
form therefore requires a ``vehicle`` reference, ``central_body = "earth"``,
and a ``[[sequence]]`` containing exactly one ``pad_release`` entry.

Event sequence (FR-14): ordered named entries, exactly enough vocabulary for
a scripted pad-to-LEO ascent and a TLI burn with no GNC in the loop::

    [[sequence]]
    name = "release"            # unique, referenced by after_event triggers
    trigger = "elapsed"         # "elapsed" | "after_event" | "condition"
    t_s = 0.0
    action = "pad_release"      # see the action vocabulary below

Triggers:

- ``elapsed`` -- ``t_s`` seconds since t0 (must not exceed duration_s).
- ``after_event`` -- ``event`` names an earlier sequence entry;
  ``offset_s >= 0`` after it fires.
- ``condition`` -- ``condition`` is one of ``altitude_above`` /
  ``altitude_below`` (with ``altitude_m``, ellipsoid-relative, ascending or
  descending crossing), ``apoapsis`` / ``periapsis`` (apsis crossing),
  ``perigee_above`` (osculating perigee altitude rises through
  ``perigee_alt_m`` -- the orbit-insertion terminal condition), or
  ``soi_transition`` (with ``body``, entering that body's sphere of
  influence -- the TLI terminal condition; ``body`` differs from the central
  body).

Actions:

- ``pad_release`` -- release the pad-fixed attitude constraint (geodetic
  missions only; exactly one per sequence).
- ``ignite_engine`` / ``cutoff_engine`` -- ``stage`` and ``engine`` name a
  vehicle engine.
- ``separate_stage`` -- ``stage`` names a vehicle stage (FR-10 state remap).
- ``jettison`` -- ``stage`` and ``item`` name a vehicle jettison item.
- ``pitch_program`` -- open-loop pitch-over in the launch-pad tangent frame:
  ``azimuth_deg`` (flight azimuth, degrees east of north), index-matched
  tables ``pitch_t_s`` (>= 2 strictly increasing times) and ``pitch_deg``
  (pitch above the local horizontal, [-90, 90], linearly interpolated,
  held at the end values outside the table). Geodetic missions only.
- ``attitude_hold`` -- hold the attitude at the event inertially fixed.
- ``prograde_hold`` -- velocity-pointing open-loop steering: body +X tracks
  the current inertial velocity each control cycle, so a finite burn stays
  prograde (used by the trans-lunar injection burn).
- ``rate_command`` -- open-loop rate: ``frame`` ("gcrf" | "body"),
  ``omega_dps`` (vec3).
- ``terminate`` -- end the run early (duration_s stays the hard ceiling).

Every action except ``terminate`` requires a vehicle reference. The resolved
config carries the sequence entries in file order.
"""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1

# TOML decimal literals (0.1) are not exactly representable in binary, so the
# duration/step and decimation integer checks cannot use exact arithmetic;
# 1e-9 relative is the contract tolerance (Phase 1 contract section 3).
_REL_TOL = 1e-9

_TOP_TABLES = (
    "mission",
    "run",
    "integrator",
    "spacecraft",
    "initial_state",
    "environment",
    "logging",
)

# D-2 defaults: applied only here, and always recorded in the resolved config
# so no value is ever silently defaulted out of sight.
_DEFAULT_MASS_KG = 1.0
_DEFAULT_TRUTH_RATE_HZ = 10
_DEFAULT_EPHEMERIS = "data/de440s_2020_2060.sreph"  # `star data fetch de440s`
_DEFAULT_HP_EXPONENT_N = 4.0  # Orekit-compatible bulge exponent (ch:harrispriester)

_U64_MAX = 2**64 - 1

_CENTRAL_BODIES = ("earth", "moon", "mars")

# FR-14 v1 sequence vocabulary. Deliberately small: exactly enough for a
# scripted pad-to-LEO ascent and a TLI burn with no GNC in the loop; every
# addition must be documented in the module docstring and earns its own
# validation branch.
_TRIGGERS = ("elapsed", "after_event", "condition")
_CONDITIONS = {
    "altitude_above": ("altitude_m",),
    "altitude_below": ("altitude_m",),
    "apoapsis": (),
    "periapsis": (),
    "perigee_above": ("perigee_alt_m",),
    "soi_transition": ("body",),
}
_ACTIONS = {
    "pad_release": (),
    "ignite_engine": ("stage", "engine"),
    "cutoff_engine": ("stage", "engine"),
    "separate_stage": ("stage",),
    "jettison": ("stage", "item"),
    "pitch_program": ("azimuth_deg", "pitch_t_s", "pitch_deg"),
    "attitude_hold": (),
    "prograde_hold": (),
    "rate_command": ("frame", "omega_dps"),
    "terminate": (),
}
# Everything an attitude or a vehicle part is involved in needs the vehicle;
# only bare termination is meaningful for a point-mass mission.
_VEHICLE_FREE_ACTIONS = ("terminate",)
_RATE_FRAMES = ("gcrf", "body")
# Canonical third-body order: the resolved config records the enabled set in
# this order, which is also the core's fixed force-summation order (D-10), so
# two missions enabling the same set hash identically regardless of file
# order.
_THIRD_BODY_ORDER = ("sun", "earth", "moon", "venus", "mars", "jupiter")
_OCCULTER_BODIES = ("earth", "moon", "mars")
_EARTH_ATMOSPHERES = ("ussa76", "harris_priester")
_MARS_ATMOSPHERES = ("mars_exponential",)


class MissionValidationError(Exception):
    """Carries the accumulated, fully formatted validation error lines."""

    def __init__(self, errors: list[str]):
        super().__init__(f"{len(errors)} validation error(s)")
        self.errors = list(errors)


class _Errors:
    """Accumulates DX-2 formatted error lines for one mission file."""

    def __init__(self, source: str):
        self.source = source
        self.items: list[str] = []

    def add(
        self,
        table: str,
        key: str,
        message: str,
        *,
        units: str | None = None,
        typical: str | None = None,
        hint: str | None = None,
    ) -> None:
        # DX-2 line shape: file, [table.path], key, message, units/typical
        # range where they are meaningful, and the closing no-default
        # statement so a truncated read can never imply a default was used.
        # Top-level keys report their table path as "root".
        if units is not None:
            detail = f" (units: {units}; typical range {typical})"
        elif hint is not None:
            detail = f" ({hint})"
        else:
            detail = ""
        self.items.append(
            f"{self.source}: [{table}] {key}: {message}{detail}. "
            f"No default applied; run aborted."
        )


def _is_number(v) -> bool:
    # TOML booleans arrive as Python bool, a subclass of int; they are never
    # acceptable where a number is required.
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _valid_epoch(value: str) -> bool:
    # datetime.fromisoformat on Python >= 3.11 accepts both "Z" and numeric
    # offsets; requiring tzinfo rejects naive timestamps, whose epoch would
    # be ambiguous. The string itself is carried verbatim into the log header.
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _get_table(doc: dict, name: str, errs: _Errors, *, required: bool, hint: str) -> dict | None:
    if name not in doc:
        if required:
            errs.add("root", name, "missing required table", hint=hint)
        return None
    value = doc[name]
    if not isinstance(value, dict):
        errs.add("root", name, f"expected a table, got {type(value).__name__}", hint=hint)
        return None
    return value


def _reject_unknown(table: dict, path: str, allowed: set[str], errs: _Errors) -> None:
    for key in table:
        if key not in allowed:
            errs.add(
                path,
                key,
                "unknown key",
                hint=f"remove it or fix the spelling; allowed keys here: {', '.join(sorted(allowed))}",
            )


def _req_str(table: dict, path: str, key: str, errs: _Errors, *, hint: str) -> str | None:
    if key not in table:
        errs.add(path, key, "missing required key", hint=hint)
        return None
    value = table[key]
    if not isinstance(value, str):
        errs.add(path, key, f"expected a string, got {type(value).__name__}", hint=hint)
        return None
    return value


def _req_num(
    table: dict,
    path: str,
    key: str,
    errs: _Errors,
    *,
    units: str,
    typical: str,
    positive: bool = False,
) -> float | None:
    if key not in table:
        errs.add(path, key, "missing required key", units=units, typical=typical)
        return None
    value = table[key]
    if not _is_number(value):
        errs.add(
            path,
            key,
            f"expected a number, got {type(value).__name__}",
            units=units,
            typical=typical,
        )
        return None
    value = float(value)
    if not math.isfinite(value):
        errs.add(path, key, f"must be finite, got {value!r}", units=units, typical=typical)
        return None
    if positive and value <= 0.0:
        errs.add(path, key, f"must be > 0, got {value!r}", units=units, typical=typical)
        return None
    return value


def _req_vec3(
    table: dict, path: str, key: str, errs: _Errors, *, units: str, typical: str
) -> list[float] | None:
    if key not in table:
        errs.add(path, key, "missing required key", units=units, typical=typical)
        return None
    value = table[key]
    if (
        not isinstance(value, list)
        or len(value) != 3
        or not all(_is_number(x) for x in value)
    ):
        errs.add(
            path,
            key,
            "expected an array of exactly 3 numbers",
            units=units,
            typical=typical,
        )
        return None
    out = [float(x) for x in value]
    if not all(math.isfinite(x) for x in out):
        errs.add(path, key, f"all components must be finite, got {out!r}", units=units, typical=typical)
        return None
    return out


def _validate_cartesian(table: object, errs: _Errors) -> dict | None:
    path = "initial_state.cartesian"
    if not isinstance(table, dict):
        errs.add("initial_state", "cartesian", "expected a table", hint="e.g. [initial_state.cartesian] with r_m, v_mps, frame")
        return None
    _reject_unknown(table, path, {"r_m", "v_mps", "frame"}, errs)
    r_m = _req_vec3(table, path, "r_m", errs, units="m", typical="magnitude 6.5e6 to 1e9")
    v_mps = _req_vec3(table, path, "v_mps", errs, units="m/s", typical="magnitude 0 to 15000")
    frame = _req_str(table, path, "frame", errs, hint='only "GCRF" is accepted in Phase 1')
    if frame is not None and frame != "GCRF":
        errs.add(
            path,
            "frame",
            f'only "GCRF" is accepted in Phase 1, got {frame!r}',
            hint="additional frames land with the Phase 2 frame family",
        )
        frame = None
    if r_m is not None and math.hypot(*r_m) == 0.0:
        # The two-body acceleration divides by |r|^3; a zero position vector
        # is a guaranteed singularity, not a usable state.
        errs.add(path, "r_m", "position vector must be non-zero", units="m", typical="magnitude 6.5e6 to 1e9")
        r_m = None
    if r_m is None or v_mps is None or frame is None:
        return None
    return {"r_m": r_m, "v_mps": v_mps, "frame": frame}


def _validate_keplerian(table: object, errs: _Errors) -> dict | None:
    path = "initial_state.keplerian"
    if not isinstance(table, dict):
        errs.add(
            "initial_state",
            "keplerian",
            "expected a table",
            hint="e.g. [initial_state.keplerian] with sma_m, ecc, inc_deg, raan_deg, argp_deg, ta_deg",
        )
        return None
    allowed = {"sma_m", "ecc", "inc_deg", "raan_deg", "argp_deg", "ta_deg"}
    _reject_unknown(table, path, allowed, errs)
    sma_m = _req_num(table, path, "sma_m", errs, units="m", typical="6.6e6 to 4.5e8", positive=True)
    ecc = _req_num(table, path, "ecc", errs, units="1", typical="0 to 1 (exclusive)")
    inc_deg = _req_num(table, path, "inc_deg", errs, units="deg", typical="0 to 180")
    raan_deg = _req_num(table, path, "raan_deg", errs, units="deg", typical="0 to 360")
    argp_deg = _req_num(table, path, "argp_deg", errs, units="deg", typical="0 to 360")
    ta_deg = _req_num(table, path, "ta_deg", errs, units="deg", typical="0 to 360")
    if ecc is not None and not (0.0 <= ecc < 1.0):
        # Phase 1 accepts bounded orbits only: the conversion's semi-latus
        # rectum a(1 - e^2) must stay positive with a positive sma_m.
        errs.add(
            path,
            "ecc",
            f"must satisfy 0 <= ecc < 1 in Phase 1 (elliptical orbits only), got {ecc!r}",
            units="1",
            typical="0 to 1 (exclusive)",
        )
        ecc = None
    if inc_deg is not None and not (0.0 <= inc_deg <= 180.0):
        errs.add(path, "inc_deg", f"must be within [0, 180], got {inc_deg!r}", units="deg", typical="0 to 180")
        inc_deg = None
    values = {
        "sma_m": sma_m,
        "ecc": ecc,
        "inc_deg": inc_deg,
        "raan_deg": raan_deg,
        "argp_deg": argp_deg,
        "ta_deg": ta_deg,
    }
    if any(v is None for v in values.values()):
        return None
    return values


def _validate_geodetic(table: object, errs: _Errors) -> dict | None:
    path = "initial_state.geodetic"
    if not isinstance(table, dict):
        errs.add(
            "initial_state",
            "geodetic",
            "expected a table",
            hint="e.g. [initial_state.geodetic] with lat_deg, lon_deg, alt_m",
        )
        return None
    _reject_unknown(table, path, {"lat_deg", "lon_deg", "alt_m"}, errs)
    lat_deg = _req_num(table, path, "lat_deg", errs, units="deg", typical="-90 to 90")
    if lat_deg is not None and not (-90.0 <= lat_deg <= 90.0):
        errs.add(path, "lat_deg", f"must be within [-90, 90], got {lat_deg!r}", units="deg", typical="-90 to 90")
        lat_deg = None
    lon_deg = _req_num(table, path, "lon_deg", errs, units="deg", typical="-180 to 180 (east positive)")
    if lon_deg is not None and not (-180.0 <= lon_deg <= 180.0):
        errs.add(
            path,
            "lon_deg",
            f"must be within [-180, 180], got {lon_deg!r}",
            units="deg",
            typical="-180 to 180 (east positive)",
        )
        lon_deg = None
    alt_m = _req_num(table, path, "alt_m", errs, units="m", typical="0 to 3000 (height above the ellipsoid)")
    if alt_m is not None and not (-500.0 <= alt_m <= 10000.0):
        # Launch sites live between the Dead Sea shore and high plateaus; a
        # value outside this band is a unit mistake, not a pad.
        errs.add(
            path,
            "alt_m",
            f"must be within [-500, 10000], got {alt_m!r}",
            units="m",
            typical="0 to 3000 (height above the ellipsoid)",
        )
        alt_m = None
    values = {"lat_deg": lat_deg, "lon_deg": lon_deg, "alt_m": alt_m}
    if any(v is None for v in values.values()):
        return None
    return values


def _validate_number_array(
    entry: dict, path: str, key: str, errs: _Errors, *, units: str, typical: str
) -> list[float] | None:
    """A required array of >= 2 finite numbers (the pitch-program tables)."""
    if key not in entry:
        errs.add(path, key, "missing required key", units=units, typical=typical)
        return None
    v = entry[key]
    if not isinstance(v, list) or len(v) < 2 or not all(_is_number(x) for x in v):
        errs.add(path, key, "expected an array of at least 2 numbers", units=units, typical=typical)
        return None
    out = [float(x) for x in v]
    if not all(math.isfinite(x) for x in out):
        errs.add(path, key, f"all entries must be finite, got {out!r}", units=units, typical=typical)
        return None
    return out


def _vehicle_ref_maps(vehicle_resolved: dict) -> dict:
    """Stage name -> engine/jettison name sets, for sequence resolution."""
    return {
        stage["name"]: {
            "engines": {e["name"] for e in stage.get("engine", [])},
            "jettison": {j["name"] for j in stage.get("jettison", [])},
        }
        for stage in vehicle_resolved["stage"]
    }


def _validate_sequence_entry(
    entry: dict,
    path: str,
    errs: _Errors,
    *,
    earlier_names: set,
    duration_s: float | None,
    central_body: str | None,
    vehicle_present: bool,
    vehicle_stages: dict | None,
    initial_form: str | None,
) -> dict | None:
    name = _req_str(entry, path, "name", errs, hint='unique event name, e.g. "meco"')
    if name is not None and not name.strip():
        errs.add(path, "name", "must be a non-empty string", hint='e.g. "meco"')
        name = None
    if name is not None and name in earlier_names:
        errs.add(
            path,
            "name",
            f"duplicate event name {name!r}",
            hint="after_event triggers resolve by name; names must be unique",
        )
        name = None

    trigger = _req_str(entry, path, "trigger", errs, hint='"elapsed", "after_event", or "condition"')
    if trigger is not None and trigger not in _TRIGGERS:
        errs.add(
            path,
            "trigger",
            f'must be one of "elapsed", "after_event", "condition", got {trigger!r}',
            hint="the FR-14 v1 trigger vocabulary",
        )
        trigger = None
    action = _req_str(entry, path, "action", errs, hint=", ".join(sorted(_ACTIONS)))
    if action is not None and action not in _ACTIONS:
        errs.add(
            path,
            "action",
            f"unknown action {action!r}",
            hint=f"the FR-14 v1 action vocabulary: {', '.join(sorted(_ACTIONS))}",
        )
        action = None

    condition = None
    if trigger == "condition":
        condition = _req_str(
            entry, path, "condition", errs, hint=", ".join(sorted(_CONDITIONS))
        )
        if condition is not None and condition not in _CONDITIONS:
            errs.add(
                path,
                "condition",
                f"unknown condition {condition!r}",
                hint=f"the FR-14 v1 condition vocabulary: {', '.join(sorted(_CONDITIONS))}",
            )
            condition = None

    # Unknown-key rejection needs the per-type key sets; with an unresolved
    # trigger/action the precise set is unknowable and the type error above
    # already aborts, so the check is skipped rather than guessed.
    if trigger is not None and action is not None and (trigger != "condition" or condition is not None):
        allowed = {"name", "trigger", "action"}
        if trigger == "elapsed":
            allowed |= {"t_s"}
        elif trigger == "after_event":
            allowed |= {"event", "offset_s"}
        else:
            allowed |= {"condition", *_CONDITIONS[condition]}
        allowed |= set(_ACTIONS[action])
        _reject_unknown(entry, path, allowed, errs)

    resolved: dict = {}

    if trigger == "elapsed":
        t_s = _req_num(entry, path, "t_s", errs, units="s", typical="0 to duration_s")
        if t_s is not None and t_s < 0.0:
            errs.add(path, "t_s", f"must be >= 0, got {t_s!r}", units="s", typical="0 to duration_s")
            t_s = None
        if t_s is not None and duration_s is not None and t_s > duration_s:
            errs.add(
                path,
                "t_s",
                f"exceeds [mission] duration_s = {duration_s!r}, so the event can never fire",
                units="s",
                typical="0 to duration_s",
            )
            t_s = None
        resolved["t_s"] = t_s
    elif trigger == "after_event":
        event = _req_str(entry, path, "event", errs, hint="name of an earlier sequence entry")
        if event is not None and event not in earlier_names:
            errs.add(
                path,
                "event",
                f"references {event!r}, which is not an earlier sequence entry",
                hint="triggers may only chain to entries defined earlier in the sequence",
            )
            event = None
        offset_s = _req_num(entry, path, "offset_s", errs, units="s", typical="0 to 600")
        if offset_s is not None and offset_s < 0.0:
            errs.add(path, "offset_s", f"must be >= 0, got {offset_s!r}", units="s", typical="0 to 600")
            offset_s = None
        resolved["event"] = event
        resolved["offset_s"] = offset_s
    elif trigger == "condition" and condition is not None:
        resolved["condition"] = condition
        if condition in ("altitude_above", "altitude_below"):
            altitude_m = _req_num(
                entry, path, "altitude_m", errs, units="m", typical="1e3 to 5e5 (above the ellipsoid)"
            )
            if altitude_m is not None and altitude_m < 0.0:
                errs.add(
                    path,
                    "altitude_m",
                    f"must be >= 0, got {altitude_m!r}",
                    units="m",
                    typical="1e3 to 5e5 (above the ellipsoid)",
                )
                altitude_m = None
            resolved["altitude_m"] = altitude_m
        elif condition == "perigee_above":
            perigee_alt_m = _req_num(
                entry, path, "perigee_alt_m", errs, units="m", typical="1.5e5 to 5e5"
            )
            if perigee_alt_m is not None and perigee_alt_m < 0.0:
                errs.add(
                    path,
                    "perigee_alt_m",
                    f"must be >= 0, got {perigee_alt_m!r}",
                    units="m",
                    typical="1.5e5 to 5e5",
                )
                perigee_alt_m = None
            resolved["perigee_alt_m"] = perigee_alt_m
        elif condition == "soi_transition":
            body = _req_str(entry, path, "body", errs, hint='"earth", "moon", or "mars"')
            if body is not None and body not in _CENTRAL_BODIES:
                errs.add(
                    path,
                    "body",
                    f'must be one of "earth", "moon", "mars", got {body!r}',
                    hint="the FR-12 SOI-transition event set",
                )
                body = None
            if body is not None and central_body is not None and body == central_body:
                errs.add(
                    path,
                    "body",
                    f"the mission already starts inside the SOI of the central body {body!r}",
                    hint="name the body whose sphere of influence is being entered",
                )
                body = None
            resolved["body"] = body

    if action is not None:
        if action not in _VEHICLE_FREE_ACTIONS and not vehicle_present:
            errs.add(
                path,
                "action",
                f"{action!r} requires a vehicle reference",
                hint='set the root key vehicle = "vehicles/<file>.toml"',
            )
        if action in ("pad_release", "pitch_program") and initial_form != "geodetic":
            errs.add(
                path,
                "action",
                f"{action!r} requires the geodetic (launch-site) initial-state form",
                hint="pad release and the pad-frame pitch program are defined "
                "relative to the launch pad (FR-14)",
            )

    if action in ("ignite_engine", "cutoff_engine", "separate_stage", "jettison"):
        stage = _req_str(entry, path, "stage", errs, hint="a stage name from the vehicle file")
        if (
            stage is not None
            and vehicle_stages is not None
            and stage not in vehicle_stages
        ):
            errs.add(
                path,
                "stage",
                f"unknown stage {stage!r} in the referenced vehicle",
                hint=f"stages defined there: {', '.join(repr(s) for s in vehicle_stages)}",
            )
            stage = None
        resolved["stage"] = stage
        if action in ("ignite_engine", "cutoff_engine"):
            engine = _req_str(entry, path, "engine", errs, hint="an engine name from that stage")
            if (
                engine is not None
                and stage is not None
                and vehicle_stages is not None
                and engine not in vehicle_stages[stage]["engines"]
            ):
                errs.add(
                    path,
                    "engine",
                    f"unknown engine {engine!r} in vehicle stage {stage!r}",
                    hint=f"engines defined there: "
                    f"{', '.join(repr(e) for e in sorted(vehicle_stages[stage]['engines'])) or 'none'}",
                )
                engine = None
            resolved["engine"] = engine
        elif action == "jettison":
            item = _req_str(entry, path, "item", errs, hint="a jettison-item name from that stage")
            if (
                item is not None
                and stage is not None
                and vehicle_stages is not None
                and item not in vehicle_stages[stage]["jettison"]
            ):
                errs.add(
                    path,
                    "item",
                    f"unknown jettison item {item!r} in vehicle stage {stage!r}",
                    hint=f"items defined there: "
                    f"{', '.join(repr(j) for j in sorted(vehicle_stages[stage]['jettison'])) or 'none'}",
                )
                item = None
            resolved["item"] = item
    elif action == "pitch_program":
        azimuth_deg = _req_num(
            entry, path, "azimuth_deg", errs, units="deg", typical="0 to 360 (east of north)"
        )
        if azimuth_deg is not None and not (0.0 <= azimuth_deg < 360.0):
            errs.add(
                path,
                "azimuth_deg",
                f"must be within [0, 360), got {azimuth_deg!r}",
                units="deg",
                typical="0 to 360 (east of north)",
            )
            azimuth_deg = None
        pitch_t_s = _validate_number_array(
            entry, path, "pitch_t_s", errs, units="s", typical="strictly increasing from >= 0"
        )
        if pitch_t_s is not None and (
            pitch_t_s[0] < 0.0 or any(b <= a for a, b in zip(pitch_t_s, pitch_t_s[1:]))
        ):
            errs.add(
                path,
                "pitch_t_s",
                f"must be strictly increasing from >= 0, got {pitch_t_s!r}",
                units="s",
                typical="strictly increasing from >= 0",
            )
            pitch_t_s = None
        pitch_deg = _validate_number_array(
            entry, path, "pitch_deg", errs, units="deg", typical="-90 to 90 (above local horizontal)"
        )
        if pitch_deg is not None and not all(-90.0 <= p <= 90.0 for p in pitch_deg):
            errs.add(
                path,
                "pitch_deg",
                f"every entry must be within [-90, 90], got {pitch_deg!r}",
                units="deg",
                typical="-90 to 90 (above local horizontal)",
            )
            pitch_deg = None
        if (
            pitch_t_s is not None
            and pitch_deg is not None
            and len(pitch_t_s) != len(pitch_deg)
        ):
            errs.add(
                path,
                "pitch_deg",
                f"must have one entry per pitch_t_s breakpoint, got {len(pitch_deg)} "
                f"for {len(pitch_t_s)}",
                hint="the two arrays are index-matched",
            )
            pitch_deg = None
        resolved["azimuth_deg"] = azimuth_deg
        resolved["pitch_t_s"] = pitch_t_s
        resolved["pitch_deg"] = pitch_deg
    elif action == "rate_command":
        frame = _req_str(entry, path, "frame", errs, hint='"gcrf" or "body"')
        if frame is not None and frame not in _RATE_FRAMES:
            errs.add(
                path,
                "frame",
                f'must be "gcrf" or "body", got {frame!r}',
                hint="the FR-14 v1 rate-command frames",
            )
            frame = None
        omega_dps = _req_vec3(
            entry, path, "omega_dps", errs, units="deg/s", typical="component magnitudes 0 to 10"
        )
        resolved["frame"] = frame
        resolved["omega_dps"] = omega_dps

    resolved["name"] = name
    resolved["trigger"] = trigger
    resolved["action"] = action
    if any(v is None for v in resolved.values()):
        return None
    return resolved


def _validate_sequence(
    entries: object,
    errs: _Errors,
    *,
    duration_s: float | None,
    central_body: str | None,
    vehicle_present: bool,
    vehicle_stages: dict | None,
    initial_form: str | None,
) -> list | None:
    if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
        errs.add(
            "root",
            "sequence",
            "expected an array of tables ([[sequence]] entries)",
            hint="write each event as its own [[sequence]] table",
        )
        return None
    if not entries:
        errs.add(
            "root",
            "sequence",
            "must contain at least one entry",
            hint="remove the empty sequence or add an event",
        )
        return None
    resolved = []
    earlier_names: set = set()
    releases = 0
    ok = True
    for i, entry in enumerate(entries, 1):
        path = f"sequence.{i}"
        r = _validate_sequence_entry(
            entry,
            path,
            errs,
            earlier_names=earlier_names,
            duration_s=duration_s,
            central_body=central_body,
            vehicle_present=vehicle_present,
            vehicle_stages=vehicle_stages,
            initial_form=initial_form,
        )
        # Raw names still register for after_event chaining and duplicate
        # detection even when the entry has other defects.
        raw_name = entry.get("name")
        if isinstance(raw_name, str) and raw_name.strip():
            earlier_names.add(raw_name)
        if isinstance(entry.get("action"), str) and entry["action"] == "pad_release":
            releases += 1
            if releases > 1:
                errs.add(
                    path,
                    "action",
                    "duplicate pad_release: the pad constraint can only be released once",
                    hint="keep exactly one pad_release entry",
                )
                ok = False
        if r is None:
            ok = False
        else:
            resolved.append(r)
    return resolved if ok else None


def _validate_str_list(
    table: dict, path: str, key: str, errs: _Errors, *, allowed: tuple, hint: str
) -> list[str] | None:
    """A list of unique strings drawn from ``allowed``; None on any defect."""
    value = table[key]
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        errs.add(path, key, "expected an array of strings", hint=hint)
        return None
    bad = [x for x in value if x not in allowed]
    if bad:
        errs.add(
            path,
            key,
            f"unknown name(s) {bad!r}",
            hint=f"allowed: {', '.join(allowed)}",
        )
        return None
    if len(set(value)) != len(value):
        errs.add(path, key, f"duplicate name(s) in {value!r}", hint=hint)
        return None
    return list(value)


def _validate_environment(
    env: dict, errs: _Errors, *, cd_a_over_m: float | None, cr_a_over_m: float | None
) -> dict | None:
    """Validate [environment] (FR-5..FR-9 model selection, FR-6/FR-15 regime
    rules) and return the resolved sub-dict.

    Feature keys enter the resolved config only when the mission enables the
    feature, so pre-Phase-3 mission files resolve byte-identically to their
    Phase 1/2 form (the config-SHA reproducibility anchor, FR-15).
    """
    path = "environment"
    _reject_unknown(
        env, path, {"central_body", "ephemeris", "third_bodies", "gravity", "srp", "drag"}, errs
    )
    central = _req_str(env, path, "central_body", errs, hint='"earth", "moon", or "mars"')
    if central is not None and central not in _CENTRAL_BODIES:
        errs.add(
            path,
            "central_body",
            f'must be one of "earth", "moon", "mars", got {central!r}',
            hint="FR-3/FR-5 central bodies",
        )
        central = None

    resolved: dict = {"central_body": central}

    # --- gravity model selection (FR-5 tiers) ------------------------------
    gravity_resolved = None
    if "gravity" in env:
        gpath = "environment.gravity"
        gtable = env["gravity"]
        if not isinstance(gtable, dict):
            errs.add(path, "gravity", "expected a table", hint='e.g. [environment.gravity] with model = "harmonic", field, degree, order')
            gtable = None
        if gtable is not None:
            _reject_unknown(gtable, gpath, {"model", "field", "degree", "order"}, errs)
            model = _req_str(gtable, gpath, "model", errs, hint='"pointmass", "j2", or "harmonic"')
            if model is not None and model not in ("pointmass", "j2", "harmonic"):
                errs.add(
                    gpath,
                    "model",
                    f'must be "pointmass", "j2", or "harmonic", got {model!r}',
                    hint="the FR-5 fidelity tiers",
                )
                model = None
            field_header = None
            field = None
            if model in ("j2", "harmonic"):
                field = _req_str(
                    gtable,
                    gpath,
                    "field",
                    errs,
                    hint="path to an SRGRAV v1 field file, e.g. data/egm2008_n70.srgrav "
                    "(produced by `star data fetch egm2008|grgm1200a|mro120f`)",
                )
                if field is not None:
                    from star_reacher.data_fetch import DataFetchError, read_srgrav

                    try:
                        field_header = read_srgrav(Path(field))
                    except (OSError, DataFetchError) as exc:
                        errs.add(
                            gpath,
                            "field",
                            f"cannot load SRGRAV field {field!r}: {exc}",
                            hint="fetch it with `star data fetch <dataset>` or fix the path "
                            "(relative paths resolve against the working directory)",
                        )
                        field = None
            elif model == "pointmass":
                for key in ("field", "degree", "order"):
                    if key in gtable:
                        errs.add(
                            gpath,
                            key,
                            'not accepted for model = "pointmass"',
                            hint="the point-mass tier uses the central body's GM only; remove it",
                        )
            if model == "j2":
                for key in ("degree", "order"):
                    if key in gtable:
                        errs.add(
                            gpath,
                            key,
                            'not accepted for model = "j2"',
                            hint="the J2 tier evaluates exactly degree 0 plus C(2,0); remove it",
                        )
                if field_header is not None and field_header.n_max < 2:
                    errs.add(
                        gpath,
                        "field",
                        f"field stores n_max = {field_header.n_max}, but the J2 tier needs C(2,0)",
                        hint="use a field of degree >= 2",
                    )
                    field = None
                if field is not None:
                    gravity_resolved = {"model": "j2", "field": field}
            elif model == "harmonic":
                degree = order = None
                for key, lo in (("degree", 2), ("order", 0)):
                    if key not in gtable:
                        errs.add(gpath, key, "missing required key", units="1", typical=f"{lo} to the field's stored band")
                    elif not _is_int(gtable[key]) or gtable[key] < lo:
                        errs.add(
                            gpath,
                            key,
                            f"expected an integer >= {lo}, got {gtable[key]!r}",
                            units="1",
                            typical=f"{lo} to the field's stored band",
                        )
                    elif key == "degree":
                        degree = gtable[key]
                    else:
                        order = gtable[key]
                if degree is not None and order is not None and order > degree:
                    errs.add(
                        gpath,
                        "order",
                        f"must be <= degree, got order = {order} > degree = {degree}",
                        units="1",
                        typical="0 to degree",
                    )
                    order = None
                if field_header is not None and degree is not None and degree > field_header.n_max:
                    errs.add(
                        gpath,
                        "degree",
                        f"exceeds the field's stored degree band (requested {degree}, "
                        f"stored n_max = {field_header.n_max})",
                        hint="the file carries no information above its band; fetch a deeper "
                        "repack or lower the request (FR-5: the core never silently degrades fidelity)",
                    )
                    degree = None
                if field_header is not None and order is not None and order > field_header.m_max:
                    errs.add(
                        gpath,
                        "order",
                        f"exceeds the field's stored order band (requested {order}, "
                        f"stored m_max = {field_header.m_max})",
                        hint="the file carries no information above its band",
                    )
                    order = None
                if field is not None and degree is not None and order is not None:
                    gravity_resolved = {"model": "harmonic", "field": field, "degree": degree, "order": order}
            elif model == "pointmass":
                gravity_resolved = {"model": "pointmass"}
    if gravity_resolved is not None:
        resolved["gravity"] = gravity_resolved

    # --- third bodies (FR-6) -------------------------------------------------
    third_bodies: list[str] | None = []
    if "third_bodies" in env:
        third_bodies = _validate_str_list(
            env,
            path,
            "third_bodies",
            errs,
            allowed=_THIRD_BODY_ORDER,
            hint='e.g. third_bodies = ["sun", "moon"]',
        )
        if third_bodies is not None and central is not None and central in third_bodies:
            errs.add(
                path,
                "third_bodies",
                f"the central body {central!r} cannot also be a third body",
                hint="remove it from the list",
            )
            third_bodies = None
    # Regime rules (FR-6/FR-15), applied to the effective set (absent = off):
    if third_bodies is not None and central is not None:
        enabled = set(third_bodies)
        if central == "earth" and enabled and not {"sun", "moon"} <= enabled:
            errs.add(
                path,
                "third_bodies",
                f"in the Earth regime the Sun and Moon are always on when third-body "
                f"perturbations are enabled, got {sorted(enabled)!r}",
                hint='FR-6; add "sun" and "moon" (or disable third bodies entirely for '
                "a validation isolation case by omitting the key)",
            )
        if central == "moon" and not {"sun", "earth"} <= enabled:
            errs.add(
                path,
                "third_bodies",
                f"lunar-regime configurations require the Earth and Sun third bodies, "
                f"got {sorted(enabled)!r}",
                hint='FR-15 regime consistency; set third_bodies = ["sun", "earth"] at minimum',
            )
    if third_bodies:
        resolved["third_bodies"] = [b for b in _THIRD_BODY_ORDER if b in third_bodies]

    # --- SRP (FR-7) -----------------------------------------------------------
    srp_enabled = False
    if "srp" in env:
        spath = "environment.srp"
        stable = env["srp"]
        if not isinstance(stable, dict):
            errs.add(path, "srp", "expected a table", hint="e.g. [environment.srp] with optional occulters")
        else:
            srp_enabled = True
            _reject_unknown(stable, spath, {"occulters"}, errs)
            occulters = [central] if central is not None else None
            if "occulters" in stable:
                occulters = _validate_str_list(
                    stable,
                    spath,
                    "occulters",
                    errs,
                    allowed=_OCCULTER_BODIES,
                    hint='e.g. occulters = ["earth", "moon"]',
                )
                if occulters is not None and central is not None and central not in occulters:
                    errs.add(
                        spath,
                        "occulters",
                        f"must include the central body {central!r} (FR-7: the current "
                        f"central body always occults), got {occulters!r}",
                        hint="add it to the list",
                    )
                    occulters = None
            if cr_a_over_m is None:
                errs.add(
                    spath,
                    "occulters" if "occulters" in stable else "srp",
                    "SRP is enabled but [spacecraft] cr_a_over_m_m2pkg is missing",
                    units="m^2/kg",
                    typical="0.001 to 0.05 (Cr*A/m)",
                )
            elif occulters is not None:
                resolved["srp"] = {"occulters": [b for b in _OCCULTER_BODIES if b in occulters]}

    # --- drag (FR-8/FR-9) ------------------------------------------------------
    drag_atmosphere = None
    if "drag" in env:
        dpath = "environment.drag"
        dtable = env["drag"]
        if not isinstance(dtable, dict):
            errs.add(path, "drag", "expected a table", hint='e.g. [environment.drag] with atmosphere = "harris_priester"')
        else:
            _reject_unknown(dtable, dpath, {"atmosphere", "hp_exponent_n"}, errs)
            atmo = _req_str(
                dtable,
                dpath,
                "atmosphere",
                errs,
                hint='"ussa76" or "harris_priester" (Earth), "mars_exponential" (Mars)',
            )
            if atmo is not None and central is not None:
                if central == "earth" and atmo not in _EARTH_ATMOSPHERES:
                    errs.add(
                        dpath,
                        "atmosphere",
                        f'must be "ussa76" or "harris_priester" for central_body = "earth", got {atmo!r}',
                        hint="FR-8 Earth atmospheres",
                    )
                    atmo = None
                elif central == "mars" and atmo not in _MARS_ATMOSPHERES:
                    errs.add(
                        dpath,
                        "atmosphere",
                        f'must be "mars_exponential" for central_body = "mars", got {atmo!r}',
                        hint="FR-8 Mars atmosphere (PRD A-3, confidence low)",
                    )
                    atmo = None
                elif central == "moon":
                    errs.add(
                        dpath,
                        "atmosphere",
                        "the Moon has no atmosphere model; drag cannot be enabled in the lunar regime",
                        hint="remove the [environment.drag] table",
                    )
                    atmo = None
            hp_n = _DEFAULT_HP_EXPONENT_N
            if "hp_exponent_n" in dtable:
                if atmo is not None and atmo != "harris_priester":
                    errs.add(
                        dpath,
                        "hp_exponent_n",
                        'only meaningful for atmosphere = "harris_priester"',
                        hint="remove it",
                    )
                value = dtable["hp_exponent_n"]
                if not _is_number(value) or not (2.0 <= float(value) <= 6.0):
                    errs.add(
                        dpath,
                        "hp_exponent_n",
                        f"expected a number within [2, 6], got {value!r}",
                        units="1",
                        typical="2 (equatorial) to 6 (polar); 4 matches the Orekit default",
                    )
                    hp_n = None
                else:
                    hp_n = float(value)
            if cd_a_over_m is None:
                errs.add(
                    dpath,
                    "atmosphere",
                    "drag is enabled but [spacecraft] cd_a_over_m_m2pkg is missing",
                    units="m^2/kg",
                    typical="0.001 to 0.05 (Cd*A/m)",
                )
            elif atmo is not None and hp_n is not None:
                drag_atmosphere = atmo
                drag_resolved = {"atmosphere": atmo}
                if atmo == "harris_priester":
                    # The default exponent is recorded so it is never a
                    # silent, out-of-sight default (D-2).
                    drag_resolved["hp_exponent_n"] = hp_n
                resolved["drag"] = drag_resolved

    # --- ephemeris ---------------------------------------------------------------
    # Consumers: any third body, SRP (Sun position), Harris-Priester (Sun
    # direction), and the Moon central body (PA-frame librations).
    needs_ephemeris = bool(third_bodies) or srp_enabled or (
        drag_atmosphere == "harris_priester"
    ) or central == "moon"
    if "ephemeris" in env and not needs_ephemeris:
        errs.add(
            path,
            "ephemeris",
            "no configured model consumes an ephemeris (no third bodies, no SRP, "
            "no Harris-Priester drag, central body not moon)",
            hint="remove it: an unused path would perturb the resolved-config hash "
            "without changing the physics",
        )
    elif needs_ephemeris:
        eph = env.get("ephemeris", _DEFAULT_EPHEMERIS)
        if not isinstance(eph, str):
            errs.add(
                path,
                "ephemeris",
                f"expected a string path, got {type(eph).__name__}",
                hint=f'e.g. ephemeris = "{_DEFAULT_EPHEMERIS}"',
            )
        elif not Path(eph).is_file():
            errs.add(
                path,
                "ephemeris",
                f"ephemeris file not found: {eph!r}",
                hint="fetch it with `star data fetch de440s` (default path "
                f"{_DEFAULT_EPHEMERIS!r}) or point at a committed excerpt; relative "
                "paths resolve against the working directory",
            )
        else:
            resolved["ephemeris"] = eph

    return resolved


def _validate_document(doc: dict, errs: _Errors, *, strict: bool = False) -> dict | None:
    # Pass 2: schema shape and unknown-key rejection (typos are errors, DX-2).
    for key in doc:
        if key not in ("schema_version", "vehicle", "sequence") and key not in _TOP_TABLES:
            errs.add(
                "root",
                key,
                "unknown key",
                hint=(
                    "remove it or fix the spelling; allowed top-level entries are "
                    "schema_version, vehicle, [[sequence]], and the tables "
                    + ", ".join(f"[{t}]" for t in _TOP_TABLES)
                ),
            )

    # --- vehicle reference (FR-13/FR-14) -----------------------------------
    # Validated before the initial state and sequence, which cross-reference
    # it. The vehicle file's errors accumulate into this report (each line
    # already names the vehicle file as its source); its config hash enters
    # the resolved config so the mission hash covers the vehicle (FR-15).
    vehicle_present = "vehicle" in doc
    vehicle_entry = None
    vehicle_stages = None
    if vehicle_present:
        vpath = doc["vehicle"]
        if not isinstance(vpath, str) or not vpath.strip():
            errs.add(
                "root",
                "vehicle",
                f"expected a non-empty string path to a vehicle TOML file, got {vpath!r}",
                hint='e.g. vehicle = "vehicles/electron_class.toml"; relative paths '
                "resolve against the working directory",
            )
        else:
            from star_reacher.vehicle import validate_vehicle_file

            vres, verrs, vwarns = validate_vehicle_file(vpath, strict=strict)
            errs.items.extend(verrs)
            if not strict:
                for line in vwarns:
                    warnings.warn(line, UserWarning, stacklevel=4)
            if vres is not None:
                vehicle_stages = _vehicle_ref_maps(vres)
                vehicle_entry = {"path": vpath, "config_sha256": config_sha256(vres)}

    if "schema_version" not in doc:
        errs.add("root", "schema_version", "missing required key", hint="must equal 1, e.g. schema_version = 1")
    elif not _is_int(doc["schema_version"]) or doc["schema_version"] != SCHEMA_VERSION:
        errs.add(
            "root",
            "schema_version",
            f"must equal {SCHEMA_VERSION}, got {doc['schema_version']!r}",
            hint=f"this validator implements mission schema version {SCHEMA_VERSION}",
        )

    mission = _get_table(doc, "mission", errs, required=True, hint="must define name, epoch_utc, duration_s")
    name = epoch = duration_s = None
    # Optional acceptance targets (Phase 4). Present only in vehicle missions
    # whose exit-criterion gate compares the achieved orbit against a
    # mission-file target; pre-Phase-4 missions omit them and resolve
    # byte-identically (the config-SHA reproducibility anchor).
    target_apoapsis_alt_m = None
    target_perilune_alt_m = None
    if mission is not None:
        _reject_unknown(
            mission,
            "mission",
            {
                "name",
                "epoch_utc",
                "duration_s",
                "target_apoapsis_alt_m",
                "target_perilune_alt_m",
            },
            errs,
        )
        name = _req_str(mission, "mission", "name", errs, hint='non-empty string, e.g. "twobody-leo"')
        if name is not None and not name.strip():
            errs.add("mission", "name", "must be a non-empty string", hint='e.g. "twobody-leo"')
            name = None
        epoch = _req_str(
            mission,
            "mission",
            "epoch_utc",
            errs,
            hint='ISO-8601 UTC date-time with Z or a numeric offset, e.g. "2026-01-01T00:00:00Z"',
        )
        if epoch is not None and not _valid_epoch(epoch):
            errs.add(
                "mission",
                "epoch_utc",
                f"not a valid ISO-8601 date-time with an explicit timezone, got {epoch!r}",
                hint='e.g. "2026-01-01T00:00:00Z" or "2026-01-01T00:00:00+00:00"',
            )
            epoch = None
        duration_s = _req_num(
            mission, "mission", "duration_s", errs, units="s", typical="60 to 604800", positive=True
        )
        if "target_apoapsis_alt_m" in mission:
            target_apoapsis_alt_m = _req_num(
                mission,
                "mission",
                "target_apoapsis_alt_m",
                errs,
                units="m",
                typical="1.5e5 to 4e5 (ascent insertion apoapsis altitude)",
                positive=True,
            )
        if "target_perilune_alt_m" in mission:
            target_perilune_alt_m = _req_num(
                mission,
                "mission",
                "target_perilune_alt_m",
                errs,
                units="m",
                typical="1e5 to 5e6 (trans-lunar arrival perilune altitude)",
                positive=True,
            )

    run = _get_table(doc, "run", errs, required=True, hint="must define seed")
    seed = None
    if run is not None:
        _reject_unknown(run, "run", {"seed"}, errs)
        if "seed" not in run:
            errs.add("run", "seed", "missing required key", units="1", typical=f"0 to {_U64_MAX}")
        elif not _is_int(run["seed"]):
            errs.add(
                "run",
                "seed",
                f"expected an integer, got {type(run['seed']).__name__}",
                units="1",
                typical=f"0 to {_U64_MAX}",
            )
        elif not (0 <= run["seed"] <= _U64_MAX):
            # The master seed crosses the binding as a u64 (D-9); anything
            # outside that range cannot be represented faithfully.
            errs.add(
                "run",
                "seed",
                f"must be within [0, 2^64 - 1], got {run['seed']!r}",
                units="1",
                typical=f"0 to {_U64_MAX}",
            )
        else:
            seed = run["seed"]

    integrator = _get_table(
        doc,
        "integrator",
        errs,
        required=True,
        hint='must define type ("rk4" with dt_s, or "rkf78" with rtol, '
        "atol_pos_m, atol_vel_mps, h_init_s, h_max_s)",
    )
    integrator_resolved = None
    dt_s = None
    h_max_s = None
    if integrator is not None:
        allowed_integ = {"type", "dt_s", "rtol", "atol_pos_m", "atol_vel_mps", "h_init_s", "h_max_s"}
        _reject_unknown(integrator, "integrator", allowed_integ, errs)
        itype = _req_str(integrator, "integrator", "type", errs, hint='"rk4" or "rkf78"')
        if itype is not None and itype not in ("rk4", "rkf78"):
            errs.add(
                "integrator",
                "type",
                f'must be "rk4" or "rkf78", got {itype!r}',
                hint="fixed-step classical RK4 or the adaptive Fehlberg RKF7(8) (FR-11)",
            )
            itype = None
        if itype == "rk4":
            for key in ("rtol", "atol_pos_m", "atol_vel_mps", "h_init_s", "h_max_s"):
                if key in integrator:
                    errs.add(
                        "integrator",
                        key,
                        'only meaningful for type = "rkf78"',
                        hint="remove it, or select the adaptive integrator",
                    )
            dt_s = _req_num(integrator, "integrator", "dt_s", errs, units="s", typical="0.01 to 10", positive=True)
            if dt_s is not None:
                integrator_resolved = {"type": "rk4", "dt_s": dt_s}
        elif itype == "rkf78":
            if "dt_s" in integrator:
                errs.add(
                    "integrator",
                    "dt_s",
                    'only meaningful for type = "rk4"',
                    hint="the adaptive step is controlled by rtol/atol and h_init_s/h_max_s",
                )
            rtol = _req_num(integrator, "integrator", "rtol", errs, units="1", typical="1e-13 to 1e-8", positive=True)
            if rtol is not None and rtol > 1e-3:
                errs.add(
                    "integrator",
                    "rtol",
                    f"must be <= 1e-3, got {rtol!r}",
                    units="1",
                    typical="1e-13 to 1e-8",
                )
                rtol = None
            atol_pos_m = _req_num(
                integrator, "integrator", "atol_pos_m", errs, units="m", typical="1e-9 to 1e-3", positive=True
            )
            atol_vel_mps = _req_num(
                integrator, "integrator", "atol_vel_mps", errs, units="m/s", typical="1e-12 to 1e-6", positive=True
            )
            h_init_s = _req_num(
                integrator, "integrator", "h_init_s", errs, units="s", typical="1 to 300", positive=True
            )
            h_max_s = _req_num(
                integrator, "integrator", "h_max_s", errs, units="s", typical="10 to 900", positive=True
            )
            if h_init_s is not None and h_max_s is not None and h_init_s > h_max_s:
                errs.add(
                    "integrator",
                    "h_init_s",
                    f"must be <= h_max_s, got h_init_s = {h_init_s!r} > h_max_s = {h_max_s!r}",
                    units="s",
                    typical="1 to 300",
                )
                h_init_s = None
            values = {
                "rtol": rtol,
                "atol_pos_m": atol_pos_m,
                "atol_vel_mps": atol_vel_mps,
                "h_init_s": h_init_s,
                "h_max_s": h_max_s,
            }
            if all(v is not None for v in values.values()):
                integrator_resolved = {"type": "rkf78", **values}

    mass_kg = _DEFAULT_MASS_KG
    cd_a_over_m = None
    cr_a_over_m = None
    spacecraft = _get_table(
        doc,
        "spacecraft",
        errs,
        required=False,
        hint="optional table with mass_kg, cd_a_over_m_m2pkg, cr_a_over_m_m2pkg",
    )
    if spacecraft is not None:
        _reject_unknown(
            spacecraft, "spacecraft", {"mass_kg", "cd_a_over_m_m2pkg", "cr_a_over_m_m2pkg"}, errs
        )
        if "mass_kg" in spacecraft:
            value = _req_num(spacecraft, "spacecraft", "mass_kg", errs, units="kg", typical="1 to 1e6", positive=True)
            mass_kg = value if value is not None else None
        if "cd_a_over_m_m2pkg" in spacecraft:
            cd_a_over_m = _req_num(
                spacecraft,
                "spacecraft",
                "cd_a_over_m_m2pkg",
                errs,
                units="m^2/kg",
                typical="0.001 to 0.05 (Cd*A/m, FR-9 cannonball)",
                positive=True,
            )
        if "cr_a_over_m_m2pkg" in spacecraft:
            cr_a_over_m = _req_num(
                spacecraft,
                "spacecraft",
                "cr_a_over_m_m2pkg",
                errs,
                units="m^2/kg",
                typical="0.001 to 0.05 (Cr*A/m, FR-7 cannonball)",
                positive=True,
            )

    initial_state = _get_table(
        doc,
        "initial_state",
        errs,
        required=True,
        hint="must contain exactly one of the sub-tables cartesian, keplerian, geodetic",
    )
    initial_state_resolved = None
    initial_form = None
    if initial_state is not None:
        known_forms = ("cartesian", "keplerian", "geodetic")
        forms = [k for k in known_forms if k in initial_state]
        if len(forms) == 1:
            initial_form = forms[0]
        for key in initial_state:
            if key not in known_forms:
                errs.add(
                    "initial_state",
                    key,
                    "unknown key",
                    hint="allowed sub-tables: cartesian, keplerian, geodetic",
                )
        if len(forms) == 0:
            errs.add(
                "initial_state",
                "cartesian|keplerian|geodetic",
                "exactly one initial-state form is required, found none",
                hint="provide [initial_state.cartesian], [initial_state.keplerian], "
                "or [initial_state.geodetic] (the FR-14 launch-site form)",
            )
        elif len(forms) > 1:
            errs.add(
                "initial_state",
                "|".join(forms),
                f"exactly one initial-state form is required, found {len(forms)}",
                hint="keep one of the sub-tables and delete the others",
            )
        cart = kep = geo = None
        if "cartesian" in forms:
            cart = _validate_cartesian(initial_state["cartesian"], errs)
        if "keplerian" in forms:
            kep = _validate_keplerian(initial_state["keplerian"], errs)
        if "geodetic" in forms:
            geo = _validate_geodetic(initial_state["geodetic"], errs)
        if len(forms) == 1:
            if forms[0] == "cartesian" and cart is not None:
                initial_state_resolved = {"cartesian": cart}
            elif forms[0] == "keplerian" and kep is not None:
                initial_state_resolved = {"keplerian": kep}
            elif forms[0] == "geodetic" and geo is not None:
                initial_state_resolved = {"geodetic": geo}

    environment = _get_table(doc, "environment", errs, required=True, hint="must define central_body")
    environment_resolved = None
    if environment is not None:
        environment_resolved = _validate_environment(
            environment, errs, cd_a_over_m=cd_a_over_m, cr_a_over_m=cr_a_over_m
        )
    central_body = environment_resolved["central_body"] if environment_resolved else None

    # --- event sequence (FR-14) ---------------------------------------------
    sequence_resolved = None
    if "sequence" in doc:
        sequence_resolved = _validate_sequence(
            doc["sequence"],
            errs,
            duration_s=duration_s,
            central_body=central_body,
            vehicle_present=vehicle_present,
            vehicle_stages=vehicle_stages,
            initial_form=initial_form,
        )

    # Geodetic cross rules (FR-14): the launch-site form starts on a rotating
    # Earth pad with pad-fixed attitude, so it is meaningless without a
    # vehicle, an Earth central body, and a release event to end the
    # constraint.
    if initial_form == "geodetic":
        if not vehicle_present:
            errs.add(
                "root",
                "vehicle",
                "the geodetic (launch-site) initial-state form requires a vehicle reference",
                hint='set vehicle = "vehicles/<file>.toml"',
            )
        if central_body is not None and central_body != "earth":
            errs.add(
                "initial_state.geodetic",
                "lat_deg",
                f'the geodetic launch form requires central_body = "earth", got {central_body!r}',
                hint="the pad co-rotation velocity v = omega_earth x r is Earth-specific (FR-14)",
            )
        raw_sequence = doc.get("sequence")
        has_release = isinstance(raw_sequence, list) and any(
            isinstance(e, dict) and e.get("action") == "pad_release" for e in raw_sequence
        )
        if not has_release:
            errs.add(
                "root",
                "sequence",
                "the geodetic initial-state form requires a [[sequence]] entry with "
                'action = "pad_release"',
                hint="the vehicle holds pad-fixed attitude until released (FR-14)",
            )

    truth_rate_hz = _DEFAULT_TRUTH_RATE_HZ
    # v1.1 vehicle channel-group rates (FR-16): a value of 0 disables the group,
    # any nonzero value must divide truth_rate_hz (the log decimates from the
    # truth grid). Absent for pre-Phase-4 missions, so those resolve unchanged.
    group_rates: dict[str, int] = {}
    logging_tbl = _get_table(
        doc,
        "logging",
        errs,
        required=False,
        hint="optional table with truth_rate_hz and the vehicle-group rates "
        "forces_rate_hz, mass_rate_hz, env_rate_hz",
    )
    if logging_tbl is not None:
        _reject_unknown(
            logging_tbl,
            "logging",
            {"truth_rate_hz", "forces_rate_hz", "mass_rate_hz", "env_rate_hz"},
            errs,
        )
        if "truth_rate_hz" in logging_tbl:
            value = logging_tbl["truth_rate_hz"]
            if not _is_int(value) or value < 1:
                errs.add(
                    "logging",
                    "truth_rate_hz",
                    f"expected an integer >= 1, got {value!r}",
                    units="Hz",
                    typical="1 to 100",
                )
                truth_rate_hz = None
            else:
                truth_rate_hz = value
        for gkey in ("forces_rate_hz", "mass_rate_hz", "env_rate_hz"):
            if gkey not in logging_tbl:
                continue
            gval = logging_tbl[gkey]
            if not _is_int(gval) or gval < 0:
                errs.add(
                    "logging",
                    gkey,
                    f"expected an integer >= 0 (0 disables the group), got {gval!r}",
                    units="Hz",
                    typical="0 to truth_rate_hz",
                )
            elif gval > 0 and truth_rate_hz is not None and truth_rate_hz % gval != 0:
                errs.add(
                    "logging",
                    gkey,
                    f"must be 0 or an exact divisor of truth_rate_hz "
                    f"({truth_rate_hz}), got {gval!r}; vehicle groups decimate "
                    f"from the truth grid",
                    units="Hz",
                    typical="0 to truth_rate_hz",
                )
            else:
                group_rates[gkey] = gval

    # Pass 4: cross-field checks, run only on fields that survived passes 2-3.
    if duration_s is not None and dt_s is not None:
        steps = duration_s / dt_s
        steps_int = round(steps)
        if steps_int < 1 or abs(steps - steps_int) > _REL_TOL * abs(steps):
            errs.add(
                "mission",
                "duration_s",
                f"must be an integer multiple of [integrator] dt_s within 1e-9 relative "
                f"(duration_s / dt_s = {steps!r})",
                units="s",
                typical="60 to 604800",
            )
    if dt_s is not None and truth_rate_hz is not None:
        decim = 1.0 / (dt_s * truth_rate_hz)
        decim_int = round(decim)
        if decim_int < 1 or abs(decim - decim_int) > _REL_TOL * abs(decim):
            errs.add(
                "logging",
                "truth_rate_hz",
                f"1 / (dt_s * truth_rate_hz) must be an exact positive integer, got {decim!r}; "
                f"the truth log is decimated from integrator steps and never interpolated, "
                f"so its rate cannot exceed or divide unevenly into the step rate",
                units="Hz",
                typical="1 to 100",
            )
    if (
        integrator_resolved is not None
        and integrator_resolved["type"] == "rkf78"
        and duration_s is not None
    ):
        if truth_rate_hz is not None:
            # With adaptive steps the truth log is sampled from the dense
            # output at k / truth_rate_hz; the final record must land exactly
            # on the duration.
            records = duration_s * truth_rate_hz
            if abs(records - round(records)) > _REL_TOL * abs(records):
                errs.add(
                    "logging",
                    "truth_rate_hz",
                    f"duration_s * truth_rate_hz must be an integer, got {records!r}; "
                    f"the adaptive truth log is sampled at k / truth_rate_hz and the "
                    f"final record must land on the duration",
                    units="Hz",
                    typical="1 to 100",
                )
        if h_max_s is not None and h_max_s > duration_s:
            errs.add(
                "integrator",
                "h_max_s",
                f"must be <= [mission] duration_s, got {h_max_s!r} > {duration_s!r}",
                units="s",
                typical="10 to 900",
            )

    if errs.items:
        return None
    spacecraft_resolved = {"mass_kg": mass_kg}
    # Optional ballistic parameters enter the resolved config only when
    # present, so pre-Phase-3 missions keep their byte-identical resolution.
    if cd_a_over_m is not None:
        spacecraft_resolved["cd_a_over_m_m2pkg"] = cd_a_over_m
    if cr_a_over_m is not None:
        spacecraft_resolved["cr_a_over_m_m2pkg"] = cr_a_over_m
    mission_resolved = {"name": name, "epoch_utc": epoch, "duration_s": duration_s}
    if target_apoapsis_alt_m is not None:
        mission_resolved["target_apoapsis_alt_m"] = target_apoapsis_alt_m
    if target_perilune_alt_m is not None:
        mission_resolved["target_perilune_alt_m"] = target_perilune_alt_m
    resolved = {
        "schema_version": SCHEMA_VERSION,
        "mission": mission_resolved,
        "run": {"seed": seed},
        "integrator": integrator_resolved,
        "spacecraft": spacecraft_resolved,
        "initial_state": initial_state_resolved,
        "environment": environment_resolved,
        "logging": {"truth_rate_hz": truth_rate_hz, **group_rates},
    }
    # Phase 4 keys enter the resolved config only when the mission uses them,
    # so pre-Phase-4 missions keep their byte-identical resolution and hash.
    if vehicle_entry is not None:
        resolved["vehicle"] = vehicle_entry
    if sequence_resolved is not None:
        resolved["sequence"] = sequence_resolved
    return resolved


def _warn_if_epoch_past_leap_expiry(epoch_utc: str) -> None:
    """Warn (never error) when the epoch lies beyond the leap-table expiry.

    The bundled leap-second table can only be verified against IERS
    Bulletin C up to its release horizon; the core exposes that expiry
    date programmatically (FR-2, D-6) because it never reads the clock
    (D-2) - the warning decision belongs here. Beyond the expiry the
    conversion silently assumes TAI - UTC stays 37 s, which is why this is
    a warning and not an error: the epoch is still perfectly usable, it is
    merely no longer guaranteed leap-second-exact.
    """
    from star_reacher._corelink import CoreMissingError, import_core

    try:
        core = import_core()
    except CoreMissingError:
        # The table and its expiry live only in the compiled core (one home
        # per constant), and validation must stay fully usable without it.
        # Any code path that goes on to propagate raises the actionable
        # core-missing error itself, so the advisory warning is skipped
        # rather than duplicated in a degraded form.
        return
    info = core.leap_table_info()
    expiry = tuple(info["expiry_utc"])
    # The epoch string was already validated: aware ISO-8601. Comparison is
    # by UTC calendar date, matching how leap-second steps take effect.
    moment = datetime.fromisoformat(epoch_utc).astimezone(timezone.utc)
    if (moment.year, moment.month, moment.day) >= expiry:
        warnings.warn(
            f"epoch_utc {epoch_utc!r} is on or after "
            f"{expiry[0]:04d}-{expiry[1]:02d}-{expiry[2]:02d}, the expiry of "
            f"the bundled leap-second table ({info['version']}); TAI-UTC = "
            f"37 s is assumed for this epoch. Update star_reacher if a leap "
            f"second has been announced since.",
            UserWarning,
            stacklevel=3,
        )


def validate_mission_file(path, *, strict: bool = False) -> tuple[dict | None, list[str]]:
    """Validate one mission TOML file.

    Returns ``(resolved, errors)``: on success ``resolved`` is the
    defaults-applied configuration dict and ``errors`` is empty; on failure
    ``resolved`` is None and ``errors`` holds every DX-2 formatted error line.
    A referenced vehicle file is validated too: its errors join this report,
    and ``strict=True`` promotes its warning tier to errors (FR-15); without
    it, vehicle warnings surface through ``warnings.warn``.
    """
    source = str(path)
    errs = _Errors(source)
    try:
        with open(path, "rb") as fh:
            doc = tomllib.load(fh)
    except OSError as exc:
        errs.items.append(f"{source}: cannot read mission file: {exc}. No default applied; run aborted.")
        return None, errs.items
    except tomllib.TOMLDecodeError as exc:
        # A parse failure leaves no structure to walk, so it is the one
        # class of error that cannot be accumulated with others.
        errs.items.append(f"{source}: TOML parse error: {exc}. No default applied; run aborted.")
        return None, errs.items
    resolved = _validate_document(doc, errs, strict=strict)
    if errs.items:
        return None, errs.items
    _warn_if_epoch_past_leap_expiry(resolved["mission"]["epoch_utc"])
    return resolved, []


def canonical_bytes(resolved: dict) -> bytes:
    """Serialize a resolved config to its canonical byte form (FR-15).

    Sorted keys, compact separators, UTF-8, floats via ``repr`` (json uses
    ``float.__repr__``, the shortest round-trip form): the byte stream, and
    therefore the SHA-256 reproducibility anchor, is independent of TOML
    table order and dict insertion order.
    """
    return json.dumps(
        resolved, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def config_sha256(resolved: dict) -> str:
    """SHA-256 hex digest over exactly the canonical bytes."""
    return hashlib.sha256(canonical_bytes(resolved)).hexdigest()


def keplerian_to_cartesian(kep: dict, gm: float) -> tuple[np.ndarray, np.ndarray]:
    """Classical orbital elements to inertial cartesian position and velocity.

    Standard conversion per Vallado, "Fundamentals of Astrodynamics and
    Applications", 4th ed., Algorithm 10 (COE2RV): build the perifocal-frame
    state from the conic equation, then rotate to the inertial frame with
    R3(-raan) R1(-inc) R3(-argp). The validator restricts inputs to
    elliptical orbits (sma_m > 0, 0 <= ecc < 1), so the semi-latus rectum
    p = a(1 - e^2) is always positive. ``gm`` comes from ``_core.gm`` so the
    gravitational parameter has exactly one home (contract section 3).
    """
    a = float(kep["sma_m"])
    e = float(kep["ecc"])
    inc = math.radians(float(kep["inc_deg"]))
    raan = math.radians(float(kep["raan_deg"]))
    argp = math.radians(float(kep["argp_deg"]))
    nu = math.radians(float(kep["ta_deg"]))

    p = a * (1.0 - e * e)
    r_mag = p / (1.0 + e * math.cos(nu))
    r_pf = np.array([r_mag * math.cos(nu), r_mag * math.sin(nu), 0.0])
    v_pf = math.sqrt(gm / p) * np.array([-math.sin(nu), e + math.cos(nu), 0.0])

    co, so = math.cos(raan), math.sin(raan)
    ci, si = math.cos(inc), math.sin(inc)
    cw, sw = math.cos(argp), math.sin(argp)
    # Perifocal (PQW) to inertial (IJK) direction cosine matrix, written out
    # so the composition order is auditable against the cited algorithm.
    rot = np.array(
        [
            [co * cw - so * sw * ci, -co * sw - so * cw * ci, so * si],
            [so * cw + co * sw * ci, -so * sw + co * cw * ci, -co * si],
            [sw * si, cw * si, ci],
        ]
    )
    return rot @ r_pf, rot @ v_pf
