"""Mission TOML validation, canonicalization, and hashing (D-2, FR-14/FR-15 lite).

All parsing and validation live in Python so the C++ core never touches text
(D-2). Validation follows the four-pass-lite discipline: parse, schema with
unknown-key rejection, field ranges, cross-field checks. Every error in the
file is accumulated and reported together (DX-2) so the user fixes the file
once, not once per rerun; a missing critical input always aborts and is never
silently defaulted. Exit-code policy is enforced by the CLI: 2 for validation
failures, 1 for runtime failures.
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

_U64_MAX = 2**64 - 1


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


def _validate_document(doc: dict, errs: _Errors) -> dict | None:
    # Pass 2: schema shape and unknown-key rejection (typos are errors, DX-2).
    for key in doc:
        if key != "schema_version" and key not in _TOP_TABLES:
            errs.add(
                "root",
                key,
                "unknown key",
                hint=(
                    "remove it or fix the spelling; allowed top-level entries are "
                    "schema_version and the tables "
                    + ", ".join(f"[{t}]" for t in _TOP_TABLES)
                ),
            )

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
    if mission is not None:
        _reject_unknown(mission, "mission", {"name", "epoch_utc", "duration_s"}, errs)
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

    integrator = _get_table(doc, "integrator", errs, required=True, hint="must define type and dt_s")
    dt_s = None
    if integrator is not None:
        _reject_unknown(integrator, "integrator", {"type", "dt_s"}, errs)
        itype = _req_str(integrator, "integrator", "type", errs, hint='only "rk4" is accepted in Phase 1')
        if itype is not None and itype != "rk4":
            errs.add(
                "integrator",
                "type",
                f'only "rk4" is accepted in Phase 1, got {itype!r}',
                hint="the adaptive RKF7(8) integrator lands in Phase 2",
            )
        dt_s = _req_num(integrator, "integrator", "dt_s", errs, units="s", typical="0.01 to 10", positive=True)

    mass_kg = _DEFAULT_MASS_KG
    spacecraft = _get_table(doc, "spacecraft", errs, required=False, hint="optional table with mass_kg")
    if spacecraft is not None:
        _reject_unknown(spacecraft, "spacecraft", {"mass_kg"}, errs)
        if "mass_kg" in spacecraft:
            value = _req_num(spacecraft, "spacecraft", "mass_kg", errs, units="kg", typical="1 to 1e6", positive=True)
            mass_kg = value if value is not None else None

    initial_state = _get_table(
        doc,
        "initial_state",
        errs,
        required=True,
        hint="must contain exactly one of the sub-tables cartesian, keplerian, geodetic",
    )
    initial_state_resolved = None
    if initial_state is not None:
        known_forms = ("cartesian", "keplerian", "geodetic")
        forms = [k for k in known_forms if k in initial_state]
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
                hint="provide [initial_state.cartesian] or [initial_state.keplerian] in Phase 1",
            )
        elif len(forms) > 1:
            errs.add(
                "initial_state",
                "|".join(forms),
                f"exactly one initial-state form is required, found {len(forms)}",
                hint="keep one of the sub-tables and delete the others",
            )
        if "geodetic" in forms:
            # Recognized so the message can be specific, but its schema is
            # defined in Phase 4 (FR-14 launch-site form), so its contents
            # are not walked here.
            errs.add(
                "initial_state.geodetic",
                "geodetic",
                "recognized but not accepted: the geodetic launch-site form is supported from Phase 4",
                hint="use cartesian or keplerian in Phase 1",
            )
        cart = kep = None
        if "cartesian" in forms:
            cart = _validate_cartesian(initial_state["cartesian"], errs)
        if "keplerian" in forms:
            kep = _validate_keplerian(initial_state["keplerian"], errs)
        if len(forms) == 1:
            if forms[0] == "cartesian" and cart is not None:
                initial_state_resolved = {"cartesian": cart}
            elif forms[0] == "keplerian" and kep is not None:
                initial_state_resolved = {"keplerian": kep}

    environment = _get_table(doc, "environment", errs, required=True, hint="must define central_body")
    if environment is not None:
        _reject_unknown(environment, "environment", {"central_body"}, errs)
        body = _req_str(environment, "environment", "central_body", errs, hint='only "earth" is accepted in Phase 1')
        if body is not None and body != "earth":
            errs.add(
                "environment",
                "central_body",
                f'only "earth" is accepted in Phase 1, got {body!r}',
                hint="lunar and Mars central bodies land with the Phase 2/3 ephemerides",
            )

    truth_rate_hz = _DEFAULT_TRUTH_RATE_HZ
    logging_tbl = _get_table(doc, "logging", errs, required=False, hint="optional table with truth_rate_hz")
    if logging_tbl is not None:
        _reject_unknown(logging_tbl, "logging", {"truth_rate_hz"}, errs)
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

    if errs.items:
        return None
    return {
        "schema_version": SCHEMA_VERSION,
        "mission": {"name": name, "epoch_utc": epoch, "duration_s": duration_s},
        "run": {"seed": seed},
        "integrator": {"type": "rk4", "dt_s": dt_s},
        "spacecraft": {"mass_kg": mass_kg},
        "initial_state": initial_state_resolved,
        "environment": {"central_body": "earth"},
        "logging": {"truth_rate_hz": truth_rate_hz},
    }


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


def validate_mission_file(path) -> tuple[dict | None, list[str]]:
    """Validate one mission TOML file.

    Returns ``(resolved, errors)``: on success ``resolved`` is the
    defaults-applied configuration dict and ``errors`` is empty; on failure
    ``resolved`` is None and ``errors`` holds every DX-2 formatted error line.
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
    resolved = _validate_document(doc, errs)
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
