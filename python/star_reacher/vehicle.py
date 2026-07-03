"""Vehicle TOML validation, canonicalization, and hashing (D-2/D-3, FR-13/FR-15).

All parsing and validation live in Python so the C++ core never touches text
(D-2). Validation follows the FR-15 four-pass discipline -- (1) parse/schema
with unknown-key rejection everywhere in the file, (2) field ranges, (3)
cross-field physics, (4) vehicle-level sanity -- with every error accumulated
and reported together (DX-2). Required parameters abort when missing and are
never silently defaulted (FR-13). A warning tier carries plausibility
findings (liftoff thrust-to-weight, Isp outside the chemical range, ...);
``strict=True`` promotes warnings to errors.

Schema reference (vehicle TOML v1)
==================================

Frame convention: one structural frame per vehicle (FR-13), +X forward
(toward the nose), +Y/+Z completing a right-handed triad, origin at the aft
plane of the assembled stack. Every position, CG, inertia orientation, axis,
and aero center-of-pressure station in the file is expressed in this frame.
Stage inertia tensors are about the block's own CG, axis-aligned with the
structural frame; the core composes the stack by parallel-axis transport.

Units are SI and appear as suffixes in key names (DX-3), so a wrong unit is
visible at the key. Dimensionless quantities carry no suffix.

Top level::

    schema_version = 1                  # required; this module implements 1
    provenance = "representative"       # required, non-empty (FR-13); states
                                        # where the numbers come from, e.g.
                                        # "representative" for round
                                        # class-representative values

    [vehicle]
    name = "..."                        # required, non-empty
    description = "..."                 # optional free text

Stages are an ordered array of tables, bottom (first-burning) stage first::

    [[stage]]
    name = "stage1"                     # required, unique across stages
    dry_mass_kg = 1000.0                # required, > 0
    dry_cg_m = [x, y, z]                # required, structural frame
    dry_inertia_kgm2 = [[...3x3...]]    # required; symmetric positive-
                                        # definite about the dry CG; the
                                        # principal moments must satisfy the
                                        # rigid-body triangle inequality

Each stage owns optional arrays of sub-blocks (absent arrays are simply
omitted; an empty block list cannot be expressed and is never needed):

``[[stage.tank]]`` -- settled cylindrical propellant tank, axis +X (A-2):
    name (unique in stage), radius_m > 0, length_m > 0, position_m (vec3,
    cylinder center), propellant_mass_kg > 0, density_kgpm3 > 0 (bulk
    density of the load). Cross-check: the propellant must fit the cylinder
    volume implied by radius/length/density.

``[[stage.engine]]`` -- one engine (or one rigid cluster modeled as an
equivalent single engine); F = thrust_vac_N - p_amb * exit_area_m2 (FR-10):
    name (unique in stage), feeds_tank (name of a tank in the same stage),
    thrust_vac_N > 0, isp_vac_s > 0 (constant vacuum Isp), exit_area_m2 > 0,
    position_m (vec3 mount point), axis (unit vec3, nominal thrust force
    direction, usually [1.0, 0.0, 0.0]), gimbal_max_deg in [0, 45] (0 =
    fixed engine), gimbal_rate_dps >= 0, throttle_min and throttle_max in
    (0, 1] with min <= max (fraction of rated thrust; a non-throttleable
    engine states 1.0 for both), spool_time_s >= 0 (linear ramp), ignitions
    (integer >= 1).

``[[stage.rcs]]`` -- one thruster cluster sharing a common thruster size:
    name (unique in stage), thrust_N > 0 (per thruster),
    min_impulse_bit_Ns > 0, thruster_positions_m (array of >= 1 vec3),
    thruster_directions (array of unit vec3, same count; direction of the
    force applied to the vehicle).

``[[stage.wheel]]`` -- one reaction wheel:
    name (unique in stage), axis (unit vec3, spin axis), max_torque_Nm > 0,
    max_momentum_Nms > 0.

``[[stage.sensor]]`` -- one sensor instance; placement plus preset reference
(FR-13). Validated structurally in Phase 4; the preset files and runtime
error models land with the Phase 6 sensor suite, so the preset string is not
dereferenced here:
    name (unique in stage), preset (non-empty string), position_m (vec3),
    axis (unit vec3, boresight or sensitive axis).

``[[stage.jettison]]`` -- one discretely droppable item (fairing, adapter,
payload allowance). Until jettisoned it rides as part of the stack mass:
    name (unique in stage), mass_kg > 0, cg_m (vec3), inertia_kgm2 (3x3,
    same SPD/triangle rules, about the item's own CG).

Aero blocks are an optional array, one per stack configuration (FR-9)::

    [[aero]]
    config = "full_stack"               # required, unique across blocks
    ref_area_m2 = 1.13                  # required, > 0
    ref_diameter_m = 1.2                # required, > 0
    mach_table_csv = "vehicles/x.csv"   # required; path resolves against
                                        # the working directory (same rule
                                        # as mission [environment] paths)
    cmq_per_rad = -0.4                  # optional, <= 0 (constant pitch
                                        # damping; omit for none)

The Mach-table CSV's first non-comment line must be exactly the header
``mach,ca,cnalpha_per_rad,xcp_m`` (units declared by the column names: Mach
and CA dimensionless, CNalpha per radian, xcp meters in the structural
frame). ``#``-prefixed lines are comments. At least two rows; Mach strictly
increasing from >= 0.

Resolved-config echo and hash
=============================

On success the validator returns the resolved configuration: canonical key
order, every number coerced to its canonical type (floats everywhere except
``schema_version`` and ``ignitions``), optional keys present only when given
(FR-13 defines no defaulted physical parameters -- a missing required
parameter is always an abort, so the echo is the canonicalized input).
``canonical_vehicle_toml`` serializes it to the canonical TOML echo:
re-validating the echo reproduces the identical bytes (Phase 4 exit
criterion 1), and ``star_reacher.mission.config_sha256`` over the resolved
dict is the vehicle's reproducibility hash, which mission validation embeds
in the mission's resolved config so the run's config SHA-256 covers the
vehicle (FR-15).
"""

from __future__ import annotations

import json
import math
import tomllib
from pathlib import Path

import numpy as np

# One error-message implementation for the whole config layer: the DX-2 line
# shape is pinned by the mission validator's tests, and a second copy here
# would inevitably drift from it.
from star_reacher.mission import (
    _Errors,
    _is_int,
    _is_number,
    _req_num,
    _req_str,
    _req_vec3,
    _reject_unknown,
)

VEHICLE_SCHEMA_VERSION = 1

# Standard gravity for the warning-tier thrust-to-weight plausibility check
# only; dynamical constants used inside the time loop live in the core (one
# home per constant), but this check never crosses the binding.
_G0_MPS2 = 9.80665

# Sea-level standard pressure for the warning-tier back-pressure check; same
# advisory-only role as _G0_MPS2.
_P_SEA_LEVEL_PA = 101325.0

# Hand-typed direction vectors are accepted to within this norm defect; unit
# vectors written with repr-precision components land at ~1e-16, so 1e-9
# catches genuinely wrong axes without demanding symbolic exactness.
_AXIS_NORM_TOL = 1e-9

# The propellant-fit and triangle-inequality comparisons tolerate 1e-12
# relative so a load computed to exactly match capacity (or a lamina-like
# body) is not rejected for the last floating-point ulp.
_REL_TOL = 1e-12

AERO_CSV_COLUMNS = ("mach", "ca", "cnalpha_per_rad", "xcp_m")

# Required keys per block kind. This is the registry the Phase 4 mutation
# gate walks ("deleting any required key from any starter vehicle yields
# nonzero exit naming that exact key"): tests enumerate instances from the
# fleet files against this table, so a key added to the schema without a
# matching abort is caught by the gate, not silently defaulted.
REQUIRED_KEYS = {
    "root": ("schema_version", "provenance"),
    "vehicle": ("name",),
    "stage": ("name", "dry_mass_kg", "dry_cg_m", "dry_inertia_kgm2"),
    "stage.tank": (
        "name",
        "radius_m",
        "length_m",
        "position_m",
        "propellant_mass_kg",
        "density_kgpm3",
    ),
    "stage.engine": (
        "name",
        "feeds_tank",
        "thrust_vac_N",
        "isp_vac_s",
        "exit_area_m2",
        "position_m",
        "axis",
        "gimbal_max_deg",
        "gimbal_rate_dps",
        "throttle_min",
        "throttle_max",
        "spool_time_s",
        "ignitions",
    ),
    "stage.rcs": (
        "name",
        "thrust_N",
        "min_impulse_bit_Ns",
        "thruster_positions_m",
        "thruster_directions",
    ),
    "stage.wheel": ("name", "axis", "max_torque_Nm", "max_momentum_Nms"),
    "stage.sensor": ("name", "preset", "position_m", "axis"),
    "stage.jettison": ("name", "mass_kg", "cg_m", "inertia_kgm2"),
    "aero": ("config", "ref_area_m2", "ref_diameter_m", "mach_table_csv"),
}

_SUB_BLOCKS = ("tank", "engine", "rcs", "wheel", "sensor", "jettison")

# Canonical key emission order for the TOML echo: schema order (scalars in
# declaration order), which TOML also requires structurally (scalar keys
# before sub-tables). Optional keys are listed so a present value has a
# fixed slot.
_ORDER = {
    "root": ("schema_version", "provenance"),
    "vehicle": ("name", "description"),
    "stage": ("name", "dry_mass_kg", "dry_cg_m", "dry_inertia_kgm2"),
    "stage.tank": REQUIRED_KEYS["stage.tank"],
    "stage.engine": REQUIRED_KEYS["stage.engine"],
    "stage.rcs": REQUIRED_KEYS["stage.rcs"],
    "stage.wheel": REQUIRED_KEYS["stage.wheel"],
    "stage.sensor": REQUIRED_KEYS["stage.sensor"],
    "stage.jettison": REQUIRED_KEYS["stage.jettison"],
    "aero": ("config", "ref_area_m2", "ref_diameter_m", "mach_table_csv", "cmq_per_rad"),
}


class _Report(_Errors):
    """Error accumulator plus the FR-15 warning tier.

    ``items`` (inherited) carries DX-2 error lines; ``warnings`` carries
    warning bases without a closing sentence, so the caller can suffix them
    for the advisory case or promote them verbatim under ``--strict``.
    """

    def __init__(self, source: str):
        super().__init__(source)
        self.warnings: list[str] = []

    def warn(
        self,
        table: str,
        key: str,
        message: str,
        *,
        units: str | None = None,
        typical: str | None = None,
        hint: str | None = None,
    ) -> None:
        if units is not None:
            detail = f" (units: {units}; typical range {typical})"
        elif hint is not None:
            detail = f" ({hint})"
        else:
            detail = ""
        self.warnings.append(f"{self.source}: [{table}] {key}: {message}{detail}")


_WARN_SUFFIX = ". Warning only; --strict promotes warnings to errors."
_STRICT_SUFFIX = ". Promoted to an error by --strict; run aborted."


def _req_unit_vec3(table: dict, path: str, key: str, errs, *, hint: str) -> list[float] | None:
    v = _req_vec3(table, path, key, errs, units="1", typical="unit vector, e.g. [1.0, 0.0, 0.0]")
    if v is None:
        return None
    norm = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if abs(norm - 1.0) > _AXIS_NORM_TOL:
        errs.add(
            path,
            key,
            f"must be unit-norm to within {_AXIS_NORM_TOL:g} (|v| = {norm!r})",
            hint=hint,
        )
        return None
    return v


def _req_inertia(table: dict, path: str, key: str, errs) -> list[list[float]] | None:
    """3x3 inertia tensor: shape, exact symmetry, SPD, triangle inequality.

    Symmetry is demanded exactly as written: the mirrored entries are literal
    values in a curated file, so any disagreement is an authoring error, not
    round-off. SPD and the principal-moment triangle inequality (I1 + I2 >=
    I3, the perpendicular-axis bound every physical mass distribution obeys)
    are checked on the eigenvalues.
    """
    typical = "3x3 symmetric positive-definite matrix about the block's CG"
    if key not in table:
        errs.add(path, key, "missing required key", units="kg*m^2", typical=typical)
        return None
    v = table[key]
    if (
        not isinstance(v, list)
        or len(v) != 3
        or not all(isinstance(row, list) and len(row) == 3 for row in v)
        or not all(_is_number(x) for row in v for x in row)
    ):
        errs.add(
            path,
            key,
            "expected a 3x3 array of numbers (array of three 3-element rows)",
            units="kg*m^2",
            typical=typical,
        )
        return None
    m = [[float(x) for x in row] for row in v]
    if not all(math.isfinite(x) for row in m for x in row):
        errs.add(path, key, "all entries must be finite", units="kg*m^2", typical=typical)
        return None
    for i in range(3):
        for j in range(i + 1, 3):
            if m[i][j] != m[j][i]:
                errs.add(
                    path,
                    key,
                    f"must be symmetric: entry [{i}][{j}] = {m[i][j]!r} but "
                    f"[{j}][{i}] = {m[j][i]!r}",
                    hint="mirrored off-diagonal products of inertia must be written identically",
                )
                return None
    eig = np.linalg.eigvalsh(np.array(m))
    lam = sorted(float(x) for x in eig)
    if lam[0] <= 0.0:
        errs.add(
            path,
            key,
            f"must be positive definite; principal moments {lam!r}",
            units="kg*m^2",
            typical=typical,
        )
        return None
    if lam[0] + lam[1] < lam[2] * (1.0 - _REL_TOL):
        errs.add(
            path,
            key,
            f"principal moments violate the rigid-body triangle inequality "
            f"I1 + I2 >= I3 (got {lam!r})",
            hint="no physical mass distribution produces these moments; recheck the tensor",
        )
        return None
    return m


def _block_names(raw_list, kind: str) -> list[str]:
    """Names as written in a raw block array, for reference resolution.

    Collected from the raw document (not the validated output) so a block
    with an unrelated defect still contributes its name and does not cascade
    false dangling-reference errors.
    """
    names = []
    for entry in raw_list if isinstance(raw_list, list) else []:
        if isinstance(entry, dict) and isinstance(entry.get(kind), str):
            names.append(entry[kind])
    return names


def _check_unique(names: list[tuple[str, str | None]], what: str, errs) -> None:
    """Reject duplicate names; ``names`` pairs each name with its table path."""
    seen: dict[str, str] = {}
    for path, name in names:
        if name is None:
            continue
        if name in seen:
            errs.add(
                path,
                "name",
                f"duplicate {what} name {name!r} (first defined at [{seen[name]}])",
                hint="sequence events and feed mappings resolve by name; names must be unique",
            )
        else:
            seen[name] = path


def _validate_tank(tank: dict, path: str, errs) -> dict | None:
    _reject_unknown(tank, path, set(REQUIRED_KEYS["stage.tank"]), errs)
    name = _req_str(tank, path, "name", errs, hint="unique tank name within the stage")
    radius_m = _req_num(tank, path, "radius_m", errs, units="m", typical="0.1 to 3", positive=True)
    length_m = _req_num(tank, path, "length_m", errs, units="m", typical="0.2 to 20", positive=True)
    position_m = _req_vec3(
        tank, path, "position_m", errs, units="m", typical="component magnitudes 0 to 50"
    )
    propellant_mass_kg = _req_num(
        tank, path, "propellant_mass_kg", errs, units="kg", typical="1 to 5e5", positive=True
    )
    density_kgpm3 = _req_num(
        tank,
        path,
        "density_kgpm3",
        errs,
        units="kg/m^3",
        typical="70 (LH2) to 1450 (NTO); 1030 for kerolox bulk",
        positive=True,
    )
    values = {
        "name": name,
        "radius_m": radius_m,
        "length_m": length_m,
        "position_m": position_m,
        "propellant_mass_kg": propellant_mass_kg,
        "density_kgpm3": density_kgpm3,
    }
    if any(v is None for v in values.values()):
        return None
    capacity_kg = density_kgpm3 * math.pi * radius_m * radius_m * length_m
    if propellant_mass_kg > capacity_kg * (1.0 + _REL_TOL):
        errs.add(
            path,
            "propellant_mass_kg",
            f"exceeds the tank capacity {capacity_kg!r} kg implied by radius_m, "
            f"length_m, and density_kgpm3",
            units="kg",
            typical="at most density * pi * radius^2 * length",
        )
        return None
    return values


def _validate_engine(engine: dict, path: str, errs, tank_names: list[str]) -> dict | None:
    _reject_unknown(engine, path, set(REQUIRED_KEYS["stage.engine"]), errs)
    name = _req_str(engine, path, "name", errs, hint="unique engine name within the stage")
    feeds_tank = _req_str(
        engine, path, "feeds_tank", errs, hint="name of a [[stage.tank]] in the same stage"
    )
    if feeds_tank is not None and feeds_tank not in tank_names:
        errs.add(
            path,
            "feeds_tank",
            f"unknown tank {feeds_tank!r} in this stage",
            hint=(
                f"tanks defined here: {', '.join(repr(n) for n in tank_names)}"
                if tank_names
                else "this stage defines no [[stage.tank]] blocks"
            ),
        )
        feeds_tank = None
    thrust_vac_N = _req_num(
        engine, path, "thrust_vac_N", errs, units="N", typical="10 to 1e7", positive=True
    )
    isp_vac_s = _req_num(
        engine,
        path,
        "isp_vac_s",
        errs,
        units="s",
        typical="typical chemical range 200-465",
        positive=True,
    )
    exit_area_m2 = _req_num(
        engine, path, "exit_area_m2", errs, units="m^2", typical="1e-4 to 5", positive=True
    )
    position_m = _req_vec3(
        engine, path, "position_m", errs, units="m", typical="component magnitudes 0 to 50"
    )
    axis = _req_unit_vec3(
        engine,
        path,
        "axis",
        errs,
        hint="nominal thrust force direction in the structural frame, usually [1.0, 0.0, 0.0]",
    )
    gimbal_max_deg = _req_num(
        engine, path, "gimbal_max_deg", errs, units="deg", typical="0 (fixed) to 10"
    )
    if gimbal_max_deg is not None and not (0.0 <= gimbal_max_deg <= 45.0):
        errs.add(
            path,
            "gimbal_max_deg",
            f"must be within [0, 45], got {gimbal_max_deg!r}",
            units="deg",
            typical="0 (fixed) to 10",
        )
        gimbal_max_deg = None
    gimbal_rate_dps = _req_num(
        engine, path, "gimbal_rate_dps", errs, units="deg/s", typical="0 (fixed) to 30"
    )
    if gimbal_rate_dps is not None and gimbal_rate_dps < 0.0:
        errs.add(
            path,
            "gimbal_rate_dps",
            f"must be >= 0, got {gimbal_rate_dps!r}",
            units="deg/s",
            typical="0 (fixed) to 30",
        )
        gimbal_rate_dps = None
    throttle_min = _req_num(
        engine, path, "throttle_min", errs, units="1", typical="0.4 to 1", positive=True
    )
    throttle_max = _req_num(
        engine, path, "throttle_max", errs, units="1", typical="usually 1.0", positive=True
    )
    for tkey, tval in (("throttle_min", throttle_min), ("throttle_max", throttle_max)):
        if tval is not None and tval > 1.0:
            errs.add(
                path,
                tkey,
                f"must be within (0, 1] (fraction of rated thrust), got {tval!r}",
                units="1",
                typical="0.4 to 1",
            )
            if tkey == "throttle_min":
                throttle_min = None
            else:
                throttle_max = None
    if throttle_min is not None and throttle_max is not None and throttle_min > throttle_max:
        errs.add(
            path,
            "throttle_min",
            f"must be <= throttle_max, got throttle_min = {throttle_min!r} > "
            f"throttle_max = {throttle_max!r}",
            units="1",
            typical="0.4 to 1",
        )
        throttle_min = None
    spool_time_s = _req_num(engine, path, "spool_time_s", errs, units="s", typical="0 to 10")
    if spool_time_s is not None and spool_time_s < 0.0:
        errs.add(
            path,
            "spool_time_s",
            f"must be >= 0, got {spool_time_s!r}",
            units="s",
            typical="0 to 10",
        )
        spool_time_s = None
    ignitions = None
    if "ignitions" not in engine:
        errs.add(path, "ignitions", "missing required key", units="1", typical="1 to 10")
    elif not _is_int(engine["ignitions"]) or engine["ignitions"] < 1:
        errs.add(
            path,
            "ignitions",
            f"expected an integer >= 1, got {engine['ignitions']!r}",
            units="1",
            typical="1 to 10",
        )
    else:
        ignitions = engine["ignitions"]
    values = {
        "name": name,
        "feeds_tank": feeds_tank,
        "thrust_vac_N": thrust_vac_N,
        "isp_vac_s": isp_vac_s,
        "exit_area_m2": exit_area_m2,
        "position_m": position_m,
        "axis": axis,
        "gimbal_max_deg": gimbal_max_deg,
        "gimbal_rate_dps": gimbal_rate_dps,
        "throttle_min": throttle_min,
        "throttle_max": throttle_max,
        "spool_time_s": spool_time_s,
        "ignitions": ignitions,
    }
    if any(v is None for v in values.values()):
        return None
    return values


def _validate_rcs(rcs: dict, path: str, errs) -> dict | None:
    _reject_unknown(rcs, path, set(REQUIRED_KEYS["stage.rcs"]), errs)
    name = _req_str(rcs, path, "name", errs, hint="unique cluster name within the stage")
    thrust_N = _req_num(
        rcs, path, "thrust_N", errs, units="N", typical="0.1 to 500 (per thruster)", positive=True
    )
    mib_Ns = _req_num(
        rcs, path, "min_impulse_bit_Ns", errs, units="N*s", typical="1e-4 to 1", positive=True
    )
    positions = None
    if "thruster_positions_m" not in rcs:
        errs.add(
            path,
            "thruster_positions_m",
            "missing required key",
            units="m",
            typical="array of >= 1 [x, y, z] positions",
        )
    else:
        v = rcs["thruster_positions_m"]
        if (
            not isinstance(v, list)
            or len(v) < 1
            or not all(
                isinstance(p, list)
                and len(p) == 3
                and all(_is_number(x) and math.isfinite(float(x)) for x in p)
                for p in v
            )
        ):
            errs.add(
                path,
                "thruster_positions_m",
                "expected an array of >= 1 finite [x, y, z] positions",
                units="m",
                typical="array of >= 1 [x, y, z] positions",
            )
        else:
            positions = [[float(x) for x in p] for p in v]
    directions = None
    if "thruster_directions" not in rcs:
        errs.add(
            path,
            "thruster_directions",
            "missing required key",
            hint="array of unit vectors, one per thruster; direction of the force on the vehicle",
        )
    else:
        v = rcs["thruster_directions"]
        if (
            not isinstance(v, list)
            or len(v) < 1
            or not all(
                isinstance(d, list)
                and len(d) == 3
                and all(_is_number(x) and math.isfinite(float(x)) for x in d)
                for d in v
            )
        ):
            errs.add(
                path,
                "thruster_directions",
                "expected an array of >= 1 finite [x, y, z] unit vectors",
                hint="one per thruster; direction of the force on the vehicle",
            )
        else:
            directions = [[float(x) for x in d] for d in v]
            for i, d in enumerate(directions):
                norm = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
                if abs(norm - 1.0) > _AXIS_NORM_TOL:
                    errs.add(
                        path,
                        "thruster_directions",
                        f"entry {i + 1} must be unit-norm to within "
                        f"{_AXIS_NORM_TOL:g} (|v| = {norm!r})",
                        hint="direction of the force on the vehicle",
                    )
                    directions = None
                    break
    if positions is not None and directions is not None and len(positions) != len(directions):
        errs.add(
            path,
            "thruster_directions",
            f"must have one entry per thruster position, got {len(directions)} "
            f"directions for {len(positions)} positions",
            hint="the two arrays are index-matched",
        )
        directions = None
    values = {
        "name": name,
        "thrust_N": thrust_N,
        "min_impulse_bit_Ns": mib_Ns,
        "thruster_positions_m": positions,
        "thruster_directions": directions,
    }
    if any(v is None for v in values.values()):
        return None
    return values


def _validate_wheel(wheel: dict, path: str, errs) -> dict | None:
    _reject_unknown(wheel, path, set(REQUIRED_KEYS["stage.wheel"]), errs)
    values = {
        "name": _req_str(wheel, path, "name", errs, hint="unique wheel name within the stage"),
        "axis": _req_unit_vec3(
            wheel, path, "axis", errs, hint="wheel spin axis in the structural frame"
        ),
        "max_torque_Nm": _req_num(
            wheel, path, "max_torque_Nm", errs, units="N*m", typical="0.001 to 1", positive=True
        ),
        "max_momentum_Nms": _req_num(
            wheel,
            path,
            "max_momentum_Nms",
            errs,
            units="N*m*s",
            typical="0.01 to 100",
            positive=True,
        ),
    }
    if any(v is None for v in values.values()):
        return None
    return values


def _validate_sensor(sensor: dict, path: str, errs) -> dict | None:
    _reject_unknown(sensor, path, set(REQUIRED_KEYS["stage.sensor"]), errs)
    name = _req_str(sensor, path, "name", errs, hint="unique sensor name within the stage")
    # The preset path is checked for shape only: preset files and their error
    # models are the Phase 6 sensor suite, so dereferencing here would make
    # every Phase 4 vehicle invalid until that phase lands.
    preset = _req_str(
        sensor, path, "preset", errs, hint='preset reference, e.g. "presets/imu_tactical.toml"'
    )
    if preset is not None and not preset.strip():
        errs.add(path, "preset", "must be a non-empty string", hint="preset reference")
        preset = None
    values = {
        "name": name,
        "preset": preset,
        "position_m": _req_vec3(
            sensor, path, "position_m", errs, units="m", typical="component magnitudes 0 to 50"
        ),
        "axis": _req_unit_vec3(
            sensor, path, "axis", errs, hint="boresight or sensitive axis in the structural frame"
        ),
    }
    if any(v is None for v in values.values()):
        return None
    return values


def _validate_jettison(item: dict, path: str, errs) -> dict | None:
    _reject_unknown(item, path, set(REQUIRED_KEYS["stage.jettison"]), errs)
    values = {
        "name": _req_str(item, path, "name", errs, hint="unique jettison-item name within the stage"),
        "mass_kg": _req_num(item, path, "mass_kg", errs, units="kg", typical="1 to 1e4", positive=True),
        "cg_m": _req_vec3(item, path, "cg_m", errs, units="m", typical="component magnitudes 0 to 50"),
        "inertia_kgm2": _req_inertia(item, path, "inertia_kgm2", errs),
    }
    if any(v is None for v in values.values()):
        return None
    return values


def _validate_aero_csv(path_str: str, apath: str, errs) -> bool:
    """Structural check of a Mach-table CSV (FR-9 columns, declared units)."""
    try:
        text = Path(path_str).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        errs.add(
            apath,
            "mach_table_csv",
            f"cannot read aero table {path_str!r}: {exc}",
            hint="relative paths resolve against the working directory, the same "
            "rule as mission [environment] paths",
        )
        return False
    lines = [ln.strip() for ln in text.splitlines()]
    content = [ln for ln in lines if ln and not ln.startswith("#")]
    header_expected = ",".join(AERO_CSV_COLUMNS)
    if not content or [c.strip() for c in content[0].split(",")] != list(AERO_CSV_COLUMNS):
        errs.add(
            apath,
            "mach_table_csv",
            f"{path_str!r}: first non-comment line must be the header "
            f"{header_expected!r}",
            hint="units are declared by the column names: mach and ca dimensionless, "
            "cnalpha per radian, xcp in meters (structural frame)",
        )
        return False
    rows = []
    for i, ln in enumerate(content[1:], 2):
        parts = [c.strip() for c in ln.split(",")]
        try:
            vals = [float(c) for c in parts]
        except ValueError:
            vals = None
        if vals is None or len(vals) != 4 or not all(math.isfinite(v) for v in vals):
            errs.add(
                apath,
                "mach_table_csv",
                f"{path_str!r}: row {i} must hold 4 finite comma-separated numbers, "
                f"got {ln!r}",
                hint=f"columns: {header_expected}",
            )
            return False
        rows.append(vals)
    if len(rows) < 2:
        errs.add(
            apath,
            "mach_table_csv",
            f"{path_str!r}: at least 2 Mach breakpoints are required, got {len(rows)}",
            hint="the aero model interpolates between breakpoints",
        )
        return False
    machs = [r[0] for r in rows]
    if machs[0] < 0.0 or any(b <= a for a, b in zip(machs, machs[1:])):
        errs.add(
            apath,
            "mach_table_csv",
            f"{path_str!r}: the mach column must be strictly increasing from >= 0, "
            f"got {machs!r}",
            hint="sort the rows by Mach and remove duplicates",
        )
        return False
    return True


def _validate_aero(aero: dict, path: str, errs) -> dict | None:
    allowed = set(REQUIRED_KEYS["aero"]) | {"cmq_per_rad"}
    _reject_unknown(aero, path, allowed, errs)
    config = _req_str(aero, path, "config", errs, hint="stack-configuration name, unique across aero blocks")
    ref_area_m2 = _req_num(
        aero, path, "ref_area_m2", errs, units="m^2", typical="0.1 to 30", positive=True
    )
    ref_diameter_m = _req_num(
        aero, path, "ref_diameter_m", errs, units="m", typical="0.3 to 10", positive=True
    )
    table_csv = _req_str(
        aero,
        path,
        "mach_table_csv",
        errs,
        hint="path to the CA/CNalpha/xcp Mach-table CSV; relative paths resolve "
        "against the working directory",
    )
    if table_csv is not None and not _validate_aero_csv(table_csv, path, errs):
        table_csv = None
    values = {
        "config": config,
        "ref_area_m2": ref_area_m2,
        "ref_diameter_m": ref_diameter_m,
        "mach_table_csv": table_csv,
    }
    if any(v is None for v in values.values()):
        return None
    if "cmq_per_rad" in aero:
        cmq = aero["cmq_per_rad"]
        if not _is_number(cmq) or not math.isfinite(float(cmq)) or float(cmq) > 0.0:
            errs.add(
                path,
                "cmq_per_rad",
                f"expected a finite number <= 0 (pitch damping opposes the rate), "
                f"got {cmq!r}",
                units="1/rad",
                typical="-60 to 0",
            )
            return None
        values["cmq_per_rad"] = float(cmq)
    return values


def _validate_stage(stage: dict, path: str, errs) -> dict | None:
    allowed = set(REQUIRED_KEYS["stage"]) | set(_SUB_BLOCKS)
    _reject_unknown(stage, path, allowed, errs)
    resolved: dict = {
        "name": _req_str(stage, path, "name", errs, hint="unique stage name, e.g. \"stage1\""),
        "dry_mass_kg": _req_num(
            stage, path, "dry_mass_kg", errs, units="kg", typical="10 to 1e5", positive=True
        ),
        "dry_cg_m": _req_vec3(
            stage, path, "dry_cg_m", errs, units="m", typical="component magnitudes 0 to 50"
        ),
        "dry_inertia_kgm2": _req_inertia(stage, path, "dry_inertia_kgm2", errs),
    }
    if resolved["name"] is not None and not resolved["name"].strip():
        errs.add(path, "name", "must be a non-empty string", hint='e.g. "stage1"')
        resolved["name"] = None
    ok = all(v is not None for v in resolved.values())

    tank_names = _block_names(stage.get("tank"), "name")
    validators = {
        "tank": lambda blk, bpath: _validate_tank(blk, bpath, errs),
        "engine": lambda blk, bpath: _validate_engine(blk, bpath, errs, tank_names),
        "rcs": lambda blk, bpath: _validate_rcs(blk, bpath, errs),
        "wheel": lambda blk, bpath: _validate_wheel(blk, bpath, errs),
        "sensor": lambda blk, bpath: _validate_sensor(blk, bpath, errs),
        "jettison": lambda blk, bpath: _validate_jettison(blk, bpath, errs),
    }
    for kind in _SUB_BLOCKS:
        if kind not in stage:
            continue
        raw = stage[kind]
        if not isinstance(raw, list) or not all(isinstance(b, dict) for b in raw):
            errs.add(
                path,
                kind,
                f"expected an array of tables ([[stage.{kind}]] entries)",
                hint=f"write each block as its own [[stage.{kind}]] table",
            )
            ok = False
            continue
        out = []
        for j, blk in enumerate(raw, 1):
            bpath = f"{path}.{kind}.{j}"
            v = validators[kind](blk, bpath)
            if v is None:
                ok = False
            else:
                out.append(v)
        _check_unique(
            [(f"{path}.{kind}.{j}", b.get("name") if isinstance(b.get("name"), str) else None)
             for j, b in enumerate(raw, 1)],
            f"stage.{kind}",
            errs,
        )
        if ok and len(out) == len(raw):
            resolved[kind] = out
    return resolved if ok else None


def _vehicle_sanity(resolved: dict, rep: _Report) -> None:
    """Pass 4: vehicle-level sanity. Hard errors for impossibilities,
    warnings for implausibilities (promoted under --strict).

    Runs only on a document that survived passes 1-3, so every number here
    is present and validated.
    """
    stages = resolved["stage"]
    wet_kg = 0.0
    for stage in stages:
        wet_kg += stage["dry_mass_kg"]
        wet_kg += sum(t["propellant_mass_kg"] for t in stage.get("tank", []))
        wet_kg += sum(j["mass_kg"] for j in stage.get("jettison", []))
    if wet_kg <= 0.0:
        # Unreachable while pass 2 demands positive component masses; kept so
        # the vehicle-level invariant does not silently depend on that.
        rep.add(
            "root",
            "stage",
            f"total wet mass must be > 0, got {wet_kg!r}",
            units="kg",
            typical="10 to 1e6",
        )
        return

    has_aero = bool(resolved.get("aero"))
    first = stages[0]
    first_engines = first.get("engine", [])
    if has_aero and first_engines:
        # An aero block marks an ascent vehicle; a bottom stage that cannot
        # lift the stack is almost certainly a data-entry error, but a hop
        # test or a deliberately clamped case is conceivable: warning tier.
        tw = sum(e["thrust_vac_N"] for e in first_engines) / (_G0_MPS2 * wet_kg)
        if tw < 1.2:
            rep.warn(
                "stage.1",
                "thrust_vac_N",
                f"liftoff thrust-to-weight {tw:.3f} is below 1.2 for an ascent "
                f"vehicle (vacuum-thrust basis, wet mass {wet_kg!r} kg)",
                units="1",
                typical="1.2 to 2.5 at liftoff",
            )
        for j, engine in enumerate(first_engines, 1):
            if engine["thrust_vac_N"] - _P_SEA_LEVEL_PA * engine["exit_area_m2"] <= 0.0:
                rep.warn(
                    f"stage.1.engine.{j}",
                    "exit_area_m2",
                    "sea-level thrust (thrust_vac_N - 101325 Pa * exit_area_m2) is "
                    "not positive; this engine cannot run at sea level",
                    units="m^2",
                    typical="sized so vacuum thrust exceeds the back-pressure loss",
                )

    for i, stage in enumerate(stages, 1):
        for j, engine in enumerate(stage.get("engine", []), 1):
            if not (200.0 <= engine["isp_vac_s"] <= 500.0):
                rep.warn(
                    f"stage.{i}.engine.{j}",
                    "isp_vac_s",
                    f"outside the typical chemical range, got {engine['isp_vac_s']!r}",
                    units="s",
                    typical="typical chemical range 200-465",
                )
        prop = sum(t["propellant_mass_kg"] for t in stage.get("tank", []))
        if prop > 0.0:
            frac = prop / (prop + stage["dry_mass_kg"])
            if frac > 0.95:
                rep.warn(
                    f"stage.{i}",
                    "dry_mass_kg",
                    f"stage propellant mass fraction {frac:.3f} exceeds 0.95, an "
                    f"implausibly light structure for this schema's vehicle classes",
                    units="1",
                    typical="0.75 to 0.93",
                )


def _validate_document(doc: dict, rep: _Report) -> dict | None:
    for key in doc:
        if key not in ("schema_version", "provenance", "vehicle", "stage", "aero"):
            rep.add(
                "root",
                key,
                "unknown key",
                hint="remove it or fix the spelling; allowed top-level entries are "
                "schema_version, provenance, [vehicle], [[stage]], [[aero]]",
            )

    if "schema_version" not in doc:
        rep.add(
            "root",
            "schema_version",
            "missing required key",
            hint=f"must equal {VEHICLE_SCHEMA_VERSION}, e.g. schema_version = 1",
        )
    elif not _is_int(doc["schema_version"]) or doc["schema_version"] != VEHICLE_SCHEMA_VERSION:
        rep.add(
            "root",
            "schema_version",
            f"must equal {VEHICLE_SCHEMA_VERSION}, got {doc['schema_version']!r}",
            hint=f"this validator implements vehicle schema version {VEHICLE_SCHEMA_VERSION}",
        )

    provenance = None
    if "provenance" not in doc:
        rep.add(
            "root",
            "provenance",
            "missing required key",
            hint='FR-13 mandatory provenance, e.g. provenance = "representative"',
        )
    elif not isinstance(doc["provenance"], str) or not doc["provenance"].strip():
        rep.add(
            "root",
            "provenance",
            f"expected a non-empty string, got {doc['provenance']!r}",
            hint='e.g. provenance = "representative"',
        )
    else:
        provenance = doc["provenance"]

    vehicle_tbl = None
    name = description = None
    if "vehicle" not in doc:
        rep.add("root", "vehicle", "missing required table", hint="must define name")
    elif not isinstance(doc["vehicle"], dict):
        rep.add("root", "vehicle", f"expected a table, got {type(doc['vehicle']).__name__}", hint="must define name")
    else:
        vehicle_tbl = doc["vehicle"]
        _reject_unknown(vehicle_tbl, "vehicle", {"name", "description"}, rep)
        name = _req_str(vehicle_tbl, "vehicle", "name", rep, hint='non-empty string, e.g. "electron-class"')
        if name is not None and not name.strip():
            rep.add("vehicle", "name", "must be a non-empty string", hint='e.g. "electron-class"')
            name = None
        if "description" in vehicle_tbl:
            if not isinstance(vehicle_tbl["description"], str):
                rep.add(
                    "vehicle",
                    "description",
                    f"expected a string, got {type(vehicle_tbl['description']).__name__}",
                    hint="optional free text",
                )
            else:
                description = vehicle_tbl["description"]

    stages_resolved: list | None = []
    if "stage" not in doc:
        rep.add(
            "root",
            "stage",
            "missing required table array",
            hint="at least one [[stage]] block is required",
        )
        stages_resolved = None
    elif not isinstance(doc["stage"], list) or not all(isinstance(s, dict) for s in doc["stage"]):
        rep.add(
            "root",
            "stage",
            "expected an array of tables ([[stage]] blocks)",
            hint="write each stage as its own [[stage]] table",
        )
        stages_resolved = None
    elif not doc["stage"]:
        rep.add("root", "stage", "at least one [[stage]] block is required", hint="add a stage")
        stages_resolved = None
    else:
        raw_stages = doc["stage"]
        for i, stage in enumerate(raw_stages, 1):
            s = _validate_stage(stage, f"stage.{i}", rep)
            if s is None:
                stages_resolved = None
            elif stages_resolved is not None:
                stages_resolved.append(s)
        _check_unique(
            [
                (f"stage.{i}", s.get("name") if isinstance(s.get("name"), str) else None)
                for i, s in enumerate(raw_stages, 1)
            ],
            "stage",
            rep,
        )

    aero_resolved: list | None = None
    if "aero" in doc:
        if not isinstance(doc["aero"], list) or not all(isinstance(a, dict) for a in doc["aero"]):
            rep.add(
                "root",
                "aero",
                "expected an array of tables ([[aero]] blocks)",
                hint="write each stack configuration as its own [[aero]] table",
            )
        else:
            aero_resolved = []
            for i, aero in enumerate(doc["aero"], 1):
                a = _validate_aero(aero, f"aero.{i}", rep)
                if a is None:
                    aero_resolved = None
                elif aero_resolved is not None:
                    aero_resolved.append(a)
            _check_unique(
                [
                    (f"aero.{i}", a.get("config") if isinstance(a.get("config"), str) else None)
                    for i, a in enumerate(doc["aero"], 1)
                ],
                "aero configuration",
                rep,
            )

    if rep.items:
        return None
    resolved: dict = {
        "schema_version": VEHICLE_SCHEMA_VERSION,
        "provenance": provenance,
        "vehicle": {"name": name},
        "stage": stages_resolved,
    }
    if description is not None:
        resolved["vehicle"]["description"] = description
    if aero_resolved:
        resolved["aero"] = aero_resolved

    _vehicle_sanity(resolved, rep)
    if rep.items:
        return None
    return resolved


def validate_vehicle_file(path, *, strict: bool = False) -> tuple[dict | None, list[str], list[str]]:
    """Validate one vehicle TOML file (FR-13/FR-15).

    Returns ``(resolved, errors, warnings)``: on success ``resolved`` is the
    canonicalized configuration dict and ``errors`` is empty; on failure
    ``resolved`` is None and ``errors`` holds every DX-2 formatted error
    line. ``warnings`` always carries the advisory findings; with
    ``strict=True`` they are additionally promoted into ``errors``, so a
    vehicle with any warning fails strict validation.
    """
    source = str(path)
    rep = _Report(source)
    try:
        with open(path, "rb") as fh:
            doc = tomllib.load(fh)
    except OSError as exc:
        rep.items.append(
            f"{source}: cannot read vehicle file: {exc}. No default applied; run aborted."
        )
        return None, rep.items, []
    except tomllib.TOMLDecodeError as exc:
        # A parse failure leaves no structure to walk, so it is the one class
        # of error that cannot be accumulated with others.
        rep.items.append(
            f"{source}: TOML parse error: {exc}. No default applied; run aborted."
        )
        return None, rep.items, []
    resolved = _validate_document(doc, rep)
    warnings_out = [w + _WARN_SUFFIX for w in rep.warnings]
    if strict:
        rep.items.extend(w + _STRICT_SUFFIX for w in rep.warnings)
    if rep.items:
        return None, rep.items, warnings_out
    return resolved, [], warnings_out


def _toml_value(value) -> str:
    """Canonical TOML rendering of a schema value.

    Floats via ``repr`` (shortest round-trip form, also valid TOML), so the
    echo re-parses to bit-identical doubles; strings via JSON encoding, whose
    escape set is a subset of TOML basic-string escapes.
    """
    if isinstance(value, bool):
        raise ValueError("the vehicle schema has no boolean-valued keys")
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise ValueError(f"unsupported value type {type(value).__name__}")


def _emit_keys(
    lines: list[str], table: dict, order: tuple[str, ...], container_keys: tuple[str, ...] = ()
) -> None:
    # container_keys are sub-table keys emitted by the caller as their own
    # [[...]] sections, never as inline values on this table.
    unknown = set(table) - set(order) - set(container_keys)
    if unknown:
        raise ValueError(f"cannot serialize unknown keys {sorted(unknown)!r}")
    for key in order:
        if key in table:
            lines.append(f"{key} = {_toml_value(table[key])}")


def canonical_vehicle_toml(doc: dict) -> str:
    """Serialize a schema-shaped vehicle dict to canonical TOML (FR-15 echo).

    Fixed key order and formatting: validating a file, echoing the resolved
    config through this function, and re-validating the echo reproduces the
    identical bytes (Phase 4 exit criterion 1). The function accepts any
    dict shaped like the schema (keys present or absent), which the mutation
    tests use to re-serialize surgically edited documents.
    """
    lines: list[str] = []
    _emit_keys(lines, doc, _ORDER["root"], container_keys=("vehicle", "stage", "aero"))
    if "vehicle" in doc:
        lines.extend(["", "[vehicle]"])
        _emit_keys(lines, doc["vehicle"], _ORDER["vehicle"])
    for stage in doc.get("stage", []):
        lines.extend(["", "[[stage]]"])
        _emit_keys(lines, stage, _ORDER["stage"], container_keys=_SUB_BLOCKS)
        for kind in _SUB_BLOCKS:
            for block in stage.get(kind, []):
                lines.extend(["", f"[[stage.{kind}]]"])
                _emit_keys(lines, block, _ORDER[f"stage.{kind}"])
    for aero in doc.get("aero", []):
        lines.extend(["", "[[aero]]"])
        _emit_keys(lines, aero, _ORDER["aero"])
    return "\n".join(lines) + "\n"
