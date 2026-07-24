"""Monte Carlo sweep-spec parsing and case generation (FR-27, D-3).

A *sweep spec* is a TOML file that expands into N single-run cases, each a
``{dotted.path: value}`` override dict in the FR-24/FR-27 vocabulary
(``star_reacher.overrides``). ``star mc`` runs the base mission once per case;
the manifest records each case's overrides so any run is individually
reproducible via ``star run --seed ... --set ...``.

Grammar (D-3: TOML everywhere)::

    schema_version = 1
    [sweep]
    mission = "missions/leo_gravity_8x8.toml"   # base mission, path resolved
                                                # relative to the sweep file's
                                                # directory then cwd
    master_seed = 20260723
    method = "lhs"           # "grid" | "list" | "lhs"
    n_runs = 256             # required for lhs; derived for grid/list, and if
                             # given must match the derived count
    [[sweep.parameter]]
    path = "mission.duration_s"     # a dotted override path (same vocabulary
                                    # as --set)
    min = 3600.0                    # lhs/grid range endpoints
    max = 7200.0
    # integer = true                # optional: sample/round to integer leaves

    [[sweep.parameter]]
    path = "spacecraft.cd_a_over_m_m2pkg"
    values = [0.01, 0.02, 0.03]     # list/grid explicit values

Methods:

* **grid** -- the Cartesian product of each parameter's ``values``; ``n_runs``
  is the product of the value-list lengths.
* **list** -- the parallel (zipped) ``values``; every parameter's list must
  have the same length, which is ``n_runs``.
* **lhs** -- a Latin hypercube over each parameter's ``[min, max]`` with
  ``n_runs`` samples. The sampling is deterministic in ``master_seed``: the
  per-dimension stratum permutations and in-stratum jitter are drawn from the
  core PCG64 stream ``rng_stream_u64(master_seed, "mc.lhs", ...)``, never from
  ``random``/``numpy.random`` default entropy, so the same spec always expands
  to the same cases.

Validation follows the FR-15/DX-2 discipline: every error in the spec is
accumulated and reported together, each line naming the file, the table path,
and the reason, and unknown keys are rejected.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

__all__ = [
    "SweepError",
    "SweepSpec",
    "load_sweep_spec",
]

SCHEMA_VERSION = 1

# The named stream the LHS sampler draws from. Fixed as a constant so the
# stream identity is part of the reproducibility contract rather than an
# incidental string, the same way the core names its subsystem streams.
_LHS_STREAM = "mc.lhs"

# 2^-53 as the exact multiplier that maps a u64's top 53 bits into a double in
# [0, 1); identical construction to cpp/src/rng.cpp u64_to_unit, so a unit draw
# here equals the core's for the same u64.
_TWO_POW_NEG_53 = 2.0**-53

_METHODS = ("grid", "list", "lhs")


class SweepError(Exception):
    """Carries the accumulated, formatted sweep-spec validation error lines."""

    def __init__(self, errors: list[str]):
        super().__init__(f"{len(errors)} sweep-spec error(s)")
        self.errors = list(errors)


class SweepSpec:
    """A validated sweep spec and the cases it expands to.

    ``cases`` is the list of per-run override dicts, in run-index order;
    ``parameters`` is the resolved parameter list recorded in the manifest.
    """

    def __init__(self, *, mission, mission_path, master_seed, method, n_runs,
                 parameters, cases, source):
        self.mission = mission
        self.mission_path = mission_path
        self.master_seed = master_seed
        self.method = method
        self.n_runs = n_runs
        self.parameters = parameters
        self.cases = cases
        self.source = source


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _unit_draws(master_seed: int, count: int) -> list[float]:
    """``count`` unit floats in [0, 1) from the core LHS stream.

    Drawn from the core PCG64 stream so the sampling shares the core's exact,
    platform-independent generator (D-9); mapping the u64 through the same
    ``(x >> 11) * 2^-53`` construction the core's ``u64_to_unit`` uses makes a
    unit value here identical to the core's for the same draw.
    """
    from star_reacher._corelink import import_core

    if count == 0:
        return []
    raw = import_core().rng_stream_u64(master_seed, _LHS_STREAM, count)
    return [(x >> 11) * _TWO_POW_NEG_53 for x in raw]


def _lhs_cases(master_seed, params, n_runs):
    """Latin-hypercube cases over each parameter's [min, max].

    One stratum permutation and one jitter value per (dimension, sample) are
    drawn from the LHS stream, dimension-major: all n permutation draws for
    dimension 0, then its n jitter draws, then dimension 1, and so on. Fixing
    that consumption order is what makes the expansion reproducible; a reader
    regenerating the sweep gets the identical cases from the identical stream.
    """
    d = len(params)
    # 2n unit draws per dimension: n to permute the strata (Fisher-Yates on the
    # first n), n to jitter within each chosen stratum.
    draws = _unit_draws(master_seed, 2 * d * n_runs)
    columns = []  # per-dimension list of n sampled values, in stratum order
    perms = []    # per-dimension permutation of stratum indices
    cursor = 0
    for _pi, p in enumerate(params):
        lo, hi = p["min"], p["max"]
        width = (hi - lo) / n_runs
        perm_draws = draws[cursor:cursor + n_runs]
        cursor += n_runs
        jitter_draws = draws[cursor:cursor + n_runs]
        cursor += n_runs
        # Fisher-Yates permutation of [0, n): the i-th draw selects a swap
        # partner in [i, n). A unit draw scaled by the remaining count and
        # floored is an unbiased index for this generator.
        order = list(range(n_runs))
        for i in range(n_runs - 1):
            span = n_runs - i
            j = i + min(span - 1, int(perm_draws[i] * span))
            order[i], order[j] = order[j], order[i]
        perms.append(order)
        col = []
        for stratum in range(n_runs):
            value = lo + (stratum + jitter_draws[stratum]) * width
            if p.get("integer"):
                value = float(round(value))
            col.append(value)
        columns.append(col)
    cases = []
    for run in range(n_runs):
        case = {}
        for dim, p in enumerate(params):
            stratum = perms[dim][run]
            case[p["path"]] = columns[dim][stratum]
        cases.append(case)
    return cases


def _grid_cases(params):
    """Cartesian product of each parameter's ``values`` (row-major)."""
    cases = [{}]
    for p in params:
        expanded = []
        for prefix in cases:
            for value in p["values"]:
                nxt = dict(prefix)
                nxt[p["path"]] = value
                expanded.append(nxt)
        cases = expanded
    return cases


def _list_cases(params, n_runs):
    """Zipped parameter values: case k takes each parameter's k-th value."""
    return [
        {p["path"]: p["values"][k] for p in params}
        for k in range(n_runs)
    ]


def _validate_parameter(entry, index, errs, *, method):
    """Validate one [[sweep.parameter]] and return its resolved dict or None.

    Accumulates into ``errs`` rather than raising, so a spec with several bad
    parameters reports them all at once.
    """
    path_label = f"sweep.parameter[{index}]"
    if not isinstance(entry, dict):
        errs.append(f"[{path_label}]: expected a table, got "
                    f"{type(entry).__name__}")
        return None
    resolved = {}
    path = entry.get("path")
    if not isinstance(path, str) or not path.strip():
        errs.append(f"[{path_label}] path: missing or not a non-empty string "
                    f"(a dotted override path, e.g. \"mission.duration_s\")")
        path = None
    else:
        resolved["path"] = path

    has_values = "values" in entry
    has_range = "min" in entry or "max" in entry
    allowed = {"path", "values", "min", "max", "integer"}
    for key in entry:
        if key not in allowed:
            errs.append(f"[{path_label}] {key}: unknown key; allowed: "
                        f"{', '.join(sorted(allowed))}")

    if "integer" in entry and not isinstance(entry["integer"], bool):
        errs.append(f"[{path_label}] integer: expected true or false")
    if entry.get("integer") is True:
        resolved["integer"] = True

    if method in ("grid", "list"):
        if not has_values:
            errs.append(f"[{path_label}] values: the {method!r} method needs "
                        f"an explicit values array for every parameter")
            return None
        values = entry["values"]
        if (not isinstance(values, list) or len(values) == 0
                or not all(_is_number(v) for v in values)):
            errs.append(f"[{path_label}] values: expected a non-empty array "
                        f"of numbers, got {values!r}")
            return None
        resolved["values"] = list(values)
        return resolved if path is not None else None

    # method == "lhs": a numeric [min, max] range is required.
    if not has_range or has_values:
        errs.append(f"[{path_label}]: the 'lhs' method needs a numeric range "
                    f"(min and max), not a values array")
        return None
    lo, hi = entry.get("min"), entry.get("max")
    if not _is_number(lo) or not _is_number(hi):
        errs.append(f"[{path_label}] min/max: both must be numbers, got "
                    f"min={lo!r}, max={hi!r}")
        return None
    if not hi > lo:
        errs.append(f"[{path_label}] max: must be greater than min, got "
                    f"min={lo!r}, max={hi!r}")
        return None
    resolved["min"] = float(lo)
    resolved["max"] = float(hi)
    return resolved if path is not None else None


def load_sweep_spec(spec_path) -> SweepSpec:
    """Parse and validate a sweep-spec TOML file into a :class:`SweepSpec`.

    Raises :class:`SweepError` carrying every accumulated error, or
    ``FileNotFoundError`` if the spec file is absent.
    """
    spec_path = Path(spec_path)
    source = str(spec_path)
    with spec_path.open("rb") as fh:
        doc = tomllib.load(fh)

    errs: list[str] = []

    sv = doc.get("schema_version")
    if sv != SCHEMA_VERSION:
        errs.append(f"{source}: schema_version: expected {SCHEMA_VERSION}, "
                    f"got {sv!r}")
    for key in doc:
        if key not in ("schema_version", "sweep"):
            errs.append(f"{source}: {key}: unknown top-level key; the spec "
                        f"holds schema_version and [sweep]")

    sweep = doc.get("sweep")
    if not isinstance(sweep, dict):
        errs.append(f"{source}: [sweep]: missing or not a table")
        raise SweepError(errs)

    allowed = {"mission", "master_seed", "method", "n_runs", "parameter"}
    for key in sweep:
        if key not in allowed:
            errs.append(f"{source}: [sweep] {key}: unknown key; allowed: "
                        f"{', '.join(sorted(allowed))}")

    mission = sweep.get("mission")
    if not isinstance(mission, str) or not mission.strip():
        errs.append(f"{source}: [sweep] mission: missing or not a non-empty "
                    f"string (the base mission TOML path)")
        mission = None

    master_seed = sweep.get("master_seed")
    if not _is_int(master_seed) or not (0 <= master_seed <= 2**64 - 1):
        errs.append(f"{source}: [sweep] master_seed: expected an integer in "
                    f"[0, 2**64-1], got {master_seed!r}")
        master_seed = None

    method = sweep.get("method")
    if method not in _METHODS:
        errs.append(f"{source}: [sweep] method: must be one of "
                    f"{', '.join(_METHODS)}, got {method!r}")
        method = None

    raw_params = sweep.get("parameter")
    params = []
    if not isinstance(raw_params, list) or not raw_params:
        errs.append(f"{source}: [[sweep.parameter]]: at least one parameter "
                    f"table is required")
    elif method is not None:
        param_errs: list[str] = []
        for i, entry in enumerate(raw_params):
            resolved = _validate_parameter(entry, i, param_errs, method=method)
            if resolved is not None:
                params.append(resolved)
        errs.extend(f"{source}: {line}" for line in param_errs)

    n_runs_given = sweep.get("n_runs")
    if n_runs_given is not None and (not _is_int(n_runs_given)
                                     or n_runs_given <= 0):
        errs.append(f"{source}: [sweep] n_runs: expected a positive integer, "
                    f"got {n_runs_given!r}")
        n_runs_given = None

    # Derive n_runs per method and cross-check against any given value. Only
    # attempted once the pieces it depends on are individually sound, so a
    # cascade of secondary errors does not bury the primary one.
    n_runs = None
    cases = None
    if method is not None and params and not _params_have_gaps(params, method):
        if method == "grid":
            n_runs = 1
            for p in params:
                n_runs *= len(p["values"])
        elif method == "list":
            lengths = {len(p["values"]) for p in params}
            if len(lengths) != 1:
                errs.append(f"{source}: [[sweep.parameter]] values: the "
                            f"'list' method zips parameters, so every values "
                            f"array must have the same length; got lengths "
                            f"{sorted(len(p['values']) for p in params)}")
            else:
                n_runs = lengths.pop()
        else:  # lhs
            if n_runs_given is None:
                errs.append(f"{source}: [sweep] n_runs: required for the "
                            f"'lhs' method (the number of hypercube samples)")
            else:
                n_runs = n_runs_given
        if (n_runs is not None and n_runs_given is not None
                and method in ("grid", "list") and n_runs_given != n_runs):
            errs.append(f"{source}: [sweep] n_runs: the {method!r} method "
                        f"derives {n_runs} run(s) from the parameter values, "
                        f"but n_runs = {n_runs_given} was given")

    if errs:
        raise SweepError(errs)

    if method == "grid":
        cases = _grid_cases(params)
    elif method == "list":
        cases = _list_cases(params, n_runs)
    else:
        cases = _lhs_cases(master_seed, params, n_runs)

    # The manifest records only the identifying fields of each parameter, in
    # the spec's order; the sampled values live per run in the cases.
    manifest_params = []
    for p in params:
        mp = {"path": p["path"]}
        if "values" in p:
            mp["values"] = list(p["values"])
        else:
            mp["min"] = p["min"]
            mp["max"] = p["max"]
        if p.get("integer"):
            mp["integer"] = True
        manifest_params.append(mp)

    return SweepSpec(
        mission=mission,
        mission_path=_resolve_mission_path(mission, spec_path),
        master_seed=master_seed,
        method=method,
        n_runs=n_runs,
        parameters=manifest_params,
        cases=cases,
        source=source,
    )


def _params_have_gaps(params, method) -> bool:
    """True when a parameter lacks the fields its method's count needs.

    Guards the n_runs derivation so it never indexes a key an earlier
    per-parameter check already flagged as missing.
    """
    if method in ("grid", "list"):
        return any("values" not in p for p in params)
    return any("min" not in p or "max" not in p for p in params)


def _resolve_mission_path(mission, spec_path):
    """Locate the base mission: relative to the spec's directory, then cwd.

    Returns the first path that exists, or the spec-relative candidate when
    neither does (so a downstream FileNotFoundError names a real location
    rather than a bare relative string).
    """
    if mission is None:
        return None
    candidate = Path(mission)
    if candidate.is_absolute():
        return candidate
    beside = spec_path.parent / candidate
    if beside.exists():
        return beside
    if candidate.exists():
        return candidate
    return beside
