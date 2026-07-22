"""Regenerate the FR-18 plot-feeding-array golden files in this directory.

These goldens are a data-level REGRESSION instrument (Phase 5 exit
criterion 1), not an independent rederivation: each value file freezes the
named plot-feeding arrays that ``star_reacher.plotting.prepare_all``
produces for the two committed reference missions, probed at fixed sample
indices, together with the resolved-config hash that binds the freeze to
its exact inputs. The consuming test (tests/python/test_plot_golden.py)
re-runs the mission on the test platform, re-prepares the arrays, and
compares at the committed probes under the per-array rule recorded in each
entry (tolerance policy: docs/formats/plots.md section on goldens, and
manifest.toml here). Physical correctness of the underlying states is gated
elsewhere (the golden suites of Phases 1-4 and the cross-tool campaigns);
what THIS instrument catches is any silent change in the log -> plot-array
pipeline.

Regenerating requires the compiled core (the missions must run):

    .venv/Scripts/python tests/golden/plots/generate.py

The script writes twobody_leo.toml and ascent_leo.toml. Regeneration is
byte-deterministic on a given platform and binary; cross-platform reruns
may move libm-bearing values inside the recorded tolerances, which is why
the consuming test compares within tolerances instead of bitwise
(manifest.toml records the derivation of every bound).
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
# Same rule as tests/python/conftest.py: an installed wheel carries the
# compiled _core and must not be shadowed by the pure-Python source tree,
# because a stale core in python/star_reacher would silently refreeze the
# goldens against the wrong binary. Regeneration needs the core either way.
if importlib.util.find_spec("star_reacher") is None:
    sys.path.insert(0, str(REPO_ROOT / "python"))

MISSIONS = ["missions/twobody_leo.toml", "missions/ascent_leo.toml"]

# Probes per array: enough spread to catch a shape or content change
# anywhere along the run while keeping the committed files small.
N_PROBES = 9

# Uniform probes alone resolve nothing shorter than the gap between them --
# over the 400 s ascent that is 50 s, which left the whole t <= 10 s vertical
# hold, and the pitch-over that follows it, between two probes and therefore
# unfrozen. A guidance discontinuity does not appear at an arbitrary time: it
# appears where the commanded attitude changes definition. So each mission may
# name the times whose neighbourhoods must be probed regardless of the uniform
# grid -- here every breakpoint of the ascent pitch table, bracketed by one
# control cycle either side, plus the pad-release cycle where attitude
# authority passes to the pitch program. Probing t and t +/- dt across a
# breakpoint is what makes a single-cycle step in the frozen truth visible.
_PITCH_TABLE_BREAKS_S = [0.0, 10.0, 25.0, 60.0, 121.0, 200.0, 290.0, 370.0]
_ASCENT_PROBE_TIMES_S = sorted(
    {2.0, 2.1}
    | {round(b + k * 0.1, 1) for b in _PITCH_TABLE_BREAKS_S for k in (-1, 0, 1)}
)

# Comparison rules per mission tier (recorded per array entry; derivations
# in manifest.toml):
# - twobody_leo: the reference propagation is basic-IEEE-op only and its
#   cross-platform final-state divergence is committed as 0.0
#   (tests/golden/determinism/cross_platform.toml), so run-fed arrays get
#   rtol 1e-9 (>= 6 orders above the libm-ulp spread the array preparation
#   itself can add) and angle-valued arrays 1e-6 deg.
# - ascent_leo: the vehicle run bears libm models (USSA76, frames chain),
#   so no bit-identity claim exists; rtol 1e-6 / 1e-3 deg absorb the
#   estimated accumulated libm divergence (<= ~1e-8 relative over the
#   7,600-step ascent) with >= 2 orders of margin while still failing on
#   any >= 1 ppm model or pipeline change.
_RULES = {
    "missions/twobody_leo.toml": {"rtol": 1e-9, "angle_tol_deg": 1e-6},
    "missions/ascent_leo.toml": {
        "rtol": 1e-6,
        "angle_tol_deg": 1e-3,
        "probe_times_s": _ASCENT_PROBE_TIMES_S,
    },
}

# Angle-valued feeding arrays, compared by circular difference (mod 360) so
# a value sitting at the 0/360 wrap cannot fail on representation alone.
_CIRCULAR_KEYS = {"raan_deg", "argp_deg", "nu_deg", "lon_deg", "ev_lon_deg"}
# Angle-valued but non-wrapping (bounded ranges): plain absolute degrees.
_ABS_DEG_KEYS = {"i_deg", "lat_deg", "ev_lat_deg"}


def _probe_indices(n: int, rule: dict, t_s: np.ndarray | None) -> list[int]:
    if n <= N_PROBES:
        return list(range(n))
    idx = set(np.linspace(0, n - 1, N_PROBES).round().astype(int).tolist())
    # Anchor the named times to the array's own time base. Arrays logged at a
    # lower rate than the control cycle collapse several requested times onto
    # one sample; the set absorbs that.
    times = rule.get("probe_times_s")
    if times is not None and t_s is not None and len(t_s) == n:
        base = np.asarray(t_s, dtype=np.float64)
        for want in times:
            if base[0] <= want <= base[-1]:
                idx.add(int(np.argmin(np.abs(base - want))))
    return sorted(idx)


def _toml_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _emit_array(out: list[str], plot: str, name: str, values: np.ndarray,
                rule: dict, t_s: np.ndarray | None = None) -> None:
    n = len(values)
    idx = _probe_indices(n, rule, t_s)
    probes = np.asarray(values, dtype=np.float64)[idx]
    if not np.all(np.isfinite(probes)):
        raise SystemExit(
            f"non-finite probe in {plot}.{name}: choose different probes or fix "
            f"the pipeline before freezing"
        )
    out.append("[[array]]")
    out.append(f"plot = {_toml_str(plot)}")
    out.append(f"name = {_toml_str(name)}")
    out.append(f"n = {n}")
    out.append(f"indices = [{', '.join(str(i) for i in idx)}]")
    if name in _CIRCULAR_KEYS:
        out.append('compare = "circular_deg"')
        out.append(f"tol = {rule['angle_tol_deg']:.1e}")
    elif name in _ABS_DEG_KEYS:
        out.append('compare = "abs"')
        out.append(f"tol = {rule['angle_tol_deg']:.1e}")
    else:
        scale = float(np.max(np.abs(probes))) if len(probes) else 0.0
        out.append('compare = "reltol"')
        out.append(f"rtol = {rule['rtol']:.1e}")
        # tol = rtol * scale: an all-zero array (scale 0) is required to
        # stay exactly zero, which run-fed structural zeros do.
        out.append(f"scale_hex = {_toml_str(float(scale).hex())}")
    out.append(
        "values_hex = [" + ", ".join(_toml_str(float(v).hex()) for v in probes) + "]"
    )
    out.append("")


def _emit_events(out: list[str], run) -> None:
    ev = run.events
    out.append("[[events]]")
    out.append(f"n = {len(ev)}")
    # Event times are fixed-step grid arithmetic (IEEE basic ops), so they
    # are compared bit-exactly; a flipped condition-trigger step is a real
    # regression signal, not tolerance-worthy noise.
    out.append(
        "t_s_hex = ["
        + ", ".join(_toml_str(float(t).hex()) for t in ev["t_s"])
        + "]"
    )
    out.append("codes = [" + ", ".join(str(int(c)) for c in ev["code"]) + "]")
    out.append(
        "details = [" + ", ".join(_toml_str(str(d)) for d in ev["detail"]) + "]"
    )
    out.append("")


def main() -> None:
    from datetime import date

    from star_reacher.plotting import prepare_all
    from star_reacher.runner import run_mission
    from star_reacher.srlog import load

    for mission in MISSIONS:
        rule = _RULES[mission]
        with tempfile.TemporaryDirectory() as td:
            result = run_mission(REPO_ROOT / mission, pathlib.Path(td) / "run")
            run = load(result.srlog_path)
            prepared = prepare_all(run)
            out: list[str] = [
                "# Frozen plot-feeding arrays for "
                f"{pathlib.Path(mission).name} (FR-18, Phase 5 exit criterion 1).",
                "# Generated by tests/golden/plots/generate.py; provenance and",
                "# tolerance derivations in tests/golden/plots/manifest.toml.",
                "# Hand-editing is forbidden (tests/golden/README.md update policy).",
                "",
                "schema_version = 1",
                f"mission = {_toml_str(mission)}",
                f"config_sha256 = {_toml_str(result.config_sha256)}",
                f"generated = {_toml_str(date.today().isoformat())}",
                "",
            ]
            for plot_name, prep in prepared.items():
                if prep.arrays is None:
                    continue
                # A plot may carry several series at different log rates; each
                # is anchored to the time array of its own length.
                bases = {
                    len(v): v
                    for k, v in prep.arrays.items()
                    if k.endswith("t_s")
                }
                for array_name in sorted(prep.arrays):
                    values = prep.arrays[array_name]
                    if len(values) == 0:
                        continue
                    _emit_array(out, plot_name, array_name, values, rule,
                                bases.get(len(values)))
            _emit_events(out, run)
            target = HERE / (pathlib.Path(mission).stem + ".toml")
            target.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
            print(f"wrote {target}")


if __name__ == "__main__":
    main()
