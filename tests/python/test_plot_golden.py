"""Data-level golden regression for the FR-18 plot-feeding arrays.

Phase 5 exit criterion 1: the arrays feeding every named quicklook plot must
match committed golden vectors. Each test re-runs a committed reference
mission through the real runner, reduces the log with the array-preparation
layer behind ``star plot`` (``star_reacher.plotting.prepare_all``), and
compares at the frozen probe indices under the per-array rule recorded in
``tests/golden/plots/`` (tolerance derivations in that directory's
manifest.toml).

These tests REQUIRE the compiled core (the missions must run and the
groundtrack uses the core frame chain); they fail cleanly, never skip, when
it is absent - the project's agent-honesty gate, same as the Phase 4
mission tests.
"""

import tomllib
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = REPO_ROOT / "tests" / "golden" / "plots"

_CORE_MISSING_MESSAGE = (
    "star_reacher._core is not built in this environment. The plot golden "
    "tests require the compiled core: build and install it with 'pip "
    "install .' from the repository root. This failure is expected on a "
    "core-less checkout and must be green at integration/CI."
)


def _core_or_fail():
    try:
        from star_reacher import _core
    except ImportError:
        pytest.fail(_CORE_MISSING_MESSAGE)
    return _core


@pytest.fixture(scope="module")
def reference_runs(tmp_path_factory):
    """Both reference missions run once per module; (RunResult, Run) pairs."""
    _core_or_fail()
    from star_reacher.runner import run_mission
    from star_reacher.srlog import load

    out = {}
    base = tmp_path_factory.mktemp("plot_golden_runs")
    for mission in ("twobody_leo", "ascent_leo"):
        result = run_mission(REPO_ROOT / "missions" / f"{mission}.toml", base / mission)
        out[mission] = (result, load(result.srlog_path))
    return out


def _golden(mission: str) -> dict:
    with open(GOLDEN_DIR / f"{mission}.toml", "rb") as fh:
        return tomllib.load(fh)


def _compare_entry(entry: dict, got: np.ndarray) -> None:
    where = f"{entry['plot']}.{entry['name']}"
    assert len(got) == entry["n"], (
        f"{where}: array length {len(got)} != frozen {entry['n']} "
        f"(the run itself changed; regenerate the goldens deliberately)"
    )
    idx = np.asarray(entry["indices"], dtype=int)
    probes = np.asarray(got, dtype=np.float64)[idx]
    ref = np.array([float.fromhex(h) for h in entry["values_hex"]])
    assert np.all(np.isfinite(probes)), f"{where}: non-finite probe values {probes}"
    if entry["compare"] == "reltol":
        scale = float.fromhex(entry["scale_hex"])
        tol = entry["rtol"] * scale
        err = np.max(np.abs(probes - ref))
        assert err <= tol, (
            f"{where}: max probe error {err:.6e} > tol {tol:.6e} "
            f"(rtol {entry['rtol']:.1e} * scale {scale:.6e})"
        )
    elif entry["compare"] == "abs":
        err = np.max(np.abs(probes - ref))
        assert err <= entry["tol"], (
            f"{where}: max probe error {err:.6e} deg > tol {entry['tol']:.1e} deg"
        )
    elif entry["compare"] == "circular_deg":
        # Wrap-safe angular difference: 359.9999 vs 0.0001 deg must compare
        # as 2e-4 deg apart, not 359.9998.
        d = np.abs(probes - ref) % 360.0
        err = np.max(np.minimum(d, 360.0 - d))
        assert err <= entry["tol"], (
            f"{where}: max circular error {err:.6e} deg > tol {entry['tol']:.1e} deg"
        )
    else:  # pragma: no cover - schema guard
        pytest.fail(f"{where}: unknown compare rule {entry['compare']!r}")


@pytest.mark.parametrize("mission", ["twobody_leo", "ascent_leo"])
def test_feeding_arrays_match_goldens(reference_runs, mission):
    from star_reacher.plotting import prepare_all

    result, run = reference_runs[mission]
    golden = _golden(mission)
    # The freeze is bound to its exact resolved inputs: a hash mismatch
    # means the mission or resolver changed and the goldens must be
    # regenerated deliberately, not silently re-tolerated.
    assert result.config_sha256 == golden["config_sha256"]

    prepared = prepare_all(run)
    entries = golden["array"]
    assert entries, "golden file carries no array entries"
    # Every named plot the mission can feed is covered by at least one entry.
    frozen_plots = {e["plot"] for e in entries}
    feedable = {name for name, prep in prepared.items() if prep.arrays is not None}
    assert frozen_plots == feedable, (
        f"golden coverage {sorted(frozen_plots)} != feedable plot set "
        f"{sorted(feedable)}"
    )
    for entry in entries:
        prep = prepared[entry["plot"]]
        assert prep.arrays is not None, f"{entry['plot']}: no arrays prepared"
        assert entry["name"] in prep.arrays, (
            f"{entry['plot']}: feeding array {entry['name']!r} disappeared; "
            f"has {sorted(prep.arrays)}"
        )
        _compare_entry(entry, prep.arrays[entry["name"]])


@pytest.mark.parametrize("mission", ["twobody_leo", "ascent_leo"])
def test_event_stream_matches_goldens_exactly(reference_runs, mission):
    _result, run = reference_runs[mission]
    golden = _golden(mission)["events"][0]
    ev = run.events
    assert len(ev) == golden["n"]
    # Bit-exact: event times are fixed-step grid arithmetic (IEEE basic
    # ops); a shifted condition-trigger step is a real regression.
    assert [float(t).hex() for t in ev["t_s"]] == golden["t_s_hex"]
    assert [int(c) for c in ev["code"]] == golden["codes"]
    assert [str(d) for d in ev["detail"]] == golden["details"]


def test_geodetic_transcription_pinned_to_core(reference_runs):
    """The NumPy Bowring latitude/altitude mirrors the core's algorithm.

    ``plotting._geodetic_lat_alt`` is a vectorized transcription of the
    core's ``geodetic_altitude`` (same two fixed Bowring passes); this pin
    keeps the copies from drifting apart, the same discipline as the GM
    table cross-check in test_gm_crosscheck.py.
    """
    core = _core_or_fail()
    from star_reacher.plotting import (
        _WGS84_A_M,
        _WGS84_INV_F,
        _earth_fixed_positions,
        _geodetic_lat_alt,
    )

    _result, run = reference_runs["ascent_leo"]
    r_ecef = _earth_fixed_positions(run)
    idx = np.linspace(0, len(r_ecef) - 1, 25).round().astype(int)
    _lat, alt = _geodetic_lat_alt(r_ecef[idx])
    for k, i in enumerate(idx):
        ref = core.geodetic_altitude(tuple(r_ecef[i]), _WGS84_A_M, _WGS84_INV_F)
        # Identical algorithm and inputs: only NumPy-vs-libm ulp spread and
        # operation-order differences remain, bounded far below 1e-9 m at
        # LEO magnitudes.
        assert abs(alt[k] - ref) <= max(1e-9, 1e-12 * abs(ref)), (
            f"sample {i}: numpy Bowring {alt[k]!r} != core {ref!r}"
        )


def test_golden_manifest_lint():
    """FR-22 layer 1: every golden file here is covered and cited."""
    with open(GOLDEN_DIR / "manifest.toml", "rb") as fh:
        manifest = tomllib.load(fh)
    assert manifest["schema_version"] == 1
    golden = manifest["golden"]
    assert golden["directory"] == "plots"
    for key in ("date", "generation"):
        assert str(golden[key]).strip(), f"[golden] {key} is empty"
    covered = {}
    for entry in manifest["file"]:
        for key in ("name", "source", "citation", "generation", "date", "tolerance"):
            text = str(entry[key]).strip()
            assert text, f"file {entry.get('name')!r}: {key} is empty"
            assert "TBD" not in text, (
                f"file {entry.get('name')!r}: {key} carries a TBD placeholder "
                f"(forbidden by tests/golden/README.md)"
            )
        covered[entry["name"]] = entry
    on_disk = {
        p.name
        for p in GOLDEN_DIR.iterdir()
        if p.is_file() and p.name != "manifest.toml"
    }
    assert on_disk == set(covered), (
        f"manifest coverage {sorted(covered)} != directory contents "
        f"{sorted(on_disk)}: every golden file needs a cited manifest entry"
    )
