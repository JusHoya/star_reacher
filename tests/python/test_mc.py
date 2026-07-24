"""FR-27 ``star mc``: sweep-spec parsing, LHS determinism, and the manifest.

The end-to-end tests satisfy Phase 7 exit criterion 1 in miniature: a small
sweep finishes with every manifest entry ``status: success``, and re-executing
one entry via ``run_mission`` with its recorded seed and overrides reproduces
its logged SHA-256. The pure parsing/determinism tests need no core; the
end-to-end tests fail (never skip) when the core is absent.
"""

import contextlib
import hashlib
import json
import os
from pathlib import Path

import pytest

from star_reacher.mc import run_sweep
from star_reacher.sweep import SweepError, load_sweep_spec

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED_SWEEP = REPO_ROOT / "missions" / "leo_gravity_8x8_sweep.toml"

_CORE_MISSING = (
    "star_reacher._core is not built in this environment. These tests require "
    "the compiled core: build and install it with 'pip install .'. This "
    "failure is expected on a core-less checkout and must be green at CI."
)


def _core_or_fail():
    try:
        from star_reacher import _core  # noqa: F401
    except ImportError:
        pytest.fail(_CORE_MISSING)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextlib.contextmanager
def _in_repo_root():
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        yield
    finally:
        os.chdir(cwd)


def _write_spec(tmp_path, body, name="sweep.toml"):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


# --- sweep-spec parsing (no core) ------------------------------------------


_GRID = """\
schema_version = 1
[sweep]
mission = "twobody_leo.toml"
master_seed = 7
method = "grid"
[[sweep.parameter]]
path = "mission.duration_s"
values = [100.0, 200.0]
[[sweep.parameter]]
path = "spacecraft.mass_kg"
values = [10.0, 20.0, 30.0]
"""


def test_grid_is_the_cartesian_product(tmp_path):
    spec = load_sweep_spec(_write_spec(tmp_path, _GRID))
    assert spec.method == "grid"
    assert spec.n_runs == 6  # 2 * 3
    got = {
        (c["mission.duration_s"], c["spacecraft.mass_kg"]) for c in spec.cases
    }
    assert got == {
        (d, m) for d in (100.0, 200.0) for m in (10.0, 20.0, 30.0)
    }


_LIST = """\
schema_version = 1
[sweep]
mission = "twobody_leo.toml"
master_seed = 7
method = "list"
[[sweep.parameter]]
path = "mission.duration_s"
values = [100.0, 200.0, 300.0]
[[sweep.parameter]]
path = "spacecraft.mass_kg"
values = [10.0, 20.0, 30.0]
"""


def test_list_zips_the_values(tmp_path):
    spec = load_sweep_spec(_write_spec(tmp_path, _LIST))
    assert spec.method == "list"
    assert spec.n_runs == 3
    assert spec.cases == [
        {"mission.duration_s": 100.0, "spacecraft.mass_kg": 10.0},
        {"mission.duration_s": 200.0, "spacecraft.mass_kg": 20.0},
        {"mission.duration_s": 300.0, "spacecraft.mass_kg": 30.0},
    ]


def test_list_rejects_unequal_lengths(tmp_path):
    body = _LIST.replace("values = [10.0, 20.0, 30.0]", "values = [10.0, 20.0]")
    with pytest.raises(SweepError) as exc:
        load_sweep_spec(_write_spec(tmp_path, body))
    assert any("same length" in e for e in exc.value.errors)


_LHS = """\
schema_version = 1
[sweep]
mission = "twobody_leo.toml"
master_seed = 20260723
method = "lhs"
n_runs = 16
[[sweep.parameter]]
path = "mission.duration_s"
min = 100.0
max = 200.0
[[sweep.parameter]]
path = "spacecraft.mass_kg"
min = 10.0
max = 30.0
"""


def test_lhs_is_a_latin_hypercube(tmp_path):
    _core_or_fail()  # LHS draws from the core PCG64 stream
    spec = load_sweep_spec(_write_spec(tmp_path, _LHS))
    assert spec.method == "lhs"
    assert spec.n_runs == 16
    # Each of the n strata is used exactly once per dimension.
    for path, lo, hi in (
        ("mission.duration_s", 100.0, 200.0),
        ("spacecraft.mass_kg", 10.0, 30.0),
    ):
        width = (hi - lo) / 16
        strata = sorted(int((c[path] - lo) // width) for c in spec.cases)
        assert strata == list(range(16))
        assert all(lo <= c[path] < hi for c in spec.cases)


def test_lhs_is_deterministic_in_the_master_seed(tmp_path):
    """Same master_seed -> same cases, bit-for-bit; a different seed differs."""
    _core_or_fail()
    a = load_sweep_spec(_write_spec(tmp_path, _LHS, "a.toml"))
    b = load_sweep_spec(_write_spec(tmp_path, _LHS, "b.toml"))
    assert a.cases == b.cases
    other = _LHS.replace("master_seed = 20260723", "master_seed = 20260724")
    c = load_sweep_spec(_write_spec(tmp_path, other, "c.toml"))
    assert c.cases != a.cases


def test_lhs_integer_dimension_rounds_to_whole_numbers(tmp_path):
    _core_or_fail()
    body = _LHS.replace(
        'path = "mission.duration_s"\nmin = 100.0\nmax = 200.0',
        'path = "mission.duration_s"\nmin = 100.0\nmax = 200.0\ninteger = true',
    )
    spec = load_sweep_spec(_write_spec(tmp_path, body))
    for c in spec.cases:
        assert c["mission.duration_s"] == float(round(c["mission.duration_s"]))


def test_sweep_rejects_unknown_key(tmp_path):
    """An unknown key under [sweep] is rejected (FR-15/DX-2 discipline)."""
    body = _LHS.replace('method = "lhs"', 'method = "lhs"\nbogus_key = 1')
    with pytest.raises(SweepError) as exc:
        load_sweep_spec(_write_spec(tmp_path, body))
    assert any("unknown key" in e for e in exc.value.errors)


def test_sweep_rejects_bad_override_path_at_run_time(tmp_path):
    """A path naming no existing leaf fails when the sweep is run, by name."""
    _core_or_fail()
    body = _LHS.replace(
        'path = "mission.duration_s"', 'path = "mission.no_such_key"'
    )
    (tmp_path / "twobody_leo.toml").write_text(
        (REPO_ROOT / "missions" / "twobody_leo.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    spec_path = _write_spec(tmp_path, body)
    manifest = run_sweep(
        spec_path, workers=1, outdir=str(tmp_path / "out"), force=True
    )
    # Every run fails the same way: the override names no existing leaf.
    assert all(r["status"] == "failed" for r in manifest["runs"])
    assert all("no such key" in r["error"] for r in manifest["runs"])


def test_sweep_requires_matching_n_runs_for_grid(tmp_path):
    body = _GRID.replace('method = "grid"', 'method = "grid"\nn_runs = 99')
    with pytest.raises(SweepError) as exc:
        load_sweep_spec(_write_spec(tmp_path, body))
    assert any("derives 6 run" in e for e in exc.value.errors)


def test_sweep_lhs_requires_n_runs(tmp_path):
    body = _LHS.replace("n_runs = 16\n", "")
    with pytest.raises(SweepError) as exc:
        load_sweep_spec(_write_spec(tmp_path, body))
    assert any("n_runs" in e and "lhs" in e for e in exc.value.errors)


def test_sweep_lhs_rejects_values_array(tmp_path):
    body = _LHS.replace("min = 100.0\nmax = 200.0", "values = [1.0, 2.0]")
    with pytest.raises(SweepError) as exc:
        load_sweep_spec(_write_spec(tmp_path, body))
    assert any("numeric range" in e for e in exc.value.errors)


def test_sweep_accumulates_multiple_errors(tmp_path):
    body = """\
schema_version = 2
[sweep]
method = "nope"
"""
    with pytest.raises(SweepError) as exc:
        load_sweep_spec(_write_spec(tmp_path, body))
    # schema_version, mission, master_seed, method, missing parameters: several.
    assert len(exc.value.errors) >= 3


def test_shipped_sweep_parses(tmp_path):
    """The committed example sweep is a valid spec expanding to 256 cases."""
    _core_or_fail()  # LHS draws from the core stream
    with _in_repo_root():
        spec = load_sweep_spec(SHIPPED_SWEEP)
    assert spec.method == "lhs"
    assert spec.n_runs == 256
    assert len(spec.cases) == 256


# --- end-to-end: exit criterion 1 in miniature -----------------------------


_FAST_SWEEP = """\
schema_version = 1
[sweep]
mission = "twobody_leo.toml"
master_seed = 20260723
method = "lhs"
n_runs = 8
[[sweep.parameter]]
path = "mission.duration_s"
min = 600.0
max = 1200.0
integer = true
[[sweep.parameter]]
path = "spacecraft.mass_kg"
min = 100.0
max = 200.0
"""


def _run_fast_sweep(tmp_path, workers):
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    spec_path = tmp_path / "fast_sweep.toml"
    # The base mission path resolves relative to the spec's directory then cwd;
    # write the spec beside a copy of the mission so it resolves without cwd.
    (tmp_path / "twobody_leo.toml").write_text(
        (REPO_ROOT / "missions" / "twobody_leo.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    spec_path.write_text(_FAST_SWEEP, encoding="utf-8")
    out = tmp_path / "mc_out"
    manifest = run_sweep(spec_path, workers=workers, outdir=str(out), force=True)
    return spec_path, out, manifest


def test_small_sweep_all_success_and_manifest_shape(tmp_path):
    """An 8-run sweep finishes 8/8 success with a well-formed manifest."""
    _core_or_fail()
    _spec, out, manifest = _run_fast_sweep(tmp_path, workers=4)

    assert manifest["schema_version"] == 1
    runs = manifest["runs"]
    assert len(runs) == 8
    assert all(r["status"] == "success" for r in runs)
    assert [r["index"] for r in runs] == list(range(8))  # sorted, contiguous
    assert len({r["log_sha256"] for r in runs}) == 8  # all distinct
    assert manifest["sweep"]["method"] == "lhs"
    assert manifest["sweep"]["n_runs"] == 8
    assert len(manifest["binary"]["binary_sha256"]) == 64
    assert "sweep_spec_sha256" in manifest["sweep"]

    # The manifest on disk matches what run_sweep returned.
    on_disk = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk == manifest

    # Each run's own log re-hashes to the recorded log_sha256.
    for r in runs:
        log = out / r["outdir"] / "run.srlog"
        assert _sha256(log) == r["log_sha256"]


def test_reexecuting_a_manifest_entry_reproduces_its_hash(tmp_path):
    """Criterion 1: re-run one entry with its seed+overrides, hashes match.

    Through run_mission (the star run API), into a fresh directory, so the
    reproduction is genuine and not the sweep's own output being reread.
    """
    _core_or_fail()
    from star_reacher.runner import run_mission

    _spec, _out, manifest = _run_fast_sweep(tmp_path, workers=4)
    entry = manifest["runs"][3]  # any entry

    result = run_mission(
        tmp_path / "twobody_leo.toml",
        outdir=str(tmp_path / "reexec"),
        force=True,
        seed=entry["seed"],
        overrides=entry["overrides"],
    )
    assert result.srlog_sha256 == entry["log_sha256"], (
        "re-executing the manifest entry with its recorded seed and overrides "
        "did not reproduce the logged SHA-256 (Phase 7 exit criterion 1)"
    )
    assert result.config_sha256 == entry["config_sha256"]


def test_worker_count_does_not_change_the_result(tmp_path):
    """A run is bit-identical whether pooled or standalone (D-10).

    The same sweep at 1 worker (in-process) and 4 workers (a process pool)
    must produce the same per-run log hashes.
    """
    _core_or_fail()
    _s1, _o1, single = _run_fast_sweep(tmp_path / "single", workers=1)
    _s4, _o4, pooled = _run_fast_sweep(tmp_path / "pooled", workers=4)
    single_hashes = {r["index"]: r["log_sha256"] for r in single["runs"]}
    pooled_hashes = {r["index"]: r["log_sha256"] for r in pooled["runs"]}
    assert single_hashes == pooled_hashes


def test_manifest_run_matches_standalone_star_run(tmp_path):
    """A pooled worker's log equals a standalone star run with the same flags.

    The contract that a worker calls the same run_mission a CLI star run does:
    take a manifest entry and reproduce it through run_mission directly, then
    assert the bytes match the sweep's own output for that run.
    """
    _core_or_fail()
    from star_reacher.runner import run_mission

    _spec, out, manifest = _run_fast_sweep(tmp_path, workers=4)
    entry = manifest["runs"][0]
    standalone = run_mission(
        tmp_path / "twobody_leo.toml",
        outdir=str(tmp_path / "standalone"),
        force=True,
        seed=entry["seed"],
        overrides=entry["overrides"],
    )
    pooled_log = out / entry["outdir"] / "run.srlog"
    assert _sha256(pooled_log) == standalone.srlog_sha256
