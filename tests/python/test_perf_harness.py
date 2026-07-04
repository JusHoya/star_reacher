"""Tests for the Phase 5 performance and minimality gate tooling.

Covers scripts/perf_gate.py (EC-4 absolute gates, the EC-5 rolling-median
compare rule, measurement JSON schema) and scripts/check_min_deps.py plus
the committed dependency-closure expectation (EC-6). The measurement smoke
test runs on the tiny two-body reference mission, never Mission A, so the
suite's wall time stays flat; the real missions are exercised by the
nightly workflow and the Pi 5 checklist, not by pytest.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    """Import a scripts/ module by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


perf_gate = _load_script("perf_gate")
check_min_deps = _load_script("check_min_deps")


# ---------------------------------------------------------------------------
# Absolute gates (EC-4 thresholds, both senses)
# ---------------------------------------------------------------------------


def test_gate_senses_and_thresholds():
    # Wall time: strictly below 60 s passes, 60.0 itself fails ("< 60 s").
    assert perf_gate.gate_passes("mission_a_wall_s", 59.999)
    assert not perf_gate.gate_passes("mission_a_wall_s", 60.0)
    # Real-time factor: 100.0 itself passes (">= 100x").
    assert perf_gate.gate_passes("ascent_rt_factor", 100.0)
    assert not perf_gate.gate_passes("ascent_rt_factor", 99.999)
    # Write throughput: 50.0 itself passes (">= 50 MB/s").
    assert perf_gate.gate_passes("srlog_write_mbps", 50.0)
    assert not perf_gate.gate_passes("srlog_write_mbps", 49.999)


# ---------------------------------------------------------------------------
# EC-5 compare rule (direction-aware, strict > 10 % boundary)
# ---------------------------------------------------------------------------


def _measurement(**values) -> dict:
    return {"metrics": {k: {"value": v} for k, v in values.items()}}


def _median(**values) -> dict:
    return {"metrics": {k: {"median": v} for k, v in values.items()}}


def test_compare_wall_time_regresses_up():
    # Wall time is lower-is-better: +10 % exactly passes, just past it fails.
    ok, _ = perf_gate.compare_metric("mission_a_wall_s", 55.0, 50.0, 0.10)
    assert ok
    ok, reg = perf_gate.compare_metric("mission_a_wall_s", 55.001, 50.0, 0.10)
    assert not ok
    assert reg > 0.10
    # Getting faster is never a regression, however large the change.
    ok, reg = perf_gate.compare_metric("mission_a_wall_s", 25.0, 50.0, 0.10)
    assert ok
    assert reg < 0


def test_compare_rt_factor_and_mbps_regress_down():
    # Higher-is-better metrics: -10 % exactly passes, just past it fails.
    for metric in ("ascent_rt_factor", "srlog_write_mbps"):
        ok, _ = perf_gate.compare_metric(metric, 180.0, 200.0, 0.10)
        assert ok
        ok, reg = perf_gate.compare_metric(metric, 179.99, 200.0, 0.10)
        assert not ok
        assert reg > 0.10
        # Getting faster is never a regression.
        ok, reg = perf_gate.compare_metric(metric, 400.0, 200.0, 0.10)
        assert ok
        assert reg < 0


def test_compare_documents_verdicts():
    current = _measurement(mission_a_wall_s=40.0, ascent_rt_factor=150.0)
    median = _median(mission_a_wall_s=41.0, ascent_rt_factor=140.0)
    ok, lines = perf_gate.compare_documents(current, median, 0.10)
    assert ok
    assert len(lines) == 2

    # A regressed metric flips the verdict.
    current = _measurement(mission_a_wall_s=50.0, ascent_rt_factor=150.0)
    ok, _ = perf_gate.compare_documents(current, median, 0.10)
    assert not ok


def test_compare_documents_missing_metric_fails():
    # A metric with history that vanishes from the current measurement must
    # fail: silently dropping a metric would hide its regressions forever.
    current = _measurement(mission_a_wall_s=40.0)
    median = _median(mission_a_wall_s=41.0, ascent_rt_factor=140.0)
    ok, lines = perf_gate.compare_documents(current, median, 0.10)
    assert not ok
    assert any("missing from the current measurement" in ln for ln in lines)


def test_compare_documents_no_history_skips():
    # A brand-new metric has no median yet: reported, not failed (the
    # rolling gate for it becomes live once history exists).
    current = _measurement(mission_a_wall_s=40.0, srlog_write_mbps=500.0)
    median = _median(mission_a_wall_s=41.0)
    ok, lines = perf_gate.compare_documents(current, median, 0.10)
    assert ok
    assert any("no history, skipped" in ln for ln in lines)


def test_median_of_documents():
    docs = [
        _measurement(mission_a_wall_s=40.0, ascent_rt_factor=100.0),
        _measurement(mission_a_wall_s=44.0, ascent_rt_factor=300.0),
        _measurement(mission_a_wall_s=41.0),
    ]
    doc = perf_gate.median_of_documents(docs)
    assert doc["schema"] == perf_gate.SCHEMA_MEDIAN
    assert doc["metrics"]["mission_a_wall_s"] == {"median": 41.0, "n": 3}
    # Even-count median interpolates (statistics.median).
    assert doc["metrics"]["ascent_rt_factor"] == {"median": 200.0, "n": 2}


# ---------------------------------------------------------------------------
# Measurement smoke (tiny mission; schema, identity, gate evaluation)
# ---------------------------------------------------------------------------


def test_measure_smoke_schema_and_gates(tmp_path):
    out_json = tmp_path / "measurement.json"
    twobody = REPO / "missions" / "twobody_leo.toml"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "perf_gate.py"),
            "measure",
            # The tiny reference mission stands in for both mission metrics
            # so the suite stays fast; the gate MATH is identical.
            "--mission-a", str(twobody),
            "--ascent", str(twobody),
            "--srlog-records", "20000",
            "--json", str(out_json),
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    # A gate verdict (0 pass / 1 fail) is machine-dependent and both are
    # legitimate here; a crash never writes the JSON and fails below.
    assert proc.returncode in (0, 1), proc.stdout + proc.stderr
    assert out_json.exists(), proc.stdout + proc.stderr
    doc = json.loads(out_json.read_text(encoding="utf-8"))

    assert doc["schema"] == perf_gate.SCHEMA_MEASUREMENT
    assert set(doc["metrics"]) == set(perf_gate.DEFAULT_METRICS)

    # Runner identity: every field populated (this is what makes a number
    # attributable to the machine that produced it).
    runner = doc["runner"]
    for key in ("platform", "machine", "system", "python_version"):
        assert isinstance(runner[key], str) and runner[key]
    assert isinstance(runner["cpu_count"], int) and runner["cpu_count"] >= 1
    assert doc["package_version"]
    assert doc["core_version"]
    assert doc["git_sha"]

    # Each metric entry is self-consistent with the gate rule it claims.
    for name, entry in doc["metrics"].items():
        assert entry["value"] > 0.0
        assert entry["threshold"] == perf_gate.GATES[name]["threshold"]
        assert entry["sense"] == perf_gate.GATES[name]["sense"]
        assert entry["pass"] == perf_gate.gate_passes(name, entry["value"])
    assert doc["all_pass"] == all(e["pass"] for e in doc["metrics"].values())
    assert (proc.returncode == 0) == doc["all_pass"]

    # PASS/FAIL is printed for humans as well as encoded in the JSON.
    assert "PERF: PASS" in proc.stdout or "PERF: FAIL" in proc.stdout


# ---------------------------------------------------------------------------
# EC-6: dep-minimality checker and the committed closure expectation
# ---------------------------------------------------------------------------


def test_check_min_deps_set_comparison():
    ok, lines = check_min_deps.compare_sets({"a", "b"}, {"a", "b"})
    assert ok and not lines
    ok, lines = check_min_deps.compare_sets({"a", "b", "c"}, {"a", "b"})
    assert not ok
    assert any(ln.startswith("UNEXPECTED: c") for ln in lines)
    ok, lines = check_min_deps.compare_sets({"a"}, {"a", "b"})
    assert not ok
    assert any(ln.startswith("MISSING: b") for ln in lines)


def test_check_min_deps_name_normalization():
    # PEP 503: dashes, underscores, dots, and case all collapse.
    assert check_min_deps.normalize("Python_Dateutil") == "python-dateutil"
    assert check_min_deps.normalize("ruamel.yaml") == "ruamel-yaml"


def _requirement_name(spec: str) -> str:
    """Distribution name from a requirement string ('numpy>=1.26' -> 'numpy')."""
    match = re.match(r"\s*([A-Za-z0-9._-]+)", spec)
    assert match, f"unparseable requirement: {spec!r}"
    return check_min_deps.normalize(match.group(1))


def test_runtime_deps_expectation_matches_declared_closure():
    """The committed closure equals {declared deps + recursive Requires-Dist}.

    This is the self-explanation layer for the CI dep-minimality gate: when
    [project].dependencies drifts from tests/golden/packaging/runtime_deps.toml,
    this test either fails naming the new dependency (added without
    re-deriving the closure) or skips naming the missing one (a root the
    expectation anticipates that pyproject does not declare yet - the
    Phase 5 integration window where matplotlib lands from a concurrent
    workstream - or a root that was removed, in which case the expectation
    must be re-derived per its header).
    """
    import importlib.metadata as md

    expectation = tomllib.loads(
        (REPO / "tests" / "golden" / "packaging" / "runtime_deps.toml").read_text(
            encoding="utf-8"
        )
    )
    roots = {check_min_deps.normalize(r) for r in expectation["roots"]}
    expected = {check_min_deps.normalize(d) for d in expectation["distributions"]}
    assert "star-reacher" in expected
    assert roots <= expected

    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    declared = {_requirement_name(d) for d in pyproject["project"]["dependencies"]}

    assert declared <= roots, (
        f"pyproject.toml declares {sorted(declared - roots)} which is not in the "
        f"committed closure roots - re-derive tests/golden/packaging/"
        f"runtime_deps.toml per its header"
    )
    if declared != roots:
        pytest.skip(
            f"pyproject.toml does not (yet) declare {sorted(roots - declared)}: "
            f"either the concurrent Phase 5 workstream adding it has not merged "
            f"here, or it was removed and the expectation must be re-derived; "
            f"the CI dep-minimality job still enforces the full committed closure"
        )

    # declared == roots: walk the recursive Requires-Dist closure the way pip
    # resolved it. Extra-gated requirements are not runtime dependencies; a
    # requirement carrying any other environment marker is counted only if
    # pip actually installed it here (mirroring pip's marker evaluation
    # without re-implementing it).
    closure: set[str] = set()
    stack = sorted(roots)
    while stack:
        name = stack.pop()
        if name in closure:
            continue
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            pytest.skip(
                f"declared dependency {name!r} is not installed in this venv; "
                f"the CI venv (pip install .) resolves the closure"
            )
        closure.add(name)
        for req in dist.requires or []:
            if "extra ==" in req:
                continue
            req_name = _requirement_name(req)
            if ";" in req:
                try:
                    md.distribution(req_name)
                except md.PackageNotFoundError:
                    continue  # marker evaluated false on this platform
            stack.append(req_name)

    assert closure | {"star-reacher"} == expected, (
        "the committed closure no longer matches the resolved Requires-Dist "
        "closure of the declared dependencies - re-derive "
        "tests/golden/packaging/runtime_deps.toml per its header "
        f"(resolved here: {sorted(closure)})"
    )
