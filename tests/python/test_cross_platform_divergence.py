"""Unit tests for scripts/cross_platform_divergence.py with synthetic inputs.

The script is the comparison instrument behind PRD Phase 2 exit criterion 8
(the cross-platform-divergence CI job), so its arithmetic and its gate rules
are pinned here against hand-computed values and hand-built records. The
script lives outside the package (it is CI tooling, not runtime surface), so
it is imported from its file path.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from star_reacher import _fixtures

REPO_ROOT = Path(__file__).resolve().parents[2]

_SPEC = importlib.util.spec_from_file_location(
    "cross_platform_divergence",
    REPO_ROOT / "scripts" / "cross_platform_divergence.py",
)
cpd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cpd)

LEGS = ["ubuntu-24.04", "ubuntu-24.04-arm", "macos-15", "windows-2022"]


def _state(leg: str, r: list[float], v: list[float], t: float = 5400.0) -> dict:
    return {
        "leg": leg,
        "t_s_hex": t.hex(),
        "t_s": t,
        "r_m": list(r),
        "v_mps": list(v),
    }


def _write_finalstate(root: Path, state: dict) -> None:
    d = root / f"finalstate-{state['leg']}"
    d.mkdir(parents=True)
    payload = {
        "schema": 1,
        "leg": state["leg"],
        "t_s_hex": state["t_s_hex"],
        "r_m_hex": [x.hex() for x in state["r_m"]],
        "v_mps_hex": [x.hex() for x in state["v_mps"]],
    }
    (d / "finalstate.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_record(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


R_BASE = [6.7e6, 1.0e6, -2.0e5]
V_BASE = [100.0, 7.5e3, -50.0]


# ---------------------------------------------------------------------------
# Divergence arithmetic
# ---------------------------------------------------------------------------


def test_divergence_zero_for_identical_states():
    states = [_state(leg, R_BASE, V_BASE) for leg in LEGS]
    result = cpd.max_pairwise_divergence(states)
    assert result["max_rel"] == 0.0
    assert result["max_rel_pos"] == 0.0
    assert result["max_rel_vel"] == 0.0


def test_divergence_matches_hand_computed_value():
    # One leg perturbed by +1e-3 m along x: the worst pair is (perturbed,
    # any other), rel_pos = 1e-3 / min(|r|, |r'|) with the unperturbed norm
    # smaller. Velocity is identical, so max_rel is the position value.
    r_perturbed = [R_BASE[0] + 1e-3, R_BASE[1], R_BASE[2]]
    states = [_state(LEGS[0], r_perturbed, V_BASE)] + [
        _state(leg, R_BASE, V_BASE) for leg in LEGS[1:]
    ]
    expected = 1e-3 / math.hypot(*R_BASE)
    result = cpd.max_pairwise_divergence(states)
    assert result["max_rel"] == pytest.approx(expected, rel=1e-9)
    assert result["max_rel_vel"] == 0.0
    assert result["worst_quantity"] == "position"
    assert LEGS[0] in result["worst_pair"]


def test_divergence_velocity_dominates_when_larger():
    v_perturbed = [V_BASE[0], V_BASE[1] + 1e-4, V_BASE[2]]
    states = [_state(LEGS[0], R_BASE, v_perturbed)] + [
        _state(leg, R_BASE, V_BASE) for leg in LEGS[1:]
    ]
    expected = 1e-4 / math.hypot(*V_BASE)
    result = cpd.max_pairwise_divergence(states)
    assert result["max_rel"] == pytest.approx(expected, rel=1e-9)
    assert result["worst_quantity"] == "velocity"


def test_divergence_rejects_zero_norm_scale():
    states = [
        _state(LEGS[0], [0.0, 0.0, 0.0], V_BASE),
        _state(LEGS[1], R_BASE, V_BASE),
    ]
    with pytest.raises(ValueError, match="zero-norm"):
        cpd.max_pairwise_divergence(states)


# ---------------------------------------------------------------------------
# measure subcommand
# ---------------------------------------------------------------------------


def _run_measure(tmp_path: Path, states: list[dict], expect_legs: int = 4) -> tuple[int, Path]:
    art = tmp_path / "finalstates"
    for s in states:
        _write_finalstate(art, s)
    out = tmp_path / "measurement.json"
    rc = cpd.main(
        [
            "measure",
            "--dir", str(art),
            "--expect-legs", str(expect_legs),
            "--bound", "1e-9",
            "--out", str(out),
        ]
    )
    return rc, out


def test_measure_end_to_end(tmp_path, capsys):
    r_perturbed = [R_BASE[0] + 1e-3, R_BASE[1], R_BASE[2]]
    states = [_state(LEGS[0], r_perturbed, V_BASE)] + [
        _state(leg, R_BASE, V_BASE) for leg in LEGS[1:]
    ]
    rc, out = _run_measure(tmp_path, states)
    assert rc == 0
    stdout = capsys.readouterr().out
    measurement = json.loads(out.read_text(encoding="utf-8"))
    # Hand-computed: 1e-3 m along x over the unperturbed norm (the smaller
    # of the pair); the 1e-6 slack covers the representation of x + 1e-3.
    assert measurement["max_rel"] == pytest.approx(1e-3 / math.hypot(*R_BASE), rel=1e-6)
    assert sorted(measurement["legs"]) == sorted(LEGS)
    # Machine-readable lines the workflow appends to $GITHUB_OUTPUT.
    assert f"max_rel={measurement['max_rel']:.3e}" in stdout
    assert "status_state=success" in stdout


def test_measure_reports_failure_state_above_bound(tmp_path, capsys):
    r_perturbed = [R_BASE[0] + 1.0, R_BASE[1], R_BASE[2]]  # 1 m: rel ~ 1.5e-7
    states = [_state(LEGS[0], r_perturbed, V_BASE)] + [
        _state(leg, R_BASE, V_BASE) for leg in LEGS[1:]
    ]
    rc, _ = _run_measure(tmp_path, states)
    assert rc == 0  # measure records honestly; the gate subcommand fails
    assert "status_state=failure" in capsys.readouterr().out


def test_measure_fails_on_missing_leg(tmp_path, capsys):
    states = [_state(leg, R_BASE, V_BASE) for leg in LEGS[:3]]
    rc, _ = _run_measure(tmp_path, states, expect_legs=4)
    assert rc != 0
    assert "expected 4 leg artifacts" in capsys.readouterr().err


def test_measure_fails_on_duplicate_leg(tmp_path, capsys):
    states = [_state(leg, R_BASE, V_BASE) for leg in LEGS[:3]]
    dup = _state(LEGS[0], R_BASE, V_BASE)
    dup_dir = tmp_path / "finalstates" / "finalstate-dup"
    dup_dir.mkdir(parents=True)
    (dup_dir / "finalstate.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "leg": dup["leg"],
                "t_s_hex": dup["t_s_hex"],
                "r_m_hex": [x.hex() for x in dup["r_m"]],
                "v_mps_hex": [x.hex() for x in dup["v_mps"]],
            }
        ),
        encoding="utf-8",
    )
    rc, _ = _run_measure(tmp_path, states, expect_legs=4)
    assert rc != 0
    assert "duplicate leg identifiers" in capsys.readouterr().err


def test_measure_fails_on_epoch_mismatch(tmp_path, capsys):
    states = [_state(leg, R_BASE, V_BASE) for leg in LEGS[:3]]
    states.append(_state(LEGS[3], R_BASE, V_BASE, t=5400.1))
    rc, _ = _run_measure(tmp_path, states)
    assert rc != 0
    assert "not bit-identical" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# gate subcommand
# ---------------------------------------------------------------------------


def _measurement_file(tmp_path: Path, max_rel: float) -> Path:
    path = tmp_path / "measurement.json"
    path.write_text(json.dumps({"schema": 1, "max_rel": max_rel}), encoding="utf-8")
    return path


MEASURED_RECORD = """\
schema_version = 1

[record]
status = "measured"
mission = "missions/twobody_leo.toml"
legs = ["a", "b", "c", "d"]
bound_rel = 1e-9
measured_max_rel = {value}
ci_run_url = "https://example.invalid/run/1"
date = "2026-07-02"
"""

PENDING_RECORD = """\
schema_version = 1

[record]
status = "pending-first-measurement"
mission = "missions/twobody_leo.toml"
legs = ["a", "b", "c", "d"]
bound_rel = 1e-9
"""


def _run_gate(measurement: Path, record: Path) -> int:
    return cpd.main(
        [
            "gate",
            "--measurement", str(measurement),
            "--record", str(record),
            "--bound", "1e-9",
        ]
    )


def test_gate_passes_on_measured_record(tmp_path, capsys):
    m = _measurement_file(tmp_path, 2.0e-10)
    r = _write_record(tmp_path / "rec.toml", MEASURED_RECORD.format(value="3.0e-10"))
    assert _run_gate(m, r) == 0
    assert "criterion-8 gate passed" in capsys.readouterr().out


def test_gate_fails_on_pending_record(tmp_path, capsys):
    m = _measurement_file(tmp_path, 2.0e-10)
    r = _write_record(tmp_path / "rec.toml", PENDING_RECORD)
    assert _run_gate(m, r) != 0
    assert "pending-first-measurement" in capsys.readouterr().err


def test_gate_fails_on_missing_record(tmp_path, capsys):
    m = _measurement_file(tmp_path, 2.0e-10)
    assert _run_gate(m, tmp_path / "absent.toml") != 0
    assert "missing" in capsys.readouterr().err


def test_gate_fails_when_measurement_exceeds_bound(tmp_path, capsys):
    m = _measurement_file(tmp_path, 2.0e-9)
    r = _write_record(tmp_path / "rec.toml", MEASURED_RECORD.format(value="3.0e-10"))
    assert _run_gate(m, r) != 0
    assert "exceeds the D-10 bound" in capsys.readouterr().err


def test_gate_fails_when_record_value_exceeds_bound(tmp_path, capsys):
    m = _measurement_file(tmp_path, 2.0e-10)
    r = _write_record(tmp_path / "rec.toml", MEASURED_RECORD.format(value="2.0e-9"))
    assert _run_gate(m, r) != 0
    assert "committed measured_max_rel" in capsys.readouterr().err


def test_gate_fails_when_record_lacks_value(tmp_path, capsys):
    m = _measurement_file(tmp_path, 2.0e-10)
    body = MEASURED_RECORD.format(value="1.0e-10").replace(
        "measured_max_rel = 1.0e-10\n", ""
    )
    r = _write_record(tmp_path / "rec.toml", body)
    assert _run_gate(m, r) != 0
    assert "measured_max_rel is absent" in capsys.readouterr().err


def test_gate_fails_on_bound_mismatch(tmp_path, capsys):
    m = _measurement_file(tmp_path, 2.0e-10)
    body = MEASURED_RECORD.format(value="3.0e-10").replace(
        "bound_rel = 1e-9", "bound_rel = 1e-8"
    )
    r = _write_record(tmp_path / "rec.toml", body)
    assert _run_gate(m, r) != 0
    assert "disagrees with the enforced bound" in capsys.readouterr().err


def test_gate_warns_without_failing_on_factor_ten_drift(tmp_path, capsys):
    m = _measurement_file(tmp_path, 5.0e-11)
    r = _write_record(tmp_path / "rec.toml", MEASURED_RECORD.format(value="8.0e-10"))
    assert _run_gate(m, r) == 0
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "factor of 10" in out


def test_gate_no_warning_within_factor_ten(tmp_path, capsys):
    m = _measurement_file(tmp_path, 2.0e-10)
    r = _write_record(tmp_path / "rec.toml", MEASURED_RECORD.format(value="3.0e-10"))
    assert _run_gate(m, r) == 0
    assert "WARNING" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# extract subcommand (against a synthesized SRLOG)
# ---------------------------------------------------------------------------


def test_extract_writes_final_truth_record_at_full_precision(tmp_path, capsys):
    header = _fixtures.contract_header()
    # Two truth records: extract must take the FINAL one. Values exercise
    # full 17-significant-digit reprs.
    r_final = (6778137.000000123, -12.25, 3.0e-9)
    v_final = (-0.1, 7668.600000000001, 5e-324)
    records = [
        _fixtures.truth_record(0.0),
        _fixtures.truth_record(5400.0, r_m=r_final, v_mps=v_final),
    ]
    srlog = tmp_path / "run.srlog"
    srlog.write_bytes(_fixtures.build_srlog(header, records))
    out = tmp_path / "finalstate" / "finalstate.json"
    rc = cpd.main(
        ["extract", "--srlog", str(srlog), "--leg", "test-leg", "--out", str(out)]
    )
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["leg"] == "test-leg"
    assert data["t_s_hex"] == (5400.0).hex()
    assert [float.fromhex(h) for h in data["r_m_hex"]] == list(r_final)
    assert [float.fromhex(h) for h in data["v_mps_hex"]] == list(v_final)
    # The decimal mirrors stay consistent with the authoritative hex fields.
    assert data["r_m"] == list(r_final)
    # Round trip through the measure-side reader.
    parsed = cpd.load_finalstate(out)
    assert parsed["r_m"] == list(r_final)
    assert parsed["v_mps"] == list(v_final)
