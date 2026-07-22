"""Unit tests for scripts/cross_platform_divergence.py with synthetic inputs.

The script is the comparison instrument behind PRD Phase 2 exit criterion 8
(the cross-platform-divergence CI job), so its arithmetic and its gate rules
are pinned here against hand-computed values and hand-built records. The
script lives outside the package (it is CI tooling, not runtime surface), so
it is imported from its file path.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import math
import struct
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


# ---------------------------------------------------------------------------
# Channel-level widening: the derived tolerance
# ---------------------------------------------------------------------------


def test_channel_tolerance_is_the_derived_geometric_mean():
    """The tolerance is derived, not chosen, and its inputs are pinned here.

    A round number would be a judgement call; this one is a function of two
    quantities - one measured, one specified by D-10 - so moving it requires
    moving one of them.
    """
    assert cpd.MEASURED_WORST_REL == 1.06e-10
    assert cpd.D10_BOUND_REL == 1e-9
    assert cpd.CHANNEL_TOLERANCE_REL == math.sqrt(1.06e-10 * 1e-9)
    assert cpd.CHANNEL_TOLERANCE_REL == pytest.approx(3.2558e-10, rel=1e-4)


def test_channel_tolerance_has_equal_margin_either_way():
    """Equal multiplicative headroom against a false red and against masking.

    Below the measured worst case the gate would red on divergence the
    project has already measured and accepted; at or above the D-10 bound it
    could never fail before D-10 already had, which is the defect this
    widening exists to correct.
    """
    above = cpd.CHANNEL_TOLERANCE_REL / cpd.MEASURED_WORST_REL
    below = cpd.D10_BOUND_REL / cpd.CHANNEL_TOLERANCE_REL
    assert above == pytest.approx(below, rel=1e-12)
    assert above == pytest.approx(3.0715, rel=1e-3)
    assert cpd.MEASURED_WORST_REL < cpd.CHANNEL_TOLERANCE_REL < cpd.D10_BOUND_REL


def test_every_declared_gate_mission_exists():
    """A mission named in the gate table but absent from the tree gates nothing."""
    missing = [
        m["path"] for m in cpd.CHANNEL_MISSIONS.values()
        if not (REPO_ROOT / m["path"]).is_file()
    ]
    assert not missing, f"the gate table names missions that do not exist: {missing}"


def test_the_camera_mission_is_in_the_gate_table():
    """The one shipped camera mission must be compared, not merely committed."""
    assert "leo_optical_nav" in cpd.CHANNEL_MISSIONS
    required = cpd.CHANNEL_MISSIONS["leo_optical_nav"]["require_active"]
    assert "sensors.camera.px_uv" in required


# ---------------------------------------------------------------------------
# Channel-level widening: synthetic two-mission fixture
# ---------------------------------------------------------------------------

# Two synthetic missions exercising both arithmetic declarations. They are
# monkeypatched over the real table so the tests below pin the comparison
# logic rather than any shipped mission's current numbers.
_SYNTH_MISSIONS = {
    "synth_basic": {
        "path": "missions/twobody_leo.toml",
        "arithmetic": "basic-ops-only",
        "require_active": [],
        "min_active": 0,
    },
    "synth_libm": {
        "path": "missions/leo_attitude_gnc.toml",
        "arithmetic": "libm",
        "exact_float_channels": ["truth.r_m"],
        "require_active": ["truth.q_i2b"],
        "min_active": 1,
    },
}


@pytest.fixture
def synth(monkeypatch):
    monkeypatch.setattr(cpd, "CHANNEL_MISSIONS", _SYNTH_MISSIONS)
    return _SYNTH_MISSIONS


def _synth_srlog(path: Path, *, r_x: float = 6778137.0) -> Path:
    """A small SRLOG with a varying attitude and a fixed translational state."""
    header = _fixtures.contract_header()
    records = []
    for i in range(16):
        # q_i2b varies so the channel carries a nonzero RMS and can be
        # perturbed; r_m is constant so a one-ULP change to it is isolated.
        q = (0.5, 0.5, 0.5, 0.5 + 1e-3 * i)
        records.append(
            _fixtures.truth_record(
                0.1 * i, r_m=(r_x, -12.25, 3.0), v_mps=(0.0, 7668.6, 0.0), q_i2b=q
            )
        )
    records.append(_fixtures.event_record(0.0, 1, "start"))
    path.write_bytes(_fixtures.build_srlog(header, records))
    return path


def _extract_channels(tmp_path: Path, mission: str, leg: str, srlog: Path) -> dict:
    out = tmp_path / f"art-{leg}" / f"channels-{mission}.json"
    rc = cpd.main(["extract-channels", "--srlog", str(srlog), "--mission", mission,
                   "--leg", leg, "--out", str(out)])
    assert rc == 0
    return json.loads(out.read_text(encoding="utf-8"))


def _legs_dir(tmp_path: Path, per_leg: dict, mission: str, root: Path | None = None) -> Path:
    """Materialize one artifact per leg from a {leg: srlog} map."""
    root = root if root is not None else tmp_path / "artifacts"
    for leg, srlog in per_leg.items():
        out = root / f"xplat-{leg}" / f"channels-{mission}.json"
        rc = cpd.main(["extract-channels", "--srlog", str(srlog), "--mission", mission,
                       "--leg", leg, "--out", str(out)])
        assert rc == 0
    return root


def _all_missions_dir(tmp_path: Path, per_leg: dict) -> Path:
    """Artifacts for every synthetic mission, so measure-channels is satisfied."""
    root = tmp_path / "artifacts"
    for mission in _SYNTH_MISSIONS:
        _legs_dir(tmp_path, per_leg, mission, root=root)
    return root


def _measure_channels(tmp_path: Path, root: Path, legs: int = 4) -> tuple[int, dict]:
    out = tmp_path / "channel-measurement.json"
    rc = cpd.main(["measure-channels", "--dir", str(root), "--expect-legs", str(legs),
                   "--out", str(out)])
    return rc, json.loads(out.read_text(encoding="utf-8"))


def test_extract_channels_splits_the_two_classes(synth, tmp_path):
    """Structural classification: integers and t_s exact, floats to tolerance."""
    srlog = _synth_srlog(tmp_path / "run.srlog")
    art = _extract_channels(tmp_path, "synth_libm", "windows-2022", srlog)
    # t_s in both groups, the u32 code and the str16 detail are exact by
    # dtype or by name; truth.r_m is exact by the mission's declaration.
    assert set(art["exact"]) == {
        "truth.t_s", "events.t_s", "events.code", "events.detail", "truth.r_m",
    }
    assert "truth.q_i2b" in art["tolerance"]
    assert "truth.v_mps" in art["tolerance"]
    values = cpd._decode_f64(base64.b64decode(art["tolerance"]["truth.q_i2b"]["b64"]))
    assert len(values) == 16 * 4


def test_extract_channels_makes_a_basic_ops_mission_wholly_exact(synth, tmp_path):
    srlog = _synth_srlog(tmp_path / "run.srlog")
    art = _extract_channels(tmp_path, "synth_basic", "windows-2022", srlog)
    assert art["tolerance"] == {}
    assert "truth.q_i2b" in art["exact"]


def test_channel_gate_is_green_on_identical_legs(synth, tmp_path):
    srlog = _synth_srlog(tmp_path / "run.srlog")
    root = _all_missions_dir(tmp_path, {leg: srlog for leg in LEGS})
    rc, m = _measure_channels(tmp_path, root)
    assert rc == 0
    assert m["max_rel"] == 0.0
    assert m["violations"] == []


# ---------------------------------------------------------------------------
# Mutation battery: the widened gate must be able to fail
# ---------------------------------------------------------------------------


def _perturb_tolerance_channel(art_path: Path, key: str, factor: float) -> None:
    """Add ``factor * tolerance * rms`` to one element of a tolerance channel."""
    art = json.loads(art_path.read_text(encoding="utf-8"))
    entry = art["tolerance"][key]
    values = list(cpd._decode_f64(base64.b64decode(entry["b64"])))
    values[len(values) // 2] += cpd._rms(values) * cpd.CHANNEL_TOLERANCE_REL * factor
    entry["b64"] = base64.b64encode(
        struct.pack(f"<{len(values)}d", *values)
    ).decode("ascii")
    art_path.write_text(json.dumps(art), encoding="utf-8")


def test_gate_fails_when_a_libm_channel_exceeds_the_tolerance(synth, tmp_path):
    """The mutation the old gate could not see: a perturbed libm channel."""
    srlog = _synth_srlog(tmp_path / "run.srlog")
    root = _all_missions_dir(tmp_path, {leg: srlog for leg in LEGS})
    _perturb_tolerance_channel(
        root / f"xplat-{LEGS[0]}" / "channels-synth_libm.json", "truth.q_i2b", 1.5
    )
    rc, m = _measure_channels(tmp_path, root)
    assert rc == 0  # measure records honestly; the gate enforces
    assert m["max_rel"] > cpd.CHANNEL_TOLERANCE_REL
    assert m["worst_channel"] == "truth.q_i2b"
    assert cpd.main([
        "gate-channels",
        "--measurement", str(tmp_path / "channel-measurement.json"),
        "--record", str(_channel_record(tmp_path)),
    ]) != 0


def test_gate_stays_green_just_inside_the_tolerance(synth, tmp_path, capsys):
    """Calibration: the gate fails on a real breach, not on any perturbation.

    Without this, the test above would pass just as well for a gate hard-wired
    to fail, which would be the same defect in the opposite direction.
    """
    srlog = _synth_srlog(tmp_path / "run.srlog")
    root = _all_missions_dir(tmp_path, {leg: srlog for leg in LEGS})
    _perturb_tolerance_channel(
        root / f"xplat-{LEGS[0]}" / "channels-synth_libm.json", "truth.q_i2b", 0.5
    )
    rc, m = _measure_channels(tmp_path, root)
    assert rc == 0
    assert 0.0 < m["max_rel"] < cpd.CHANNEL_TOLERANCE_REL
    assert m["violations"] == []
    assert cpd.main([
        "gate-channels",
        "--measurement", str(tmp_path / "channel-measurement.json"),
        "--record", str(_channel_record(tmp_path)),
    ]) == 0
    assert "channel gate passed" in capsys.readouterr().out


def test_gate_fails_on_a_one_ulp_change_in_an_exact_channel(synth, tmp_path):
    """The exact-class assertion fires on the smallest possible change.

    The perturbation is one unit in the last place of ``truth.r_m`` - the
    channel the unwidened gate sampled - injected into the log itself rather
    than into the artifact, so the whole extract-and-compare path is under
    test.
    """
    clean = _synth_srlog(tmp_path / "clean.srlog")
    one_ulp = math.nextafter(6778137.0, math.inf)
    assert one_ulp != 6778137.0
    mutated = _synth_srlog(tmp_path / "mutated.srlog", r_x=one_ulp)

    per_leg = {leg: clean for leg in LEGS}
    per_leg[LEGS[0]] = mutated
    root = _all_missions_dir(tmp_path, per_leg)
    rc, m = _measure_channels(tmp_path, root)
    assert rc == 0
    # The change is 1 ULP, far below the tolerance: it is caught because the
    # channel is gated for bit-identity, not against a threshold.
    assert m["max_rel"] <= cpd.CHANNEL_TOLERANCE_REL
    assert any("truth.r_m" in v and "not bit-identical" in v for v in m["violations"]), (
        f"a one-ULP change in an exact-class channel went unreported: {m['violations']}"
    )
    assert cpd.main([
        "gate-channels",
        "--measurement", str(tmp_path / "channel-measurement.json"),
        "--record", str(_channel_record(tmp_path)),
    ]) != 0


def test_gate_fails_when_a_required_channel_is_identically_zero(synth, tmp_path):
    """Anti-degeneracy: an all-zero channel compares equal for free."""
    srlog = _synth_srlog(tmp_path / "run.srlog")
    root = _all_missions_dir(tmp_path, {leg: srlog for leg in LEGS})
    for leg in LEGS:
        p = root / f"xplat-{leg}" / "channels-synth_libm.json"
        art = json.loads(p.read_text(encoding="utf-8"))
        entry = art["tolerance"]["truth.q_i2b"]
        n = len(cpd._decode_f64(base64.b64decode(entry["b64"])))
        entry["b64"] = base64.b64encode(
            struct.pack(f"<{n}d", *([0.0] * n))
        ).decode("ascii")
        p.write_text(json.dumps(art), encoding="utf-8")
    rc, m = _measure_channels(tmp_path, root)
    assert rc == 0
    assert any("identically zero" in v for v in m["violations"]), m["violations"]


def test_gate_fails_when_the_header_differs_across_legs(synth, tmp_path):
    srlog = _synth_srlog(tmp_path / "run.srlog")
    root = _all_missions_dir(tmp_path, {leg: srlog for leg in LEGS})
    p = root / f"xplat-{LEGS[0]}" / "channels-synth_libm.json"
    art = json.loads(p.read_text(encoding="utf-8"))
    art["header_sha256"] = "0" * 64
    p.write_text(json.dumps(art), encoding="utf-8")
    _, m = _measure_channels(tmp_path, root)
    assert any("headers are not byte-identical" in v for v in m["violations"])


def test_measure_channels_fails_when_a_declared_mission_is_absent(synth, tmp_path):
    """A mission that silently stops being compared is the target failure mode."""
    srlog = _synth_srlog(tmp_path / "run.srlog")
    root = _legs_dir(tmp_path, {leg: srlog for leg in LEGS}, "synth_libm")
    assert cpd.main(["measure-channels", "--dir", str(root), "--expect-legs", "4",
                     "--out", str(tmp_path / "m.json")]) != 0


def test_measure_channels_fails_on_a_missing_leg(synth, tmp_path):
    srlog = _synth_srlog(tmp_path / "run.srlog")
    root = _all_missions_dir(tmp_path, {leg: srlog for leg in LEGS[:3]})
    assert cpd.main(["measure-channels", "--dir", str(root), "--expect-legs", "4",
                     "--out", str(tmp_path / "m.json")]) != 0


# ---------------------------------------------------------------------------
# gate-channels record rules
# ---------------------------------------------------------------------------

CHANNEL_RECORD = """\
schema_version = 2

[record]
status = "measured"
mission = "missions/twobody_leo.toml"
legs = ["a", "b", "c", "d"]
bound_rel = 1e-9
measured_max_rel = 0.0
ci_run_url = "https://example.invalid/run/1"
date = "2026-07-02"

[channels]
status = "measured"
tolerance_rel = {tolerance}
missions = {missions}
measured_max_rel = {value}
ci_run_url = "https://example.invalid/run/1"
date = "2026-07-19"
"""


def _channel_record(tmp_path: Path, **kwargs) -> Path:
    body = CHANNEL_RECORD.format(
        tolerance=kwargs.get("tolerance", repr(cpd.CHANNEL_TOLERANCE_REL)),
        missions=kwargs.get("missions", json.dumps(sorted(cpd.CHANNEL_MISSIONS))),
        value=kwargs.get("value", "1.0e-10"),
    )
    return _write_record(tmp_path / "rec.toml", body)


def _clean_channel_measurement(tmp_path: Path, max_rel: float = 1.0e-10) -> Path:
    path = tmp_path / "cm.json"
    path.write_text(
        json.dumps({
            "schema": 1, "max_rel": max_rel, "violations": [],
            "worst_mission": "m", "worst_channel": "c", "worst_pair": "p",
        }),
        encoding="utf-8",
    )
    return path


def test_gate_channels_passes_on_a_measured_record(synth, tmp_path, capsys):
    rc = cpd.main([
        "gate-channels",
        "--measurement", str(_clean_channel_measurement(tmp_path)),
        "--record", str(_channel_record(tmp_path)),
    ])
    assert rc == 0
    assert "channel gate passed" in capsys.readouterr().out


def test_gate_channels_fails_on_a_pending_record(synth, tmp_path, capsys):
    body = CHANNEL_RECORD.format(
        tolerance=repr(cpd.CHANNEL_TOLERANCE_REL),
        missions=json.dumps(sorted(cpd.CHANNEL_MISSIONS)),
        value="1.0e-10",
    ).replace(
        '[channels]\nstatus = "measured"',
        '[channels]\nstatus = "pending-first-measurement"',
    )
    rec = _write_record(tmp_path / "rec.toml", body)
    rc = cpd.main([
        "gate-channels",
        "--measurement", str(_clean_channel_measurement(tmp_path)),
        "--record", str(rec),
    ])
    assert rc != 0
    assert "pending-first-measurement" in capsys.readouterr().err


def test_gate_channels_fails_when_the_record_tolerance_disagrees(synth, tmp_path, capsys):
    """The derivation and the committed record move together or not at all."""
    rc = cpd.main([
        "gate-channels",
        "--measurement", str(_clean_channel_measurement(tmp_path)),
        "--record", str(_channel_record(tmp_path, tolerance="5.0e-10")),
    ])
    assert rc != 0
    assert "disagrees with the enforced tolerance" in capsys.readouterr().err


def test_gate_channels_fails_when_the_record_mission_list_drifts(synth, tmp_path, capsys):
    """Dropping a mission from the gate must not be possible silently."""
    rc = cpd.main([
        "gate-channels",
        "--measurement", str(_clean_channel_measurement(tmp_path)),
        "--record", str(_channel_record(tmp_path, missions=json.dumps(["synth_libm"]))),
    ])
    assert rc != 0
    assert "must update both homes" in capsys.readouterr().err


def test_gate_channels_fails_above_the_tolerance(synth, tmp_path, capsys):
    rc = cpd.main([
        "gate-channels",
        "--measurement", str(_clean_channel_measurement(tmp_path, max_rel=5.0e-10)),
        "--record", str(_channel_record(tmp_path)),
    ])
    assert rc != 0
    assert "exceeds the derived tolerance" in capsys.readouterr().err


def test_gate_channels_propagates_measurement_violations(synth, tmp_path, capsys):
    path = tmp_path / "cm.json"
    path.write_text(
        json.dumps({
            "schema": 1, "max_rel": 0.0,
            "violations": ["synth_libm: exact-class channel truth.r_m is not bit-identical"],
        }),
        encoding="utf-8",
    )
    rc = cpd.main([
        "gate-channels", "--measurement", str(path),
        "--record", str(_channel_record(tmp_path)),
    ])
    assert rc != 0
    assert "not bit-identical" in capsys.readouterr().err
