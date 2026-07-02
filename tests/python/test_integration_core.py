"""Integration tests that REQUIRE the compiled core (star_reacher._core).

These fail cleanly, never skip, when the core is absent: a green suite must
mean the whole contract holds, and a silent skip would let a broken or
missing core masquerade as verified (see the project's agent-honesty gate).
They are expected to fail on a core-less checkout and to pass at orchestrator
integration and in CI, where the wheel is installed.

The RNG cross-check reimplements the contract section 4 seeding chain
(FNV-1a-64 -> SplitMix64 -> PCG64 setseq-128 srandom) in pure Python integers
and compares the core's stream to numpy.random.PCG64 raw output with the same
derived state, proving bit-compatibility with the reference PCG64 XSL-RR
128/64 generator (O'Neill) rather than merely self-consistency.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_CORE_MISSING_MESSAGE = (
    "star_reacher._core is not built in this environment. These integration tests "
    "require the compiled core: build and install it with 'pip install .' from the "
    "repository root (CMake >= 3.26 and a C++17 compiler required). This failure is "
    "expected on a core-less checkout and must be green at integration/CI."
)


def _core_or_fail():
    try:
        from star_reacher import _core
    except ImportError:
        pytest.fail(_CORE_MISSING_MESSAGE)
    return _core


# --- Pure-Python reimplementation of the contract section 4 seeding chain ---

_MASK64 = (1 << 64) - 1
_MASK128 = (1 << 128) - 1
# PCG default multiplier for the setseq 128-bit LCG (O'Neill, PCG report
# HMC-CS-2014-0905 / pcg-random.org).
_PCG_MULT = 0x2360ED051FC65DA44385DF649FCCF645


def _fnv1a64(data: bytes) -> int:
    # FNV-1a 64-bit (IETF draft-eastlake-fnv): offset basis and prime per spec.
    h = 14695981039346656037
    for byte in data:
        h ^= byte
        h = (h * 1099511628211) & _MASK64
    return h


def _splitmix64(seed: int, n: int) -> list[int]:
    # Vigna's reference splitmix64.c (public domain), used verbatim by D-9.
    state = seed & _MASK64
    out = []
    for _ in range(n):
        state = (state + 0x9E3779B97F4A7C15) & _MASK64
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
        out.append(z ^ (z >> 31))
    return out


def _pcg64_state_for_stream(master_seed: int, stream_name: str) -> tuple[int, int]:
    """Derive the (state, inc) pair per the contract section 4 procedure."""
    h = _fnv1a64(stream_name.encode("utf-8"))
    sm1, sm2, sm3, sm4 = _splitmix64((master_seed ^ h) & _MASK64, 4)
    initstate = ((sm1 << 64) | sm2) & _MASK128
    initseq = ((sm3 << 64) | sm4) & _MASK128
    # Reference pcg_setseq_128_srandom_r: state=0; inc=(initseq<<1)|1;
    # step; state += initstate; step.
    inc = ((initseq << 1) | 1) & _MASK128
    state = 0
    state = (state * _PCG_MULT + inc) & _MASK128
    state = (state + initstate) & _MASK128
    state = (state * _PCG_MULT + inc) & _MASK128
    return state, inc


def _numpy_pcg64_raw(state: int, inc: int, n: int) -> list[int]:
    bit_gen = np.random.PCG64()
    bit_gen.state = {
        "bit_generator": "PCG64",
        "state": {"state": state, "inc": inc},
        "has_uint32": 0,
        "uinteger": 0,
    }
    return [int(x) for x in bit_gen.random_raw(n)]


@pytest.mark.parametrize(
    ("master_seed", "stream_name"),
    [
        (1234567890, "sensors.imu"),
        (1234567890, "dispersions.mass"),
        (0, "sensors.imu"),
        (0xDEADBEEF12345678, "nav.startracker"),
    ],
)
def test_rng_stream_matches_numpy_pcg64_raw(master_seed, stream_name):
    core = _core_or_fail()
    state, inc = _pcg64_state_for_stream(master_seed, stream_name)
    expected = _numpy_pcg64_raw(state, inc, 64)
    got = [int(x) for x in core.rng_stream_u64(master_seed, stream_name, 64)]
    assert got == expected


def test_rng_streams_are_independent():
    core = _core_or_fail()
    a = [int(x) for x in core.rng_stream_u64(42, "sensors.imu", 32)]
    b = [int(x) for x in core.rng_stream_u64(42, "dispersions.mass", 32)]
    assert a != b


def test_gm_earth_is_iers_2010_value():
    core = _core_or_fail()
    # IERS Conventions (2010), TN No. 36: GM_earth = 3.986004418e14 m^3/s^2.
    assert core.gm("earth") == 3.986004418e14


def test_end_to_end_double_run_is_bit_identical(tmp_path):
    _core_or_fail()
    from star_reacher.runner import run_mission

    mission = REPO_ROOT / "missions" / "twobody_leo.toml"
    r1 = run_mission(mission, tmp_path / "run1")
    r2 = run_mission(mission, tmp_path / "run2")
    assert r1.srlog_sha256 == r2.srlog_sha256
    assert r1.config_sha256 == r2.config_sha256


def test_end_to_end_outputs_and_load(tmp_path):
    _core_or_fail()
    from star_reacher import load
    from star_reacher.runner import run_mission

    mission = REPO_ROOT / "missions" / "twobody_leo.toml"
    result = run_mission(mission, tmp_path / "run")
    run = load(result.srlog_path)

    # Header binds the log to its exact inputs (FR-15/D-11).
    assert run.header["config_sha256"] == result.config_sha256
    resolved_bytes = (tmp_path / "run" / "resolved_config.json").read_bytes()
    assert hashlib.sha256(resolved_bytes).hexdigest() == result.config_sha256
    assert run.header["master_seed"] == "20260101"
    assert run.header["oracle"] is False
    assert run.header["epoch_utc"] == "2026-01-01T00:00:00Z"

    truth = run.groups["truth"]
    # 5400 s at 10 Hz plus the record at t = 0.
    assert len(truth) == 54001
    assert truth["r_m"].shape == (54001, 3)
    assert np.all(np.diff(truth["t_s"]) > 0)
    # Two-body placeholder attitude schema: identity quaternion, zero rates,
    # constant mass (contract section 2 semantics).
    assert np.all(truth["q_i2b"] == np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.all(truth["w_b_radps"] == 0.0)
    assert np.all(truth["mass_kg"] == 150.0)

    events = run.events
    codes = [int(c) for c in events["code"]]
    assert 1 in codes and 2 in codes

    # meta.json is the only home of wall-clock and host data (D-11).
    meta = json.loads((tmp_path / "run" / "meta.json").read_text(encoding="utf-8"))
    assert meta["srlog_sha256"] == result.srlog_sha256
    assert meta["config_sha256"] == result.config_sha256
    log_text = result.srlog_path.read_bytes()
    assert meta["host"]["node"].encode("utf-8") not in log_text or meta["host"]["node"] == ""
