"""Rotation-kernel and reference-frame binding tests (FR-3, D-7).

These drive the same conversions the doctest suite covers, but through the
pybind11 surface the Python frontend uses, against the identical golden
vectors (provenance and tolerances in tests/golden/rotations/manifest.toml
and tests/golden/frames/manifest.toml). Core-requiring tests fail cleanly,
never skip, when the compiled core is absent (see test_integration_core.py
for the rationale).
"""

import math
import tomllib
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_FRAMES = REPO_ROOT / "tests" / "golden" / "frames"
GOLDEN_ROT = REPO_ROOT / "tests" / "golden" / "rotations"

DAS2R = 4.848136811095359935899141e-6
DEG2RAD = math.radians(1.0)

_CORE_MISSING_MESSAGE = (
    "star_reacher._core is not built in this environment. These frame tests "
    "require the compiled core: build and install it with 'pip install .' "
    "from the repository root. This failure is expected on a core-less "
    "checkout and must be green at integration/CI."
)


def _core_or_fail():
    try:
        from star_reacher import _core
    except ImportError:
        pytest.fail(_CORE_MISSING_MESSAGE)
    return _core


def _load_cases(path: Path) -> list[dict]:
    with open(path, "rb") as fh:
        return tomllib.load(fh)["case"]


def _golden_matrix(c: dict, prefix: str = "c") -> np.ndarray:
    return np.array(
        [
            [float.fromhex(c[f"{prefix}{i}{j}"]) for j in range(3)]
            for i in range(3)
        ]
    )


def _epoch(c: dict) -> tuple[int, float]:
    return int(c["tai_day"]), float.fromhex(c["tai_sec"])


def _rotation_angle_between(core, a, b) -> float:
    # Angle of the relative rotation conj(a) (x) b - the criterion-6a
    # "rad equivalent" metric, insensitive to the q ~ -q sign.
    e = core.quat_multiply(*core.quat_conjugate(*a), *b)
    return 2.0 * math.atan2(math.hypot(e[1], e[2], e[3]), abs(e[0]))


def test_frames_bindings_match_goldens():
    core = _core_or_fail()
    cases = _load_cases(GOLDEN_FRAMES / "earth_chain.toml")
    assert len(cases) == 14
    for c in cases:
        day, sec = core.utc_to_tai(
            int(c["year"]),
            int(c["month"]),
            int(c["day"]),
            int(c["hour"]),
            int(c["minute"]),
            float.fromhex(c["second"]),
        )
        # Epoch plumbing bit-identical to the committed two-part TAI.
        assert (day, sec) == _epoch(c), c["name"]
        dut1 = float.fromhex(c["dut1_s"])

        # Chain components at the manifest tolerance (1e-12 rad).
        dpsi, deps = core.nutation_00b(core.tt_julian_centuries(day, sec))
        x, y, s = core.cip_cio_06b(day, sec)
        era = core.era_00(day, sec, dut1)
        assert abs(dpsi - float.fromhex(c["dpsi"])) <= 1e-12, c["name"]
        assert abs(deps - float.fromhex(c["deps"])) <= 1e-12, c["name"]
        assert abs(x - float.fromhex(c["x"])) <= 1e-12, c["name"]
        assert abs(y - float.fromhex(c["y"])) <= 1e-12, c["name"]
        assert abs(s - float.fromhex(c["s"])) <= 1e-12, c["name"]
        assert abs(era - float.fromhex(c["era"])) <= 1e-12, c["name"]

        # Phase 2 exit criterion 1: matrix elements <= 1e-11 vs ERFA.
        m = np.array(core.gcrf_to_itrf(day, sec, dut1)).reshape(3, 3)
        assert np.max(np.abs(m - _golden_matrix(c))) <= 1e-11, c["name"]

        # Orthonormality at machine precision (a few ulp).
        assert np.max(np.abs(m.T @ m - np.eye(3))) <= 2e-15, c["name"]
        assert abs(np.linalg.det(m) - 1.0) <= 2e-15, c["name"]


def test_frames_cookbook_bindings():
    core = _core_or_fail()
    (c,) = _load_cases(GOLDEN_FRAMES / "cookbook_2006_2000a.toml")
    day, sec = _epoch(c)
    dut1 = float.fromhex(c["dut1_s"])
    pub = np.array(
        [[float(c[f"pub_c{i}{j}"]) for j in range(3)] for i in range(3)]
    )
    m = np.array(core.gcrf_to_itrf(day, sec, dut1)).reshape(3, 3)
    # Documented 2000B-vs-published-2000A bound, not the 1e-11 ERFA gate.
    assert np.max(np.abs(m - pub)) <= float(c["tol_matrix"])
    x, y, s = core.cip_cio_06b(day, sec)
    assert abs(x - float(c["pub_x"])) <= 1e-8
    assert abs(y - float(c["pub_y"])) <= 1e-8
    assert abs(s / DAS2R - float(c["pub_s_arcsec"])) <= 1e-6
    era = core.era_00(day, sec, dut1)
    assert abs(era - float(c["pub_era_deg"]) * DEG2RAD) <= 5e-14


def test_frames_eci_ecef_roundtrip_bindings():
    # Phase 2 exit criterion 6b through the binding surface: ECI -> ECEF ->
    # ECI position round trip at LEO radius errs <= 1e-8 m over multiple
    # epochs and directions.
    core = _core_or_fail()
    cases = _load_cases(GOLDEN_FRAMES / "earth_chain.toml")
    r_leo = 6778137.0
    dirs = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [-2.0, 0.5, 1.5],
            [0.3, -0.9, -0.6],
        ]
    )
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    worst = 0.0
    for c in cases:
        day, sec = _epoch(c)
        m = np.array(
            core.gcrf_to_itrf(day, sec, float.fromhex(c["dut1_s"]))
        ).reshape(3, 3)
        for d in dirs:
            r_eci = r_leo * d
            back = m.T @ (m @ r_eci)
            worst = max(worst, float(np.linalg.norm(back - r_eci)))
    assert worst <= 1e-8


def test_frames_moon_mars_bindings():
    core = _core_or_fail()
    for c in _load_cases(GOLDEN_FRAMES / "moon_pa.toml"):
        m = np.array(
            core.gcrf_to_moonpa(
                float.fromhex(c["phi"]),
                float.fromhex(c["theta"]),
                float.fromhex(c["psi"]),
            )
        ).reshape(3, 3)
        assert np.max(np.abs(m - _golden_matrix(c))) <= 1e-15, c["name"]

    for c in _load_cases(GOLDEN_FRAMES / "mars_iau.toml"):
        day, sec = _epoch(c)
        ra, dec, w = core.mars_elements(day, sec)
        assert abs(ra - float.fromhex(c["alpha0"])) <= 1e-13, c["name"]
        assert abs(dec - float.fromhex(c["delta0"])) <= 1e-13, c["name"]
        assert abs(w - float.fromhex(c["w"])) <= 1e-13, c["name"]
        m = np.array(core.gcrf_to_marsfixed(day, sec)).reshape(3, 3)
        assert np.max(np.abs(m - _golden_matrix(c))) <= 1e-14, c["name"]


def test_rotation_bindings_match_goldens():
    core = _core_or_fail()
    for c in _load_cases(GOLDEN_ROT / "quat_dcm.toml"):
        q = tuple(float.fromhex(c[f"q{k}"]) for k in "wxyz")
        golden = _golden_matrix(c)
        m = np.array(core.quat_to_dcm(*q)).reshape(3, 3)
        assert np.max(np.abs(m - golden)) <= 1e-15, c["name"]
        back = core.dcm_to_quat(list(golden.reshape(9)))
        assert back[0] >= 0.0, c["name"]
        sign = -1.0 if sum(a * b for a, b in zip(back, q)) < 0.0 else 1.0
        assert max(abs(b - sign * a) for b, a in zip(back, q)) <= 1e-15, c["name"]

    for c in _load_cases(GOLDEN_ROT / "euler.toml"):
        a1 = float.fromhex(c["a1"])
        a2 = float.fromhex(c["a2"])
        a3 = float.fromhex(c["a3"])
        build = (
            core.dcm_from_euler321
            if c["sequence"] == "321"
            else core.dcm_from_euler313
        )
        m = np.array(build(a1, a2, a3)).reshape(3, 3)
        assert np.max(np.abs(m - _golden_matrix(c))) <= 1e-15, c["name"]


def test_rotation_bindings_roundtrip():
    # Criterion 6a exercised through the binding surface, on the identical
    # seeded stream the doctest uses (master seed 20260702, stream
    # "tests.rotation"), so the attitudes are the same doubles in both
    # suites. 250 draws keep the pytest tier fast; the full 1,000 live in
    # the doctest gate.
    core = _core_or_fail()
    n = 250
    normals = core.rng_stream_normal(20260702, "tests.rotation", 4 * n)
    worst = 0.0
    for i in range(n):
        q = core.quat_normalize(*normals[4 * i : 4 * i + 4])
        c = core.quat_to_dcm(*q)
        # DCM round trip.
        worst = max(
            worst, _rotation_angle_between(core, q, core.dcm_to_quat(c))
        )
        # Euler round trips, both sequences.
        for extract, build in (
            (core.euler321_from_dcm, core.dcm_from_euler321),
            (core.euler313_from_dcm, core.dcm_from_euler313),
        ):
            angles = extract(c)
            q2 = core.dcm_to_quat(build(*angles))
            worst = max(worst, _rotation_angle_between(core, q, q2))
    assert worst <= 1e-13

    # A near-gimbal-lock draw stays inside the criterion bound.
    q_lock = core.dcm_to_quat(
        core.dcm_from_euler321(0.7, math.pi / 2 - 1e-2, -0.4)
    )
    c_lock = core.quat_to_dcm(*q_lock)
    q_back = core.dcm_to_quat(core.dcm_from_euler321(*core.euler321_from_dcm(c_lock)))
    assert _rotation_angle_between(core, q_lock, q_back) <= 1e-13

    # ValueError (not a silent default) for a degenerate quaternion.
    with pytest.raises(ValueError):
        core.quat_normalize(0.0, 0.0, 0.0, 0.0)
