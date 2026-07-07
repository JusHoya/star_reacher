"""SRLOG v1.2 loader coverage (docs/formats/srlog_v1.md section 3.2).

Two layers, per the format-conformance conventions:

- synthetic fixtures (star_reacher._fixtures) exercise the reader's
  dict-driven handling of the new groups without the compiled core - the
  reader has no v1.2-specific code by design, so these prove the general
  f64[N]/aperiodic machinery carries the new layouts;
- a writer-produced log (a short run of the committed GNC reference
  mission) pins the real header dictionary and record content end to end
  through the same Run public surface every existing group uses.
"""

from pathlib import Path

import numpy as np
import pytest

from star_reacher import _fixtures
from star_reacher.srlog import _parse_bytes

REPO_ROOT = Path(__file__).resolve().parents[2]
MISSION = REPO_ROOT / "missions" / "leo_attitude_gnc.toml"

_CORE_MISSING_MESSAGE = (
    "star_reacher._core is not built in this environment. The "
    "writer-produced-log tests require the compiled core: build and install "
    "it with 'pip install .' from the repository root."
)

# The v1.2 group dictionaries as the format doc specifies them, used as
# extra_groups on the synthetic fixture header (the fixture builder packs
# whatever it is told; no writer code is involved).
_IMU_GROUP = {
    "name": "sensors.imu",
    "rate_hz": 100,
    "channels": [
        {"name": "t_s", "dtype": "f64", "units": "s", "frame": ""},
        {"name": "dtheta_b_rad", "dtype": "f64[3]", "units": "rad", "frame": "body"},
        {"name": "dv_b_mps", "dtype": "f64[3]", "units": "m/s", "frame": "body"},
    ],
}
# nav.est with the error-state EKF layout reservation: state dimension 16
# with an independently declared covariance dimension 15, so P is f64[120]
# (the format doc's m(m+1)/2 packing) - proving the reader carries
# header-declared dimensions where n and m differ.
_NAV_EST_GROUP = {
    "name": "nav.est",
    "rate_hz": 100,
    "channels": [
        {"name": "t_s", "dtype": "f64", "units": "s", "frame": ""},
        {"name": "x_hat", "dtype": "f64[16]", "units": "", "frame": ""},
        {"name": "P", "dtype": "f64[120]", "units": "", "frame": ""},
    ],
}
_NAV_INNOV_GROUP = {
    "name": "nav.innov",
    "rate_hz": 0,
    "channels": [
        {"name": "t_s", "dtype": "f64", "units": "s", "frame": ""},
        {"name": "sensor_id", "dtype": "u32", "units": "1", "frame": ""},
        {"name": "m", "dtype": "u32", "units": "1", "frame": ""},
        {"name": "y", "dtype": "f64[3]", "units": "", "frame": ""},
        {"name": "S", "dtype": "f64[6]", "units": "", "frame": ""},
    ],
}


def test_synthetic_v12_groups_load_dict_driven():
    header = _fixtures.contract_header(
        minor=2,
        extra_groups=[_IMU_GROUP, _NAV_EST_GROUP, _NAV_INNOV_GROUP],
    )
    header["gnc"] = {"cycle_rate_hz": 100, "latency_cycles": 2, "sensors": ["imu"]}
    imu_i = _fixtures.group_index(header, "sensors.imu")
    est_i = _fixtures.group_index(header, "nav.est")
    innov_i = _fixtures.group_index(header, "nav.innov")
    x_hat = tuple(float(i) for i in range(16))
    p = tuple(0.5 * i for i in range(120))
    records = [
        _fixtures.event_record(0.0, 1, "run_start"),
        (imu_i, (0.01, (1e-4, -2e-4, 3e-4), (0.05, 0.0, -0.01))),
        (est_i, (0.01, x_hat, p)),
        # Two same-cycle aiding updates from different sensors: nav.innov's
        # documented exception to strictly increasing t_s.
        (innov_i, (0.01, 0, 2, (0.25, -0.5, 0.0), (2.0, 0.1, 0.0, 3.0, 0.0, 4.0))),
        (innov_i, (0.01, 1, 1, (0.125, 0.0, 0.0), (9.0, 0.0, 0.0, 0.0, 0.0, 0.0))),
        _fixtures.event_record(1.0, 2, "run_end"),
    ]
    run = _parse_bytes(_fixtures.build_srlog(header, records), "synthetic-v12")

    # The unknown top-level "gnc" key is carried, not rejected (additive
    # minor-version evolution), and the version words surface.
    assert run.header["format"]["minor"] == 2
    assert run.header["gnc"]["sensors"] == ["imu"]

    imu = run.groups["sensors.imu"]
    assert imu["dtheta_b_rad"].shape == (1, 3)
    assert imu["dtheta_b_rad"][0, 2] == 3e-4
    est = run.groups["nav.est"]
    assert est["x_hat"].shape == (1, 16)
    assert est["P"].shape == (1, 120)
    assert est["P"][0, 119] == 59.5
    innov = run.groups["nav.innov"]
    assert innov.shape == (2,)
    assert list(innov["sensor_id"]) == [0, 1]
    assert list(innov["m"]) == [2, 1]
    # Zero-padding beyond m is data, not structure: the reader returns it
    # verbatim and consumers mask on m.
    assert innov["y"][0, 2] == 0.0
    assert innov["S"][1, 0] == 9.0
    assert innov["t_s"][0] == innov["t_s"][1] == 0.01


def _core_or_fail():
    try:
        from star_reacher import _core  # noqa: F401
    except ImportError:
        pytest.fail(_CORE_MISSING_MESSAGE)


def test_writer_produced_v12_log_roundtrips_through_loader(tmp_path):
    """Writer-produced bytes: a short run of the reference GNC mission must
    expose every declared v1.2 group through the same Run surface the
    existing groups use, with exactly the format doc's channel names."""
    _core_or_fail()
    import os

    from star_reacher import load
    from star_reacher.runner import run_mission

    text = MISSION.read_text(encoding="utf-8").replace(
        "duration_s = 60.0", "duration_s = 2.0"
    )
    mission = tmp_path / "short_gnc.toml"
    mission.write_text(text, encoding="utf-8")
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        result = run_mission(mission, tmp_path / "run")
    finally:
        os.chdir(cwd)
    run = load(result.srlog_path)

    # Channel names are the loader's structured-array field names - the
    # normative section 3.2 layouts, straight from the header dictionary.
    assert run.groups["sensors.imu"].dtype.names == (
        "t_s", "dtheta_b_rad", "dv_b_mps",
    )
    assert run.groups["nav.est"].dtype.names == ("t_s", "x_hat", "P")
    assert run.groups["nav.err"].dtype.names == ("t_s", "e")
    assert run.groups["gnc.cmd"].dtype.names == (
        "t_s", "tau_b_nm", "q_cmd_i2b", "w_cmd_b_radps", "valid",
    )
    # 2 s at 10 Hz: cycles 0..20; sensors start at their first sample
    # instant, nav/cmd at activation (format doc record-start semantics).
    assert len(run.groups["sensors.imu"]) == 20
    assert len(run.groups["nav.est"]) == 21
    assert len(run.groups["nav.err"]) == 21
    assert len(run.groups["gnc.cmd"]) == 21
    # nav.err.e shares nav.est's dimension n = 7 by construction.
    assert run.groups["nav.err"]["e"].shape == (21, 7)
    # Run.time_s works on the new groups exactly like the old ones.
    t = run.time_s("gnc.cmd")
    assert t[0] == 0.0 and len(t) == 21
