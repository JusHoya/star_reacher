"""SRLOG v1 reader tests against synthesized in-memory fixtures (contract
section 2; Phase 1 exit criterion 3).

Fixture bytes are built by ``star_reacher._fixtures`` (never committed as
binaries) with independent struct-based packing, so a reader bug cannot be
masked by a mirrored writer bug. No compiled core is required.
"""

import struct

import numpy as np
import pytest

from star_reacher import _fixtures
from star_reacher.srlog import (
    MAGIC,
    SrlogCorruptError,
    SrlogVersionError,
    load,
)


def _write(tmp_path, data, name="test.srlog"):
    path = tmp_path / name
    path.write_bytes(data)
    return path


def _standard_records():
    return [
        _fixtures.event_record(0.0, 1, "run_start"),
        (0, (0.0, (6778137.0, 0.0, 0.0), (0.0, 7668.6, 0.0), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 150.0)),
        (0, (0.1, (6778136.0, 766.9, 0.0), (-0.9, 7668.6, 0.0), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 150.0)),
        _fixtures.event_record(600.0, 2, "run_end"),
    ]


def test_valid_file_loads_header_groups_and_events(tmp_path):
    data = _fixtures.build_srlog(_fixtures.contract_header(), _standard_records())
    run = load(_write(tmp_path, data))
    assert run.header["format"] == {"name": "SRLOG", "major": 1, "minor": 0}
    assert run.header["central_body"] == "earth"
    truth = run.groups["truth"]
    assert len(truth) == 2
    assert truth["t_s"].tolist() == [0.0, 0.1]
    # Vector channels surface as fixed-size subarrays (D-12).
    assert truth["r_m"].shape == (2, 3)
    assert truth["q_i2b"].shape == (2, 4)
    assert truth["r_m"][0].tolist() == [6778137.0, 0.0, 0.0]
    assert truth["v_mps"][1].tolist() == [-0.9, 7668.6, 0.0]
    assert truth["mass_kg"].dtype == np.float64
    events = run.events
    assert len(events) == 2
    assert events["code"].tolist() == [1, 2]
    assert list(events["detail"]) == ["run_start", "run_end"]
    assert events is run.groups["events"]


def test_load_accepts_str_and_path(tmp_path):
    data = _fixtures.build_srlog(_fixtures.contract_header(), _standard_records())
    path = _write(tmp_path, data)
    assert load(str(path)).groups["truth"].shape == load(path).groups["truth"].shape


def test_empty_record_stream_gives_zero_length_arrays(tmp_path):
    data = _fixtures.build_srlog(_fixtures.contract_header(), [])
    run = load(_write(tmp_path, data))
    assert len(run.groups["truth"]) == 0
    assert len(run.events) == 0


def test_minor_version_ahead_loads_and_exposes_added_channel(tmp_path):
    header = _fixtures.contract_header(
        minor=999,
        extra_truth_channels=[{"name": "flux_w", "dtype": "f64", "units": "W", "frame": ""}],
    )
    header["future_top_level_key"] = [1, 2, 3]  # readers must ignore unknown header keys
    records = [
        (0, (0.0, (1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.5, 99.5)),
    ]
    run = load(_write(tmp_path, _fixtures.build_srlog(header, records)))
    assert run.header["format"]["minor"] == 999
    truth = run.groups["truth"]
    assert "flux_w" in truth.dtype.names
    assert truth["flux_w"][0] == 99.5
    assert truth["r_m"][0].tolist() == [1.0, 2.0, 3.0]


def test_major_version_mismatch_raises_naming_both_versions(tmp_path):
    data = _fixtures.build_srlog(_fixtures.contract_header(major=2), [])
    with pytest.raises(SrlogVersionError) as exc_info:
        load(_write(tmp_path, data))
    message = str(exc_info.value)
    assert "2" in message and "1" in message


def test_bad_magic_raises_corrupt(tmp_path):
    data = bytearray(_fixtures.build_srlog(_fixtures.contract_header(), []))
    data[0] ^= 0xFF
    with pytest.raises(SrlogCorruptError, match="magic"):
        load(_write(tmp_path, bytes(data)))


def test_magic_with_crlf_mangling_raises_corrupt(tmp_path):
    # The CR/LF bytes in the magic exist to catch text-mode transfers that
    # rewrite line endings; simulate one.
    data = _fixtures.build_srlog(_fixtures.contract_header(), [])
    mangled = data.replace(b"\r\n", b"\n", 1)
    with pytest.raises(SrlogCorruptError):
        load(_write(tmp_path, mangled))


def test_header_json_len_beyond_eof_raises_corrupt(tmp_path):
    data = _fixtures.build_srlog(_fixtures.contract_header(), [])
    truncated = data[:24]  # cuts inside the header JSON
    with pytest.raises(SrlogCorruptError, match="truncated"):
        load(_write(tmp_path, truncated))


def test_undecodable_header_json_raises_corrupt(tmp_path):
    header_json = b'{"broken": '
    data = MAGIC + struct.pack("<HHI", 1, 0, len(header_json)) + header_json
    with pytest.raises(SrlogCorruptError, match="not decodable"):
        load(_write(tmp_path, data))


def test_header_without_groups_raises_corrupt(tmp_path):
    header_json = b'{"format":{"name":"SRLOG","major":1,"minor":0}}'
    data = MAGIC + struct.pack("<HHI", 1, 0, len(header_json)) + header_json
    with pytest.raises(SrlogCorruptError, match="groups"):
        load(_write(tmp_path, data))


def test_unknown_dtype_raises_corrupt_naming_dtype(tmp_path):
    import json

    header = _fixtures.contract_header()
    header["groups"][0]["channels"][0]["dtype"] = "i128"
    # Serialize by hand with no records: the reader must reject the channel
    # dictionary itself, before any payload parsing.
    header_json = json.dumps(header, separators=(",", ":")).encode()
    data = MAGIC + struct.pack("<HHI", 1, 0, len(header_json)) + header_json
    with pytest.raises(SrlogCorruptError, match="i128"):
        load(_write(tmp_path, data))


def test_trailing_partial_record_raises_corrupt(tmp_path):
    data = _fixtures.build_srlog(_fixtures.contract_header(), _standard_records())
    with pytest.raises(SrlogCorruptError, match="partial"):
        load(_write(tmp_path, data[:-3]))


def test_dangling_group_index_byte_raises_corrupt(tmp_path):
    data = _fixtures.build_srlog(_fixtures.contract_header(), _standard_records())
    with pytest.raises(SrlogCorruptError, match="partial"):
        load(_write(tmp_path, data + b"\x00"))


def test_truncated_str16_payload_raises_corrupt(tmp_path):
    header = _fixtures.contract_header()
    data = _fixtures.build_srlog(header, [_fixtures.event_record(0.0, 1, "run_start")])
    with pytest.raises(SrlogCorruptError, match="str16"):
        load(_write(tmp_path, data[:-4]))


def test_out_of_range_group_index_raises_corrupt(tmp_path):
    data = _fixtures.build_srlog(_fixtures.contract_header(), []) + struct.pack("<H", 7)
    with pytest.raises(SrlogCorruptError, match="group index 7"):
        load(_write(tmp_path, data))


def test_loaded_arrays_are_writable_copies(tmp_path):
    # Downstream analysis mutates arrays freely; frombuffer views over the
    # file bytes would be read-only.
    data = _fixtures.build_srlog(_fixtures.contract_header(), _standard_records())
    run = load(_write(tmp_path, data))
    run.groups["truth"]["mass_kg"][0] = 0.0
    assert run.groups["truth"]["mass_kg"][0] == 0.0
