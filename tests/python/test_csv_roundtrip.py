"""CSV exporter bit-exactness tests (FR-17, Phase 1 exit criterion 4).

Bit-exact means the IEEE-754 bytes of the re-parsed value equal the original
bytes (compared via struct packing), not merely numerical closeness.
No compiled core is required.
"""

import csv
import math
import struct

import numpy as np

from star_reacher import _fixtures
from star_reacher.export import export_csv
from star_reacher.srlog import load

# Shortest-repr corner cases: subnormal minimum, negative zero,
# non-terminating binary fractions, 17-significant-digit values.
TRICKY = [
    0.1,
    1.0 / 3.0,
    -0.0,
    5e-324,
    math.pi,
    6.02214076e23,
    -7668.6,
    1.0000000000000002,
    2.0**-1022,
]


def _build_srlog(tmp_path):
    n = len(TRICKY)
    records = []
    for i in range(n):
        records.append(
            (
                0,
                (
                    TRICKY[i],
                    (TRICKY[(i + 1) % n], TRICKY[(i + 2) % n], TRICKY[(i + 3) % n]),
                    (TRICKY[(i + 4) % n], TRICKY[(i + 5) % n], TRICKY[(i + 6) % n]),
                    (1.0, -0.0, 0.0, 0.0),
                    (TRICKY[(i + 7) % n], TRICKY[(i + 8) % n], TRICKY[i]),
                    TRICKY[(i + 2) % n],
                ),
            )
        )
    records.append(_fixtures.event_record(0.0, 1, "run_start"))
    records.append(_fixtures.event_record(0.5, 7, 'comma, "quote" and\nnewline'))
    records.append(_fixtures.event_record(600.0, 2, "run_end"))
    path = tmp_path / "run.srlog"
    path.write_bytes(_fixtures.build_srlog(_fixtures.contract_header(), records))
    return path


def _bits(x: float) -> bytes:
    return struct.pack("<d", x)


def test_truth_csv_round_trips_bit_exactly(tmp_path):
    path = _build_srlog(tmp_path)
    export_csv(path, tmp_path)
    run = load(path)
    truth = run.groups["truth"]
    with open(tmp_path / "truth.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header, data_rows = rows[0], rows[1:]
    assert header == [
        "t_s",
        "r_m_0",
        "r_m_1",
        "r_m_2",
        "v_mps_0",
        "v_mps_1",
        "v_mps_2",
        "q_i2b_0",
        "q_i2b_1",
        "q_i2b_2",
        "q_i2b_3",
        "w_b_radps_0",
        "w_b_radps_1",
        "w_b_radps_2",
        "mass_kg",
    ]
    assert len(data_rows) == len(truth)
    for ri, rec in enumerate(truth):
        flat = [float(rec["t_s"])]
        for field in ("r_m", "v_mps", "q_i2b", "w_b_radps"):
            flat.extend(float(x) for x in rec[field])
        flat.append(float(rec["mass_kg"]))
        for ci, original in enumerate(flat):
            reparsed = float(data_rows[ri][ci])
            assert _bits(reparsed) == _bits(original), (
                f"row {ri} col {ci}: {data_rows[ri][ci]!r} reparses to a different bit pattern"
            )


def test_negative_zero_sign_preserved(tmp_path):
    path = _build_srlog(tmp_path)
    export_csv(path, tmp_path)
    with open(tmp_path / "truth.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    q0_col = rows[0].index("q_i2b_1")
    # repr(-0.0) is "-0.0"; a formatter that dropped the sign would still
    # compare equal numerically, so assert on the bit pattern.
    assert _bits(float(rows[1][q0_col])) == _bits(-0.0)


def test_events_csv_round_trips_including_quoting(tmp_path):
    path = _build_srlog(tmp_path)
    export_csv(path, tmp_path)
    run = load(path)
    with open(tmp_path / "events.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["t_s", "code", "detail"]
    events = run.events
    assert len(rows) - 1 == len(events)
    for ri, rec in enumerate(events):
        cells = rows[ri + 1]
        assert _bits(float(cells[0])) == _bits(float(rec["t_s"]))
        assert int(cells[1]) == int(rec["code"])
        assert cells[2] == str(rec["detail"])


def test_export_default_outdir_is_input_parent(tmp_path):
    path = _build_srlog(tmp_path)
    written = export_csv(path)
    assert {p.name for p in written} == {"truth.csv", "events.csv"}
    assert all(p.parent == tmp_path for p in written)


def test_export_writes_one_csv_per_group(tmp_path):
    path = _build_srlog(tmp_path)
    outdir = tmp_path / "exported"
    written = export_csv(path, outdir)
    assert sorted(p.name for p in written) == ["events.csv", "truth.csv"]
    assert all(p.exists() for p in written)


def test_round_trip_via_numpy_loadtxt_matches(tmp_path):
    # The documented consumer path: NumPy reads the CSV back and the float
    # columns match the log arrays bit-for-bit.
    path = _build_srlog(tmp_path)
    export_csv(path, tmp_path)
    run = load(path)
    table = np.loadtxt(tmp_path / "truth.csv", delimiter=",", skiprows=1)
    t_col = table[:, 0]
    assert t_col.tobytes() == run.groups["truth"]["t_s"].tobytes()
