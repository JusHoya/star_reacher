"""In-memory SRLOG byte synthesis for format-conformance checks.

This lives inside the package, not the test tree, because ``star verify``
must synthesize its fixture files when run from an installed wheel on a user
machine (DX-5), and because binary fixtures are never committed to the
repository (the contract normalizes line endings repo-wide; fixture bytes are
always constructed in code). The pytest suite reuses these builders so the
reader is exercised against one shared, contract-shaped byte source.

The packing here is written against the contract's byte layout directly (via
``struct``), independent of the reader's parsing tables, so a bug in the
reader cannot be masked by a mirrored bug in the fixtures.
"""

from __future__ import annotations

import copy
import json
import struct

from star_reacher.srlog import MAGIC

# The contract section 2 reference channel dictionary for the Phase 1
# two-body placeholder: a truth group and an events group.
_TRUTH_CHANNELS = [
    {"name": "t_s", "dtype": "f64", "units": "s", "frame": ""},
    {"name": "r_m", "dtype": "f64[3]", "units": "m", "frame": "GCRF"},
    {"name": "v_mps", "dtype": "f64[3]", "units": "m/s", "frame": "GCRF"},
    {"name": "q_i2b", "dtype": "f64[4]", "units": "1", "frame": "GCRF->body Hamilton scalar-first"},
    {"name": "w_b_radps", "dtype": "f64[3]", "units": "rad/s", "frame": "body"},
    {"name": "mass_kg", "dtype": "f64", "units": "kg", "frame": ""},
]
_EVENTS_CHANNELS = [
    {"name": "t_s", "dtype": "f64", "units": "s", "frame": ""},
    {"name": "code", "dtype": "u32", "units": "1", "frame": ""},
    {"name": "detail", "dtype": "str16", "units": "", "frame": ""},
]


def contract_header(
    *,
    major: int = 1,
    minor: int = 0,
    truth_rate_hz: int = 10,
    master_seed: str = "1",
    config_sha256: str = "0" * 64,
    epoch_utc: str = "2026-01-01T00:00:00Z",
    extra_truth_channels: list[dict] | None = None,
) -> dict:
    """Build the contract section 2 header dict for a synthesized file.

    Every call deep-copies the channel templates so tests that mutate a
    returned header (e.g. to plant an unknown dtype) cannot poison the
    module-level templates for later callers.
    """
    truth_channels = copy.deepcopy(_TRUTH_CHANNELS)
    if extra_truth_channels:
        truth_channels.extend(copy.deepcopy(extra_truth_channels))
    return {
        "format": {"name": "SRLOG", "major": major, "minor": minor},
        "producer": {"core_version": "0.1.0", "git_hash": "unknown"},
        "config_sha256": config_sha256,
        "master_seed": master_seed,
        "oracle": False,
        "epoch_utc": epoch_utc,
        "central_body": "earth",
        "groups": [
            {"name": "truth", "rate_hz": truth_rate_hz, "channels": truth_channels},
            {"name": "events", "rate_hz": 0, "channels": copy.deepcopy(_EVENTS_CHANNELS)},
        ],
    }


def _pack_channel(dtype: str, value) -> bytes:
    if dtype == "f64":
        return struct.pack("<d", value)
    if dtype.startswith("f64[") and dtype.endswith("]"):
        n = int(dtype[4:-1])
        return struct.pack(f"<{n}d", *value)
    if dtype == "u32":
        return struct.pack("<I", value)
    if dtype == "u64":
        return struct.pack("<Q", value)
    if dtype == "str16":
        encoded = value.encode("utf-8")
        return struct.pack("<H", len(encoded)) + encoded
    raise ValueError(f"fixture builder cannot pack dtype {dtype!r}")


def build_srlog(
    header: dict,
    records: list[tuple[int, tuple]],
    *,
    major: int | None = None,
    minor: int | None = None,
) -> bytes:
    """Serialize a header dict and (group_index, values) records to SRLOG bytes.

    The binary version fields default to the header's ``format`` entry so the
    two stay consistent unless a test deliberately desynchronizes them.
    """
    fmt = header.get("format", {})
    bin_major = fmt.get("major", 1) if major is None else major
    bin_minor = fmt.get("minor", 0) if minor is None else minor
    groups = header["groups"]
    payload = bytearray()
    for group_index, values in records:
        channels = groups[group_index]["channels"]
        if len(values) != len(channels):
            raise ValueError(
                f"group {group_index} has {len(channels)} channels, got {len(values)} values"
            )
        payload += struct.pack("<H", group_index)
        for channel, value in zip(channels, values):
            payload += _pack_channel(channel["dtype"], value)
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return (
        MAGIC
        + struct.pack("<HHI", bin_major, bin_minor, len(header_json))
        + header_json
        + bytes(payload)
    )


# SREPH v1 magic (docs/formats/sreph_v1.md section 2): ASCII SREPH, then the
# NUL + CRLF tripwire shared with SRLOG.
SREPH_MAGIC = b"SREPH\x00\r\n"


def build_sreph(
    segments: list[dict],
    *,
    major: int = 1,
    minor: int = 0,
) -> bytes:
    """Serialize segment dicts to SREPH v1 bytes (docs/formats/sreph_v1.md).

    Packed against the format document's byte layout directly, independent
    of both the repack writer (``data_fetch.write_sreph``) and the C++
    loader, so a defect in either cannot be masked by a mirrored defect
    here. Each segment dict carries ``name``, ``target``, ``center``,
    ``kind``, ``init_tdb_s``, ``intlen_s``, and ``records``: a list of
    records, each a list of exactly 3 per-component coefficient lists in
    ascending Chebyshev order.
    """
    header_size = 96
    dir_entry_size = 64
    directory = bytearray()
    blocks = bytearray()
    base = header_size + dir_entry_size * len(segments)
    for seg in segments:
        records = seg["records"]
        n_coeffs = len(records[0][0])
        block = bytearray()
        for record in records:
            if len(record) != 3:
                raise ValueError("SREPH records carry exactly 3 components")
            for component in record:
                block += struct.pack(f"<{n_coeffs}d", *component)
        directory += struct.pack(
            "<16sIIIIddIIQ",
            seg["name"].encode("ascii"),
            seg["target"],
            seg["center"],
            seg["kind"],
            n_coeffs,
            seg["init_tdb_s"],
            seg["intlen_s"],
            len(records),
            0,  # reserved, = 0 per the format
            base + len(blocks),
        )
        blocks += block
    span_start = max(s["init_tdb_s"] for s in segments)
    span_end = min(
        s["init_tdb_s"] + len(s["records"]) * s["intlen_s"] for s in segments
    )
    # The two 32-byte source-kernel digests are in-band provenance that
    # readers never interpret; synthesized files carry zero digests.
    header = struct.pack(
        "<8sHHIdd32s32s",
        SREPH_MAGIC,
        major,
        minor,
        len(segments),
        span_start,
        span_end,
        bytes(32),
        bytes(32),
    )
    return header + bytes(directory) + bytes(blocks)


def truth_record(
    t_s: float,
    r_m: tuple[float, float, float] = (6778137.0, 0.0, 0.0),
    v_mps: tuple[float, float, float] = (0.0, 7668.6, 0.0),
    q_i2b: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    w_b_radps: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mass_kg: float = 1.0,
    *extra,
) -> tuple[int, tuple]:
    """A truth-group record tuple; extras append for added-channel fixtures."""
    return (0, (t_s, r_m, v_mps, q_i2b, w_b_radps, mass_kg, *extra))


def event_record(t_s: float, code: int, detail: str) -> tuple[int, tuple]:
    return (1, (t_s, code, detail))
