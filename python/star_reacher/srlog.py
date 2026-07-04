"""SRLOG v1 binary log reader (D-11, FR-16/FR-17).

Pure NumPy plus stdlib: this module must import and work without the compiled
core so a log stays readable on any machine with only NumPy installed (FR-31).

The byte layout is normative in the Phase 1 interface contract and in
``docs/formats/srlog_v1.md``: little-endian, an 8-byte magic, u16 major/minor
version, u32 header-JSON length, a UTF-8 JSON channel dictionary, then a
record stream of (u16 group index, payload) records. The reader is entirely
dict-driven off the header's channel dictionary, which is what lets files with
a minor version ahead of this reader (additive channels) load without code
changes; a major-version change means the byte layout itself moved, so it is
refused loudly instead of producing garbage.
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ASCII "SRLOG", NUL, CR, LF: the trailing CR/LF detects text-mode transfer
# mangling (a corrupted magic is rejected before any other field is trusted).
MAGIC = b"SRLOG\x00\r\n"
READER_MAJOR = 1

# Fixed-size channel dtypes: dtype string -> (numpy field dtype, struct
# format, payload bytes). str16 is variable-length and handled separately.
_SCALAR_DTYPES = {
    "f64": ("<f8", "<d", 8),
    "u32": ("<u4", "<I", 4),
    "u64": ("<u8", "<Q", 8),
}
_VEC_RE = re.compile(r"f64\[([1-9][0-9]*)\]\Z")

# Standard events dtype, used when a file carries no events group so that
# Run.events is always a structured array with a stable interface.
_EVENTS_EMPTY_DTYPE = np.dtype([("t_s", "<f8"), ("code", "<u4"), ("detail", object)])


class SrlogError(Exception):
    """Base class for SRLOG read failures."""


class SrlogVersionError(SrlogError):
    """Major-version mismatch: the byte layout is not the one this reader implements."""


class SrlogCorruptError(SrlogError):
    """Structurally invalid file: bad magic, truncated data, or an unknown dtype."""


def _flat_columns(arr: np.ndarray) -> list[tuple[str, str, int | None]]:
    """Flattened column plan for a structured group array.

    Returns ``(column_name, field_name, index)`` triples in field order,
    where vector channels expand to indexed columns (``r_m_0, r_m_1,
    r_m_2``) and scalar channels carry ``index = None``. This is the one
    shared definition of the tabular-export column convention, used by
    ``Run.to_pandas()`` and the Parquet exporter so every flat-table view
    of a log agrees with the CSV exporter's layout.
    """
    columns: list[tuple[str, str, int | None]] = []
    for field_name in arr.dtype.names:
        shape = arr.dtype[field_name].shape
        if shape:
            columns.extend((f"{field_name}_{i}", field_name, i) for i in range(shape[0]))
        else:
            columns.append((field_name, field_name, None))
    return columns


@dataclass
class Run:
    """One loaded SRLOG file.

    ``header`` is the raw channel-dictionary JSON; ``groups`` maps group name
    to a NumPy structured array (vector channels appear as fixed-size
    subarrays, e.g. ``run.groups["truth"]["r_m"]`` has shape ``(n, 3)``);
    ``events`` is the events group with str16 details decoded to Python
    strings (an empty array when the file has no events group).

    Derived quantities are computed lazily, never logged (FR-16):
    ``elements()`` reduces a group's ``r_m``/``v_mps`` channels to osculating
    orbital elements about the header's central body, ``time_s()`` returns a
    group's time axis, and ``to_pandas()`` gives per-group DataFrames when
    pandas is installed (D-12 optional extra).
    """

    header: dict
    groups: dict[str, np.ndarray]
    events: np.ndarray

    def _group(self, group: str) -> np.ndarray:
        try:
            return self.groups[group]
        except KeyError:
            available = ", ".join(sorted(self.groups)) or "none"
            raise KeyError(
                f"this log has no channel group named {group!r}; "
                f"available groups: {available}"
            ) from None

    def time_s(self, group: str = "truth") -> np.ndarray:
        """The ``t_s`` time axis [s] of a channel group.

        Every group the format defines carries a leading ``t_s`` channel
        (docs/formats/srlog_v1.md section 3); a synthetic or third-party
        group without one raises ``ValueError`` naming the group.
        """
        arr = self._group(group)
        if "t_s" not in arr.dtype.names:
            raise ValueError(
                f"group {group!r} carries no 't_s' channel; its channels are "
                f"{list(arr.dtype.names)}"
            )
        return arr["t_s"]

    def elements(self, group: str = "truth") -> dict[str, np.ndarray]:
        """Osculating elements derived from a group's ``r_m``/``v_mps``.

        Computed lazily on first call and cached per group (FR-16:
        "osculating elements are derived in the loader, not logged"). The
        central body's GM comes from the log header's ``central_body`` via
        ``star_reacher.derived.central_body_gm``. Element definitions,
        angle ranges, and singular-geometry conventions are documented in
        ``star_reacher.derived.osculating_elements`` and
        ``docs/formats/derived_elements.md``.
        """
        cache = getattr(self, "_elements_cache", None)
        if cache is None:
            cache = {}
            self._elements_cache = cache
        if group not in cache:
            # Local import keeps plain log reading free of the derived-math
            # module until elements are actually requested.
            from star_reacher import derived

            arr = self._group(group)
            for channel in ("r_m", "v_mps"):
                if channel not in arr.dtype.names:
                    raise ValueError(
                        f"group {group!r} carries no {channel!r} channel, so "
                        f"osculating elements cannot be derived from it; its "
                        f"channels are {list(arr.dtype.names)}"
                    )
            gm = derived.central_body_gm(self.header.get("central_body"))
            cache[group] = derived.osculating_elements(arr["r_m"], arr["v_mps"], gm)
        return cache[group]

    def to_pandas(self) -> dict:
        """Per-group pandas DataFrames with CSV-convention flat columns.

        Vector channels expand to indexed columns (``r_m_0, r_m_1, r_m_2``)
        exactly like the CSV exporter, so column names agree across every
        tabular view of a log. pandas is a documented optional extra
        (D-12): when it is not installed this raises an actionable
        ``ImportError`` instead of adding a hard dependency.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "Run.to_pandas() requires pandas, which star_reacher treats "
                "as an optional dependency (D-12); install it with "
                'pip install "star-reacher[pandas]" (or pip install pandas)'
            ) from exc
        frames = {}
        for group_name, arr in self.groups.items():
            data = {}
            for column_name, field_name, index in _flat_columns(arr):
                if index is None:
                    data[column_name] = arr[field_name]
                else:
                    data[column_name] = arr[field_name][:, index]
            frames[group_name] = pd.DataFrame(data)
        return frames


@dataclass
class _Channel:
    name: str
    dtype: str
    size: int  # fixed payload bytes; -1 marks variable-length (str16)
    fmt: str | None  # struct format for fixed channels
    vec_len: int  # 0 for scalars, element count for f64[N]


@dataclass
class _Group:
    name: str
    channels: list[_Channel]
    np_dtype: np.dtype
    fixed_size: int | None  # None when any channel is variable-length


def _compile_channel(source: str, gname: str, meta: object) -> tuple[_Channel, tuple]:
    if (
        not isinstance(meta, dict)
        or not isinstance(meta.get("name"), str)
        or not meta.get("name")
        or not isinstance(meta.get("dtype"), str)
    ):
        raise SrlogCorruptError(
            f"{source}: malformed channel entry in group '{gname}': every channel "
            f"needs a non-empty string 'name' and a string 'dtype'"
        )
    name = meta["name"]
    dt = meta["dtype"]
    if dt in _SCALAR_DTYPES:
        np_dt, fmt, size = _SCALAR_DTYPES[dt]
        return _Channel(name, dt, size, fmt, 0), (name, np_dt)
    vec = _VEC_RE.fullmatch(dt)
    if vec:
        n = int(vec.group(1))
        return _Channel(name, dt, 8 * n, f"<{n}d", n), (name, "<f8", (n,))
    if dt == "str16":
        # Object dtype keeps decoded strings lossless regardless of length;
        # fixed-width unicode would silently truncate future long details.
        return _Channel(name, dt, -1, None, 0), (name, object)
    raise SrlogCorruptError(
        f"{source}: unknown channel dtype '{dt}' for channel '{gname}.{name}'; "
        f"known dtypes are f64, f64[N], u32, u64, str16"
    )


def _compile_group(source: str, meta: object, index: int) -> _Group:
    if not isinstance(meta, dict) or not isinstance(meta.get("name"), str) or not meta.get("name"):
        raise SrlogCorruptError(
            f"{source}: group entry {index} in the header is malformed: a string 'name' is required"
        )
    gname = meta["name"]
    chan_meta = meta.get("channels")
    if not isinstance(chan_meta, list) or not chan_meta:
        raise SrlogCorruptError(
            f"{source}: group '{gname}' declares no channels; a self-describing "
            f"channel dictionary cannot be empty"
        )
    channels: list[_Channel] = []
    fields: list[tuple] = []
    for cm in chan_meta:
        ch, field = _compile_channel(source, gname, cm)
        channels.append(ch)
        fields.append(field)
    names = [c.name for c in channels]
    if len(set(names)) != len(names):
        raise SrlogCorruptError(f"{source}: group '{gname}' declares duplicate channel names")
    fixed: int | None = 0
    for ch in channels:
        if ch.size < 0:
            fixed = None
            break
        fixed += ch.size
    return _Group(gname, channels, np.dtype(fields), fixed)


def _parse_bytes(data: bytes, source: str) -> Run:
    if data[:8] != MAGIC:
        raise SrlogCorruptError(
            f"{source}: bad magic: expected {MAGIC!r} at byte 0; this is not an "
            f"SRLOG file, or it was mangled by a text-mode transfer"
        )
    if len(data) < 16:
        raise SrlogCorruptError(
            f"{source}: truncated fixed header: file is {len(data)} bytes, the fixed header is 16"
        )
    major, minor, jlen = struct.unpack_from("<HHI", data, 8)
    if major != READER_MAJOR:
        # Minor versions ahead of the reader are additive and load fine
        # (dict-driven layout); a major bump means the layout itself changed,
        # so reading on would produce garbage rather than data (D-11).
        raise SrlogVersionError(
            f"{source}: SRLOG major version {major} (file is v{major}.{minor}) is not "
            f"readable by this reader, which implements major version {READER_MAJOR}; "
            f"a major-version change breaks the byte layout and is refused"
        )
    if 16 + jlen > len(data):
        raise SrlogCorruptError(
            f"{source}: truncated header JSON: header_json_len={jlen} runs past "
            f"the end of the file ({len(data)} bytes)"
        )
    try:
        header = json.loads(data[16 : 16 + jlen].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SrlogCorruptError(f"{source}: header JSON is not decodable: {exc}") from exc
    # Unknown header keys are ignored by construction (only the keys the
    # reader needs are accessed): that is the additive minor-version path.
    if not isinstance(header, dict) or not isinstance(header.get("groups"), list):
        raise SrlogCorruptError(
            f"{source}: header JSON lacks the 'groups' channel dictionary; the "
            f"file is not self-describing and cannot be parsed"
        )
    groups = [_compile_group(source, g, i) for i, g in enumerate(header["groups"])]
    gnames = [g.name for g in groups]
    if len(set(gnames)) != len(gnames):
        raise SrlogCorruptError(f"{source}: header declares duplicate group names")

    buffers: list[bytearray] = [bytearray() for _ in groups]
    rows: list[list[tuple]] = [[] for _ in groups]
    pos = 16 + jlen
    end = len(data)
    while pos < end:
        if pos + 2 > end:
            raise SrlogCorruptError(
                f"{source}: trailing partial record at byte {pos}: 1 byte remains "
                f"where a 2-byte group index is required (append-only truncation is corruption)"
            )
        (gi,) = struct.unpack_from("<H", data, pos)
        pos += 2
        if gi >= len(groups):
            raise SrlogCorruptError(
                f"{source}: record at byte {pos - 2} names group index {gi}, but the "
                f"header declares only {len(groups)} group(s)"
            )
        grp = groups[gi]
        if grp.fixed_size is not None:
            rec_end = pos + grp.fixed_size
            if rec_end > end:
                raise SrlogCorruptError(
                    f"{source}: trailing partial record: group '{grp.name}' needs "
                    f"{grp.fixed_size} payload bytes at byte {pos}, only {end - pos} remain"
                )
            buffers[gi] += data[pos:rec_end]
            pos = rec_end
        else:
            row: list = []
            for ch in grp.channels:
                if ch.dtype == "str16":
                    if pos + 2 > end:
                        raise SrlogCorruptError(
                            f"{source}: trailing partial record: str16 length prefix of "
                            f"'{grp.name}.{ch.name}' truncated at byte {pos}"
                        )
                    (slen,) = struct.unpack_from("<H", data, pos)
                    pos += 2
                    if pos + slen > end:
                        raise SrlogCorruptError(
                            f"{source}: trailing partial record: str16 payload of "
                            f"'{grp.name}.{ch.name}' ({slen} bytes) truncated at byte {pos}"
                        )
                    try:
                        row.append(data[pos : pos + slen].decode("utf-8"))
                    except UnicodeDecodeError as exc:
                        raise SrlogCorruptError(
                            f"{source}: str16 payload of '{grp.name}.{ch.name}' at byte "
                            f"{pos} is not valid UTF-8: {exc}"
                        ) from exc
                    pos += slen
                else:
                    if pos + ch.size > end:
                        raise SrlogCorruptError(
                            f"{source}: trailing partial record: channel "
                            f"'{grp.name}.{ch.name}' needs {ch.size} bytes at byte {pos}, "
                            f"only {end - pos} remain"
                        )
                    vals = struct.unpack_from(ch.fmt, data, pos)
                    row.append(vals if ch.vec_len else vals[0])
                    pos += ch.size
            rows[gi].append(tuple(row))

    arrays: dict[str, np.ndarray] = {}
    for gi, grp in enumerate(groups):
        if grp.fixed_size is not None:
            # frombuffer views are read-only; copy so callers get ordinary
            # writable arrays decoupled from the file bytes.
            arrays[grp.name] = np.frombuffer(bytes(buffers[gi]), dtype=grp.np_dtype).copy()
        else:
            arr = np.empty(len(rows[gi]), dtype=grp.np_dtype)
            for i, row in enumerate(rows[gi]):
                arr[i] = row
            arrays[grp.name] = arr

    events = arrays.get("events")
    if events is None:
        events = np.empty(0, dtype=_EVENTS_EMPTY_DTYPE)
    return Run(header=header, groups=arrays, events=events)


def load(path) -> Run:
    """Read an SRLOG v1 file into NumPy structured arrays.

    Raises ``SrlogVersionError`` on a major-version mismatch and
    ``SrlogCorruptError`` on structural damage (bad magic, truncated header,
    unknown dtype, trailing partial record). Minor versions ahead of this
    reader load normally.
    """
    data = Path(path).read_bytes()
    return _parse_bytes(data, str(path))
