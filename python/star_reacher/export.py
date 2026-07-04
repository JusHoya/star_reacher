"""SRLOG exporters: CSV, NPZ, and Parquet (FR-17, D-13).

CSV (mandatory, Phase 1): one file per channel group. Floats are written via
``repr``, the shortest string that round-trips to the identical IEEE-754
double, so the exported text preserves every stored bit (Phase 1 exit
criterion 4) while remaining human-readable. Vector channels expand to
indexed columns (``r_m_0, r_m_1, r_m_2``) so the CSV stays flat for pandas,
MATLAB, and spreadsheet consumers.

NPZ (mandatory, Phase 5): one archive holding every group's structured array,
the events, and the header JSON, written and read back without pickle so the
archive stays loadable by any NumPy (layout normative in
``docs/formats/npz_v1.md``). ``load_npz`` reproduces the ``Run`` bit-exactly
(Phase 5 exit criterion 3).

Parquet (Phase 5, behind the ``pyarrow`` optional extra, D-13): one file per
group with the CSV column-flattening convention (layout normative in
``docs/formats/parquet_v1.md``).

This module stays importable with NumPy alone, like ``star_reacher.srlog``:
pyarrow is imported lazily inside the Parquet path only.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from star_reacher.srlog import _EVENTS_EMPTY_DTYPE, Run, _flat_columns, load


def _format_cell(field_dtype: np.dtype, value) -> str:
    if field_dtype.kind == "f":
        return repr(float(value))
    if field_dtype.kind in ("u", "i"):
        return str(int(value))
    # str16 channels arrive as decoded Python strings (object dtype); the
    # csv writer handles quoting of commas, quotes, and newlines.
    return str(value)


def export_csv(srlog_path, outdir=None) -> list[Path]:
    """Export every channel group of an SRLOG file to ``<group>.csv``.

    Returns the list of written paths. Raises the srlog reader errors
    unchanged so the CLI can map them to a nonzero exit.
    """
    run = load(srlog_path)
    out = Path(outdir) if outdir is not None else Path(srlog_path).parent
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for group_name, arr in run.groups.items():
        columns: list[str] = []
        for field_name in arr.dtype.names:
            shape = arr.dtype[field_name].shape
            if shape:
                columns.extend(f"{field_name}_{i}" for i in range(shape[0]))
            else:
                columns.append(field_name)
        path = out / f"{group_name}.csv"
        # newline="" hands line-ending control to the csv module, which is
        # required for correct quoting of embedded newlines on every platform.
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(columns)
            for rec in arr:
                row: list[str] = []
                for field_name in arr.dtype.names:
                    fdt = arr.dtype[field_name]
                    if fdt.shape:
                        row.extend(repr(float(x)) for x in rec[field_name])
                    else:
                        row.append(_format_cell(fdt, rec[field_name]))
                writer.writerow(row)
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# NPZ (D-13 mandatory; layout normative in docs/formats/npz_v1.md)
# ---------------------------------------------------------------------------

# Layout marker checked on read so a future incompatible NPZ layout is
# refused loudly instead of half-parsed (the same refuse-don't-guess rule
# SRLOG applies to its major version).
_NPZ_LAYOUT_KEY = "srnpz_layout"
_NPZ_LAYOUT_VERSION = "1"
_NPZ_HEADER_KEY = "srlog_header_json"
_NPZ_GROUP_PREFIX = "group/"


class NpzFormatError(Exception):
    """The file is not a layout-1 star_reacher NPZ archive."""


def _field_is_object(dtype: np.dtype, name: str) -> bool:
    # .base strips a subarray wrapper such as ('<f8', (3,)) down to '<f8',
    # so the kind test sees the element type for vector channels too.
    return dtype[name].base.kind == "O"


def write_npz(run: Run, path) -> Path:
    """Write a ``Run`` to one ``.npz`` archive, without pickle.

    Numeric structured arrays are stored natively (NumPy's structured-array
    save round-trips them bit-exactly). Object-dtype string channels (str16,
    events ``detail``) cannot be saved without pickle, so each is decomposed
    into a concatenated UTF-8 byte array plus a u64 offsets array, per the
    documented convention in docs/formats/npz_v1.md; non-string object
    values are refused with ``TypeError`` rather than silently coerced.
    """
    payload: dict[str, np.ndarray] = {
        _NPZ_LAYOUT_KEY: np.array(_NPZ_LAYOUT_VERSION),
        # Compact separators mirror the on-disk SRLOG header serialization;
        # json.loads on read reproduces the header dict exactly.
        _NPZ_HEADER_KEY: np.array(json.dumps(run.header, separators=(",", ":"))),
    }
    for group_name, arr in run.groups.items():
        object_fields = [f for f in arr.dtype.names if _field_is_object(arr.dtype, f)]
        if not object_fields:
            payload[f"{_NPZ_GROUP_PREFIX}{group_name}"] = arr
            continue
        fixed_fields = [f for f in arr.dtype.names if f not in object_fields]
        # The fields manifest preserves the full field order (the fixed
        # subarray alone cannot say where the string fields sat) and the
        # row count (needed when a group is all-string).
        manifest = {
            "n": int(len(arr)),
            "fields": [
                {"name": f, "utf8": f in object_fields} for f in arr.dtype.names
            ],
        }
        payload[f"{_NPZ_GROUP_PREFIX}{group_name}/fields"] = np.array(
            json.dumps(manifest, separators=(",", ":"))
        )
        if fixed_fields:
            fixed_dtype = np.dtype([(f, arr.dtype[f]) for f in fixed_fields])
            fixed_arr = np.empty(len(arr), dtype=fixed_dtype)
            for f in fixed_fields:
                fixed_arr[f] = arr[f]
            payload[f"{_NPZ_GROUP_PREFIX}{group_name}"] = fixed_arr
        for f in object_fields:
            encoded = []
            for i, value in enumerate(arr[f]):
                if not isinstance(value, str):
                    raise TypeError(
                        f"group {group_name!r} channel {f!r} row {i} holds "
                        f"{type(value).__name__}, not str; only decoded str16 "
                        f"string channels can be exported without pickle"
                    )
                encoded.append(value.encode("utf-8"))
            offsets = np.zeros(len(encoded) + 1, dtype="<u8")
            # The explicit dtype matters: cumsum of a plain (possibly empty)
            # Python list would default to a dtype that cannot cast into the
            # u64 out array.
            np.cumsum(
                np.array([len(b) for b in encoded], dtype="<u8"), out=offsets[1:]
            )
            payload[f"{_NPZ_GROUP_PREFIX}{group_name}/utf8/{f}"] = np.frombuffer(
                b"".join(encoded), dtype=np.uint8
            )
            payload[f"{_NPZ_GROUP_PREFIX}{group_name}/offsets/{f}"] = offsets
    path = Path(path)
    # Uncompressed savez: the round-trip contract is about array content,
    # not container bytes (the zip container embeds wall-clock member
    # timestamps, so NPZ files are not covered by the D-10 byte-determinism
    # contract; docs/formats/npz_v1.md section 5).
    with open(path, "wb") as fh:
        np.savez(fh, **payload)
    return path


def load_npz(path) -> Run:
    """Read a ``write_npz`` archive back into a ``Run``.

    The inverse of ``write_npz``: every numeric array is reproduced
    bit-exactly and every string channel value-exactly (Phase 5 exit
    criterion 3). ``np.load`` runs with its default ``allow_pickle=False``,
    which doubles as proof the archive is pickle-free. Raises
    ``NpzFormatError`` for archives this module did not write or a layout
    version it does not implement.
    """
    with np.load(Path(path)) as npz:
        if _NPZ_LAYOUT_KEY not in npz:
            raise NpzFormatError(
                f"{path}: not a star_reacher NPZ archive (missing "
                f"'{_NPZ_LAYOUT_KEY}'); expected a file written by "
                f"star export --npz"
            )
        layout = str(npz[_NPZ_LAYOUT_KEY][()])
        if layout != _NPZ_LAYOUT_VERSION:
            raise NpzFormatError(
                f"{path}: NPZ layout version {layout!r} is not readable by "
                f"this reader, which implements layout "
                f"{_NPZ_LAYOUT_VERSION!r}"
            )
        header = json.loads(str(npz[_NPZ_HEADER_KEY][()]))
        # npz.files preserves zip member order, which is payload insertion
        # order, so group order survives the round trip.
        group_order: list[str] = []
        string_groups: set[str] = set()
        for key in npz.files:
            if not key.startswith(_NPZ_GROUP_PREFIX):
                continue
            rest = key[len(_NPZ_GROUP_PREFIX):]
            if "/" not in rest:
                if rest not in group_order:
                    group_order.append(rest)
            elif rest.endswith("/fields") and rest.count("/") == 1:
                name = rest[: -len("/fields")]
                string_groups.add(name)
                if name not in group_order:
                    group_order.append(name)
        groups: dict[str, np.ndarray] = {}
        for name in group_order:
            if name not in string_groups:
                groups[name] = npz[f"{_NPZ_GROUP_PREFIX}{name}"]
                continue
            manifest = json.loads(str(npz[f"{_NPZ_GROUP_PREFIX}{name}/fields"][()]))
            n = manifest["n"]
            fixed_key = f"{_NPZ_GROUP_PREFIX}{name}"
            fixed_arr = npz[fixed_key] if fixed_key in npz else None
            fields = []
            for entry in manifest["fields"]:
                if entry["utf8"]:
                    fields.append((entry["name"], object))
                else:
                    fields.append((entry["name"], fixed_arr.dtype[entry["name"]]))
            out = np.empty(n, dtype=np.dtype(fields))
            for entry in manifest["fields"]:
                fname = entry["name"]
                if not entry["utf8"]:
                    out[fname] = fixed_arr[fname]
                    continue
                blob = npz[f"{_NPZ_GROUP_PREFIX}{name}/utf8/{fname}"].tobytes()
                offsets = npz[f"{_NPZ_GROUP_PREFIX}{name}/offsets/{fname}"]
                out[fname] = [
                    blob[int(offsets[i]) : int(offsets[i + 1])].decode("utf-8")
                    for i in range(n)
                ]
            groups[name] = out
    events = groups.get("events")
    if events is None:
        # Mirror star_reacher.load: Run.events is always a structured array
        # with the standard events interface, even when the log had none.
        events = np.empty(0, dtype=_EVENTS_EMPTY_DTYPE)
    return Run(header=header, groups=groups, events=events)


def export_npz(srlog_path, outdir=None) -> Path:
    """Export an SRLOG file to ``<stem>.npz`` and return the written path.

    Mirrors ``export_csv``'s path conventions: the default output directory
    is alongside the input, and srlog reader errors propagate unchanged so
    the CLI can map them to a nonzero exit.
    """
    run = load(srlog_path)
    out = Path(outdir) if outdir is not None else Path(srlog_path).parent
    out.mkdir(parents=True, exist_ok=True)
    return write_npz(run, out / f"{Path(srlog_path).stem}.npz")


# ---------------------------------------------------------------------------
# Parquet (D-13 optional, pyarrow extra; layout normative in
# docs/formats/parquet_v1.md)
# ---------------------------------------------------------------------------


def export_parquet(srlog_path, outdir=None) -> list[Path]:
    """Export every channel group to ``<group>.parquet``.

    Vector channels flatten to indexed columns (``r_m_0, r_m_1, r_m_2``)
    exactly like the CSV exporter, so the two tabular exports share one
    column convention. Requires pyarrow, a documented optional extra
    (D-13): when it is missing this raises an actionable ``ImportError``
    naming the extra instead of adding a hard dependency.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Parquet export requires pyarrow, which star_reacher treats as "
            'an optional dependency (D-13); install it with pip install '
            '"star-reacher[parquet]" (or pip install pyarrow)'
        ) from exc
    run = load(srlog_path)
    out = Path(outdir) if outdir is not None else Path(srlog_path).parent
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for group_name, arr in run.groups.items():
        columns = {}
        for column_name, field_name, index in _flat_columns(arr):
            if _field_is_object(arr.dtype, field_name):
                # Decoded str16 values become a native Parquet string
                # column (str16 is scalar-only per the format, so index is
                # always None here).
                columns[column_name] = pa.array(
                    [str(v) for v in arr[field_name]], type=pa.string()
                )
            elif index is None:
                columns[column_name] = pa.array(arr[field_name])
            else:
                # Subarray column slices are strided views; pyarrow needs a
                # contiguous buffer.
                columns[column_name] = pa.array(
                    np.ascontiguousarray(arr[field_name][:, index])
                )
        path = out / f"{group_name}.parquet"
        pq.write_table(pa.table(columns), path)
        written.append(path)
    return written
