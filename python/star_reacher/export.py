"""CSV exporter: one file per channel group (FR-17, D-13).

Floats are written via ``repr``, the shortest string that round-trips to the
identical IEEE-754 double, so the exported text preserves every stored bit
(Phase 1 exit criterion 4) while remaining human-readable. Vector channels
expand to indexed columns (``r_m_0, r_m_1, r_m_2``) so the CSV stays flat for
pandas, MATLAB, and spreadsheet consumers.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from star_reacher.srlog import load


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
