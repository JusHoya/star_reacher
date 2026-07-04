# Parquet export layout, version 1

Normative specification of the Parquet files written by `star export
--parquet` (PRD FR-17, D-13) — implemented by `export_parquet` in
`python/star_reacher/export.py`. Parquet is the optional columnar exporter
behind the `pyarrow` extra: when pyarrow is not installed the exporter
raises an actionable `ImportError` naming `pip install
"star-reacher[parquet]"` and the CLI exits 1; the format never becomes a
hard dependency (D-12 allowed-list discipline).

## 1. Files

One Parquet file per channel group, named `<group>.parquet` (e.g.
`truth.parquet`, `events.parquet`), in the output directory (default:
alongside the input log) — the CSV exporter's one-file-per-group and path
conventions.

## 2. Columns

Column names and order follow the shared tabular-flattening convention
(`star_reacher.srlog._flat_columns`, the same layout the CSV exporter and
`Run.to_pandas()` emit):

- scalar channels keep their channel name (`t_s`, `mass_kg`, `code`);
- vector channels `f64[N]` expand to `N` indexed columns in element order:
  `r_m_0, r_m_1, r_m_2`;
- columns appear in the header dictionary's channel order.

## 3. Types

| SRLOG dtype | Arrow/Parquet type |
|---|---|
| `f64`, `f64[N]` elements | `double` (IEEE-754 binary64, value-preserving) |
| `u32` | `uint32` |
| `u64` | `uint64` |
| `str16` | `string` (UTF-8) |

Float columns are written from the loaded arrays without reformatting, so a
pandas `read_parquet` reproduces every stored double bit-exactly; unsigned
integer widths survive (pandas reads `uint32`/`uint64` columns, not
silently-widened signed ints); string values survive exactly, including
embedded quotes and newlines.

## 4. Acceptance

The binding gate is Phase 5 exit criterion 3: exported Parquet loads in
pandas with row counts equal to each group's record count and column sets
equal to the flattened channel list (`tests/python/test_export_parquet.py`,
CI-gated on a leg with the extras installed; `star verify` V019 additionally
read-checks the files whenever pyarrow is importable). MATLAB `parquetread`
is a maintainer-run D-15 checklist item recorded under `tests/interop/`.

## 5. Determinism scope

As with NPZ (`docs/formats/npz_v1.md` section 5), the contract is content
fidelity, not container bytes: Parquet metadata includes writer version
strings, so export files are not covered by the D-10/FR-21 byte-determinism
contract, which applies to `run.srlog` only.
