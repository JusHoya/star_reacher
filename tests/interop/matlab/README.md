# MATLAB `parquetread` checklist item (Phase 5 exit criterion 3, D-15)

Verifies that the Parquet files written by `star export --parquet`
(`docs/formats/parquet_v1.md`) load in MATLAB with the documented schema and
bit-exact values. MATLAB is not installed on the maintainer machine that
runs the GMAT item (`tools/bootstrap-notes/README.md` records GMAT's
`libMatlabInterface` failing to load for that reason), so this item is fully
prepared here -- validation script, bit-exact expected values, and pinned
input hashes -- and executes on any machine with MATLAB R2019a or newer
(`parquetread` is documented by MathWorks as introduced in R2019a). The
as-run console transcript is committed here when the run is performed, per
the exit criterion.

## Contents

- `read_truth_parquet.m` -- self-contained validation script. Reads
  `truth.parquet` and `events.parquet`, checks row counts, column names and
  order, column classes, and compares spot values bit-exactly
  (`num2hex` against IEEE-754 binary64 hex constants captured from the
  generating run; a tolerance would hide exactly the kind of type or
  round-trip corruption this item exists to catch). Prints one `ok`/`FAIL`
  line per check and a final `MATLAB-PARQUET: PASS (N/N)` verdict.
- `manifest.toml` -- provenance: how the Parquet inputs were generated, the
  SHA-256 of each input as generated, tool version pins, and how every
  expected constant in the `.m` script was derived.

## Maintainer procedure

1. From the repo root, with the package installed (`pip install .` plus the
   `pyarrow` extra), regenerate the inputs:

       star run missions/twobody_leo.toml --outdir out/twobody-leo --force
       star export out/twobody-leo/run.srlog --parquet

   `run.srlog` is byte-deterministic (D-10/FR-21) on a given binary;
   `meta.json` must show the `srlog_sha256` recorded in `manifest.toml`. If
   it differs, stop and reconcile before running MATLAB -- the simulator
   binary is not the one the expected constants were captured from (D-10 is
   a same-binary contract), and a transcript produced against different
   inputs would be evidence of nothing. The Parquet container bytes
   are NOT covered by the byte-determinism contract
   (`docs/formats/parquet_v1.md` section 5): with the pinned pyarrow the
   file hashes should reproduce, but with a different pyarrow a hash
   mismatch on `.parquet` files alone is expected and acceptable -- the
   `.m` script checks content, which is what the exit criterion gates.

2. In MATLAB (R2019a+), from this directory:

       setenv('STAR_REACHER_EXPORT_DIR', '<repo>/out/twobody-leo');
       diary transcript.txt
       read_truth_parquet
       diary off

3. On `MATLAB-PARQUET: PASS`, commit `transcript.txt` and add a `[[file]]`
   entry for it in `manifest.toml` recording the MATLAB version
   (`version` output is printed by the script) and the SHA-256 of the two
   `.parquet` files actually read (printed by the script).
