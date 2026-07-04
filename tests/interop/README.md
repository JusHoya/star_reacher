# Maintainer-run interop checklist items (Phase 5 exit criterion 3, D-15)

This directory holds the two external-tool interop items of Phase 5 exit
criterion 3: the scripted GMAT ephemeris import (`gmat/`) and the MATLAB
`parquetread` check (`matlab/`). Per D-15 the external tools run offline on
the maintainer machine only; what is committed here is the exact
configuration, the command transcripts, and the SHA-256 hashes of every
input and output, so the runs are reviewable and repeatable without CI ever
installing GMAT or MATLAB. CI consumes nothing from this directory.

Each subdirectory carries a `manifest.toml` following the provenance schema
in `tests/golden/README.md` and, where the tool is installed on the
maintainer machine, the as-run transcript. The pandas/pyarrow side of the
same exit criterion is CI-gated separately
(`tests/python/test_export_parquet.py`); these items cover only the two
tools CI cannot run.
