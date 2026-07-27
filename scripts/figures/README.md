# Report case-study figures

`casestudy_figures.py` regenerates the two case-study figures of the
scientific report (FR-30): the translunar transfer and the Earth--Mars
transfer. It is the mechanism behind Phase 8 exit criterion 3 --- *with the
pinned matplotlib version and `SOURCE_DATE_EPOCH` set, regenerating all report
figures from committed seeds reproduces the committed figures bit-for-bit.*

## What is committed

- `docs/report/figures/translunar_transfer.png`, `mars_transfer.png` --- the
  figures included by `docs/report/report.tex`.
- `docs/report/figures/figure_data/*.npz` --- the decimated, derived
  figure-input arrays the renderer reads. Committing these makes the
  `--render` path independent of the compiled core.
- `docs/report/figures/figure_data/manifest.json` --- provenance: for each
  case, the source mission, its resolved-config SHA-256 (which labels the
  figure), the full-run `run.srlog` SHA-256, and the `.npz` SHA-256.

Both figures derive from committed missions (nothing fetched, no network):

| Figure | Mission | Central body |
|---|---|---|
| `translunar_transfer.png` | `missions/tli.toml` | Earth (GCRF) |
| `mars_transfer.png` | `missions/mars_cruise.toml` | Sun (heliocentric) |

## Regenerating

From the repository root, with the project's Python:

```
python scripts/figures/casestudy_figures.py            # render PNGs from committed .npz
python scripts/figures/casestudy_figures.py --extract  # re-run missions, rewrite .npz
python scripts/figures/casestudy_figures.py --verify-repro     # portable determinism gate (CI)
python scripts/figures/casestudy_figures.py --check-committed  # pinned-toolchain equality gate
```

- `--render` (the default) needs only the committed `.npz` and matplotlib.
- `--extract` needs the compiled core and the committed ephemeris excerpts;
  it re-propagates the missions with `star run`, decimates each 1 Hz truth
  log to a fixed-stride sample (~2000 points), derives the plotted quantities
  (geocentric/heliocentric distance, osculating semi-major axis via vis-viva,
  the x--y projection), and rewrites the `.npz` set and the manifest. Run it
  only when a mission or a model changes; commit the regenerated `.npz`,
  `manifest.json`, and PNGs together.
- Two gates carry criterion 3, and they are complementary:
  - `--verify-repro` renders twice to fresh temporary directories and asserts
    every output PNG is byte-identical by SHA-256. This proves *regeneration is
    deterministic* and holds on any runner regardless of the toolchain, so it is
    the gate wired into CI (`report-figures` job).
  - `--check-committed` renders once and asserts each PNG byte-matches the
    committed `docs/report/figures/*.png`. This is the literal criterion-3
    clause ("reproduces *the committed figures* bit-for-bit") and additionally
    catches a script edit that silently stales the committed PNGs, which
    `--verify-repro` cannot. Its equality is scoped to the pinned toolchain
    below (a different matplotlib/FreeType rasterizes different bytes by
    design), so it is the maintainer's on-toolchain gate rather than a
    cross-platform CI gate; run it before committing regenerated figures.

## Determinism requirements (Phase 8 criterion 3)

The PNG bytes are a pure function of the committed `.npz` inputs and the
rendering toolchain only when:

- **`SOURCE_DATE_EPOCH` is set.** matplotlib stamps the PNG `tIME` chunk from
  it; a fixed value fixes those bytes. The script sets a fixed fallback
  (`1577836800`, 2020-01-01T00:00:00Z) when the caller has not exported one,
  so a bare `--render` is reproducible; the FR-29 docs build and CI export it
  from the HEAD commit time. No wall-clock quantity is *plotted*, so the value
  does not affect the figure content.
- **The Agg backend is forced** before `pyplot` import (headless; no display,
  no inherited backend).
- **`savefig(..., metadata={"Software": None})`** drops the matplotlib
  version string from the PNG.
- **Figure geometry and DPI are fixed** in the script, not read from a host
  `matplotlibrc`.

### Pinned toolchain

PNG bytes depend on the matplotlib **and FreeType** versions (FreeType
rasterizes the text; a different FreeType shifts glyph pixels and changes the
bytes). The committed figures were generated and verified bit-for-bit with:

| Component | Version |
|---|---|
| Python | 3.12.10 |
| matplotlib | 3.11.1 |
| FreeType (via matplotlib) | 2.14.3 |
| numpy | 2.5.1 |

`pyproject.toml` declares the runtime floor `matplotlib>=3.8`; the bit-for-bit
guarantee is against the exact versions above, so the CI figure-regeneration
gate pins them. Reproducing the committed bytes on a host with a different
matplotlib or FreeType may require re-extracting and re-rendering under the
pinned versions and re-committing the figures.

Check the installed versions with:

```
python -c "import matplotlib, matplotlib.ft2font as f; print(matplotlib.__version__, f.__freetype_version__)"
```
