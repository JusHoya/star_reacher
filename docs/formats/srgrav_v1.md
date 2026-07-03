# SRGRAV v1.0 binary gravity-coefficient format

Normative specification of the SRGRAV on-disk format (PRD FR-5): the
repacked spherical-harmonic gravity-field container written by the Python
pipeline (`star data fetch egm2008 | grgm1200a | mro120f`,
`python/star_reacher/data_fetch.py`) and read by the C++ core
(`cpp/src/models/gravity.cpp`). Writer and reader are both implemented
against this document; a change here is a format change and follows the
versioning rules in section 6.

SRGRAV exists for the same reason SREPH does (D-2 boundary rule): the
deterministic core never parses text, so the coefficient sets it consumes
must be fixed-layout binary files fully decodable from fixed-width fields.
The coefficients inside are the source model's fully normalized (4-pi
geodesy normalization) Stokes coefficients C-bar/S-bar, converted from the
source text representation by one decimal-to-binary64 conversion each --
the same conversion any consumer of the text file performs -- and never
rescaled or renormalized. GM and the reference radius are taken from the
source file's own header (each field is a self-consistent triple; mixing a
field's coefficients with another source's GM changes the model).

## 1. Overview

- Byte order: **little-endian throughout**. Every supported platform is
  little-endian (same rule as SRLOG and SREPH).
- File = HEADER, then one COEFFICIENT BLOCK. The total size is fully
  determined by the header; there is no footer.
- Floating-point values are IEEE-754 binary64, stored little-endian.
- Angular/positional conventions: coefficients are interpreted in the
  body-fixed frame of the source model (ITRS-aligned for EGM2008, lunar
  principal axes for GRGM1200A, IAU Mars body-fixed for MRO120F); the
  frame binding is provenance recorded in the fetch manifest, not encoded
  in the file.

## 2. Header (96 bytes)

| Bytes | Type | Content |
|---|---|---|
| 0-7 | bytes | Magic: `53 52 47 52 56 00 0D 0A` (ASCII `SRGRV`, NUL, CR, LF) |
| 8-9 | u16 | `version_major` = 1 |
| 10-11 | u16 | `version_minor` = 0 |
| 12-15 | u32 | `n_max`: maximum stored degree |
| 16-19 | u32 | `m_max`: maximum stored order (`m_max <= n_max`) |
| 20-23 | u32 | `tide_system` (section 4) |
| 24-27 | u32 | reserved, = 0 |
| 28-35 | f64 | `gm_m3ps2`: gravitational parameter GM [m^3/s^2], from the source header |
| 36-43 | f64 | `ref_radius_m`: reference radius R [m], from the source header |
| 44-59 | bytes | `name`: ASCII, NUL-padded (max 15 chars + NUL), e.g. `EGM2008` |
| 60-91 | bytes | SHA-256 digest (raw 32 bytes) of the source coefficient file |
| 92-95 | u32 | reserved, = 0 |

The magic carries the same tripwire design as SRLOG/SREPH: the NUL stops
C-string readers, and the CR/LF pair is altered by any text-mode transfer,
so a mangled file fails the magic check immediately. The source digest is
provenance carried in-band for auditability; readers do not interpret it.

## 3. Coefficient block

Fully normalized coefficient pairs in a fixed order: ascending degree `n`
from 0 to `n_max`, and within each degree ascending order `m` from 0 to
`min(n, m_max)`; each (n, m) entry is the pair `C_bar(n,m)` then
`S_bar(n,m)`, both binary64:

```
n=0: C(0,0) S(0,0)
n=1: C(1,0) S(1,0) C(1,1) S(1,1)
n=2: C(2,0) S(2,0) C(2,1) S(2,1) C(2,2) S(2,2)
...
```

The number of (n, m) entries is `sum over n of (min(n, m_max) + 1)`; for a
square field (`m_max = n_max`) this is `(n_max+1)(n_max+2)/2`. The file
size must equal exactly `96 + 16 * n_entries`; readers must reject any
other size.

Every degree and order in the range is present -- there is no sparse
encoding. Where the source file omits rows the repack tool fills the
mathematically defined values: `C(0,0) = 1` exactly (the monopole is
carried entirely by GM, so its normalized coefficient is unity by
definition), zero for all other omitted entries (the shipped models are
center-of-mass fields, whose degree-1 coefficients are zero; the tool
records in the fetch manifest which rows were filled). `S(n,0)` is stored
as `0.0` for every degree, as the sine coefficient of order zero is
identically zero.

## 4. Tide-system codes

| code | meaning |
|---|---|
| 0 | tide-free (the source states its C(2,0) excludes the permanent tide) |
| 1 | zero-tide |
| 2 | mean-tide |
| 3 | not stated by the source |

The tide system does not enter evaluation -- it documents the convention
of the stored `C(2,0)` so cross-tool comparisons use consistent
coefficients. Readers must reject codes above 3 (an unknown convention is
an unknown semantic, mirroring the unknown-`kind` rule of SREPH).

## 5. Evaluation contract

The evaluator (Pines formulation, singularity-free at the poles; see the
gravity chapter of the math library) treats the stored coefficients,
`gm_m3ps2`, and `ref_radius_m` as one indivisible model. The degree-0 term
is evaluated from the stored `C(0,0)` -- the returned acceleration always
includes the central term. Requested truncation degree/order beyond the
stored `n_max`/`m_max` is an error (`std::invalid_argument` in the core):
the file carries no information above its stored band and the core never
silently degrades fidelity.

## 6. Versioning

- **Minor version bump** = additive change only: new `tide_system` codes
  or new trailing header content that old readers can ignore without
  misreading layout. Existing field meanings never change within a major.
- **Major version bump** = layout break. Readers must refuse a major
  version they do not implement, loudly, naming both versions.

## 7. Provenance and reproducibility

The standard repacks are produced by `star data fetch <dataset>` for the
three FR-5 fields (Earth EGM2008 to 70x70, Moon GRGM1200A to 120x120, Mars
MRO120F to 80x80), which pins each source file by SHA-256 (constants with
citations in `python/star_reacher/data_fetch.py`), records the fetch in
`data/<dataset>_manifest.json`, and is idempotent: re-running verifies
checksums instead of re-downloading. The repack output bytes are a pure
function of the source file and the requested truncation. `data/` is
git-ignored by design; the committed test fixtures under
`tests/golden/gravity/` are excerpts in this same format with their own
provenance manifest.
