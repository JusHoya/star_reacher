# SREPH v1.0 binary ephemeris format

Normative specification of the SREPH on-disk format (PRD D-8, FR-4): the
repacked Chebyshev ephemeris container written by the Python pipeline
(`star data fetch de440s`, `python/star_reacher/data_fetch.py`) and read by
the C++ core (`cpp/src/ephemeris.cpp`). Writer and reader are both
implemented against this document; a change here is a format change and
follows the versioning rules in section 6.

SREPH exists because of the D-2 boundary rule: the deterministic core never
parses text, so the ephemeris it consumes must be a fixed-layout binary file
whose entire structure is decodable from fixed-width fields. The Chebyshev
coefficients inside are copied **verbatim** from the source JPL kernels
(SPK/PCK Type 2) - never refitted, never rescaled - so evaluation accuracy is
inherited from DE440 exactly; the container contributes nothing but layout.

## 1. Overview

- Byte order: **little-endian throughout**. Every supported platform is
  little-endian (same rule as SRLOG).
- File = HEADER, then SEGMENT DIRECTORY, then COEFFICIENT BLOCKS. All sizes
  are fully determined by the header and directory; there is no footer.
- Floating-point values are IEEE-754 binary64, stored little-endian.
- **Time argument convention: TDB seconds since J2000 TDB** (JD 2451545.0
  TDB), as a binary64 - identical to the native epoch representation inside
  SPK/PCK Type 2 segments, so epochs are copied verbatim too. Near the 2060
  end of the standard span (|t| ~ 1.9e9 s) one ulp is ~2.4e-7 s; at the
  fastest stored motion (~35 km/s) that quantizes positions at the ~8 mm
  level, far below the 1 m validation gate (see the ephemeris chapter of the
  math library for the full precision budget).

## 2. Header (96 bytes)

| Bytes | Type | Content |
|---|---|---|
| 0-7 | bytes | Magic: `53 52 45 50 48 00 0D 0A` (ASCII `SREPH`, NUL, CR, LF) |
| 8-9 | u16 | `version_major` = 1 |
| 10-11 | u16 | `version_minor` = 0 |
| 12-15 | u32 | `segment_count` |
| 16-23 | f64 | `span_start_tdb_s`: max over segments of first-record start |
| 24-31 | f64 | `span_end_tdb_s`: min over segments of last-record end |
| 32-63 | bytes | SHA-256 digest (raw 32 bytes) of the source SPK kernel |
| 64-95 | bytes | SHA-256 digest (raw 32 bytes) of the source PCK kernel |

The magic carries the same tripwire design as SRLOG: the NUL stops C-string
readers, and the CR/LF pair is altered by any text-mode transfer, so a
mangled file fails the magic check immediately.

The span fields are the **intersection** of all segments' coverage. For a
standard repack (one contiguous segment per body) every body is evaluable on
this span. For excerpt files holding non-contiguous record runs (committed
test fixtures) the intersection is degenerate and the fields are
informational only; the authoritative domain check is always per-segment
containment (section 5).

The source digests are provenance carried in-band for auditability; readers
do not interpret them.

## 3. Segment directory (`segment_count` x 64 bytes)

| Bytes | Type | Content |
|---|---|---|
| 0-15 | bytes | `name`: ASCII, NUL-padded (max 15 chars + NUL) |
| 16-19 | u32 | `target`: NAIF body ID (PCK segments: NAIF frame class ID) |
| 20-23 | u32 | `center`: NAIF center ID (PCK segments: reference frame ID) |
| 24-27 | u32 | `kind` (section 4) |
| 28-31 | u32 | `n_coeffs`: Chebyshev coefficients per component per record |
| 32-39 | f64 | `init_tdb_s`: start epoch of the first record |
| 40-47 | f64 | `intlen_s`: record interval length [s] |
| 48-51 | u32 | `n_records` |
| 52-55 | u32 | reserved, = 0 |
| 56-63 | u64 | `offset_bytes`: file offset of this segment's coefficient block |

Multiple segments may share one `name` (excerpt files); their record spans
must not overlap. A reader resolves a `(name, epoch)` query to the segment
whose span contains the epoch.

### Standard `de440s` repack contents

| name | target | center | kind | components |
|---|---|---|---|---|
| `sun` | 10 | 0 (SSB) | 0 | position [km] |
| `emb` | 3 | 0 (SSB) | 0 | position [km] |
| `earth` | 399 | 3 (EMB) | 0 | position [km] |
| `moon` | 301 | 3 (EMB) | 0 | position [km] |
| `venus_bary` | 2 | 0 (SSB) | 0 | position [km] |
| `mars_bary` | 4 | 0 (SSB) | 0 | position [km] |
| `jupiter_bary` | 5 | 0 (SSB) | 0 | position [km] |
| `moon_librations` | 31008 | 1 (J2000) | 1 | Euler angles [rad] |

Moon and Earth are stored relative to the Earth-Moon barycenter exactly as
DE440 stores them; the geocentric Moon is composed at evaluation time as
`moon - earth`. `moon_librations` carries the 3-1-3 Euler angles phi, theta,
psi of the Moon principal-axis frame relative to the ICRF equator (DE440
lunar orientation); psi accumulates without wrapping, exactly as the source
kernel stores it. The standard span is 2020-01-01 through 2060-01-01 TDB,
widened outward to whole-record boundaries per segment (records are never
split). The loader accepts user-supplied wider trims (FR-4): the format
carries any span the repack tool was given.

## 4. Segment kinds

| kind | Source | Components | Units of value / rate |
|---|---|---|---|
| 0 | SPK Type 2 | x, y, z position | km, km/s (evaluator outputs m, m/s) |
| 1 | binary PCK Type 2 | phi, theta, psi Euler angles | rad, rad/s |

Coefficients are stored in the source kernel's native units (km for SPK,
rad for PCK) because scaling coefficients would round them; the C++
evaluator converts kind-0 outputs to meters with a single multiply by 1000.0
per component. A reader encountering an unknown kind must reject the file
(it cannot know the output semantics).

## 5. Coefficient blocks

Each segment's block is `n_records * 3 * n_coeffs` binary64 values, laid out
record-major, component-major within a record, ascending Chebyshev order
within a component:

```
record 0: c[x][0..n_coeffs-1], c[y][0..n_coeffs-1], c[z][0..n_coeffs-1]
record 1: ...
```

This is exactly the layout of an SPK/PCK Type 2 record with the leading MID
and RADIUS words dropped: DE Type 2 segments have constant record intervals,
so `MID = init + (k + 1/2) * intlen` and `RADIUS = intlen / 2` are redundant
with the directory fields (the repack tool verifies this before dropping
them). The retained coefficient words are byte-identical to the source
kernel's.

Record `k` covers epochs `[init + k*intlen, init + (k+1)*intlen]`. Record
selection for epoch `t`:

```
k = floor((t - init) / intlen);  if k == n_records then k = n_records - 1
```

i.e. **a shared boundary epoch belongs to the record that begins there**,
and the final epoch of the segment evaluates in the last record at scaled
time x = +1 exactly. Epochs outside every segment of the queried name are a
domain error: readers must refuse them (`std::out_of_range` in the core) and
never extrapolate. The evaluation recurrence, its derivative, and the
operation-order contract that makes C++ and Python evaluation bit-identical
are specified in the ephemeris chapter of the math library.

## 6. Versioning

- **Minor version bump** = additive change only: new `kind` values or new
  trailing header/directory content that old readers can ignore without
  misreading layout. Existing field meanings never change within a major.
- **Major version bump** = layout break. Readers must refuse a major version
  they do not implement, loudly, naming both versions.

## 7. Provenance and reproducibility

The standard repack is produced by `star data fetch de440s`, which pins both
source kernels by SHA-256 (constants with citations in
`python/star_reacher/data_fetch.py`), records the fetch in
`data/de440s_manifest.json`, and is idempotent: re-running verifies
checksums instead of re-downloading. The repack output bytes are a pure
function of the source kernels and the requested span. `data/` is
git-ignored by design (D-8); the committed test fixtures under
`tests/golden/ephemeris/` are excerpts in this same format with their own
provenance manifest.
