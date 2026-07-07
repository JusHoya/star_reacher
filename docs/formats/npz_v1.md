# NPZ export layout, version 1

Normative specification of the NPZ archive written by `star export --npz`
(PRD FR-17, D-13) — the `export_npz`/`write_npz` writer and the `load_npz`
reader in `python/star_reacher/export.py` are both implemented against this
document. NPZ is the mandatory binary interchange exporter: loading the
archive reproduces the source `Run` bit-exactly (Phase 5 exit criterion 3)
with plain NumPy, no compiled core and no pickle.

## 1. Container

One standard NumPy `.npz` file (a zip of `.npy` members), written by
`numpy.savez` (uncompressed) and readable by `numpy.load` with its default
`allow_pickle=False`. **No member is ever pickled**: an archive that any
NumPy cannot load without code-execution risk is a writer defect.

The default output path is `<input stem>.npz` alongside the input log (the
CSV exporter's path convention).

## 2. Archive members

| Key | Content |
|---|---|
| `srnpz_layout` | 0-d unicode array, the layout version: `"1"` |
| `srlog_header_json` | 0-d unicode array: the SRLOG header dict serialized as compact JSON (`json.dumps(header, separators=(",", ":"))`); `json.loads` reproduces the header dict exactly |
| `group/<name>` | the group's NumPy structured array (section 3) |
| `group/<name>/fields` | present only for groups with string channels (section 4) |
| `group/<name>/utf8/<channel>` | ditto: concatenated UTF-8 bytes of one string channel |
| `group/<name>/offsets/<channel>` | ditto: u64 row offsets into the byte array |

Member order preserves the source `Run`'s group order (zip member order is
write order), so group ordering also survives the round trip.

Readers MUST check `srnpz_layout` first and refuse any value they do not
implement (`load_npz` raises `NpzFormatError`), the same refuse-don't-guess
rule SRLOG applies to its major version.

## 3. Groups without string channels

A group whose dtype contains no object fields (every fixed-rate group: the
SRLOG format restricts `str16` to the `events` group) is stored whole as one
structured `.npy` member under `group/<name>`. NumPy's native
structured-array save/load reproduces the dtype (field names, order, vector
subarray shapes) and every IEEE-754 byte exactly, including negative zeros
and subnormals.

## 4. Groups with string channels

Object-dtype string channels (decoded `str16`) cannot be saved without
pickle, so such a group is decomposed:

- `group/<name>` — a structured array of the **non-string fields only**, in
  their original relative order (omitted entirely if every field is a
  string channel);
- `group/<name>/fields` — 0-d unicode array holding compact JSON
  `{"n": <row count>, "fields": [{"name": ..., "utf8": true|false}, ...]}`
  recording the **full original field order** and which fields are string
  channels (the row count is needed when no fixed fields remain);
- per string channel `<ch>`:
  - `group/<name>/utf8/<ch>` — u8 array, the concatenation of every row's
    UTF-8-encoded value;
  - `group/<name>/offsets/<ch>` — u64 array of length `n + 1`;
    row `i`'s value is `utf8[offsets[i]:offsets[i+1]]` decoded as UTF-8.

The reader rebuilds the original dtype (string fields as object dtype, fixed
fields from the stored subarray) and refills it, reproducing the loaded
`Run`'s array value-exactly. The writer refuses non-`str` object values with
`TypeError` rather than coercing them.

## 5. Round-trip and determinism scope

`load_npz(write_npz(run, path))` reproduces `run.header` exactly (dict
equality), every group array bit-exactly (numeric bytes and string values),
and the standard `Run.events` behavior (an archive without an `events`
group loads with the empty standard events array, mirroring
`star_reacher.load`).

The **content** round-trip is the contract. The zip container embeds
wall-clock member timestamps written by `numpy.savez`, so two exports of
the same log are not byte-identical files; the D-10/FR-21 byte-determinism
contract applies to `run.srlog` itself, never to export containers
(`docs/formats/srlog_v1.md` section 7).

## 6. Versioning

`srnpz_layout` is a single opaque version string. Any change to the member
naming, the string-channel decomposition, or the header serialization is a
new layout version; readers refuse unknown versions loudly. Adding new SRLOG
groups or channels is **not** a layout change — the member set is derived
entirely from the `Run`, so new groups appear as new `group/<name>` members
under layout 1.
