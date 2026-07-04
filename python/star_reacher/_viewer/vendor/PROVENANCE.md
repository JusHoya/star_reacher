# Vendored three.js provenance

The FR-19 viewer (decision D-16) embeds a pinned, unmodified minified
three.js build into every generated HTML file. This record pins exactly what
was vendored so the bytes stay auditable.

| Field | Value |
|---|---|
| Package | `three` 0.166.1 (npm registry) |
| Source URL | `https://registry.npmjs.org/three/-/three-0.166.1.tgz` |
| Tarball SHA-256 | `ce585bcc9ce8a33c82b98472cc4ec2e7dae9c6a0435e38eaf0d2874a8d09bd1f` |
| Vendored file | `three.module.min.js` (tarball path `package/build/three.module.min.js`, byte-verbatim) |
| File SHA-256 | `54f21cfd2d0251ad8a406fb94f290c8c8086303f20ebdbf2f261edf5f55d5e96` |
| File size | 682,185 bytes |
| License | MIT; `LICENSE` in this directory is `package/LICENSE` from the same tarball, verbatim |
| Retrieved | 2026-07-04 |

Why this build form: `three.module.min.js` at this release is a single
self-contained ES module -- verified to contain no `import`/`export ... from`
of sibling files and no `</script` byte sequence -- so the generator can
inline it verbatim into one HTML file and load it through a Blob URL with
zero network requests. The file is committed with `-text` line-ending
handling (see `.gitattributes`) so the SHA-256 above stays verifiable against
working-tree bytes on every platform.

The pytest suite (`tests/python/test_viewer.py`) re-hashes the vendored file
against the SHA-256 recorded here on every run.
