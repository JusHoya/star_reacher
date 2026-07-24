# README example-gallery images

The PNGs in this directory are the images referenced by the README
[example gallery](../../README.md#example-gallery) (PRD FR-31, a Phase 8
deliverable). They are a **documentation** artifact, not a regression
instrument: the data-level plot regression is gated separately by
`tests/golden/plots/`.

## Provenance and reproduction

Every image is regenerated from a committed mission by
[`generate.py`](generate.py) — the commands it runs are the same ones the
README Quickstart and [`docs/walkthrough.md`](../walkthrough.md) document:

```sh
.venv312/Scripts/python docs/gallery/generate.py
```

| Image | Source mission | Command driving it |
| --- | --- | --- |
| `groundtrack_inclined_leo.png` | `missions/leo_gravity_8x8.toml` (`--set mission.duration_s=18000`, first ~6 orbits) | `star plot ... --plots groundtrack` |
| `elements_inclined_leo.png` | same | `star plot ... --plots elements` |
| `altitude_speed_ascent.png` | `missions/ascent_leo.toml` | `star plot ... --plots altitude_speed` |
| `forces_by_source_ascent.png` | same | `star plot ... --plots forces_by_source` |
| `mass_thrust_ascent.png` | same | `star plot ... --plots mass_thrust_throttle` |
| `trajectory3d_cislunar.png` | `missions/mission_a_cislunar.toml` | static 3D still from the truth channel (see `generate.py`) |

The `--set mission.duration_s=18000` override is the documented DX-4 edit knob;
it shortens the 7-day inclined-LEO mission to about six orbits so the
groundtrack and osculating-element structure read cleanly. All three source
missions run on a clean clone with no fetched data (the cislunar mission uses
the committed DE440 ephemeris excerpt).

## Determinism

The `star plot` PNGs and the 3D still are a pure function of the log bytes and
the pinned matplotlib version — fixed figure geometry, fixed PNG metadata, no
timestamps — so a regeneration is byte-identical (the FR-21 discipline applied
to a derived artifact). Regenerate on the same platform/matplotlib and the
bytes do not move.

The `trajectory3d_cislunar.png` still is a preview of what the interactive
`star view` HTML plays back, rendered from the same `truth` channel the viewer
decimates; it is not a substitute for the live viewer, which the README links
alongside it.
