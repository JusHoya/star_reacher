# Shared display assets

Small committed datasets used by the Phase 5 output tools (the FR-19 viewer
groundtrack overlay and the FR-18 groundtrack plot). Display-grade only:
nothing in this directory feeds dynamics or analysis.

## `ne_110m_coastline.json`

World coastline polylines for drawing on an Earth sphere or map.

| Field | Value |
|---|---|
| Source | Natural Earth, 1:110m Physical Vectors, Coastline (`ne_110m_coastline`), version 5.1.2 |
| Source URL | `https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/geojson/ne_110m_coastline.geojson` (tag `v5.1.2` of the canonical Natural Earth repository) |
| Source SHA-256 | `851f581ff5ffb844deed8ae1a9ce22e3c4bb3d74fa342cadb5d8e39b41ae7c3c` |
| License | Public domain. Natural Earth terms of use: "All versions of Natural Earth raster + vector map data found on this website are in the public domain." (naturalearthdata.com/about/terms-of-use) |
| Committed file SHA-256 | `62948651b0b0503657efbf93002cf6bf4aa55af94aa00904e033514a1c87ba13` |
| Committed size | 76,424 bytes; 134 segments, 5,127 points |
| Retrieved | 2026-07-04 |

Simplification applied (the only transformation): every coordinate rounded to
0.01 degree (about 1.1 km at the equator, well below display resolution) and
consecutive duplicate points produced by the rounding collapsed. Conversion
code, run once against the source file above:

```python
import json

src = json.load(open("ne_110m_coastline.geojson", encoding="utf-8"))
segs = []
for feat in src["features"]:
    assert feat["geometry"]["type"] == "LineString"
    seg = [[round(lon, 2), round(lat, 2)]
           for lon, lat in feat["geometry"]["coordinates"]]
    out = [seg[0]]
    for p in seg[1:]:
        if p != out[-1]:
            out.append(p)
    if len(out) >= 2:
        segs.append(out)
doc = {
    "name": "ne_110m_coastline",
    "crs": "CRS84 longitude/latitude, degrees",
    "source": "Natural Earth 1:110m Coastline, v5.1.2 (public domain)",
    "simplification": "coordinates rounded to 0.01 degree; "
                      "consecutive duplicates collapsed",
    "segments": segs,
}
text = json.dumps(doc, separators=(",", ":"), ensure_ascii=True)
open("ne_110m_coastline.json", "w", encoding="utf-8",
     newline="\n").write(text + "\n")
```

Schema: one JSON object; `segments` is a list of polylines, each a list of
`[longitude_deg, latitude_deg]` pairs (CRS84 order, geodetic coordinates).
Consumers treat the coordinates as spherical for display.
