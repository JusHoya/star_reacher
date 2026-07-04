"""``star view``: self-contained single-file HTML playback viewer (FR-19, D-16).

The generator reads an SRLOG file through the pure-NumPy loader, decimates the
truth stream to a keyframe set with a measured position-error bound, and
assembles one HTML file embedding everything playback needs: the vendored
three.js runtime (MIT; ``_viewer/vendor/PROVENANCE.md``), the view stream as a
machine-readable JSON block (schema ``srview`` v1, normative in
``docs/formats/viewer.md``), and the public-domain coastline overlay
(``_assets/``). The emitted file makes zero network requests by construction;
``scan_external_references`` is the static check that the test suite and
``star verify`` both run against the emitted bytes.

Determinism: the HTML bytes are a pure function of the log bytes and the
committed template/vendor/asset files -- no timestamps, no filesystem paths,
no dict-order nondeterminism -- so regenerating from the same log is
byte-identical (the FR-21 discipline applied to a derived artifact).

Playback consumes only the log; there is no re-simulation. Interpolation
between keyframes (position lerp, attitude slerp) is display-only and
non-physical: analysis must read the log, never the viewer stream.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html import escape as _html_escape
from pathlib import Path

import numpy as np

from star_reacher.srlog import Run, load

_PKG_DIR = Path(__file__).resolve().parent
_VIEWER_DIR = _PKG_DIR / "_viewer"
_ASSETS_DIR = _PKG_DIR / "_assets"
_COASTLINE_ASSET = _ASSETS_DIR / "ne_110m_coastline.json"

SCHEMA_NAME = "srview"
SCHEMA_VERSION = 1

# Display radii for the central-body sphere [m]. These feed rendering and the
# HUD geocentric altitude only, never dynamics (the propagated physics lives
# in the log). Sources: Earth = WGS 84 semi-major axis (NGA TR8350.2, 2000);
# Moon = IAU/IAG recommended mean radius, Mars = IAU/IAG recommended
# equatorial radius (Archinal et al., "Report of the IAU Working Group on
# Cartographic Coordinates and Rotational Elements: 2015", Celest. Mech.
# Dyn. Astron. 130:22, 2018).
_BODY_RADIUS_M = {
    "earth": 6378137.0,
    "moon": 1737400.0,
    "mars": 3396190.0,
}

# Earth Rotation Angle: ERA(Tu) = 2*pi*(0.7790572732640 +
# 1.00273781191135448 * Tu), Tu = JD(UT1) - 2451545.0 (IERS Conventions 2010,
# TN No. 36, eq. 5.15). The viewer treats UTC as UT1 (|UT1-UTC| < 0.9 s) and
# neglects precession-nutation and polar motion: a sub-degree, DISPLAY-ONLY
# approximation that spins the coastline and groundtrack; it is never an
# analysis frame (the log's frames are the analysis frames).
_ERA_TURNS_AT_J2000 = 0.7790572732640
_ERA_TURNS_PER_UT1_DAY = 1.00273781191135448

# Beyond this many force records the embedded per-source force stream is
# stride-decimated: the forces overlay is a qualitative model-scrutiny aid,
# and an unbounded stream (multi-day runs) would dominate the file size.
_MAX_FORCE_RECORDS = 20000


class ViewerError(Exception):
    """Viewer generation failed (template/vendor integrity or assembly)."""


@dataclass
class ViewResult:
    """Summary of one generated viewer file."""

    out_path: Path
    html_bytes: int
    truth_records: int
    keyframes_kept: int
    bound_m: float
    measured_max_error_m: float
    position_span_m: float


# ---------------------------------------------------------------------------
# Epochs
# ---------------------------------------------------------------------------


def _parse_epoch_utc(text: str) -> datetime:
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _format_epoch_utc(dt: datetime) -> str:
    """ISO-8601 Z form, fractional seconds trimmed of trailing zeros."""
    base = dt.strftime("%Y-%m-%dT%H:%M:%S")
    if dt.microsecond:
        base += ("." + f"{dt.microsecond:06d}").rstrip("0")
    return base + "Z"


def first_last_epochs(header: dict, t_first_s: float, t_last_s: float) -> tuple[str, str]:
    """The scrub-extreme HUD epochs, derived from the header epoch directly.

    Each is one datetime addition from the header's ``epoch_utc`` -- never a
    float accumulation over samples -- so the Phase 5 exit-criterion-2 epoch
    equality is exact. When the first record sits at t = 0 the first epoch is
    the header string verbatim (no reformatting round trip). Leap seconds
    inside the span are not applied: the derived strings are calendar
    arithmetic on the header epoch, the documented display convention.
    """
    epoch_utc = header["epoch_utc"]
    epoch = _parse_epoch_utc(epoch_utc)
    if t_first_s == 0.0:
        utc_first = epoch_utc
    else:
        utc_first = _format_epoch_utc(epoch + timedelta(seconds=t_first_s))
    utc_last = _format_epoch_utc(epoch + timedelta(seconds=t_last_s))
    return utc_first, utc_last


def _era_rotation(epoch: datetime) -> dict:
    """Display-grade ERA rotation parameters at the run epoch (see above)."""
    unix_s = epoch.timestamp()
    jd_utc = 2440587.5 + unix_s / 86400.0
    tu = jd_utc - 2451545.0
    turns = _ERA_TURNS_AT_J2000 + _ERA_TURNS_PER_UT1_DAY * tu
    era0 = 2.0 * math.pi * (turns % 1.0)
    rate = _ERA_TURNS_PER_UT1_DAY * 2.0 * math.pi / 86400.0
    return {"model": "era", "era0_rad": era0, "rate_radps": rate}


# ---------------------------------------------------------------------------
# Decimation
# ---------------------------------------------------------------------------


def position_error_bound_m(r_m: np.ndarray) -> tuple[float, float]:
    """(bound, span): bound = max(100 m, 0.01 % of the position span).

    The span is the bounding-box diagonal of all truth positions, the
    definition documented in docs/formats/viewer.md and gated by Phase 5
    exit criterion 2.
    """
    span = float(np.linalg.norm(r_m.max(axis=0) - r_m.min(axis=0)))
    return max(100.0, 1.0e-4 * span), span


def _segment_errors(t: np.ndarray, r: np.ndarray, a: int, b: int) -> np.ndarray:
    """Distances of samples a+1..b-1 from time-linear interpolation a -> b.

    This is the playback interpolant itself (linear in t between kept
    samples), so the measured maximum is exactly the worst position error a
    viewer of the decimated stream can display at a truth-sample time.
    """
    w = (t[a + 1 : b] - t[a]) / (t[b] - t[a])
    interp = r[a] + w[:, None] * (r[b] - r[a])
    d = r[a + 1 : b] - interp
    return np.sqrt(np.einsum("ij,ij->i", d, d))


def decimate_keyframes(t: np.ndarray, r: np.ndarray, bound_m: float) -> np.ndarray:
    """Kept-sample indices via greedy largest-error keyframe insertion.

    Douglas-Peucker on the time-parameterized 3D polyline: starting from the
    endpoints, repeatedly promote the dropped sample with the largest
    interpolation error until every segment's worst error is within
    ``bound_m``. Greedy insertion is used (rather than uniform stride)
    because quiet coasts then collapse to a handful of keyframes while burns
    keep dense support, and termination itself certifies the bound.
    Deterministic: ties resolve to the lowest index (np.argmax) and the
    segment stack is processed in a fixed order.
    """
    n = len(t)
    if n <= 2:
        return np.arange(n)
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[n - 1] = True
    stack: list[tuple[int, int]] = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        err = _segment_errors(t, r, a, b)
        i = int(np.argmax(err))
        if err[i] > bound_m:
            k = a + 1 + i
            keep[k] = True
            stack.append((a, k))
            stack.append((k, b))
    return np.flatnonzero(keep)


def measure_decimation_error(t: np.ndarray, r: np.ndarray, kept: np.ndarray) -> float:
    """Exhaustive max interpolation error over every dropped truth sample.

    Recomputed over the full log after decimation (not carried over from the
    insertion loop) so the embedded value is a direct measurement, and a
    defect in the insertion bookkeeping cannot fake a passing bound.
    """
    worst = 0.0
    for a, b in zip(kept[:-1], kept[1:]):
        if b - a < 2:
            continue
        worst = max(worst, float(_segment_errors(t, r, int(a), int(b)).max()))
    return worst


# ---------------------------------------------------------------------------
# View data (schema srview v1; normative doc: docs/formats/viewer.md)
# ---------------------------------------------------------------------------


def build_view_data(run: Run) -> dict:
    """Assemble the embedded JSON document from a loaded run."""
    truth = run.groups["truth"]
    t = np.ascontiguousarray(truth["t_s"], dtype=np.float64)
    r = np.ascontiguousarray(truth["r_m"], dtype=np.float64)
    v = np.ascontiguousarray(truth["v_mps"], dtype=np.float64)
    q = np.ascontiguousarray(truth["q_i2b"], dtype=np.float64)

    bound_m, span_m = position_error_bound_m(r)
    kept = decimate_keyframes(t, r, bound_m)
    measured_m = measure_decimation_error(t, r, kept)

    header = run.header
    t_first = float(t[0])
    t_last = float(t[-1])
    utc_first, utc_last = first_last_epochs(header, t_first, t_last)

    body_name = str(header.get("central_body", ""))
    radius_m = _BODY_RADIUS_M.get(body_name)
    if body_name == "earth":
        rotation = _era_rotation(_parse_epoch_utc(header["epoch_utc"]))
    else:
        # No display rotation model for other bodies: the groundtrack then
        # traces the inertial sub-point, labeled as such in the help overlay.
        rotation = {"model": "none"}

    events = run.events
    events_data = {
        "t_s": [float(x) for x in events["t_s"]],
        "code": [int(x) for x in events["code"]],
        "detail": [str(x) for x in events["detail"]],
    }

    forces_data = None
    if "forces" in run.groups:
        forces = run.groups["forces"]
        names = [
            nm
            for nm in forces.dtype.names
            if nm.startswith("f_") and nm.endswith("_b_n")
        ]
        if names:
            m = len(forces)
            stride = max(1, -(-m // _MAX_FORCE_RECORDS))  # ceil division
            sel = np.arange(0, m, stride)
            forces_data = {
                "t_s": forces["t_s"][sel].tolist(),
                "stride": int(stride),
                # Channel-dictionary order preserved: sources render in the
                # same order the log declares them.
                "sources": [nm[2:-4] for nm in names],
                "f_b_n": [
                    np.ascontiguousarray(forces[nm], dtype=np.float64)[sel].tolist()
                    for nm in names
                ],
            }

    coastline = None
    if body_name == "earth" and _COASTLINE_ASSET.is_file():
        # The committed asset is already compact JSON; embed its parsed form
        # so the HTML carries one JSON block, not nested encoded text.
        coastline = json.loads(_COASTLINE_ASSET.read_text(encoding="utf-8"))

    return {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "header": {
            "epoch_utc": header["epoch_utc"],
            "central_body": body_name,
            "config_sha256": header.get("config_sha256", ""),
            "master_seed": header.get("master_seed", ""),
        },
        "epoch": {
            "utc_first": utc_first,
            "utc_last": utc_last,
            "t_first_s": t_first,
            "t_last_s": t_last,
        },
        "body": {
            "name": body_name,
            "radius_m": radius_m,
            "rotation": rotation,
        },
        "decimation": {
            "algorithm": "greedy largest-error keyframe insertion "
            "(Douglas-Peucker on the time-parameterized 3D polyline)",
            "bound_m": bound_m,
            "measured_max_error_m": measured_m,
            "position_span_m": span_m,
            "kept": int(len(kept)),
            "total": int(len(t)),
        },
        "frames": {
            "t_s": t[kept].tolist(),
            "r_m": r[kept].tolist(),
            "v_mps": v[kept].tolist(),
            "q_i2b": q[kept].tolist(),
        },
        "events": events_data,
        "forces": forces_data,
        "coastline": coastline,
    }


# ---------------------------------------------------------------------------
# Self-containment scan
# ---------------------------------------------------------------------------

# Attribute/call positions only: URLs inside comments or license prose are
# fine (D-16 forbids network REQUESTS, not the string "https://"), so the
# scan keys on the syntactic positions that cause a browser to fetch.
_EXTERNAL_REF_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "src/href attribute with an absolute or protocol-relative URL",
        re.compile(r"""\b(?:src|href)\s*=\s*["'](?:[a-z][a-z0-9+.-]*:)?//""", re.I),
    ),
    ("<link> element (external stylesheet/preload surface)", re.compile(r"<link\b", re.I)),
    ("<img> element", re.compile(r"<img\b", re.I)),
    ("<iframe> element", re.compile(r"<iframe\b", re.I)),
    ("CSS @import", re.compile(r"@import\b", re.I)),
    (
        "CSS url() with an absolute URL",
        re.compile(r"""\burl\(\s*["']?(?:https?:)?//""", re.I),
    ),
    (
        "fetch() of an absolute URL",
        re.compile(r"""\bfetch\s*\(\s*["'](?:https?:)?//""", re.I),
    ),
    (
        "dynamic import() of an absolute URL",
        re.compile(r"""\bimport\s*\(\s*["'](?:https?:)?//""", re.I),
    ),
    (
        "XMLHttpRequest open() of an absolute URL",
        re.compile(r"""\.open\s*\(\s*["'][A-Za-z]+["']\s*,\s*["'](?:https?:)?//"""),
    ),
    ("WebSocket constructor", re.compile(r"new\s+WebSocket\s*\(")),
    ("navigator.sendBeacon call", re.compile(r"navigator\.sendBeacon\s*\(")),
]


def scan_external_references(html_text: str) -> list[str]:
    """Findings that would violate the zero-network-request guarantee.

    Empty list = clean. Used by ``star verify`` (V020); the pytest suite
    additionally runs its own independently written scan so a defect here
    cannot mask one in the emitted HTML.
    """
    findings = []
    for label, pattern in _EXTERNAL_REF_PATTERNS:
        m = pattern.search(html_text)
        if m:
            start = max(0, m.start() - 40)
            findings.append(f"{label}: ...{html_text[start : m.end() + 40]!r}...")
    return findings


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"__SRVIEW_(?:TITLE|THREE_JS|DATA_JSON|APP_JS)__"
)


def _assemble_html(data: dict) -> bytes:
    template = (_VIEWER_DIR / "template.html").read_text(encoding="utf-8")
    app_js = (_VIEWER_DIR / "app.js").read_text(encoding="utf-8")
    three_js = (_VIEWER_DIR / "vendor" / "three.module.min.js").read_text(
        encoding="utf-8"
    )

    # ensure_ascii keeps the byte stream independent of any encoding edge
    # cases in event details; escaping "</" (a legal JSON string escape)
    # makes "</script>" unrepresentable inside the embedded JSON block.
    data_json = json.dumps(data, separators=(",", ":"), ensure_ascii=True)
    data_json = data_json.replace("</", "<\\/")

    for name, text in (("app.js", app_js), ("vendored three.js", three_js)):
        if "</script" in text.lower():
            raise ViewerError(
                f"{name} contains '</script', which would truncate the "
                f"inline script block; the embedded sources must never "
                f"contain that byte sequence"
            )

    title = _html_escape(
        f"star_reacher view - {data['header']['epoch_utc']} - "
        f"{data['body']['name'] or 'unknown body'}"
    )
    tokens = {
        "__SRVIEW_TITLE__": title,
        "__SRVIEW_THREE_JS__": three_js,
        "__SRVIEW_DATA_JSON__": data_json,
        "__SRVIEW_APP_JS__": app_js,
    }
    for token in tokens:
        if template.count(token) != 1:
            raise ViewerError(
                f"template.html must contain the token {token} exactly once "
                f"(found {template.count(token)})"
            )
    # Single pass with a function replacement: substituted content is never
    # rescanned (a token-shaped string inside the log data cannot recurse)
    # and backslashes in the vendored source are inserted literally.
    html = _TOKEN_RE.sub(lambda m: tokens[m.group(0)], template)
    return html.encode("utf-8")


def generate_view(srlog_path, out_path=None) -> ViewResult:
    """Generate the FR-19 viewer HTML for one SRLOG file.

    ``out_path`` defaults to the input path with an ``.html`` suffix. The
    output is overwritten if present: the viewer is a derived artifact,
    regenerable bit-identically from the log.
    """
    srlog_path = Path(srlog_path)
    run = load(srlog_path)
    data = build_view_data(run)
    html = _assemble_html(data)
    out = Path(out_path) if out_path is not None else srlog_path.with_suffix(".html")
    # Create missing parent directories like the exporters do (export_csv et
    # al. mkdir their outdir), so `-o new_dir/view.html` works and a missing
    # directory cannot surface as a FileNotFoundError that the CLI would
    # misattribute to the input log.
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(html)
    dec = data["decimation"]
    return ViewResult(
        out_path=out,
        html_bytes=len(html),
        truth_records=dec["total"],
        keyframes_kept=dec["kept"],
        bound_m=dec["bound_m"],
        measured_max_error_m=dec["measured_max_error_m"],
        position_span_m=dec["position_span_m"],
    )


def extract_view_data(html_text: str) -> dict:
    """Parse the embedded srview JSON block back out of a generated file.

    The inverse of assembly for tests and ``star verify``: the returned dict
    compares equal to the ``build_view_data`` output for the same log.
    """
    m = re.search(
        r'<script type="application/json" id="srview-data">(.*?)</script>',
        html_text,
        re.S,
    )
    if not m:
        raise ViewerError("no srview-data JSON block found in the HTML")
    return json.loads(m.group(1))
