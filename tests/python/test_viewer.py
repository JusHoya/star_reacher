"""Phase 5 viewer tests (FR-19, D-16): ``star view`` and the srview stream.

These REQUIRE the compiled core (they generate reference logs through the
real run path) and fail cleanly, never skip, when it is absent -- the same
agent-honesty gate as the vehicle mission tests. Two logs are exercised, per
the Phase 5 acceptance: a coast-heavy two-body orbit (aggressive decimation)
and the vehicle ascent (events, per-source forces, dense keyframes through
the burn).

The self-containment scan and the decimation-error recomputation here are
written independently of ``star_reacher.viewer`` (they re-derive their
verdicts from the emitted HTML and the raw log), so a defect in the
generator's own checks cannot mask one in the artifact.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

import star_reacher

REPO_ROOT = Path(__file__).resolve().parents[2]

_CORE_MISSING_MESSAGE = (
    "star_reacher._core is not built in this environment. These viewer tests "
    "require the compiled core to generate reference logs: build and install "
    "it with 'pip install .' from the repository root. This failure is "
    "expected on a core-less checkout and must be green at integration/CI."
)


def _core_or_fail():
    try:
        from star_reacher import _core
    except ImportError:
        pytest.fail(_CORE_MISSING_MESSAGE)
    return _core


def _generate(mission_name: str, tmp: Path):
    """Run a committed mission and generate its viewer HTML once."""
    _core_or_fail()
    from star_reacher.runner import run_mission
    from star_reacher.viewer import generate_view

    result = run_mission(REPO_ROOT / "missions" / mission_name, tmp / "run")
    view = generate_view(result.srlog_path, tmp / "view.html")
    return result.srlog_path, view


@pytest.fixture(scope="module")
def coast_view(tmp_path_factory):
    """Coast-heavy reference: the two-body LEO determinism-gate mission."""
    tmp = tmp_path_factory.mktemp("viewer_coast")
    return _generate("twobody_leo.toml", tmp)


@pytest.fixture(scope="module")
def ascent_view(tmp_path_factory):
    """Maneuver-rich reference: the ascent carries events and forces."""
    tmp = tmp_path_factory.mktemp("viewer_ascent")
    return _generate("ascent_leo.toml", tmp)


def _embedded_data(html: str) -> dict:
    m = re.search(
        r'<script type="application/json" id="srview-data">(.*?)</script>',
        html,
        re.S,
    )
    assert m, "no srview-data block in the HTML"
    return json.loads(m.group(1))


# ---------------------------------------------------------------------------
# (a) self-containment: zero external references, statically asserted
# ---------------------------------------------------------------------------

# Independent scan (not the generator's): findings key on attribute/call
# positions, so URLs inside comments or license prose do not false-positive.
_EXTERNAL_PATTERNS = [
    # any src/href attribute whose value is an absolute or protocol-relative
    # URL (http:, https:, ftp:, ws:, or bare //host)
    re.compile(r"""\b(?:src|href)\s*=\s*["'](?:[a-z][a-z0-9+.-]*:)?//""", re.I),
    # no external-fetch elements at all in a self-contained file
    re.compile(r"<link\b", re.I),
    re.compile(r"<img\b", re.I),
    re.compile(r"<iframe\b", re.I),
    re.compile(r"<embed\b", re.I),
    re.compile(r"<object\b", re.I),
    # no <script src=...> in any form (all script content must be inline)
    re.compile(r"<script\b[^>]*\bsrc\s*=", re.I),
    # CSS escape hatches
    re.compile(r"@import\b", re.I),
    re.compile(r"""\burl\(\s*["']?(?:https?:)?//""", re.I),
    # network APIs invoked on literal absolute URLs
    re.compile(r"""\bfetch\s*\(\s*["'](?:https?:)?//""", re.I),
    re.compile(r"""\bimport\s*\(\s*["'](?:https?:)?//""", re.I),
    re.compile(r"""\.open\s*\(\s*["'][A-Za-z]+["']\s*,\s*["'](?:https?:)?//"""),
    re.compile(r"new\s+WebSocket\s*\("),
    re.compile(r"navigator\.sendBeacon\s*\("),
]


def _assert_self_contained(html: str) -> None:
    for pattern in _EXTERNAL_PATTERNS:
        m = pattern.search(html)
        assert m is None, (
            f"external-reference pattern {pattern.pattern!r} matched: "
            f"...{html[max(0, m.start() - 60) : m.end() + 60]!r}..."
        )


def test_html_self_contained_coast(coast_view):
    _, view = coast_view
    _assert_self_contained(view.out_path.read_text(encoding="utf-8"))


def test_html_self_contained_ascent(ascent_view):
    _, view = ascent_view
    _assert_self_contained(view.out_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# (b) scrub-extreme epochs equal the log header's first/last epochs exactly
# ---------------------------------------------------------------------------


def _expected_last_epoch(header_epoch: str, t_last_s: float) -> str:
    """Header epoch + final truth time, one exact datetime addition."""
    epoch = datetime.fromisoformat(header_epoch.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )
    last = epoch + timedelta(seconds=t_last_s)
    text = last.strftime("%Y-%m-%dT%H:%M:%S")
    if last.microsecond:
        text += ("." + f"{last.microsecond:06d}").rstrip("0")
    return text + "Z"


@pytest.mark.parametrize("which", ["coast", "ascent"])
def test_embedded_epochs_match_header(which, coast_view, ascent_view, request):
    from star_reacher import load

    srlog_path, view = coast_view if which == "coast" else ascent_view
    run = load(srlog_path)
    data = _embedded_data(view.out_path.read_text(encoding="utf-8"))

    # First truth record is at t = 0, so the first HUD epoch must be the
    # header's epoch_utc string VERBATIM (Phase 5 exit criterion 2).
    truth_t = run.groups["truth"]["t_s"]
    assert float(truth_t[0]) == 0.0
    assert data["epoch"]["utc_first"] == run.header["epoch_utc"]
    assert data["epoch"]["utc_last"] == _expected_last_epoch(
        run.header["epoch_utc"], float(truth_t[-1])
    )
    # And the raw times the HUD anchors to are the log's, bit-exact.
    assert data["epoch"]["t_first_s"] == float(truth_t[0])
    assert data["epoch"]["t_last_s"] == float(truth_t[-1])


# ---------------------------------------------------------------------------
# (c) decimation error within the bound, recomputed against the full log
# ---------------------------------------------------------------------------


def _recompute_max_error_m(run, data) -> tuple[float, float]:
    """(max interpolation error, bound) re-derived from log + embedded data."""
    truth = run.groups["truth"]
    t = np.asarray(truth["t_s"], dtype=np.float64)
    r = np.asarray(truth["r_m"], dtype=np.float64)
    span = float(np.linalg.norm(r.max(axis=0) - r.min(axis=0)))
    bound = max(100.0, 1.0e-4 * span)

    kt = np.asarray(data["frames"]["t_s"], dtype=np.float64)
    kr = np.asarray(data["frames"]["r_m"], dtype=np.float64)
    # Keyframes must be actual truth samples (bit-exact), in order.
    idx = np.searchsorted(t, kt)
    assert np.array_equal(t[idx], kt), "keyframe times are not truth samples"
    assert np.array_equal(r[idx], kr), "keyframe positions differ from the log"
    assert idx[0] == 0 and idx[-1] == len(t) - 1, "endpoints must be kept"

    worst = 0.0
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a < 2:
            continue
        w = (t[a + 1 : b] - t[a]) / (t[b] - t[a])
        interp = r[a] + w[:, None] * (r[b] - r[a])
        err = np.sqrt(((r[a + 1 : b] - interp) ** 2).sum(axis=1))
        worst = max(worst, float(err.max()))
    return worst, bound


@pytest.mark.parametrize("which", ["coast", "ascent"])
def test_decimation_error_within_bound(which, coast_view, ascent_view):
    from star_reacher import load

    srlog_path, view = coast_view if which == "coast" else ascent_view
    run = load(srlog_path)
    data = _embedded_data(view.out_path.read_text(encoding="utf-8"))
    worst, bound = _recompute_max_error_m(run, data)

    assert worst <= bound, (worst, bound)
    # The embedded claim is a measurement of exactly this quantity.
    assert data["decimation"]["bound_m"] == bound
    assert abs(data["decimation"]["measured_max_error_m"] - worst) <= 1e-9 * bound
    # The coast must actually decimate hard (quiet orbit -> few keyframes).
    if which == "coast":
        assert data["decimation"]["kept"] < 0.02 * data["decimation"]["total"]


# ---------------------------------------------------------------------------
# (d) byte-identical regeneration
# ---------------------------------------------------------------------------


def test_regeneration_byte_identical(coast_view, tmp_path):
    from star_reacher.viewer import generate_view

    srlog_path, view = coast_view
    first = view.out_path.read_bytes()
    generate_view(srlog_path, tmp_path / "again.html")
    assert (tmp_path / "again.html").read_bytes() == first


# ---------------------------------------------------------------------------
# (e) event ticks equal the log's events
# ---------------------------------------------------------------------------


def test_event_ticks_match_log_events(ascent_view):
    from star_reacher import load

    srlog_path, view = ascent_view
    run = load(srlog_path)
    data = _embedded_data(view.out_path.read_text(encoding="utf-8"))
    assert len(run.events) > 2, "the ascent log must carry sequence events"
    assert data["events"]["t_s"] == [float(x) for x in run.events["t_s"]]
    assert data["events"]["code"] == [int(x) for x in run.events["code"]]
    assert data["events"]["detail"] == [str(x) for x in run.events["detail"]]


# ---------------------------------------------------------------------------
# forces stream (present for the vehicle log, absent-and-graceful otherwise)
# ---------------------------------------------------------------------------


def test_forces_embedded_for_vehicle_log(ascent_view):
    from star_reacher import load

    srlog_path, view = ascent_view
    run = load(srlog_path)
    data = _embedded_data(view.out_path.read_text(encoding="utf-8"))
    forces = data["forces"]
    assert forces is not None
    log_forces = run.groups["forces"]
    expected_sources = [
        nm[2:-4]
        for nm in log_forces.dtype.names
        if nm.startswith("f_") and nm.endswith("_b_n")
    ]
    assert forces["sources"] == expected_sources
    assert "thrust" in forces["sources"]
    # Below the stride cap the stream embeds every logged record, bit-exact.
    assert forces["stride"] == 1
    assert forces["t_s"] == [float(x) for x in log_forces["t_s"]]
    thrust = forces["f_b_n"][forces["sources"].index("thrust")]
    assert thrust == np.asarray(
        log_forces["f_thrust_b_n"], dtype=np.float64
    ).tolist()


def test_forces_absent_degrades_gracefully(coast_view):
    _, view = coast_view
    data = _embedded_data(view.out_path.read_text(encoding="utf-8"))
    assert data["forces"] is None
    # Attitude keyframes ride the same schedule as positions regardless.
    n = len(data["frames"]["t_s"])
    assert len(data["frames"]["q_i2b"]) == n
    assert len(data["frames"]["v_mps"]) == n
    assert len(data["frames"]["r_m"]) == n


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def _run_cli(*args, cwd=None):
    env = os.environ.copy()
    pkg_parent = Path(star_reacher.__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(pkg_parent) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "star_reacher", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
    )


def test_view_cli_writes_default_output_and_prints_bound(coast_view, tmp_path):
    srlog_path, view = coast_view
    log_copy = tmp_path / "run.srlog"
    log_copy.write_bytes(Path(srlog_path).read_bytes())
    proc = _run_cli("view", str(log_copy))
    assert proc.returncode == 0, proc.stderr
    out = tmp_path / "run.html"
    assert out.exists()
    # Both the bound and the measured error are printed (FR-19 acceptance).
    assert "decimation bound" in proc.stdout
    assert "measured max error" in proc.stdout
    # The default-path artifact matches the fixture generation byte-exactly.
    assert out.read_bytes() == view.out_path.read_bytes()


def test_view_cli_missing_file_exits_1(tmp_path):
    proc = _run_cli("view", str(tmp_path / "absent.srlog"))
    assert proc.returncode == 1
    assert "no such file" in proc.stderr


def test_view_cli_creates_missing_output_directory(coast_view, tmp_path):
    # Regression (Phase 5 red team): a missing OUTPUT directory used to
    # surface as "<input>: no such file." — the input-file handler caught
    # the writer's FileNotFoundError. The exporters mkdir their outdir, so
    # view does too; the input existence check now precedes generation.
    srlog_path, view = coast_view
    out = tmp_path / "not_yet" / "nested" / "view.html"
    proc = _run_cli("view", str(srlog_path), "-o", str(out))
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    assert out.read_bytes() == view.out_path.read_bytes()


# ---------------------------------------------------------------------------
# vendored/committed third-party content stays pinned to its provenance
# ---------------------------------------------------------------------------

_PKG = Path(star_reacher.__file__).resolve().parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_vendored_three_matches_provenance():
    provenance = (_PKG / "_viewer" / "vendor" / "PROVENANCE.md").read_text(
        encoding="utf-8"
    )
    recorded = re.findall(r"`([0-9a-f]{64})`", provenance)
    assert len(recorded) >= 2, "PROVENANCE.md must record tarball and file hashes"
    file_sha = _sha256(_PKG / "_viewer" / "vendor" / "three.module.min.js")
    assert file_sha in recorded, (
        f"vendored three.module.min.js sha256 {file_sha} is not the one "
        f"recorded in PROVENANCE.md"
    )
    assert (_PKG / "_viewer" / "vendor" / "LICENSE").read_text(
        encoding="utf-8"
    ).lstrip().startswith("The MIT License")


def test_coastline_asset_matches_readme():
    readme = (_PKG / "_assets" / "README.md").read_text(encoding="utf-8")
    recorded = re.findall(r"`([0-9a-f]{64})`", readme)
    asset = _PKG / "_assets" / "ne_110m_coastline.json"
    assert _sha256(asset) in recorded, (
        "committed coastline sha256 is not the one recorded in _assets/README.md"
    )
    doc = json.loads(asset.read_text(encoding="utf-8"))
    assert doc["segments"], "coastline asset carries no segments"
    for seg in doc["segments"]:
        assert len(seg) >= 2
        for lon, lat in seg:
            assert -180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0
