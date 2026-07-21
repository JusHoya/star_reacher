"""Phase 6 exit criteria 7 and 9 against the independent NumPy references.

These are the integration gates: they drive a real mission through the
compiled core and compare the LOGGED optical output against the blind
reference implementations of ``tests/refs`` -- modules written from the
math-library chapters alone, with no reference to the C++ they check
(provenance in ``tests/refs/manifest.toml``).

* Exit criterion 9 (``ch:sensors-optical``): the logged apparent Sun
  direction matches an independent recomputation of the normative
  first-order aberration law, equation ``eq:optical:aberration``, to better
  than 1 milliarcsecond. The comparison is first-order against first-order:
  the chapter declares that equation THE formula and specifies that the
  criterion recomputes IT, so gating against the exact relativistic form
  would measure a deliberate modelling choice rather than an implementation
  error. The size of that choice is recorded, non-normatively, by
  ``test_first_order_versus_exact_gap_is_recorded``.

* Exit criterion 7 (``ch:camera``): every logged landmark pixel matches an
  independent recomputation of equations ``eq:camera:pose`` and
  ``eq:camera:landmark``--``eq:camera:proj`` to better than 1e-6 pixels, on
  a scenario with an off-axis mount and ``fx != fy``. The criterion's
  bit-exactness clause -- the pose channels being assignments of the truth
  doubles rather than a recomputation -- is re-confirmed here under array
  equality on this mission's log.

The mission below is deliberately noise-free in its optical sensors: with
the sigmas zeroed, the logged channels carry the aberration and projection
transformations alone, so the residual against the reference measures
arithmetic and convention agreement rather than a realisation of noise.

Shared inputs versus the subject under test: the ephemeris, the time
scales, and the GCRF-to-ITRF rotation are INPUTS to both sides of every
comparison here, and each is gated by its own Phase 2 and Phase 3 golden
vectors. What these tests isolate is the optical transformation chain built
on top of them. The ephemeris is evaluated through the pure-Python
reference evaluator of ``star_reacher.data_fetch``, not through the C++
loader, so the source direction is not taken from the code under test.

They fail cleanly, never skip, when the core is absent (the project's
agent-honesty gate).
"""

import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO_ROOT / "tests" / "refs"))
import pinhole  # noqa: E402
import quaternions  # noqa: E402
from aberration import (  # noqa: E402
    MAS_PER_RAD,
    aberrate_exact,
    aberrate_first_order,
    beta_vector,
    separation_angle,
)

EPHEMERIS = "tests/golden/ephemeris/excerpt_de440s_crosstool.sreph"

# Exit criterion 9 states a REQUIREMENT of 1 milliarcsecond. That is not the
# agreement the implementation achieves, and gating at the requirement leaves
# seven orders of slack in which a wrong formula sits comfortably: the
# drop-the-transverse-projection mutation (`u + beta` for
# `u + beta - (u.beta) u`) measures 0.4696 mas on this fixture and passes a
# 1 mas gate. The gate is therefore set near the achievable agreement
# instead. The measured worst residual against the independent reference is
# 4.73e-08 mas, so 1e-5 mas keeps roughly 210x headroom over the observed
# rounding-order residual while rejecting that mutation by about 4.7e+04.
# Passing this gate implies the criterion by five orders of magnitude;
# CRITERION_9_REQUIREMENT_MAS is asserted too so the criterion's own figure
# appears in the suite rather than only in the PRD.
ABERRATION_TOL_MAS = 1e-5
CRITERION_9_REQUIREMENT_MAS = 1.0
PIXEL_TOL = 1e-6

_CORE_MISSING_MESSAGE = (
    "star_reacher._core is not built in this environment. These integration "
    "tests require the compiled core: build and install it with 'pip install .' "
    "from the repository root. This failure is expected on a core-less checkout "
    "and must be green at integration/CI."
)

# Five surface landmarks around the initial subsatellite point, in
# central-body-fixed axes. Generated once as
# ``C_GCRF->ITRF(epoch) @ [7e6, 0, 0]``, renormalized to the WGS84 equatorial
# radius, plus four neighbours displaced by 0.006 rad in the local tangent
# plane; the values are frozen here so the fixture does not depend on the
# frames path at test time. The displacement places every landmark inside
# the sensor at the initial attitude, which is what keeps the criterion-7
# gate non-vacuous (asserted by test_landmarks_are_actually_visible).
_LANDMARKS_FIXED_M = (
    "981184.3607315255, 6302193.8419025615, 16174.6749673658, "
    "1018979.3039107342, 6296193.3863468878, 16174.3838310770, "
    "981181.6291115165, 6302176.2965772515, -22093.6262959488, "
    "943354.0958690132, 6307967.4246054776, 16174.3838310770, "
    "981151.7706682307, 6301984.5143751120, 54442.3939581028"
)

# The reference GNC mission of missions/leo_attitude_gnc.toml, re-epoched
# into the committed DE440 excerpt's window and given a barycentric
# ephemeris, noise-free optics, and a nadir-pointing camera. Everything not
# named here is that mission's value and carries its rationale.
_OPTICAL_GATE_MISSION = f"""
schema_version = 1
vehicle = "vehicles/smallsat.toml"

[mission]
name = "p6-optical-gates"
# Inside the committed DE440 excerpt's span, so the run needs no fetched data.
epoch_utc = "2025-12-30T12:00:00Z"
duration_s = 60.0

[run]
seed = 20260601

[integrator]
type = "rk4"
dt_s = 0.1

[environment]
central_body = "earth"
# The Sun and Moon third bodies are enabled so the mission validator accepts
# the ephemeris: its consumer list counts force models only, and the FR-23
# optical sensors that actually require the Sun direction are not among them.
# In the Earth regime FR-6 requires the pair, not the Sun alone.
third_bodies = ["sun", "moon"]
ephemeris = "{EPHEMERIS}"

[logging]
truth_rate_hz = 10

[initial_state.cartesian]
r_m = [7.0e6, 0.0, 0.0]
v_mps = [0.0, 7546.0, 0.0]
frame = "GCRF"

[gnc]
control_rate_hz = 10
latency_cycles = 0

[gnc.nav]
component = "dead_reckoning"
q0 = [0.0, 0.7071067811865476, 0.7071067811865476, 0.0]

[gnc.guidance]
component = "attitude_hold"
# A 10-degree slew about the OFF-AXIS body direction [1, 2, -2]/3 rather than
# about body +Z. The reference mission's +Z slew leaves the attitude in the
# q_w == 0, q_z == 0 plane for the whole run, and at q_w == 0 the
# frame-transformation DCM of eq:notation:quat2dcm is exactly symmetric --
# C - C^T = -4 q_w [q_v x] vanishes identically. Criterion 9's residual is an
# angular separation, so on such a fixture a TRANSPOSED attitude convention is
# undetectable by geometry no matter which implementation supplies the DCM.
# This axis carries |q_w| up to 0.060 and |C - C^T| up to 0.18, which is what
# gives the independent-DCM substitution below something to detect. It also
# happens to place all five landmarks on-sensor at every camera sample
# (300/300 visible against 186/300 for the +Z slew), so criterion 7 gains from
# it as well.
q_cmd = [-0.061628416716219346, 0.66333041525861247, 0.74550163754690491, 0.020542805572073115]

[gnc.control]
component = "pd_attitude"
kp_nm_per_rad = [0.4, 0.4, 0.4]
kd_nm_per_radps = [3.6, 3.6, 3.6]
tau_max_nm = [0.05, 0.05, 0.05]

[sensors.imu]
sample_rate_hz = 10

[sensors.startracker]
sample_rate_hz = 5
boresight_b = [0.0, 0.0, 1.0]
sigma_rad = [0.0, 0.0, 0.0]
# Every exclusion disabled: the criterion-9 comparison is about the
# aberration arithmetic, and a gated sample would still be logged but would
# invite the reader to filter on a flag that carries no information here.
sun_exclusion_rad = 0.0
central_body_exclusion_rad = 0.0
slew_limit_radps = 0.0

[sensors.sunsensor]
sample_rate_hz = 5
boresight_b = [1.0, 0.0, 0.0]
# Full sphere, so the field-of-view gate never masks a sample.
fov_half_angle_rad = 3.141592653589793
sigma_rad = 0.0

[sensors.camera]
sample_rate_hz = 1
# fx != fy with an off-axis principal point, so the recomputation exercises
# both focal lengths and both offsets independently rather than sharing one
# scale that a transposed convention could hide behind.
fx_px = 800.0
fy_px = 600.0
cx_px = 511.5
cy_px = 383.5
width_px = 1024.0
height_px = 768.0
# A non-zero CG-relative station and a non-identity mount rotation taking the
# camera boresight (+Z_c) to nadir (-Y_b) at the initial attitude.
r_cam_b_m = [0.5, -0.25, 0.125]
q_b2c = [0.7071067811865476, 0.7071067811865476, 0.0, 0.0]
landmarks_fixed_m = [{_LANDMARKS_FIXED_M}]
"""

_J2000_JD = 2451545.0


def _core_or_fail():
    try:
        from star_reacher import _core
    except ImportError:
        pytest.fail(_CORE_MISSING_MESSAGE)
    return _core


@pytest.fixture(scope="module")
def optical_run(tmp_path_factory):
    """One run of the noise-free optical mission plus its loaded log."""
    _core_or_fail()
    import os

    from star_reacher import load
    from star_reacher.runner import run_mission

    tmp = tmp_path_factory.mktemp("p6_optical")
    mission = tmp / "optical_gates.toml"
    mission.write_text(_OPTICAL_GATE_MISSION.lstrip(), encoding="utf-8")
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)  # the vehicle and ephemeris paths are repo-relative
    try:
        result = run_mission(mission, tmp / "run")
    finally:
        os.chdir(cwd)
    return result, load(result.srlog_path), tomllib.loads(mission.read_text("utf-8"))


class _Ephemeris:
    """Barycentric states through the pure-Python SREPH reference evaluator.

    Deliberately NOT the C++ loader: the aberration gate must not take its
    source direction and observer velocity from the code it checks. This
    evaluator is the one the ephemeris golden set was generated and
    cross-checked against jplephem with (tests/golden/ephemeris/manifest.toml).
    """

    def __init__(self):
        from star_reacher.data_fetch import read_sreph

        self._eph = read_sreph(REPO_ROOT / EPHEMERIS)

    def _state_m(self, name: str, tdb_s: float):
        from star_reacher.data_fetch import evaluate_segment

        segment = self._eph.segment_for(name, tdb_s)
        position_km, rate_km_s = evaluate_segment(segment, tdb_s)
        return np.array(position_km) * 1000.0, np.array(rate_km_s) * 1000.0

    def earth_ssb(self, tdb_s: float):
        """Earth relative to the solar-system barycentre.

        DE440 stores the Earth-Moon barycentre against the SSB and the Earth
        against the EMB, so the chain is the sum of the two segments.
        """
        r_emb, v_emb = self._state_m("emb", tdb_s)
        r_earth, v_earth = self._state_m("earth", tdb_s)
        return r_emb + r_earth, v_emb + v_earth

    def sun_rel_earth(self, tdb_s: float) -> np.ndarray:
        r_sun, _ = self._state_m("sun", tdb_s)
        r_earth, _ = self.earth_ssb(tdb_s)
        return r_sun - r_earth


def _epoch_tai(core, config):
    """TAI epoch of the mission, as the (day, second) pair the core uses."""
    stamp = config["mission"]["epoch_utc"]
    date, clock = stamp.rstrip("Z").split("T")
    year, month, day = (int(p) for p in date.split("-"))
    hour, minute, second = clock.split(":")
    return core.utc_to_tai(year, month, day, int(hour), int(minute), float(second))


def _tdb_s_at(core, epoch_tai, t_s: float) -> float:
    """TDB seconds since J2000 at mission time ``t_s``."""
    tai = core.tai_add_seconds(epoch_tai[0], epoch_tai[1], t_s)
    jd1, jd2 = core.tdb_jd(tai[0], tai[1])
    return ((jd1 - _J2000_JD) + jd2) * 86400.0


def _truth_rows(truth, sample_times, cycle_s=0.1):
    """Index the truth group at each sensor sample time.

    Both grids are exact multiples of the control cycle, so the integer cycle
    index is an exact key rather than a nearest-neighbour search.
    """
    index = {int(round(t / cycle_s)): i for i, t in enumerate(truth["t_s"])}
    return [index[int(round(float(t) / cycle_s))] for t in sample_times]


def _sun_direction_residuals_mas(run, config):
    """Angle between each logged Sun direction and the reference recomputation.

    Returns the residuals alongside the deflection the aberration itself
    produces, so a caller can report the gate's signal-to-tolerance ratio.
    """
    from star_reacher import _core

    truth = run.groups["truth"]
    sun = run.groups["sensors.sunsensor"]
    ephemeris = _Ephemeris()
    epoch_tai = _epoch_tai(_core, config)
    rows = _truth_rows(truth, sun["t_s"])

    residual_mas = np.empty(len(sun))
    deflection_mas = np.empty(len(sun))
    unaberrated_mas = np.empty(len(sun))
    for j, t_s in enumerate(sun["t_s"]):
        row = rows[j]
        tdb_s = _tdb_s_at(_core, epoch_tai, float(t_s))
        _, v_earth = ephemeris.earth_ssb(tdb_s)
        # eq:optical:beta: the observer's barycentric velocity is the vehicle
        # velocity relative to the central body plus the central body's own
        # velocity relative to the SSB.
        beta = beta_vector(truth["v_mps"][row], v_earth)
        geometric = ephemeris.sun_rel_earth(tdb_s) - truth["r_m"][row]
        geometric = geometric / np.linalg.norm(geometric)
        apparent = aberrate_first_order(geometric, beta)
        # The independent DCM of tests/refs/quaternions, NOT _core.quat_to_dcm.
        # The logged direction was produced by the core using its own rotation,
        # and angular separation is invariant under a rotation applied to both
        # arguments, so putting the core's DCM on this side too would cancel it
        # exactly and no attitude-convention error could ever be detected here.
        c_i2b = quaternions.quat_to_dcm(truth["q_i2b"][row])
        logged = sun["sun_b"][j]
        residual_mas[j] = separation_angle(c_i2b @ apparent, logged) * MAS_PER_RAD
        deflection_mas[j] = separation_angle(geometric, apparent) * MAS_PER_RAD
        unaberrated_mas[j] = separation_angle(c_i2b @ geometric, logged) * MAS_PER_RAD
    return residual_mas, deflection_mas, unaberrated_mas


def test_aberration_matches_independent_reference(optical_run):
    """Exit criterion 9: logged apparent directions against the reference.

    The reference is ``tests/refs/aberration.aberrate_first_order``, the
    normative equation ``eq:optical:aberration``, written blind from the
    chapter, rotated into body axes by the equally independent
    ``tests/refs/quaternions.quat_to_dcm``. Both sides of the comparison are
    therefore free of the code under test.

    The gate is ``ABERRATION_TOL_MAS`` (1e-5 mas), not the criterion's 1 mas:
    see the constant's derivation for why gating at the requirement leaves
    room for an algebraically wrong formula to pass.
    """
    _, run, config = optical_run
    residual_mas, _, _ = _sun_direction_residuals_mas(run, config)
    assert len(residual_mas) == 300  # 60 s at 5 Hz
    worst = float(residual_mas.max())
    assert worst < ABERRATION_TOL_MAS, (
        f"worst logged-versus-reference Sun direction residual {worst:.6e} mas "
        f"exceeds the exit-criterion-9 gate of {ABERRATION_TOL_MAS} mas"
    )
    # The criterion's own figure, so the suite states the requirement it meets
    # and not only the tighter bound it is held to.
    assert worst < CRITERION_9_REQUIREMENT_MAS


def test_aberration_fixture_can_see_an_attitude_convention_error(optical_run):
    """The criterion-9 residual is sensitive to the attitude convention.

    Substituting the independent DCM is necessary but not sufficient: an
    angular separation is invariant under a rotation applied to both
    arguments, and at ``q_w == 0`` the DCM of ``eq:notation:quat2dcm`` is
    exactly symmetric, so on a fixture whose attitude stays in that plane a
    transposed convention changes nothing at all. This asserts the fixture
    carries the asymmetry that makes the substitution meaningful; without it
    the gate would read as convention-aware while remaining blind.
    """
    _, run, _ = optical_run
    truth = run.groups["truth"]
    worst_asymmetry = 0.0
    for q in truth["q_i2b"]:
        c = quaternions.quat_to_dcm(q)
        worst_asymmetry = max(worst_asymmetry, float(np.abs(c - c.T).max()))
    assert worst_asymmetry > 0.05, (
        f"the fixture's attitude keeps C_I2B within {worst_asymmetry:.3e} of "
        "symmetric, so a transposed attitude convention would be invisible to "
        "the criterion-9 residual"
    )


def test_aberration_signal_dominates_the_gate(optical_run):
    """The criterion-9 gate is not vacuous: aberration is really present.

    Comparing the GEOMETRIC direction against the same logged channel
    measures the aberration the implementation applied. If that displacement
    were absent or negligible the < 1 mas agreement above would be
    unremarkable, so this test pins the signal the gate is resolving.
    """
    _, run, config = optical_run
    _, deflection_mas, unaberrated_mas = _sun_direction_residuals_mas(run, config)
    # eq:optical:abmag gives beta sin(theta); at Earth's barycentric speed
    # plus this LEO speed the chapter's scale is ~20.5 arcsec.
    assert deflection_mas.min() > 20_000.0
    assert deflection_mas.max() < 21_000.0
    # An implementation that skipped the correction entirely would miss the
    # logged direction by that whole displacement.
    assert np.allclose(unaberrated_mas, deflection_mas, rtol=1e-3)


def test_first_order_versus_exact_gap_is_recorded(optical_run):
    """NON-NORMATIVE: the modelling gap the first-order choice accepts.

    ``ch:sensors-optical`` assumption 2 adopts the first-order law and states
    the difference from the exact relativistic form as
    ``(beta**2 / 4) sin(2 theta)``. This test measures that difference on the
    real scenario and asserts only that it stays inside the chapter's own
    bound. It is NOT the criterion-9 gate: exit criterion 9 recomputes the
    first-order equation, and gating the core against the exact form would
    spend most of the 1 mas budget on a deliberate modelling choice.
    """
    from star_reacher import _core

    _, run, config = optical_run
    truth = run.groups["truth"]
    sun = run.groups["sensors.sunsensor"]
    ephemeris = _Ephemeris()
    epoch_tai = _epoch_tai(_core, config)
    rows = _truth_rows(truth, sun["t_s"])

    gap_mas = np.empty(len(sun))
    for j, t_s in enumerate(sun["t_s"]):
        row = rows[j]
        tdb_s = _tdb_s_at(_core, epoch_tai, float(t_s))
        _, v_earth = ephemeris.earth_ssb(tdb_s)
        beta = beta_vector(truth["v_mps"][row], v_earth)
        geometric = ephemeris.sun_rel_earth(tdb_s) - truth["r_m"][row]
        geometric = geometric / np.linalg.norm(geometric)
        gap_mas[j] = separation_angle(
            aberrate_first_order(geometric, beta), aberrate_exact(geometric, beta)
        ) * MAS_PER_RAD
    # The chapter's bound is (beta^2/4) at its worst orientation; this
    # geometry is off that maximum, so the observed gap is smaller.
    assert gap_mas.max() < 0.52


def _camera_projections(run, config):
    """Recompute every logged landmark pixel through the blind reference."""
    from star_reacher import _core

    truth = run.groups["truth"]
    camera = run.groups["sensors.camera"]
    cfg = config["sensors"]["camera"]
    intrinsics = pinhole.Intrinsics(
        cfg["fx_px"], cfg["fy_px"], cfg["cx_px"], cfg["cy_px"],
        cfg["width_px"], cfg["height_px"],
    )
    r_cam_b = np.array(cfg["r_cam_b_m"])
    q_b2c = np.array(cfg["q_b2c"])
    landmarks = np.array(cfg["landmarks_fixed_m"]).reshape(-1, 3)

    ephemeris = _Ephemeris()
    epoch_tai = _epoch_tai(_core, config)
    rows = _truth_rows(truth, camera["t_s"])

    results = []
    for j, t_s in enumerate(camera["t_s"]):
        row = rows[j]
        tai = _core.tai_add_seconds(epoch_tai[0], epoch_tai[1], float(t_s))
        tdb_s = _tdb_s_at(_core, epoch_tai, float(t_s))
        _, v_earth = ephemeris.earth_ssb(tdb_s)
        beta = beta_vector(truth["v_mps"][row], v_earth)
        # Earth's body-fixed rotation is a pure time-scale function (dUT1 = 0
        # per FR-3); the central body sits at the origin of the propagation
        # frame, so its inertial position contributes nothing.
        dcm_i2t = np.array(_core.gcrf_to_itrf(tai[0], tai[1], 0.0)).reshape(3, 3)
        for k in range(len(landmarks)):
            results.append((
                j, k,
                pinhole.project_landmark(
                    camera["r_m"][j], camera["q_i2b"][j], q_b2c, r_cam_b,
                    intrinsics, np.zeros(3), dcm_i2t, landmarks[k], beta,
                ),
            ))
    return results


def _projections_from_log_alone(run, config, echo=None, truth_rows_shifted=0):
    """Recompute logged pixels from the LOG only -- header echo plus truth.

    The intrinsics and extrinsics come from the ``gnc.camera`` header echo
    and the pose from the ``truth`` channels, so neither side of the
    comparison reads the mission file. That is the distinction from
    ``_camera_projections``, which sources both from the resolved config and
    therefore cannot say anything about whether the log records the
    constants the core actually projected through.

    ``config`` still supplies the landmark table, the epoch, and the
    ephemeris path: those are shared INPUTS to both sides, disclosed here in
    the same spirit as the module docstring's shared-inputs list, and the
    criterion's clause is about intrinsics rather than landmarks.
    ``echo`` overrides the header echo so a mutation can be injected;
    ``truth_rows_shifted`` misaligns the truth row lookup for the same
    reason.
    """
    from star_reacher import _core, camera_echo

    truth = run.groups["truth"]
    camera = run.groups["sensors.camera"]
    if echo is None:
        echo = camera_echo(run.header)
    intrinsics = pinhole.Intrinsics(
        echo["fx_px"], echo["fy_px"], echo["cx_px"], echo["cy_px"],
        echo["width_px"], echo["height_px"],
    )
    q_b2c = np.asarray(echo["q_b2c"], dtype=float)
    r_cam_b = np.asarray(echo["r_cam_b_m"], dtype=float)
    landmarks = np.array(config["sensors"]["camera"]["landmarks_fixed_m"]).reshape(-1, 3)

    ephemeris = _Ephemeris()
    epoch_tai = _epoch_tai(_core, config)
    rows = _truth_rows(truth, camera["t_s"])

    results = []
    for j, t_s in enumerate(camera["t_s"]):
        row = rows[j] + truth_rows_shifted
        tai = _core.tai_add_seconds(epoch_tai[0], epoch_tai[1], float(t_s))
        tdb_s = _tdb_s_at(_core, epoch_tai, float(t_s))
        _, v_earth = ephemeris.earth_ssb(tdb_s)
        beta = beta_vector(truth["v_mps"][row], v_earth)
        dcm_i2t = np.array(_core.gcrf_to_itrf(tai[0], tai[1], 0.0)).reshape(3, 3)
        for k in range(len(landmarks)):
            results.append((
                j, k,
                pinhole.project_landmark(
                    truth["r_m"][row], truth["q_i2b"][row], q_b2c, r_cam_b,
                    intrinsics, np.zeros(3), dcm_i2t, landmarks[k], beta,
                ),
            ))
    return results


def _worst_pixel_residual(run, projections):
    """Worst absolute pixel residual against the logged ``px_uv``."""
    camera = run.groups["sensors.camera"]
    worst = 0.0
    for j, k, projection in projections:
        if not (np.isfinite(projection.u) and np.isfinite(projection.v)):
            return float("inf")
        worst = max(
            worst,
            abs(projection.u - camera["px_uv"][j][2 * k]),
            abs(projection.v - camera["px_uv"][j][2 * k + 1]),
        )
    return worst


def test_camera_echo_fixture_is_not_degenerate(optical_run):
    """The echoed constants are separately resolvable on this fixture.

    Guards the gate below against the failure shape this phase has found
    repeatedly: a fixture whose geometry lets one constant stand in for
    another, so a wrong echo reproduces the right pixels anyway. Each
    assertion pins a specific substitution the geometry must forbid.
    """
    _, run, _ = optical_run
    from star_reacher import camera_echo

    echo = camera_echo(run.header)
    # fx == fy would make an fx/fy swap invisible.
    assert echo["fx_px"] != echo["fy_px"]
    # A centred principal point is recoverable from the image dimensions, so
    # dropping cx/cy entirely would still reproduce the pixels.
    assert echo["cx_px"] != (echo["width_px"] - 1) / 2.0
    assert echo["cy_px"] != (echo["height_px"] - 1) / 2.0
    # An identity mount rotation would make a dropped or transposed q_b2c
    # undetectable; a zero mount station would do the same for r_cam_b.
    assert not np.allclose(echo["q_b2c"], [1.0, 0.0, 0.0, 0.0])
    assert np.linalg.norm(echo["r_cam_b_m"]) > 0.0


def test_camera_intrinsics_echo_reproduces_logged_pixels(optical_run):
    """Exit criterion 7, intrinsics clause: the echo is the real thing.

    The criterion asks that the camera hook's pose AND intrinsics equal the
    ``truth`` channels bit-exactly. Read literally the intrinsics half is
    ill-posed -- ``truth`` carries r, v, q, omega, and mass, so no truth
    channel is an intrinsic. What it can mean, and what this gates, is that
    the intrinsics the core used are recoverable from the log and are the
    ones the logged pixels were actually computed with.

    Both sides come off the log: intrinsics and extrinsics from the
    ``gnc.camera`` header echo, pose from the ``truth`` channels. Neither
    reads the mission file, so this is not a config-against-config
    comparison that would compare a value against itself.
    """
    _, run, config = optical_run
    worst = _worst_pixel_residual(run, _projections_from_log_alone(run, config))
    assert worst < PIXEL_TOL, (
        f"reconstructing the logged pixels from the header echo and the truth "
        f"channels leaves a worst residual of {worst:.6e} px, exceeding the "
        f"exit-criterion-7 gate of {PIXEL_TOL} px"
    )


def test_camera_echo_mutation_is_detected(optical_run):
    """The gate above cannot pass while ignoring the echo.

    Perturbs each echoed constant in turn and requires the reconstruction to
    move past the tolerance. Without this, a gate that silently fell back to
    the configuration would look identical from the outside -- which is
    exactly how a blind gate is built.
    """
    _, run, config = optical_run
    from star_reacher import camera_echo

    baseline = camera_echo(run.header)
    mutations = {
        # One part in 1e6 of the focal length moves a landmark near the image
        # edge by ~4e-4 px, far above the 1e-6 px gate.
        "fx_px": lambda e: dict(e, fx_px=e["fx_px"] * (1.0 + 1.0e-6)),
        "fy_px": lambda e: dict(e, fy_px=e["fy_px"] * (1.0 + 1.0e-6)),
        "cx_px": lambda e: dict(e, cx_px=e["cx_px"] + 1.0e-5),
        "cy_px": lambda e: dict(e, cy_px=e["cy_px"] + 1.0e-5),
        "q_b2c": lambda e: dict(e, q_b2c=np.array([e["q_b2c"][0], e["q_b2c"][1],
                                                   e["q_b2c"][2],
                                                   e["q_b2c"][3] + 1.0e-9])),
        "r_cam_b_m": lambda e: dict(
            e, r_cam_b_m=e["r_cam_b_m"] + np.array([1.0e-3, 0.0, 0.0])
        ),
    }
    for name, mutate in mutations.items():
        worst = _worst_pixel_residual(
            run, _projections_from_log_alone(run, config, echo=mutate(baseline))
        )
        assert worst > PIXEL_TOL, (
            f"perturbing the echoed {name} left a worst residual of "
            f"{worst:.6e} px, inside the {PIXEL_TOL} px gate: the criterion-7 "
            f"intrinsics gate is blind to {name}"
        )


def test_landmark_pixels_match_independent_reference(optical_run):
    """Exit criterion 7: logged pixels to < 1e-6 px of the reference.

    The reference is ``tests/refs/pinhole.project_landmark``, written blind
    from ``ch:camera``. A projection reference is largely a statement about
    conventions -- pixel origin, DCM chaining order, which quaternion is
    conjugated -- so agreement here is evidence that the C++ and the chapter
    describe the same camera, not merely the same arithmetic.
    """
    _, run, config = optical_run
    camera = run.groups["sensors.camera"]
    worst = 0.0
    for j, k, projection in _camera_projections(run, config):
        logged_u = camera["px_uv"][j][2 * k]
        logged_v = camera["px_uv"][j][2 * k + 1]
        assert np.isfinite(projection.u) and np.isfinite(projection.v), (
            f"reference declined to project landmark {k} at record {j}; the "
            "core logged a finite pixel there"
        )
        worst = max(worst, abs(projection.u - logged_u), abs(projection.v - logged_v))
    assert worst < PIXEL_TOL, (
        f"worst logged-versus-reference pixel residual {worst:.6e} px exceeds "
        f"the exit-criterion-7 gate of {PIXEL_TOL} px"
    )


def test_landmarks_are_actually_visible(optical_run):
    """The criterion-7 scenario resolves real, on-sensor landmarks.

    ``ch:camera`` gates criterion 7 on VISIBLE landmarks. The core logs a
    projection whether or not the landmark is visible, so without this check
    the pixel gate above could be satisfied entirely by off-sensor numbers
    that no consumer would ever use.
    """
    _, run, config = optical_run
    visible = sum(1 for _, _, p in _camera_projections(run, config) if p.visible)
    total = len(run.groups["sensors.camera"]) * 5
    assert visible > total // 2, (
        f"only {visible} of {total} landmark projections pass the reference's "
        "own visibility tests; the criterion-7 gate would be near-vacuous"
    )


def test_camera_pose_channels_are_bit_exact_truth(optical_run):
    """Exit criterion 7, bit-exactness clause, re-confirmed on this mission.

    The clause holds by construction -- the hook copies the truth doubles
    rather than recomputing them (``ch:camera`` implementation note 2) -- so
    this is array equality, not a tolerance. Re-asserted here because the
    property is about the code path, and this mission exercises a different
    one (ephemeris loaded, third bodies on) than the suite that first
    established it.
    """
    _, run, _ = optical_run
    camera = run.groups["sensors.camera"]
    truth = run.groups["truth"]
    rows = _truth_rows(truth, camera["t_s"])
    assert np.array_equal(camera["r_m"], truth["r_m"][rows])
    assert np.array_equal(camera["q_i2b"], truth["q_i2b"][rows])


# --- Exit criterion 6 re-gated through the blind references -----------------

# The noise-free mission above isolates the transformations; this variant
# restores representative sigmas so the per-sample statistics have a
# distribution to be gated against. The altimeter bias sigma is zeroed
# deliberately: a per-run bias makes the samples DEPENDENT and invalidates the
# ensemble-mean gate, as ch:sensors-radio states and tests/refs/sensor_stats.py
# repeats.
_STATISTICS_MISSION = _OPTICAL_GATE_MISSION.replace(
    'name = "p6-optical-gates"', 'name = "p6-optical-statistics"'
).replace(
    "sigma_rad = [0.0, 0.0, 0.0]", "sigma_rad = [1.0e-5, 1.0e-5, 5.0e-5]"
).replace(
    "sigma_rad = 0.0", "sigma_rad = 2.0e-3"
).replace(
    # Inclined 45 degrees at the same speed. The criterion-7/9 mission is
    # equatorial, and on an equatorial orbit the geodetic altitude of
    # eq:radio:alt degenerates to |r| - a exactly, so the altimeter gate
    # below would pass unchanged against a spherical radius and could not
    # detect an implementation that never applied the ellipsoid. Inclining
    # carries the ground track off the equator within the 60 s window, where
    # the two altitudes differ by tens of metres against a 0.5 m sigma.
    "v_mps = [0.0, 7546.0, 0.0]",
    "v_mps = [0.0, 5335.827770833687, 5335.827770833687]",
) + """
[sensors.navfix]
sample_rate_hz = 5
sigma_r_m = [5.0, 5.0, 9.0]
sigma_v_mps = [0.05, 0.05, 0.09]

[sensors.altimeter]
sample_rate_hz = 5
sigma_bias_m = 0.0
sigma_noise_m = 0.5
h_min_m = 0.0
h_max_m = 1.0e6
"""

# WGS84 defining parameters (NIMA TR8350.2, third edition), the ellipsoid the
# environment model resolves for an Earth central body.
_WGS84_A_M = 6378137.0
_WGS84_INV_F = 298.257223563


@pytest.fixture(scope="module")
def statistics_run(tmp_path_factory):
    """One run of the noisy variant plus its loaded log."""
    _core_or_fail()
    import os

    from star_reacher import load
    from star_reacher.runner import run_mission

    tmp = tmp_path_factory.mktemp("p6_statistics")
    mission = tmp / "optical_statistics.toml"
    mission.write_text(_STATISTICS_MISSION.lstrip(), encoding="utf-8")
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        result = run_mission(mission, tmp / "run")
    finally:
        os.chdir(cwd)
    return result, load(result.srlog_path), tomllib.loads(mission.read_text("utf-8"))


def _assert_gate(result):
    """Fail with the reference's own one-line report when a gate is rejected."""
    assert result.passed, result.describe()


def test_star_tracker_statistic_passes_the_reference_gate(statistics_run):
    """Exit criterion 1, star tracker, re-gated against the blind reference.

    The star tracker's chi-square clause belongs to criterion 1, not to
    criterion 6 -- criterion 6 names the external nav fix and the altimeter,
    which are re-gated further down this module.

    The C++ suite closed this criterion with its own statistics. Recomputing
    the per-sample chi2(3) through ``tests/refs/sensor_stats`` is stronger
    evidence, because the extraction of the error vector runs the aberration
    factor of ``eq:optical:qab`` backwards out of the logged quaternion: a
    wrong ``q_ab`` would appear here as a mean offset, not merely as noise.
    """
    import sensor_stats as stats
    from star_reacher import _core

    _, run, config = statistics_run
    truth = run.groups["truth"]
    tracker = run.groups["sensors.startracker"]
    cfg = config["sensors"]["startracker"]
    sigmas = np.array(cfg["sigma_rad"])
    boresight_b = np.array(cfg["boresight_b"])

    ephemeris = _Ephemeris()
    epoch_tai = _epoch_tai(_core, config)
    rows = _truth_rows(truth, tracker["t_s"])

    quadratic = np.empty(len(tracker))
    for j, t_s in enumerate(tracker["t_s"]):
        row = rows[j]
        tdb_s = _tdb_s_at(_core, epoch_tai, float(t_s))
        _, v_earth = ephemeris.earth_ssb(tdb_s)
        beta = beta_vector(truth["v_mps"][row], v_earth)
        q_true = truth["q_i2b"][row]
        c_i2b = np.array(_core.quat_to_dcm(*q_true)).reshape(3, 3)
        boresight_i = c_i2b.T @ boresight_b
        q_ab = stats.aberration_quaternion(
            stats.aberration_rotation_vector(boresight_i, beta)
        )
        quadratic[j] = stats.star_tracker_chi2(
            tracker["q_meas_i2b"][j], q_true, sigmas, q_ab
        )
    _assert_gate(stats.evaluate_gate("startracker", quadratic, 3))


def test_sun_sensor_statistic_passes_the_reference_gate(statistics_run):
    """Sun sensor, re-gated against the blind reference. SUPPLEMENTARY.

    No exit criterion requires this one: criterion 1 names the star tracker
    and criterion 6 the external nav fix and the altimeter. It is kept as
    supplementary coverage of the sun-sensor noise model, and the docstring
    says so rather than borrowing a criterion number it does not close.

    Sensitivity note, established by mutation: this gate is a statement about
    the noise model ONLY. At the configured 2 mrad sigma the 1e-4 rad
    aberration displaces the statistic by (beta/sigma)**2 = 0.0025 against a
    sampling standard error of 0.115, so feeding the reference the
    UNABERRATED true direction still passes. Do not read a green result here
    as corroborating the aberration path; the star tracker above does that,
    with a sigma an order of magnitude below the displacement, and exit
    criterion 9 gates it directly.
    """
    import sensor_stats as stats
    from star_reacher import _core

    _, run, config = statistics_run
    truth = run.groups["truth"]
    sun = run.groups["sensors.sunsensor"]
    sigma = float(config["sensors"]["sunsensor"]["sigma_rad"])

    ephemeris = _Ephemeris()
    epoch_tai = _epoch_tai(_core, config)
    rows = _truth_rows(truth, sun["t_s"])

    quadratic = np.empty(len(sun))
    for j, t_s in enumerate(sun["t_s"]):
        row = rows[j]
        tdb_s = _tdb_s_at(_core, epoch_tai, float(t_s))
        _, v_earth = ephemeris.earth_ssb(tdb_s)
        beta = beta_vector(truth["v_mps"][row], v_earth)
        geometric = ephemeris.sun_rel_earth(tdb_s) - truth["r_m"][row]
        geometric = geometric / np.linalg.norm(geometric)
        c_i2b = np.array(_core.quat_to_dcm(*truth["q_i2b"][row])).reshape(3, 3)
        u_body_true = c_i2b @ aberrate_first_order(geometric, beta)
        quadratic[j] = stats.sun_sensor_chi2(sun["sun_b"][j], u_body_true, sigma)
    # chi2(2): the radial component carries no information after the
    # normalization of eq:optical:sunsensor.
    _assert_gate(stats.evaluate_gate("sunsensor", quadratic, 2))


def test_nav_fix_statistics_pass_the_reference_gate(statistics_run):
    """Exit criterion 6, external nav fix, position and velocity separately."""
    import sensor_stats as stats

    _, run, config = statistics_run
    truth = run.groups["truth"]
    fix = run.groups["sensors.navfix"]
    cfg = config["sensors"]["navfix"]
    sigma_r = np.array(cfg["sigma_r_m"])
    sigma_v = np.array(cfg["sigma_v_mps"])
    rows = _truth_rows(truth, fix["t_s"])

    position = np.array([
        stats.nav_fix_chi2(fix["r_meas_m"][j], truth["r_m"][row], sigma_r)
        for j, row in enumerate(rows)
    ])
    velocity = np.array([
        stats.nav_fix_chi2(fix["v_meas_mps"][j], truth["v_mps"][row], sigma_v)
        for j, row in enumerate(rows)
    ])
    # Gated separately rather than as one chi2(6): a position fix wired to the
    # wrong truth row would otherwise be diluted by a healthy velocity fix.
    _assert_gate(stats.evaluate_gate("navfix.position", position, 3))
    _assert_gate(stats.evaluate_gate("navfix.velocity", velocity, 3))


def test_altimeter_statistic_passes_the_reference_gate(statistics_run):
    """Exit criterion 6, altimeter, re-gated against the blind reference.

    The truth altitude is recomputed from the logged inertial position through
    the body-fixed rotation and the WGS84 ellipsoid, so this also checks that
    the sensor measures geodetic altitude rather than radius.
    """
    import sensor_stats as stats
    from star_reacher import _core

    _, run, config = statistics_run
    truth = run.groups["truth"]
    altimeter = run.groups["sensors.altimeter"]
    cfg = config["sensors"]["altimeter"]
    assert cfg["sigma_bias_m"] == 0.0  # the gate is invalid with a run bias
    sigma = float(cfg["sigma_noise_m"])

    epoch_tai = _epoch_tai(_core, config)
    rows = _truth_rows(truth, altimeter["t_s"])

    quadratic = np.empty(len(altimeter))
    for j, t_s in enumerate(altimeter["t_s"]):
        tai = _core.tai_add_seconds(epoch_tai[0], epoch_tai[1], float(t_s))
        dcm = np.array(_core.gcrf_to_itrf(tai[0], tai[1], 0.0)).reshape(3, 3)
        r_fixed = dcm @ truth["r_m"][rows[j]]
        h_true = _core.geodetic_altitude(list(r_fixed), _WGS84_A_M, _WGS84_INV_F)
        quadratic[j] = stats.altimeter_chi2(
            float(altimeter["alt_meas_m"][j]), h_true, sigma
        )
    _assert_gate(stats.evaluate_gate("altimeter", quadratic, 1))
