"""Validate the independent pinhole reference of ``tests/refs/pinhole.py``.

Phase 6 exit criterion 7 requires the core's landmark pixel projections to match
an independent NumPy pinhole script to better than 1e-6 pixels. A projection
reference is almost entirely a statement about CONVENTIONS -- axis directions,
pixel origin, quaternion composition order -- so this suite validates each
convention against something that is not the implementation:

* the quaternion algebra against its own group properties (orthonormality,
  determinant, the composition rule of eq:notation:quatcomp, exp/log inversion);
* the intrinsics convention against the analytic field-of-view relation: a ray
  at exactly the horizontal half field of view must land on the sensor edge that
  the pixel-centre convention defines, ``u = W - 1/2``;
* the projection against its own inverse (unproject/reproject round trip) at a
  tolerance three orders tighter than the 1e-6 px gate;
* the four visibility tests against constructed geometries that cross each
  boundary independently.

These tests are pure NumPy and require no compiled core.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO_ROOT / "tests" / "refs"))
import pinhole  # noqa: E402
import quaternions as qt  # noqa: E402
from aberration import SPEED_OF_LIGHT_MPS  # noqa: E402

# Exit criterion 7's gate, quoted so the tests assert against the real number.
PIXEL_TOL = 1e-6


def _random_unit_quat(rng: np.random.Generator) -> np.ndarray:
    return qt.quat_normalize(rng.standard_normal(4))


# --- Quaternion conventions (ch:notation) ----------------------------------


def test_dcm_is_orthonormal_with_unit_determinant():
    rng = np.random.default_rng(11)
    for _ in range(500):
        dcm = qt.quat_to_dcm(_random_unit_quat(rng))
        assert np.max(np.abs(dcm @ dcm.T - np.eye(3))) < 1e-14
        assert float(np.linalg.det(dcm)) == pytest.approx(1.0, abs=1e-14)


def test_composition_rule_reverses_in_the_dcm_product():
    """Equation eq:notation:quatcomp: ``C(p (x) q) = C(q) C(p)``.

    The reversal is what distinguishes this project's frame-transformation
    convention from the Markley--Crassidis one; getting it backwards is the
    defect the notation chapter exists to prevent, and it would be invisible in
    any single-rotation test.
    """
    rng = np.random.default_rng(12)
    for _ in range(500):
        p, q = _random_unit_quat(rng), _random_unit_quat(rng)
        lhs = qt.quat_to_dcm(qt.quat_mul(p, q))
        rhs = qt.quat_to_dcm(q) @ qt.quat_to_dcm(p)
        assert np.max(np.abs(lhs - rhs)) < 1e-14


def test_identity_and_inverse_behave_as_a_group():
    rng = np.random.default_rng(13)
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    for _ in range(200):
        q = _random_unit_quat(rng)
        assert qt.quat_mul(q, identity) == pytest.approx(q, abs=1e-15)
        assert qt.quat_mul(q, qt.quat_conj(q)) == pytest.approx(identity, abs=1e-15)
        assert np.max(np.abs(qt.quat_to_dcm(identity) - np.eye(3))) == 0.0


@pytest.mark.parametrize("scale", [1e-8, 1e-5, 1e-2, 0.5])
def test_exponential_and_logarithmic_maps_invert_each_other(scale):
    """The exactness eq:optical:noiseq and eq:optical:extract depend on.

    Rotation vectors are kept below pi in norm: beyond it the logarithmic map
    returns the equivalent rotation in ``(-pi, pi]`` rather than the input, which
    is correct behaviour and, as the chapter notes, has probability zero for
    in-domain sensor sigmas.
    """
    rng = np.random.default_rng(14)
    for _ in range(200):
        phi = rng.standard_normal(3) * scale
        assert np.linalg.norm(phi) < np.pi
        recovered = qt.rotation_vector_from_quat(qt.quat_from_rotation_vector(phi))
        # Absolute tolerance scaled to the rotation size: the map is exact to
        # rounding, so the residual tracks the magnitude of the input.
        assert np.max(np.abs(recovered - phi)) < 1e-14 * max(scale, 1e-8)


def test_zero_rotation_maps_to_the_identity_exactly():
    assert qt.quat_from_rotation_vector(np.zeros(3)) == pytest.approx(
        np.array([1.0, 0.0, 0.0, 0.0]), abs=0.0
    )
    assert qt.rotation_vector_from_quat(np.array([1.0, 0.0, 0.0, 0.0])) == pytest.approx(
        np.zeros(3), abs=0.0
    )


# --- Intrinsics and the pixel convention (sec:camera:intrinsics) ------------


def test_optical_axis_projects_to_the_principal_point_exactly():
    """A point on the boresight lands on ``(c_x, c_y)`` with no rounding at all."""
    intr = pinhole.Intrinsics(800.0, 900.0, 511.5, 383.5, 1024, 768)
    u, v = pinhole.project(np.array([0.0, 0.0, 12345.6]), intr)
    assert (u, v) == (intr.cx, intr.cy)


@pytest.mark.parametrize("half_fov_deg", [5.0, 20.0, 45.0])
def test_field_of_view_edge_lands_on_the_sensor_edge(half_fov_deg):
    """The analytic consistency check of the pixel-CENTRE origin convention.

    ``f = (W/2)/tan(alpha_h)`` with the principal point at ``(W-1)/2`` places a
    ray at exactly ``alpha_h`` off axis at ``u = W - 1/2`` -- the right edge of
    the physical sensor under this convention. Had the principal point been put
    at ``W/2`` (the pixel-CORNER convention) this test would miss by half a
    pixel, five orders above the 1e-6 px gate.
    """
    width, height = 1024, 768
    alpha = np.radians(half_fov_deg)
    intr = pinhole.Intrinsics.from_horizontal_fov(alpha, width, height)
    u, v = pinhole.project(np.array([np.tan(alpha), 0.0, 1.0]), intr)
    assert u == pytest.approx(width - 0.5, abs=1e-9)
    assert v == pytest.approx(intr.cy, abs=1e-12)
    # The opposite edge is the mirror image, which pins the sign of +X -> u.
    u_left, _ = pinhole.project(np.array([-np.tan(alpha), 0.0, 1.0]), intr)
    assert u_left == pytest.approx(-0.5, abs=1e-9)


def test_image_row_axis_points_downward():
    """``+Y^C`` maps to increasing ``v``, the CV convention of ch:camera."""
    intr = pinhole.Intrinsics(800.0, 800.0, 511.5, 383.5, 1024, 768)
    _, v_positive_y = pinhole.project(np.array([0.0, 1.0, 10.0]), intr)
    assert v_positive_y > intr.cy


def test_intrinsics_matrix_agrees_with_the_scalar_projection():
    """``K`` of eq:camera:K applied homogeneously must equal eq:camera:proj."""
    rng = np.random.default_rng(21)
    intr = pinhole.Intrinsics(812.3, 907.1, 500.25, 380.75, 1024, 768)
    k = intr.matrix()
    for _ in range(200):
        point = np.array([rng.normal(), rng.normal(), abs(rng.normal()) + 0.5])
        homogeneous = k @ point
        u_k, v_k = homogeneous[0] / homogeneous[2], homogeneous[1] / homogeneous[2]
        u, v = pinhole.project(point, intr)
        assert abs(u - u_k) < PIXEL_TOL * 1e-3
        assert abs(v - v_k) < PIXEL_TOL * 1e-3


def test_projection_round_trips_through_unprojection():
    """Reprojecting an unprojected pixel returns it, far inside the 1e-6 px gate."""
    rng = np.random.default_rng(22)
    intr = pinhole.Intrinsics(812.3, 907.1, 500.25, 380.75, 1024, 768)
    worst = 0.0
    for _ in range(2000):
        u = rng.uniform(-0.5, intr.width - 0.5)
        v = rng.uniform(-0.5, intr.height - 0.5)
        direction = pinhole.unproject(u, v, intr)
        u2, v2 = pinhole.project(direction * rng.uniform(1.0, 1e7), intr)
        worst = max(worst, abs(u2 - u), abs(v2 - v))
    assert worst < PIXEL_TOL * 1e-3, f"round-trip worst error {worst:.3e} px"


def test_rejects_a_degenerate_configuration():
    with pytest.raises(ValueError, match="focal lengths must be positive"):
        pinhole.Intrinsics(0.0, 900.0, 0.0, 0.0, 64, 64)
    with pytest.raises(ValueError, match="image dimensions must be positive"):
        pinhole.Intrinsics(900.0, 900.0, 0.0, 0.0, 0, 64)
    with pytest.raises(ValueError, match="focal plane"):
        pinhole.project(np.array([1.0, 1.0, 0.0]), pinhole.Intrinsics(1.0, 1.0, 0.0, 0.0, 4, 4))


# --- The full landmark chain (sec:camera:projection) -----------------------


def _scenario():
    """An off-axis mount with anisotropic focal lengths, per the chapter's gate.

    The configuration deliberately exercises every term the chapter's validation
    item names -- non-zero ``r_cam^B``, non-identity ``q_b2c``, and
    ``f_x != f_y``. The target and landmark are then PLACED relative to the
    resulting boresight (a spherical body at 4,000 km, a surface landmark
    slightly off axis) so the scene is well posed rather than accidentally
    behind the camera; the arbitrary part of the geometry is the attitude, not
    the visibility.
    """
    r_i = np.array([1.9e6, -3.0e5, 4.4e5])
    q_i2b = qt.quat_normalize(np.array([0.83, 0.12, -0.44, 0.31]))
    q_b2c = qt.quat_normalize(np.array([0.5, 0.5, -0.5, 0.5]))
    r_cam_b = np.array([0.85, -0.20, 1.35])
    intr = pinhole.Intrinsics(1520.5, 1498.25, 1023.5, 767.5, 2048, 1536)
    dcm_i2t = qt.quat_to_dcm(qt.quat_normalize(np.array([0.97, 0.0, 0.0, 0.24])))

    r_cam_i = pinhole.camera_position_i(r_i, q_i2b, r_cam_b)
    dcm_i2c = qt.quat_to_dcm(pinhole.q_i2c(q_i2b, q_b2c))
    # Boresight, target centre 4,000 km down it, and a surface landmark tilted
    # 0.3 rad off the sub-camera point so it is unambiguously on the near side.
    boresight_i = dcm_i2c.T @ np.array([0.0, 0.0, 1.0])
    off_axis_i = dcm_i2c.T @ np.array([1.0, 0.0, 0.0])
    r_target_i = r_cam_i + 4.0e6 * boresight_i
    tilt = 0.3
    radius = 1.7374e6
    landmark_i = r_target_i - radius * (
        np.cos(tilt) * boresight_i + np.sin(tilt) * off_axis_i
    )
    landmark_t = dcm_i2t @ (landmark_i - r_target_i)
    return r_i, q_i2b, q_b2c, r_cam_b, intr, r_target_i, dcm_i2t, landmark_t, radius


def test_landmark_chain_matches_a_directly_composed_projection():
    """Cross-check the chain against an independent single-expression evaluation.

    The chain in ``project_landmark`` walks eq:camera:pose, eq:camera:landmark,
    eq:camera:los, eq:camera:apparent, eq:camera:proj in sequence. Here the same
    result is formed in one composed expression through ``q_i2c`` instead, which
    exercises the composition rule rather than assuming it.
    """
    r_i, q_i2b, q_b2c, r_cam_b, intr, r_target_i, dcm_i2t, landmark_t, _ = _scenario()

    result = pinhole.project_landmark(
        r_i, q_i2b, q_b2c, r_cam_b, intr, r_target_i, dcm_i2t, landmark_t
    )

    r_cam_i = r_i + qt.quat_to_dcm(q_i2b).T @ r_cam_b
    r_lm_i = r_target_i + dcm_i2t.T @ landmark_t
    d_i = r_lm_i - r_cam_i
    point_c = qt.quat_to_dcm(pinhole.q_i2c(q_i2b, q_b2c)) @ d_i
    u = intr.fx * point_c[0] / point_c[2] + intr.cx
    v = intr.fy * point_c[1] / point_c[2] + intr.cy

    assert abs(result.u - u) < PIXEL_TOL * 1e-3
    assert abs(result.v - v) < PIXEL_TOL * 1e-3


def test_aberration_shifts_the_pixel_by_the_expected_scale():
    """Aberration rotates the bearing but must not change the range.

    A 20.5 arcsec rotation at focal length ``f`` displaces the image point by
    about ``f * 20.5 arcsec`` pixels; the test checks the magnitude sits in that
    band and that the emitted range is bit-identical with and without the
    correction (eq:camera:apparent carries the APPARENT direction at the
    GEOMETRIC range).
    """
    r_i, q_i2b, q_b2c, r_cam_b, intr, r_target_i, dcm_i2t, landmark_t, _ = _scenario()
    beta = np.array([0.0, 29780.0, 0.0]) / SPEED_OF_LIGHT_MPS

    plain = pinhole.project_landmark(
        r_i, q_i2b, q_b2c, r_cam_b, intr, r_target_i, dcm_i2t, landmark_t
    )
    aberrated = pinhole.project_landmark(
        r_i, q_i2b, q_b2c, r_cam_b, intr, r_target_i, dcm_i2t, landmark_t, beta=beta
    )

    assert aberrated.range_m == plain.range_m
    shift_px = np.hypot(aberrated.u - plain.u, aberrated.v - plain.v)
    max_expected = max(intr.fx, intr.fy) * float(np.linalg.norm(beta))
    assert 0.0 < shift_px <= max_expected * 1.001
    # The effect is real at this focal length: tenths of a pixel, five orders
    # above the 1e-6 px gate, so an omitted correction cannot hide inside it.
    assert shift_px > 1e-3


def test_visibility_behind_the_camera_is_rejected():
    r_i, q_i2b, q_b2c, r_cam_b, intr, r_target_i, dcm_i2t, _, _ = _scenario()
    r_cam_i = pinhole.camera_position_i(r_i, q_i2b, r_cam_b)
    boresight_i = qt.quat_to_dcm(pinhole.q_i2c(q_i2b, q_b2c)).T @ np.array([0.0, 0.0, 1.0])
    # Place a target one kilometre BEHIND the camera along its own boresight.
    behind = r_cam_i - 1000.0 * boresight_i
    result = pinhole.project_landmark(
        r_i, q_i2b, q_b2c, r_cam_b, intr, behind, np.eye(3), np.zeros(3)
    )
    assert not result.visible
    assert np.isnan(result.u) and np.isnan(result.v)


def test_sensor_bounds_use_the_half_pixel_convention():
    intr = pinhole.Intrinsics(800.0, 800.0, 511.5, 383.5, 1024, 768)
    assert pinhole.within_sensor(-0.5, -0.5, intr)
    assert pinhole.within_sensor(1023.5, 767.5, intr)
    assert not pinhole.within_sensor(-0.5001, 0.0, intr)
    assert not pinhole.within_sensor(1023.5001, 0.0, intr)
    assert not pinhole.within_sensor(0.0, 767.5001, intr)


def test_near_side_test_is_exactly_the_spherical_limb():
    """Equation eq:camera:nearside on a sphere is exactly the limb condition.

    For a camera at range ``d`` from the centre of a sphere of radius ``R``, the
    limb sits at a surface-point angle ``arccos(R/d)`` from the sub-camera point.
    The dot-product test must flip precisely there, which is the chapter's claim
    that no separate self-occlusion test is needed.
    """
    radius, distance = 1737400.0, 4.0e6
    center = np.zeros(3)
    cam = np.array([0.0, 0.0, distance])
    limb_angle = np.arccos(radius / distance)
    for delta, expected in ((-1e-9, True), (1e-9, False)):
        angle = limb_angle + delta
        landmark = radius * np.array([np.sin(angle), 0.0, np.cos(angle)])
        assert pinhole.near_side(cam, landmark, center) is expected


def test_third_body_occlusion_is_a_segment_test_not_a_line_test():
    """The clamp in eq:camera:occlusion is what makes it a SEGMENT test.

    A sphere sitting behind the camera lies on the infinite line but not on the
    segment, so it must not occlude; without the clamp it would.
    """
    cam = np.zeros(3)
    landmark = np.array([0.0, 0.0, 1000.0])
    on_segment = (np.array([0.0, 0.0, 500.0]), 10.0)
    behind_camera = (np.array([0.0, 0.0, -500.0]), 10.0)
    beyond_landmark = (np.array([0.0, 0.0, 1500.0]), 10.0)
    assert pinhole.occluded_by_sphere(cam, landmark, *on_segment)
    assert not pinhole.occluded_by_sphere(cam, landmark, *behind_camera)
    assert not pinhole.occluded_by_sphere(cam, landmark, *beyond_landmark)
    # And a sphere on the segment but off the line of sight does not occlude.
    assert not pinhole.occluded_by_sphere(cam, landmark, np.array([50.0, 0.0, 500.0]), 10.0)
