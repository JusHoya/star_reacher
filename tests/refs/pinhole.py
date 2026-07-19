"""Independent pinhole-projection reference for Phase 6 exit criterion 7.

Part of the Phase 6 independent-reference set (``tests/refs/manifest.toml``):
written from Chapter ``ch:camera`` and the computer-vision pinhole model of
Hartley and Zisserman (2004), with no reference to the core camera hook it
gates. Exit criterion 7 requires the core's landmark pixel projections to match
an independent NumPy pinhole script to better than 1e-6 pixels.

Conventions restated from the chapter, because a projection reference is
entirely a statement about conventions:

* Camera frame ``C``: ``+Z`` along the boresight out toward the scene, ``+X``
  toward increasing image column ``u`` (right), ``+Y`` toward increasing image
  row ``v`` (down); a right-handed triad.
* Intrinsics ``K`` of equation ``eq:camera:K`` with ``(f_x, f_y, c_x, c_y)`` in
  pixels and zero skew.
* Pixel origin at the CENTRE of the top-left pixel, so the physical sensor spans
  ``u in [-1/2, W - 1/2]`` and ``v in [-1/2, H - 1/2]``. A point on the optical
  axis projects to exactly ``(c_x, c_y)``.
* Chain ``q_i2c = q_i2b (x) q_b2c`` and
  ``r_cam^I = r^I + C_I2B(q_i2b)^T r_cam^B`` (equation ``eq:camera:pose``), with
  ``r_cam^B`` measured relative to the composite centre of mass -- the point the
  ``truth`` channels carry.
* Directions carry the velocity-aberration correction of
  ``eq:optical:aberration``; the camera-frame point is the apparent direction
  carried at the GEOMETRIC range (equation ``eq:camera:apparent``), so range is
  unaffected by aberration while bearing is.
* Projection is the exact central projection of equation ``eq:camera:proj``:
  ``u = f_x X/Z + c_x``, ``v = f_y Y/Z + c_y``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aberration import aberrate_first_order
from quaternions import quat_mul, quat_to_dcm


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics and image size, equation ``eq:camera:K``."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def __post_init__(self) -> None:
        # The FR-15 validator rejects these before a run starts; the reference
        # repeats the check so a malformed fixture fails here, not silently.
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError(f"focal lengths must be positive, got fx={self.fx}, fy={self.fy}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"image dimensions must be positive, got {self.width}x{self.height}"
            )

    def matrix(self) -> np.ndarray:
        """The 3x3 ``K`` of equation ``eq:camera:K``."""
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ]
        )

    @staticmethod
    def from_horizontal_fov(
        half_fov_rad: float, width: int, height: int
    ) -> "Intrinsics":
        """Square-pixel intrinsics for horizontal half field of view ``alpha_h``.

        Chapter ``ch:camera``, section ``sec:camera:intrinsics``:
        ``f_x = f_y = (W/2) / tan(alpha_h)``. The principal point is placed at
        the sensor centre, which under the pixel-CENTRE origin convention is
        ``((W-1)/2, (H-1)/2)`` -- not ``(W/2, H/2)``, the classic off-by-half
        error this convention exists to prevent.
        """
        if not 0.0 < half_fov_rad < 0.5 * np.pi:
            raise ValueError(f"half field of view must lie in (0, pi/2), got {half_fov_rad}")
        f = (width / 2.0) / np.tan(half_fov_rad)
        return Intrinsics(f, f, (width - 1) / 2.0, (height - 1) / 2.0, width, height)


def camera_position_i(
    r_i: np.ndarray, q_i2b: np.ndarray, r_cam_b: np.ndarray
) -> np.ndarray:
    """Camera origin in inertial axes, equation ``eq:camera:pose``.

    ``C_I2B^T`` maps the body-frame mount offset into inertial axes; the
    transpose is what turns the frame transformation into the vector rotation.
    """
    return np.asarray(r_i, dtype=float) + quat_to_dcm(q_i2b).T @ np.asarray(
        r_cam_b, dtype=float
    )


def q_i2c(q_i2b: np.ndarray, q_b2c: np.ndarray) -> np.ndarray:
    """Inertial-to-camera attitude, equation ``eq:camera:pose``."""
    return quat_mul(q_i2b, q_b2c)


def landmark_position_i(
    r_target_i: np.ndarray, dcm_i2t: np.ndarray, landmark_t: np.ndarray
) -> np.ndarray:
    """Inertial position of a body-fixed landmark, equation ``eq:camera:landmark``."""
    dcm = np.asarray(dcm_i2t, dtype=float)
    if dcm.shape != (3, 3):
        raise ValueError(f"C_I2T must be 3x3, got {dcm.shape}")
    return np.asarray(r_target_i, dtype=float) + dcm.T @ np.asarray(landmark_t, dtype=float)


def camera_frame_point(
    r_cam_i: np.ndarray,
    q_i2b: np.ndarray,
    q_b2c: np.ndarray,
    r_landmark_i: np.ndarray,
    beta: np.ndarray | None = None,
) -> np.ndarray:
    """Camera-frame landmark point, equation ``eq:camera:apparent``.

    The apparent unit direction is carried at the geometric range ``|d^I|``, so
    aberration rotates the bearing without changing the range. Passing
    ``beta=None`` disables the correction, which isolates the pure pinhole path
    for the convention tests.
    """
    d_i = np.asarray(r_landmark_i, dtype=float) - np.asarray(r_cam_i, dtype=float)
    range_m = float(np.linalg.norm(d_i))
    if range_m == 0.0:
        raise ValueError("landmark coincides with the camera origin")
    u_i = d_i / range_m
    if beta is not None:
        u_i = aberrate_first_order(u_i, beta)
    # C_B2C C_I2B resolves an inertial direction into camera axes; the DCM
    # product order is the frame-chaining order, not the quaternion one.
    return range_m * (quat_to_dcm(q_b2c) @ (quat_to_dcm(q_i2b) @ u_i))


def project(point_c: np.ndarray, intrinsics: Intrinsics) -> tuple[float, float]:
    """Central projection to pixels, equation ``eq:camera:proj``."""
    p = np.asarray(point_c, dtype=float)
    if p.shape != (3,):
        raise ValueError(f"camera-frame point must be shape (3,), got {p.shape}")
    if p[2] == 0.0:
        raise ValueError("camera-frame point lies in the focal plane (Z = 0)")
    u = intrinsics.fx * p[0] / p[2] + intrinsics.cx
    v = intrinsics.fy * p[1] / p[2] + intrinsics.cy
    return float(u), float(v)


def unproject(u: float, v: float, intrinsics: Intrinsics) -> np.ndarray:
    """Unit camera-frame direction through pixel ``(u, v)``.

    The exact inverse of :func:`project` up to range; used only to verify the
    projection round-trips, which is what pins the intrinsics convention.
    """
    x = (u - intrinsics.cx) / intrinsics.fx
    y = (v - intrinsics.cy) / intrinsics.fy
    d = np.array([x, y, 1.0])
    return d / np.linalg.norm(d)


def within_sensor(u: float, v: float, intrinsics: Intrinsics) -> bool:
    """Sensor-bounds test 2 of section ``sec:camera:visibility``.

    Bounds are the half-pixel-extended interval implied by the pixel-CENTRE
    origin: ``[-1/2, W - 1/2]`` and ``[-1/2, H - 1/2]``, inclusive.
    """
    return (
        -0.5 <= u <= intrinsics.width - 0.5 and -0.5 <= v <= intrinsics.height - 0.5
    )


def near_side(
    r_cam_i: np.ndarray, r_landmark_i: np.ndarray, r_target_i: np.ndarray
) -> bool:
    """Near-side test 3, equation ``eq:camera:nearside``.

    ``(r_cam - r_l) . (r_l - r_T) > 0``: the camera lies above the landmark's
    local tangent plane. For a spherical body and a surface landmark this is
    exactly equivalent to non-occlusion by the target body itself.
    """
    cam = np.asarray(r_cam_i, dtype=float)
    lm = np.asarray(r_landmark_i, dtype=float)
    tgt = np.asarray(r_target_i, dtype=float)
    return bool(np.dot(cam - lm, lm - tgt) > 0.0)


def occluded_by_sphere(
    r_cam_i: np.ndarray,
    r_landmark_i: np.ndarray,
    center_i: np.ndarray,
    radius_m: float,
) -> bool:
    """Third-body occlusion test 4, equation ``eq:camera:occlusion``.

    The closest point of the camera-to-landmark SEGMENT (parameter clamped to
    ``[0, 1]``) to the sphere centre is tested against the radius; clamping is
    what makes it a segment test rather than an infinite-line test, so a sphere
    behind the camera or beyond the landmark never occludes.
    """
    cam = np.asarray(r_cam_i, dtype=float)
    d = np.asarray(r_landmark_i, dtype=float) - cam
    center = np.asarray(center_i, dtype=float)
    denom = float(np.dot(d, d))
    if denom == 0.0:
        raise ValueError("landmark coincides with the camera origin")
    s = float(np.clip(np.dot(center - cam, d) / denom, 0.0, 1.0))
    closest = cam + s * d
    return bool(float(np.dot(closest - center, closest - center)) < radius_m * radius_m)


@dataclass(frozen=True)
class Projection:
    """Result of the full landmark chain: pixel, camera-frame point, visibility."""

    u: float
    v: float
    point_c: np.ndarray
    range_m: float
    visible: bool


def project_landmark(
    r_i: np.ndarray,
    q_i2b: np.ndarray,
    q_b2c: np.ndarray,
    r_cam_b: np.ndarray,
    intrinsics: Intrinsics,
    r_target_i: np.ndarray,
    dcm_i2t: np.ndarray,
    landmark_t: np.ndarray,
    beta: np.ndarray | None = None,
    occluders: tuple[tuple[np.ndarray, float], ...] = (),
) -> Projection:
    """Full equation ``eq:camera:landmark`` to ``eq:camera:proj`` chain.

    Visibility applies the four tests of section ``sec:camera:visibility`` in
    the stated order. Pixel values are returned even when a later test fails,
    matching the chapter's logging semantics (consumers filter on the flag).
    """
    r_cam_i = camera_position_i(r_i, q_i2b, r_cam_b)
    r_lm_i = landmark_position_i(r_target_i, dcm_i2t, landmark_t)
    point_c = camera_frame_point(r_cam_i, q_i2b, q_b2c, r_lm_i, beta)
    range_m = float(np.linalg.norm(r_lm_i - r_cam_i))

    if point_c[2] <= 0.0:
        # Behind the camera: the projection is undefined, so report the
        # degenerate pixel as NaN rather than the mirrored point a naive
        # division would produce.
        return Projection(float("nan"), float("nan"), point_c, range_m, False)

    u, v = project(point_c, intrinsics)
    visible = (
        within_sensor(u, v, intrinsics)
        and near_side(r_cam_i, r_lm_i, r_target_i)
        and not any(
            occluded_by_sphere(r_cam_i, r_lm_i, center, radius)
            for center, radius in occluders
        )
    )
    return Projection(u, v, point_c, range_m, visible)
