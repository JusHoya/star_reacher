// Rotation-kernel golden-vector and property tests (FR-3, D-7; FR-22
// layers 1 and 2). Reference values come from tests/golden/rotations/ -
// provenance and tolerances in that directory's manifest.toml. Test IDs
// are cited by the math-library validation table (ch:rotations); do not
// rename them.
#include <cmath>
#include <string>
#include <vector>

#include "golden_io.hpp"
#include "star/rng.hpp"
#include "star/rotation.hpp"
#include "vendor/doctest.h"

namespace {

namespace rot = star::rotation;

std::string golden_path(const char* file) {
  return std::string(STAR_GOLDEN_DIR) + "/rotations/" + file;
}

double d(const star_tests::GoldenCase& c, const char* key) {
  return star_tests::parse_hex_double(c.scalar(key));
}

Eigen::Matrix3d golden_matrix(const star_tests::GoldenCase& c) {
  Eigen::Matrix3d m;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      const std::string key = "c" + std::to_string(i) + std::to_string(j);
      m(i, j) = d(c, key.c_str());
    }
  }
  return m;
}

double max_abs_diff(const Eigen::Matrix3d& a, const Eigen::Matrix3d& b) {
  return (a - b).cwiseAbs().maxCoeff();
}

// Rotation-equivalent distance between two unit quaternions: the angle of
// the relative rotation q_err = conj(a) (x) b, insensitive to the q ~ -q
// sign ambiguity. This is the "rad equivalent" metric of Phase 2 exit
// criterion 6.
double rotation_angle_between(const Eigen::Quaterniond& a,
                              const Eigen::Quaterniond& b) {
  const Eigen::Quaterniond e = rot::quat_multiply(rot::quat_conjugate(a), b);
  const double vn =
      std::sqrt(e.x() * e.x() + e.y() * e.y() + e.z() * e.z());
  return 2.0 * std::atan2(vn, std::fabs(e.w()));
}

// Deterministic random unit quaternions: four Box-Muller normals on a
// named PCG64 stream (D-9). A 4-D Gaussian normalized to the sphere is
// uniform on SO(3); the seed is committed so every run draws the same
// attitudes.
class AttitudeSampler {
 public:
  AttitudeSampler()
      : sampler_(star::rng::make_stream(20260702ULL, "tests.rotation")) {}

  Eigen::Quaterniond next() {
    const double w = sampler_.next();
    const double x = sampler_.next();
    const double y = sampler_.next();
    const double z = sampler_.next();
    return rot::quat_normalize(Eigen::Quaterniond(w, x, y, z));
  }

  double uniform_angle() {
    // A single normal folded through atan is adequate for a covering
    // spread of angles; only determinism matters here.
    return 2.0 * std::atan(sampler_.next());
  }

 private:
  star::rng::NormalSampler sampler_;
};

// One full quaternion -> DCM -> Euler -> DCM -> quaternion round trip in
// the criterion-6a metric, for either sequence.
double euler_roundtrip_error(const Eigen::Quaterniond& q, bool seq321) {
  const Eigen::Matrix3d c = rot::dcm_from_quat(q);
  double a1;
  double a2;
  double a3;
  Eigen::Matrix3d c2;
  if (seq321) {
    rot::euler321_from_dcm(c, a1, a2, a3);
    c2 = rot::dcm_from_euler321(a1, a2, a3);
  } else {
    rot::euler313_from_dcm(c, a1, a2, a3);
    c2 = rot::dcm_from_euler313(a1, a2, a3);
  }
  return rotation_angle_between(q, rot::quat_from_dcm(c2));
}

}  // namespace

TEST_CASE("rotation_quat_dcm_golden") {
  const auto cases =
      star_tests::load_golden_cases(golden_path("quat_dcm.toml"));
  REQUIRE(cases.size() == 11);
  for (const auto& c : cases) {
    CAPTURE(c.scalar("name"));
    const Eigen::Quaterniond q(d(c, "qw"), d(c, "qx"), d(c, "qy"),
                               d(c, "qz"));
    const Eigen::Matrix3d golden = golden_matrix(c);

    // Forward mapping vs the double-constructed golden (manifest: 1e-15).
    CHECK(max_abs_diff(rot::dcm_from_quat(q), golden) <= 1e-15);

    // Shepperd extraction (the four *_dominant cases exercise each pivot
    // branch). q and -q are one rotation: compare up to overall sign and
    // assert the w >= 0 output convention separately.
    const Eigen::Quaterniond back = rot::quat_from_dcm(golden);
    CHECK(back.w() >= 0.0);
    const double sign =
        (back.w() * q.w() + back.x() * q.x() + back.y() * q.y() +
         back.z() * q.z()) < 0.0
            ? -1.0
            : 1.0;
    CHECK(std::fabs(back.w() - sign * q.w()) <= 1e-15);
    CHECK(std::fabs(back.x() - sign * q.x()) <= 1e-15);
    CHECK(std::fabs(back.y() - sign * q.y()) <= 1e-15);
    CHECK(std::fabs(back.z() - sign * q.z()) <= 1e-15);

    // Vector transformation is the same operation as the DCM by
    // construction; pin it against the golden matrix on the basis vectors.
    for (int axis = 0; axis < 3; ++axis) {
      const Eigen::Vector3d v = Eigen::Vector3d::Unit(axis);
      CHECK((rot::quat_transform(q, v) - golden * v).cwiseAbs().maxCoeff() <=
            1e-15);
    }
  }
}

TEST_CASE("rotation_euler_golden") {
  const auto cases = star_tests::load_golden_cases(golden_path("euler.toml"));
  REQUIRE(cases.size() == 20);
  for (const auto& c : cases) {
    CAPTURE(c.scalar("name"));
    const double a1 = d(c, "a1");
    const double a2 = d(c, "a2");
    const double a3 = d(c, "a3");
    const Eigen::Matrix3d golden = golden_matrix(c);
    const bool seq321 = c.scalar("sequence") == "321";
    const Eigen::Matrix3d built = seq321 ? rot::dcm_from_euler321(a1, a2, a3)
                                         : rot::dcm_from_euler313(a1, a2, a3);
    // Manifest tolerance: 1e-15 per element vs the ERFA-composed golden.
    CHECK(max_abs_diff(built, golden) <= 1e-15);
  }
}

TEST_CASE("rotation_primitive_axes_hand_values") {
  // Frame-rotation convention hand checks (eq:rotations:r1r2r3): after
  // rotating the frame by +90 deg about +z, the fixed vector x-hat has
  // coordinates -y-hat in the new frame, and so on cyclically.
  const double half_pi = 1.5707963267948966;  // pi/2 rounded to binary64
  const Eigen::Vector3d ex = Eigen::Vector3d::UnitX();
  const Eigen::Vector3d ey = Eigen::Vector3d::UnitY();
  const Eigen::Vector3d ez = Eigen::Vector3d::UnitZ();

  CHECK(((rot::r3(half_pi) * ex) - (-ey)).norm() <= 1e-15);
  CHECK(((rot::r3(half_pi) * ey) - ex).norm() <= 1e-15);
  CHECK(((rot::r3(half_pi) * ez) - ez).norm() <= 1e-15);
  CHECK(((rot::r1(half_pi) * ey) - (-ez)).norm() <= 1e-15);
  CHECK(((rot::r1(half_pi) * ez) - ey).norm() <= 1e-15);
  CHECK(((rot::r2(half_pi) * ez) - (-ex)).norm() <= 1e-15);
  CHECK(((rot::r2(half_pi) * ex) - ez).norm() <= 1e-15);

  // R3(30 deg) elements against hand values cos/sin(pi/6).
  const double a = 0.5235987755982988;  // pi/6 rounded to binary64
  const Eigen::Matrix3d m = rot::r3(a);
  CHECK(std::fabs(m(0, 0) - std::cos(a)) == 0.0);
  CHECK(std::fabs(m(0, 1) - std::sin(a)) == 0.0);
  CHECK(m(1, 0) == -m(0, 1));
  CHECK(m(2, 2) == 1.0);

  // The primitives agree with the equivalent axis quaternions
  // q = [cos(a/2), sin(a/2) e], all three axes.
  const double h = 0.5 * a;
  CHECK(max_abs_diff(rot::dcm_from_quat(Eigen::Quaterniond(
                         std::cos(h), std::sin(h), 0.0, 0.0)),
                     rot::r1(a)) <= 1e-15);
  CHECK(max_abs_diff(rot::dcm_from_quat(Eigen::Quaterniond(
                         std::cos(h), 0.0, std::sin(h), 0.0)),
                     rot::r2(a)) <= 1e-15);
  CHECK(max_abs_diff(rot::dcm_from_quat(Eigen::Quaterniond(
                         std::cos(h), 0.0, 0.0, std::sin(h))),
                     rot::r3(a)) <= 1e-15);
}

TEST_CASE("rotation_quat_algebra_properties") {
  AttitudeSampler draw;
  const Eigen::Quaterniond identity(1.0, 0.0, 0.0, 0.0);
  for (int i = 0; i < 100; ++i) {
    const Eigen::Quaterniond p = draw.next();
    const Eigen::Quaterniond q = draw.next();

    // Unit norm after normalize (a few ulp: one sqrt and four divisions).
    CHECK(std::fabs(p.w() * p.w() + p.x() * p.x() + p.y() * p.y() +
                    p.z() * p.z() - 1.0) <= 1e-15);

    // conj(p (x) q) = conj(q) (x) conj(p).
    const Eigen::Quaterniond lhs =
        rot::quat_conjugate(rot::quat_multiply(p, q));
    const Eigen::Quaterniond rhs =
        rot::quat_multiply(rot::quat_conjugate(q), rot::quat_conjugate(p));
    CHECK(rotation_angle_between(lhs, rhs) <= 1e-15);

    // p (x) conj(p) = identity.
    CHECK(rotation_angle_between(rot::quat_multiply(p, rot::quat_conjugate(p)),
                                 identity) <= 1e-15);

    // Composition rule of the notation chapter (eq:notation:quatcomp):
    // C(p (x) q) = C(q) C(p) for frame transformations q_a2b = p,
    // q_b2c = q.
    CHECK(max_abs_diff(rot::dcm_from_quat(rot::quat_multiply(p, q)),
                       rot::dcm_from_quat(q) * rot::dcm_from_quat(p)) <=
          5e-15);

    // The sandwich and the DCM act identically on vectors.
    const Eigen::Vector3d v(1.25, -0.5, 2.0);
    CHECK((rot::quat_transform(p, v) - rot::dcm_from_quat(p) * v)
              .cwiseAbs()
              .maxCoeff() <= 1e-15);
  }
}

TEST_CASE("rotation_dcm_orthonormality_properties") {
  // Property gate (FR-22 layer 2): every produced DCM is orthonormal to
  // machine precision with determinant +1 - random attitudes, Euler
  // compositions, and primitives alike.
  AttitudeSampler draw;
  std::vector<Eigen::Matrix3d> mats;
  for (int i = 0; i < 200; ++i) {
    mats.push_back(rot::dcm_from_quat(draw.next()));
  }
  for (int i = 0; i < 50; ++i) {
    const double a1 = draw.uniform_angle();
    const double a2 = draw.uniform_angle();
    const double a3 = draw.uniform_angle();
    mats.push_back(rot::dcm_from_euler321(a1, a2, a3));
    mats.push_back(rot::dcm_from_euler313(a1, a2, a3));
    mats.push_back(rot::r1(a1) * rot::r2(a2) * rot::r3(a3));
  }
  // Machine precision here means a few ulp of rounding through the 3x3
  // compositions (observed maximum ~1.1e-15); 2e-15 admits exactly that
  // while failing on any structural defect.
  for (const auto& m : mats) {
    CHECK((m.transpose() * m - Eigen::Matrix3d::Identity())
              .cwiseAbs()
              .maxCoeff() <= 2e-15);
    CHECK(std::fabs(m.determinant() - 1.0) <= 2e-15);
  }
}

TEST_CASE("rotation_quat_dcm_euler_roundtrip") {
  // Phase 2 exit criterion 6a: quaternion <-> DCM <-> Euler round trips
  // recover the input to <= 1e-13 rad equivalent over 1,000 random
  // attitudes (seeded, deterministic), including near-gimbal-lock draws.
  AttitudeSampler draw;
  double worst_dcm = 0.0;
  double worst_321 = 0.0;
  double worst_313 = 0.0;
  for (int i = 0; i < 1000; ++i) {
    const Eigen::Quaterniond q = draw.next();
    const double e_dcm =
        rotation_angle_between(q, rot::quat_from_dcm(rot::dcm_from_quat(q)));
    worst_dcm = std::max(worst_dcm, e_dcm);
    worst_321 = std::max(worst_321, euler_roundtrip_error(q, true));
    worst_313 = std::max(worst_313, euler_roundtrip_error(q, false));
  }
  CAPTURE(worst_dcm);
  CAPTURE(worst_321);
  CAPTURE(worst_313);
  CHECK(worst_dcm <= 1e-13);
  CHECK(worst_321 <= 1e-13);
  CHECK(worst_313 <= 1e-13);

  // Near-gimbal-lock draws inside the 1e-13 gate: attitudes 0.1 and 0.01
  // rad from the singular middle angle of each sequence, random first and
  // third angles.
  const double half_pi = 1.5707963267948966;
  double worst_lock_gate = 0.0;
  for (const double delta : {1e-1, 1e-2}) {
    for (int i = 0; i < 20; ++i) {
      const double a1 = draw.uniform_angle();
      const double a3 = draw.uniform_angle();
      for (const double sgn : {1.0, -1.0}) {
        const Eigen::Quaterniond q321 = rot::quat_from_dcm(
            rot::dcm_from_euler321(a1, sgn * (half_pi - delta), a3));
        worst_lock_gate =
            std::max(worst_lock_gate, euler_roundtrip_error(q321, true));
      }
      const Eigen::Quaterniond q_lo =
          rot::quat_from_dcm(rot::dcm_from_euler313(a1, delta, a3));
      const Eigen::Quaterniond q_hi = rot::quat_from_dcm(rot::dcm_from_euler313(
          a1, 3.141592653589793 - delta, a3));
      worst_lock_gate =
          std::max(worst_lock_gate, euler_roundtrip_error(q_lo, false));
      worst_lock_gate =
          std::max(worst_lock_gate, euler_roundtrip_error(q_hi, false));
    }
  }
  CAPTURE(worst_lock_gate);
  CHECK(worst_lock_gate <= 1e-13);

  // Documented degradation adjacent to lock (ch:rotations, domain of
  // validity): within delta of the singularity the individual first/third
  // angles are conditioned as eps/delta, so the recomposed rotation is
  // held to a K*eps/delta envelope (K = 100, conservative), not to 1e-13.
  const double eps = 2.220446049250313e-16;  // binary64 machine epsilon
  for (const double delta : {1e-4, 1e-6, 1e-8}) {
    const double envelope = 100.0 * eps / delta;
    double worst_near = 0.0;
    for (int i = 0; i < 10; ++i) {
      const double a1 = draw.uniform_angle();
      const double a3 = draw.uniform_angle();
      const Eigen::Quaterniond q321 = rot::quat_from_dcm(
          rot::dcm_from_euler321(a1, half_pi - delta, a3));
      worst_near = std::max(worst_near, euler_roundtrip_error(q321, true));
      const Eigen::Quaterniond q313 =
          rot::quat_from_dcm(rot::dcm_from_euler313(a1, delta, a3));
      worst_near = std::max(worst_near, euler_roundtrip_error(q313, false));
    }
    CAPTURE(delta);
    CAPTURE(worst_near);
    CHECK(worst_near <= envelope);
  }

  // Exact-lock policy: with the singular elements exactly zero (hand-built
  // matrices - trigonometric construction cannot produce them, since
  // cos(pi/2 rounded) = 6.1e-17), the extraction takes the deterministic
  // a1 = 0 branch and recomposes the rotation to machine precision.
  for (const double phi : {0.4, -2.0}) {
    const double sp = std::sin(phi);
    const double cp = std::cos(phi);
    // 3-2-1 at a2 = +pi/2 exactly: C = [0 0 -1; sin f, cos f, 0;
    // cos f, -sin f, 0] with f = a3 - a1.
    Eigen::Matrix3d up;
    up << 0.0, 0.0, -1.0,  //
        sp, cp, 0.0,       //
        cp, -sp, 0.0;
    double a1;
    double a2;
    double a3;
    rot::euler321_from_dcm(up, a1, a2, a3);
    CHECK(a1 == 0.0);
    CHECK(max_abs_diff(rot::dcm_from_euler321(a1, a2, a3), up) <= 5e-16);
    // 3-1-3 at a2 = 0 exactly: C = R3(a3 + a1).
    Eigen::Matrix3d flat;
    flat << cp, sp, 0.0,  //
        -sp, cp, 0.0,     //
        0.0, 0.0, 1.0;
    rot::euler313_from_dcm(flat, a1, a2, a3);
    CHECK(a1 == 0.0);
    CHECK(a2 == 0.0);
    CHECK(max_abs_diff(rot::dcm_from_euler313(a1, a2, a3), flat) <= 5e-16);
  }
}
