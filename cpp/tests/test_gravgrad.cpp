// Gravity-gradient torque tests (FR-1, FR-22 layers 1-3, Phase 4 exit
// criterion 9): golden torque vectors, structural exact-zero geometries,
// and the pitch-libration frequency gate against the analytic value
// (eq:gravgrad:libfreq) with the finite-amplitude pendulum correction
// (eq:gravgrad:pendulum) as the tight secondary check. Test IDs are cited
// by the math-library validation table (ch:gravgrad); do not rename them.
//
// The golden references in tests/golden/attitude/ are mpmath evaluations
// from the exact committed binary64 inputs (provenance:
// tests/golden/attitude/manifest.toml). The libration reference is
// DIFFERENT mathematics from the simulated path: the closed-form planar
// pendulum reduction (eq:gravgrad:pitch) derived in ch:gravgrad, so the
// analytic frequency cannot share a defect with the torque model or the
// integrator.
#include <cmath>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "golden_io.hpp"
#include "star/constants.hpp"
#include "star/events.hpp"
#include "star/integrate.hpp"
#include "star/models/gravgrad.hpp"
#include "star/models/rigidbody.hpp"
#include "star/rotation.hpp"
#include "vendor/doctest.h"

namespace {

const std::string kGoldenDir = STAR_GOLDEN_DIR;

star_tests::GoldenCase find_case(
    const std::vector<star_tests::GoldenCase>& cases,
    const std::string& name) {
  for (const star_tests::GoldenCase& c : cases) {
    if (c.scalar("name") == name) {
      return c;
    }
  }
  throw std::runtime_error("golden case not found: " + name);
}

Eigen::Vector3d parse_vec3(const star_tests::GoldenCase& c,
                           const std::string& key) {
  const auto& a = c.array(key);
  REQUIRE(a.size() == 3);
  return Eigen::Vector3d(star_tests::parse_hex_double(a[0]),
                         star_tests::parse_hex_double(a[1]),
                         star_tests::parse_hex_double(a[2]));
}

Eigen::Quaterniond parse_quat(const star_tests::GoldenCase& c,
                              const std::string& key) {
  const auto& a = c.array(key);
  REQUIRE(a.size() == 4);
  return Eigen::Quaterniond(star_tests::parse_hex_double(a[0]),
                            star_tests::parse_hex_double(a[1]),
                            star_tests::parse_hex_double(a[2]),
                            star_tests::parse_hex_double(a[3]));
}

Eigen::Matrix3d parse_mat3(const star_tests::GoldenCase& c,
                           const std::string& key) {
  const auto& a = c.array(key);
  REQUIRE(a.size() == 9);
  Eigen::Matrix3d m;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      m(i, j) = star_tests::parse_hex_double(
          a[static_cast<std::size_t>(3 * i + j)]);
    }
  }
  return m;
}

}  // namespace

TEST_CASE("gravgrad_torque_golden") {
  // Golden gate (FR-22 layer 1): eq:gravgrad:torque against extended-
  // precision references at norm-relative 1e-12 (the C++ path accumulates
  // ~12 IEEE roundings with cross-product amplification <= 10 for the
  // committed geometries; manifest derivation). exact_zero cases carry
  // structural zeros (spherical inertia, principal-axis alignment) and
  // must be exactly zero in every component.
  const auto cases =
      star_tests::load_golden_cases(kGoldenDir + "/attitude/gravgrad.toml");
  REQUIRE(cases.size() >= 6);
  double worst = 0.0;
  for (const auto& c : cases) {
    const std::string name = c.scalar("name");
    CAPTURE(name);
    const double mu = star_tests::parse_hex_double(c.scalar("mu_m3ps2"));
    // The committed Earth-mu cases must carry exactly the constants.hpp
    // value (IERS Conventions 2010), so golden and core cannot drift
    // apart; the lunar case is representative caller-supplied geometry.
    if (name != "lunar_orbit_diag") {
      REQUIRE(mu == star::constants::GM_EARTH_M3_PER_S2);
    }
    const Eigen::Vector3d tau = star::models::gravgrad_torque(
        mu, parse_vec3(c, "r_i_m"), parse_quat(c, "q_i2b_wxyz"),
        parse_mat3(c, "i_kgm2"));
    const Eigen::Vector3d ref = parse_vec3(c, "tau_ref_nm");
    if (c.scalar("exact_zero") == "true") {
      CHECK(tau[0] == 0.0);
      CHECK(tau[1] == 0.0);
      CHECK(tau[2] == 0.0);
      continue;
    }
    const double err = (tau - ref).norm() / ref.norm();
    CAPTURE(err);
    CHECK(err <= 1e-12);
    worst = std::max(worst, err);
  }
  CAPTURE(worst);
  CHECK(worst <= 1e-12);
}

TEST_CASE("gravgrad_exact_zero_geometries") {
  // Property gate (FR-22 layer 2): the structural zeros of
  // eq:gravgrad:torque, exact in IEEE arithmetic by construction
  // (ch:gravgrad implementation notes), evaluated on states built here
  // rather than parsed, so the property holds independently of the golden
  // pipeline.
  const double mu = star::constants::GM_EARTH_M3_PER_S2;
  {  // spherical inertia (power-of-two scale), generic attitude/position
    const Eigen::Quaterniond q = star::rotation::quat_normalize(
        Eigen::Quaterniond(0.3, -0.8, 0.4, 0.33));
    const Eigen::Vector3d tau = star::models::gravgrad_torque(
        mu, Eigen::Vector3d(5.5e6, -2.2e6, 3.1e6), q,
        2.0 * Eigen::Matrix3d::Identity());
    CHECK(tau[0] == 0.0);
    CHECK(tau[1] == 0.0);
    CHECK(tau[2] == 0.0);
  }
  {  // r along a principal axis, identity attitude
    const Eigen::Matrix3d inertia = Eigen::Vector3d(9.0, 7.0, 5.0).asDiagonal();
    const Eigen::Vector3d tau = star::models::gravgrad_torque(
        mu, Eigen::Vector3d(0.0, 0.0, 7.2e6),
        Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0), inertia);
    CHECK(tau[0] == 0.0);
    CHECK(tau[1] == 0.0);
    CHECK(tau[2] == 0.0);
  }
  {  // r along a principal axis through a pure z-rotation attitude: the
    // rotated rhat_b keeps exact zeros in x/y (the DCM entries are exact
    // products of zeros), so alignment with the z principal axis survives
    // the frame transformation bit-exactly.
    const Eigen::Quaterniond q(std::cos(0.15), 0.0, 0.0, std::sin(0.15));
    const Eigen::Matrix3d inertia = Eigen::Vector3d(9.0, 7.0, 5.0).asDiagonal();
    const Eigen::Vector3d tau = star::models::gravgrad_torque(
        mu, Eigen::Vector3d(0.0, 0.0, 7.2e6), q, inertia);
    CHECK(tau[0] == 0.0);
    CHECK(tau[1] == 0.0);
    CHECK(tau[2] == 0.0);
  }
}

TEST_CASE("gravgrad_libration_frequency") {
  // Phase 4 exit criterion 9: a gravity-gradient-stabilized body on the
  // prescribed circular reference orbit (ch:gravgrad, domain of validity:
  // attitude does not feed back into translation at module level),
  // gravity-gradient torque only, small pitch offset theta0. The pitch
  // angle theta(t) = alpha - n t (alpha the body's rotation angle about
  // the orbit normal) obeys the exact pendulum eq:gravgrad:pitch; its
  // frequency is measured from the event-located increasing zero
  // crossings over ~11 full periods and gated against (a) the analytic
  // small-angle eq:gravgrad:libfreq to 0.1 % (the criterion; the known
  // finite-amplitude offset is 2.5e-5) and (b) the amplitude-corrected
  // eq:gravgrad:pendulum to 1e-7 relative (budget: dense-output
  // event-location displacement over the ~58500 s baseline; manifest
  // derivation). Observed (MSVC x64 /fp:strict, 2026-07-03): 2.50001e-5
  // vs small-angle -- reproducing the predicted finite-amplitude offset
  // to six digits -- and 1.4e-10 vs the pendulum-corrected reference
  // (the controller chose ~27 s steps, well under h_max, so the
  // interpolant term came in far below its budget).
  const auto cases =
      star_tests::load_golden_cases(kGoldenDir + "/attitude/libration.toml");
  const auto def = find_case(cases, "definition");
  const double mu = star_tests::parse_hex_double(def.scalar("mu_m3ps2"));
  REQUIRE(mu == star::constants::GM_EARTH_M3_PER_S2);
  const double r = star_tests::parse_hex_double(def.scalar("r_m"));
  const Eigen::Vector3d idiag = parse_vec3(def, "i_diag_kgm2");
  const double theta0 = star_tests::parse_hex_double(def.scalar("theta0_rad"));
  const double t_span = star_tests::parse_hex_double(def.scalar("t_span_s"));
  const double w_lib_ref =
      star_tests::parse_hex_double(def.scalar("omega_lib_radps"));
  const double w_pend_ref =
      star_tests::parse_hex_double(def.scalar("omega_lib_pendulum_radps"));

  const Eigen::Matrix3d inertia = idiag.asDiagonal();
  const Eigen::Matrix3d idot = Eigen::Matrix3d::Zero();
  const double n = std::sqrt(mu / (r * r * r));

  // Initial attitude: the LVLH-tracking base orientation C0 (body x
  // along-track, body y = +z_I the pitch axis, body z zenith) pitched by
  // theta0 about the orbit normal at nu = 0: C_I^B = C0 R3(theta0).
  Eigen::Matrix3d c0;
  c0 << 0.0, 1.0, 0.0,  //
      0.0, 0.0, 1.0,    //
      1.0, 0.0, 0.0;
  const Eigen::Quaterniond q0 =
      star::rotation::quat_from_dcm(c0 * star::rotation::r3(theta0));
  // Released at the orbit rate with zero pitch rate: omega_b = (0, n, 0).
  const Eigen::Vector3d w0(0.0, n, 0.0);

  // RHS: prescribed circular orbit r_i(t), gravity-gradient torque only.
  auto rhs = [&](double t, const double* y, double* ydot) {
    const Eigen::Vector3d r_i(r * std::cos(n * t), r * std::sin(n * t),
                              0.0);
    const Eigen::Quaterniond q(y[0], y[1], y[2], y[3]);
    const Eigen::Vector3d tau =
        star::models::gravgrad_torque(mu, r_i, q, inertia);
    star::models::rigidbody_rhs(y, inertia, idot, tau, ydot);
  };
  const star::integrate::RhsRef f(rhs);

  // Pitch angle from the state: the body's rotation angle about the
  // inertial z axis is alpha = atan2(C(2,1), C(2,0)) (third row of C_I^B
  // is the body z axis in inertial coordinates, (cos alpha, sin alpha,
  // 0)); theta = alpha - n t wrapped to [-pi, pi]. The atan2 ratio is
  // insensitive to the quaternion's norm drift.
  auto pitch = [&](double t, const double* y) {
    const Eigen::Quaterniond q(y[0], y[1], y[2], y[3]);
    const Eigen::Matrix3d c = star::rotation::dcm_from_quat(q);
    const double alpha = std::atan2(c(2, 1), c(2, 0));
    return std::remainder(alpha - n * t, star::constants::TWO_PI);
  };
  star::events::EventSpec specs[1] = {
      {"pitch_zero_up", star::events::EventFnRef(pitch),
       star::events::Direction::kIncreasing, {}, false},
  };

  star::events::PropagateOptions opt;
  opt.method = star::events::Method::kRkf78;
  opt.mode = star::events::StepMode::kAdaptive;
  // Rate scale is n ~ 1e-3 rad/s, so the rate group gets a matching
  // absolute tolerance; h_max caps the dense-output event-location error
  // (see the budget above).
  opt.adaptive.groups = {
      star::integrate::StateGroup{"attitude", 0, 4, 1e-13, 1e-13},
      star::integrate::StateGroup{"rate", 4, 3, 1e-13, 1e-15},
  };
  opt.adaptive.h_init = 10.0;
  opt.adaptive.h_max = 120.0;

  double y0[7];
  y0[0] = q0.w();
  y0[1] = q0.x();
  y0[2] = q0.y();
  y0[3] = q0.z();
  y0[4] = w0[0];
  y0[5] = w0[1];
  y0[6] = w0[2];
  double yf[7];
  std::vector<star::events::EventRecord> log;
  star::events::propagate(f, 0.0, t_span, y0, 7, yf, opt, specs, 1, &log);

  // theta(0) = +theta0 with thetadot(0) = 0 gives one increasing zero
  // crossing per libration period, at 3T/4 + kT: expect 12 over the span.
  REQUIRE(log.size() >= 10);
  const double t_first = log.front().t;
  const double t_last = log.back().t;
  const double period =
      (t_last - t_first) / static_cast<double>(log.size() - 1);
  const double w_meas = star::constants::TWO_PI / period;

  const double err_smallangle = std::fabs(w_meas / w_lib_ref - 1.0);
  const double err_pendulum = std::fabs(w_meas / w_pend_ref - 1.0);
  CAPTURE(log.size());
  CAPTURE(w_meas);
  CAPTURE(w_lib_ref);
  CAPTURE(w_pend_ref);
  CAPTURE(err_smallangle);
  CAPTURE(err_pendulum);
  CHECK(err_smallangle < 1e-3);  // Phase 4 exit criterion 9
  CHECK(err_pendulum < 1e-7);    // tight secondary gate

  // Amplitude sanity: the pitch history must actually librate at ~theta0
  // (guard against a dead torque path passing the frequency gates by
  // accident of initial conditions).
  const double theta_end = pitch(t_span, yf);
  CAPTURE(theta_end);
  CHECK(std::fabs(theta_end) <= 1.05 * theta0);
  CHECK(std::fabs(theta0) > 0.0);
}
