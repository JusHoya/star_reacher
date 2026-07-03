// Rigid-body attitude dynamics tests (FR-1, FR-22 layers 1-3, Phase 4
// exit criterion 4): golden RHS vectors, structural zero-torque
// properties, the torque-free axisymmetric coning gate against the closed
// form (eq:rigidbody:coning), the intermediate-axis (Dzhanibekov) flip
// with H and T conservation (eq:rigidbody:H / eq:rigidbody:T), and the
// quaternion norm-drift bound (eq:rigidbody:normdrift). Test IDs are
// cited by the math-library validation table (ch:rigidbody); do not
// rename them.
//
// The golden references in tests/golden/attitude/ are mpmath evaluations
// from the exact committed binary64 inputs (provenance:
// tests/golden/attitude/manifest.toml). The coning reference is closed-
// form mathematics machine-verified at generation time against the
// kinematics and dynamics ODEs, so it cannot share a defect with the
// integration path exercised here; the Dzhanibekov trajectory reference
// is an independent mpmath Taylor-series integration.
#include <cmath>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "golden_io.hpp"
#include "star/events.hpp"
#include "star/integrate.hpp"
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

// Scalar-first [w, x, y, z] per the notation chapter (D-7).
Eigen::Quaterniond parse_quat(const star_tests::GoldenCase& c,
                              const std::string& key) {
  const auto& a = c.array(key);
  REQUIRE(a.size() == 4);
  return Eigen::Quaterniond(star_tests::parse_hex_double(a[0]),
                            star_tests::parse_hex_double(a[1]),
                            star_tests::parse_hex_double(a[2]),
                            star_tests::parse_hex_double(a[3]));
}

// Row-major 9-element array.
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

// Geodesic attitude error angle [rad] between two frame-transformation
// quaternions: 2 atan2(|dq_v|, |dq_w|) of dq = q_ref^-1 (x) q. Insensitive
// to overall scale and to the +-q sign ambiguity, and accurate for small
// angles (no acos cancellation).
double attitude_error_rad(const Eigen::Quaterniond& q_ref,
                          const Eigen::Quaterniond& q) {
  const Eigen::Quaterniond dq = star::rotation::quat_multiply(
      star::rotation::quat_conjugate(q_ref), q);
  const double vn =
      std::sqrt(dq.x() * dq.x() + dq.y() * dq.y() + dq.z() * dq.z());
  return 2.0 * std::atan2(vn, std::fabs(dq.w()));
}

// Per-group tolerances for the 7-component attitude slice
// [q(4), omega(3)] (FR-11 per-state-group tolerances).
std::vector<star::integrate::StateGroup> attitude_groups(double rtol,
                                                         double atol_q,
                                                         double atol_w) {
  return {
      star::integrate::StateGroup{"attitude", 0, 4, rtol, atol_q},
      star::integrate::StateGroup{"rate", 4, 3, rtol, atol_w},
  };
}

void pack_state(const Eigen::Quaterniond& q, const Eigen::Vector3d& w,
                double* y) {
  y[0] = q.w();
  y[1] = q.x();
  y[2] = q.y();
  y[3] = q.z();
  y[4] = w[0];
  y[5] = w[1];
  y[6] = w[2];
}

}  // namespace

TEST_CASE("rigidbody_rhs_golden") {
  // Golden gate (FR-22 layer 1): the quaternion kinematics
  // (eq:rigidbody:qdot) and Euler's equation with time-varying inertia
  // (eq:rigidbody:euler / eq:rigidbody:omegadot) against extended-
  // precision references at norm-relative 1e-12 (the C++ path accumulates
  // ~10 IEEE roundings plus the kappa(I)*eps <= 1e-14 cofactor-inverse
  // solve; manifest derivation). Cases flagged qdot_exact / wdot_exact
  // carry structural zeros or exact binary products and must match bit
  // for bit: any deviation there is an algebra bug, not roundoff.
  const auto cases =
      star_tests::load_golden_cases(kGoldenDir + "/attitude/rhs.toml");
  REQUIRE(cases.size() >= 6);
  double worst_q = 0.0;
  double worst_w = 0.0;
  for (const auto& c : cases) {
    const std::string name = c.scalar("name");
    CAPTURE(name);
    const Eigen::Quaterniond q = parse_quat(c, "q_i2b_wxyz");
    const Eigen::Vector3d w = parse_vec3(c, "w_b_radps");
    const Eigen::Matrix3d inertia = parse_mat3(c, "i_kgm2");
    const Eigen::Matrix3d idot = parse_mat3(c, "idot_kgm2ps");
    const Eigen::Vector3d tau = parse_vec3(c, "tau_b_nm");
    const auto& qd_ref_a = c.array("qdot_ref");
    REQUIRE(qd_ref_a.size() == 4);
    Eigen::Vector4d qd_ref;
    for (int i = 0; i < 4; ++i) {
      qd_ref[i] = star_tests::parse_hex_double(
          qd_ref_a[static_cast<std::size_t>(i)]);
    }
    const Eigen::Vector3d wd_ref = parse_vec3(c, "wdot_ref_radps2");

    double y[7];
    double ydot[7];
    pack_state(q, w, y);
    star::models::rigidbody_rhs(y, inertia, idot, tau, ydot);
    const Eigen::Vector4d qd(ydot[0], ydot[1], ydot[2], ydot[3]);
    const Eigen::Vector3d wd(ydot[4], ydot[5], ydot[6]);

    if (c.scalar("qdot_exact") == "true") {
      for (int i = 0; i < 4; ++i) {
        CHECK(qd[i] == qd_ref[i]);
      }
    } else {
      const double err = (qd - qd_ref).norm() / qd_ref.norm();
      CAPTURE(err);
      CHECK(err <= 1e-12);
      worst_q = std::max(worst_q, err);
    }
    if (c.scalar("wdot_exact") == "true") {
      for (int i = 0; i < 3; ++i) {
        CHECK(wd[i] == wd_ref[i]);
      }
    } else {
      const double err = (wd - wd_ref).norm() / wd_ref.norm();
      CAPTURE(err);
      CHECK(err <= 1e-12);
      worst_w = std::max(worst_w, err);
    }
  }
  CAPTURE(worst_q);
  CAPTURE(worst_w);
  CHECK(worst_q <= 1e-12);
  CHECK(worst_w <= 1e-12);
}

TEST_CASE("rigidbody_zero_torque_properties") {
  // Property gate (FR-22 layer 2): the structural zeros of the dynamics,
  // exact in IEEE arithmetic by construction (every product carries a
  // zero factor, or two identically rounded copies of the same real
  // product cancel; ch:rigidbody implementation notes), plus the
  // integrated consequence: a torque-free spherical body's rate stays
  // BITWISE constant under RK4 because every stage derivative of the rate
  // slice is exactly zero.
  {  // principal-axis spin of a diagonal inertia: omega_dot exactly zero
    const Eigen::Matrix3d inertia =
        Eigen::Vector3d(4.0, 2.5, 1.5).asDiagonal();
    const Eigen::Vector3d wd = star::models::rigidbody_omega_dot(
        Eigen::Vector3d(0.0, 0.0, 0.7), inertia, Eigen::Matrix3d::Zero(),
        Eigen::Vector3d::Zero());
    CHECK(wd[0] == 0.0);
    CHECK(wd[1] == 0.0);
    CHECK(wd[2] == 0.0);
  }
  {  // spherical inertia (power-of-two scale), generic rate: exactly zero
    const Eigen::Matrix3d inertia = 2.0 * Eigen::Matrix3d::Identity();
    const Eigen::Vector3d wd = star::models::rigidbody_omega_dot(
        Eigen::Vector3d(0.3, -0.5, 0.7), inertia, Eigen::Matrix3d::Zero(),
        Eigen::Vector3d::Zero());
    CHECK(wd[0] == 0.0);
    CHECK(wd[1] == 0.0);
    CHECK(wd[2] == 0.0);
  }
  {  // rest state: qdot = 1/2 q (x) [0, 0] exactly zero, any attitude
    const Eigen::Quaterniond q = star::rotation::quat_normalize(
        Eigen::Quaterniond(0.7, -0.2, 0.4, -0.5));
    const Eigen::Quaterniond qd =
        star::models::rigidbody_qdot(q, Eigen::Vector3d::Zero());
    CHECK(qd.w() == 0.0);
    CHECK(qd.x() == 0.0);
    CHECK(qd.y() == 0.0);
    CHECK(qd.z() == 0.0);
  }
  {  // RK4 integration of the torque-free spherical body: rate bitwise
    // constant over 400 steps while the attitude precesses.
    const Eigen::Matrix3d inertia = 2.0 * Eigen::Matrix3d::Identity();
    const Eigen::Matrix3d idot = Eigen::Matrix3d::Zero();
    const Eigen::Vector3d tau = Eigen::Vector3d::Zero();
    auto rhs = [&](double /*t*/, const double* y, double* ydot) {
      star::models::rigidbody_rhs(y, inertia, idot, tau, ydot);
    };
    const star::integrate::RhsRef f(rhs);
    star::integrate::Rk4 rk4(7);
    double y[7];
    pack_state(star::rotation::quat_normalize(
                   Eigen::Quaterniond(0.9, 0.1, -0.3, 0.2)),
               Eigen::Vector3d(0.3, -0.5, 0.7), y);
    const double w0[3] = {y[4], y[5], y[6]};
    const double q0[4] = {y[0], y[1], y[2], y[3]};
    double t = 0.0;
    const double h = 0.25;
    for (int i = 0; i < 400; ++i) {
      rk4.step(f, t, y, h, y);
      t += h;
    }
    CHECK(y[4] == w0[0]);
    CHECK(y[5] == w0[1]);
    CHECK(y[6] == w0[2]);
    // The attitude must actually have moved (guard against a dead RHS).
    const double dq = std::fabs(y[0] - q0[0]) + std::fabs(y[1] - q0[1]) +
                      std::fabs(y[2] - q0[2]) + std::fabs(y[3] - q0[3]);
    CHECK(dq > 0.1);
  }
}

TEST_CASE("rigidbody_coning_closed_form") {
  // Phase 4 exit criterion 4 (coning clause): RKF7(8) adaptive propagation
  // of the rigid-body dynamics from the committed initial state against
  // the closed-form axisymmetric solution (eq:rigidbody:coning) at eight
  // checkpoints over five body-precession periods; attitude error angle
  // <= 1e-9 rad and rate error <= 1e-9 norm-relative. Budget (manifest):
  // per-step local error <= the controller tolerance 1e-13 accumulating
  // linearly over the few hundred accepted steps, expected O(1e-11) rad
  // worst case -- two orders under the gate. Observed worst case
  // (MSVC x64 /fp:strict, 2026-07-03): 1.5e-13 rad attitude, 2.0e-15
  // norm-relative rate, four orders inside the gate.
  const auto cases =
      star_tests::load_golden_cases(kGoldenDir + "/attitude/coning.toml");
  const auto def = find_case(cases, "definition");
  const double it = star_tests::parse_hex_double(def.scalar("it_kgm2"));
  const double ia = star_tests::parse_hex_double(def.scalar("ia_kgm2"));
  const Eigen::Matrix3d inertia = Eigen::Vector3d(it, it, ia).asDiagonal();
  const Eigen::Matrix3d idot = Eigen::Matrix3d::Zero();
  const Eigen::Vector3d tau = Eigen::Vector3d::Zero();

  auto rhs = [&](double /*t*/, const double* y, double* ydot) {
    star::models::rigidbody_rhs(y, inertia, idot, tau, ydot);
  };
  const star::integrate::RhsRef f(rhs);

  star::events::PropagateOptions opt;
  opt.method = star::events::Method::kRkf78;
  opt.mode = star::events::StepMode::kAdaptive;
  opt.adaptive.groups = attitude_groups(1e-13, 1e-13, 1e-13);
  opt.adaptive.h_init = 0.5;
  opt.adaptive.h_max = 10.0;

  double y[7];
  pack_state(parse_quat(def, "q0_i2b_wxyz"), parse_vec3(def, "w0_b_radps"),
             y);

  double t_prev = 0.0;
  double worst_att = 0.0;
  double worst_rate = 0.0;
  int checkpoints = 0;
  for (const auto& c : cases) {
    if (c.scalar("name") == "definition") {
      continue;
    }
    const double t_k = star_tests::parse_hex_double(c.scalar("t_s"));
    double y_out[7];
    star::events::propagate(f, t_prev, t_k, y, 7, y_out, opt, nullptr, 0,
                            nullptr);
    for (int i = 0; i < 7; ++i) {
      y[i] = y_out[i];
    }
    t_prev = t_k;
    ++checkpoints;

    const Eigen::Quaterniond q(y[0], y[1], y[2], y[3]);
    const Eigen::Vector3d w(y[4], y[5], y[6]);
    const Eigen::Quaterniond q_ref = parse_quat(c, "q_ref_i2b_wxyz");
    const Eigen::Vector3d w_ref = parse_vec3(c, "w_ref_b_radps");
    const double att_err = attitude_error_rad(q_ref, q);
    const double rate_err = (w - w_ref).norm() / w_ref.norm();
    CAPTURE(t_k);
    CAPTURE(att_err);
    CAPTURE(rate_err);
    CHECK(att_err <= 1e-9);
    CHECK(rate_err <= 1e-9);
    worst_att = std::max(worst_att, att_err);
    worst_rate = std::max(worst_rate, rate_err);
  }
  REQUIRE(checkpoints == 8);
  CAPTURE(worst_att);
  CAPTURE(worst_rate);
  CHECK(worst_att <= 1e-9);
  CHECK(worst_rate <= 1e-9);
}

TEST_CASE("rigidbody_intermediate_axis_flip") {
  // Phase 4 exit criterion 4 (flip clause): a distinct-inertia body spun
  // about its intermediate axis with a small perturbation flips
  // (Dzhanibekov effect), while the inertial angular momentum
  // (eq:rigidbody:H, magnitude AND direction) and the rotational kinetic
  // energy (eq:rigidbody:T) stay conserved to 1e-10 relative at every
  // accepted step. Early-time rates are additionally gated against an
  // independent mpmath trajectory inside the exp(lambda t) amplification
  // horizon (eq:rigidbody:lambda; manifest derivation). Observed
  // (MSVC x64 /fp:strict, 2026-07-03): max |dH|/H 8.4e-14 vector
  // (2.1e-15 magnitude-only), max |dT|/T 4.0e-15, worst checkpoint rate
  // error 4.4e-16, min w2 = -0.45 (full reversal).
  const auto cases = star_tests::load_golden_cases(
      kGoldenDir + "/attitude/dzhanibekov.toml");
  const auto def = find_case(cases, "definition");
  const Eigen::Vector3d idiag = parse_vec3(def, "i_diag_kgm2");
  const Eigen::Matrix3d inertia = idiag.asDiagonal();
  const Eigen::Matrix3d idot = Eigen::Matrix3d::Zero();
  const Eigen::Vector3d tau = Eigen::Vector3d::Zero();
  const Eigen::Vector3d w0 = parse_vec3(def, "w0_b_radps");
  const double t_span = star_tests::parse_hex_double(def.scalar("t_span_s"));
  const Eigen::Vector3d h0_ref = parse_vec3(def, "h0_i_kgm2ps");
  const double h0_mag = star_tests::parse_hex_double(def.scalar("h0_mag_kgm2ps"));
  const double t0_ref = star_tests::parse_hex_double(def.scalar("t0_j"));

  auto rhs = [&](double /*t*/, const double* y, double* ydot) {
    star::models::rigidbody_rhs(y, inertia, idot, tau, ydot);
  };
  const star::integrate::RhsRef f(rhs);

  // Conservation and flip monitor on every accepted step endpoint. The
  // quaternion is normalized before forming the DCM so its ~1e-13 norm
  // drift (a pure bookkeeping error, ch:rigidbody) does not alias into
  // the physical H comparison.
  double max_dh = 0.0;
  double max_dhmag = 0.0;
  double max_dt = 0.0;
  double min_w2 = w0[1];
  auto monitor = [&](const star::integrate::DenseStep& d) {
    const Eigen::Quaterniond q = star::rotation::quat_normalize(
        Eigen::Quaterniond(d.y1[0], d.y1[1], d.y1[2], d.y1[3]));
    const Eigen::Vector3d w(d.y1[4], d.y1[5], d.y1[6]);
    const Eigen::Vector3d h_b = inertia * w;
    // H^I = (C_I^B)^T H^B (eq:rigidbody:H).
    const Eigen::Vector3d h_i =
        star::rotation::dcm_from_quat(q).transpose() * h_b;
    const double t_kin = 0.5 * w.dot(h_b);  // eq:rigidbody:T
    max_dh = std::max(max_dh, (h_i - h0_ref).norm() / h0_mag);
    max_dhmag = std::max(max_dhmag,
                         std::fabs(h_i.norm() - h0_mag) / h0_mag);
    max_dt = std::max(max_dt, std::fabs(t_kin - t0_ref) / t0_ref);
    min_w2 = std::min(min_w2, w[1]);
  };
  star::events::StepObserverRef obs(monitor);

  star::events::PropagateOptions opt;
  opt.method = star::events::Method::kRkf78;
  opt.mode = star::events::StepMode::kAdaptive;
  opt.adaptive.groups = attitude_groups(1e-13, 1e-13, 1e-14);
  opt.adaptive.h_init = 0.1;
  opt.adaptive.h_max = 2.0;

  double y[7];
  pack_state(Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0), w0, y);

  // Segment at the golden checkpoint times, comparing the rate vector to
  // the independent mpmath trajectory, then run out the full span.
  double t_prev = 0.0;
  double worst_ck = 0.0;
  for (const auto& c : cases) {
    if (c.scalar("name") == "definition") {
      continue;
    }
    const double t_k = star_tests::parse_hex_double(c.scalar("t_s"));
    double y_out[7];
    star::events::propagate(f, t_prev, t_k, y, 7, y_out, opt, nullptr, 0,
                            nullptr, obs);
    for (int i = 0; i < 7; ++i) {
      y[i] = y_out[i];
    }
    t_prev = t_k;
    const Eigen::Vector3d w(y[4], y[5], y[6]);
    const Eigen::Vector3d w_ref = parse_vec3(c, "w_ref_b_radps");
    const double err = (w - w_ref).norm() / w_ref.norm();
    CAPTURE(t_k);
    CAPTURE(err);
    CHECK(err <= 1e-9);
    worst_ck = std::max(worst_ck, err);
  }
  double y_final[7];
  star::events::propagate(f, t_prev, t_span, y, 7, y_final, opt, nullptr, 0,
                          nullptr, obs);

  CAPTURE(max_dh);
  CAPTURE(max_dhmag);
  CAPTURE(max_dt);
  CAPTURE(min_w2);
  CAPTURE(worst_ck);
  CHECK(max_dh <= 1e-10);     // H conserved in magnitude AND direction
  CHECK(max_dhmag <= 1e-10);  // magnitude clause separately
  CHECK(max_dt <= 1e-10);     // T conserved
  CHECK(min_w2 < -0.9 * w0[1]);  // the flip actually happened
}

TEST_CASE("rigidbody_quaternion_norm_drift") {
  // FR-1 documented norm-drift bound. Part 1: for constant omega
  // (torque-free spherical body, so the rate slice is bitwise constant),
  // the per-step RK4 norm factor is exactly |R(i theta)| with
  // theta = h|omega|/2 (eq:rigidbody:rk4norm), giving the N-step drift
  // -N theta^6/144 (1 + O(theta^2)) (eq:rigidbody:normdrift) -- an exact
  // prediction, gated at 5 % (the neglected terms enter at
  // theta^2 ~ 4e-3 relative). Part 2: post-step renormalization restores
  // unit norm to machine rounding, leaves the rate slice bitwise intact,
  // and changes the represented attitude by nothing beyond one rounding.
  // Part 3: the adaptive driver's drift on the coning problem stays below
  // the linear-accumulation bound N*(rtol + atol_q) with an order of
  // headroom (ch:rigidbody, Sec. norm drift).
  const Eigen::Matrix3d inertia = 2.0 * Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d idot = Eigen::Matrix3d::Zero();
  const Eigen::Vector3d tau = Eigen::Vector3d::Zero();
  const Eigen::Vector3d w(0.12, -0.32, 0.24);
  auto rhs = [&](double /*t*/, const double* y, double* ydot) {
    star::models::rigidbody_rhs(y, inertia, idot, tau, ydot);
  };
  const star::integrate::RhsRef f(rhs);

  {  // Part 1: exact RK4 drift law.
    star::integrate::Rk4 rk4(7);
    double y[7];
    pack_state(star::rotation::quat_normalize(
                   Eigen::Quaterniond(0.6, 0.5, -0.4, 0.2)),
               w, y);
    const double h = 0.25;
    const int n_steps = 400;
    double t = 0.0;
    for (int i = 0; i < n_steps; ++i) {
      rk4.step(f, t, y, h, y);
      t += h;
    }
    const double norm_after = std::sqrt(y[0] * y[0] + y[1] * y[1] +
                                        y[2] * y[2] + y[3] * y[3]);
    const double drift = norm_after - 1.0;
    const double theta = 0.5 * h * w.norm();
    const double predicted = -static_cast<double>(n_steps) *
                             std::pow(theta, 6) / 144.0;
    CAPTURE(drift);
    CAPTURE(predicted);
    CHECK(drift < 0.0);  // the norm shrinks, per eq:rigidbody:rk4norm
    CHECK(std::fabs(drift - predicted) <= 0.05 * std::fabs(predicted));

    // Part 2: renormalization semantics.
    const Eigen::Quaterniond q_before(y[0], y[1], y[2], y[3]);
    const double w_before[3] = {y[4], y[5], y[6]};
    const double reported = star::models::rigidbody_renormalize(y);
    CHECK(reported == drift);  // same quantity, returned for monitoring
    const double norm_post = std::sqrt(y[0] * y[0] + y[1] * y[1] +
                                       y[2] * y[2] + y[3] * y[3]);
    CHECK(std::fabs(norm_post - 1.0) <= 3e-16);
    CHECK(y[4] == w_before[0]);  // rate slice untouched
    CHECK(y[5] == w_before[1]);
    CHECK(y[6] == w_before[2]);
    const Eigen::Quaterniond q_after(y[0], y[1], y[2], y[3]);
    CHECK(attitude_error_rad(q_before, q_after) <= 1e-15);
  }

  {  // Part 3: adaptive-driver drift under the linear-accumulation bound.
    const auto cases = star_tests::load_golden_cases(
        kGoldenDir + "/attitude/coning.toml");
    const auto def = find_case(cases, "definition");
    const double it = star_tests::parse_hex_double(def.scalar("it_kgm2"));
    const double ia = star_tests::parse_hex_double(def.scalar("ia_kgm2"));
    const Eigen::Matrix3d i_cone = Eigen::Vector3d(it, it, ia).asDiagonal();
    auto rhs_cone = [&](double /*t*/, const double* y, double* ydot) {
      star::models::rigidbody_rhs(y, i_cone, idot, tau, ydot);
    };
    const star::integrate::RhsRef f_cone(rhs_cone);
    const double rtol = 1e-13;
    const double atol_q = 1e-13;
    star::events::PropagateOptions opt;
    opt.method = star::events::Method::kRkf78;
    opt.mode = star::events::StepMode::kAdaptive;
    opt.adaptive.groups = attitude_groups(rtol, atol_q, 1e-13);
    opt.adaptive.h_init = 0.5;
    opt.adaptive.h_max = 10.0;
    double y[7];
    pack_state(parse_quat(def, "q0_i2b_wxyz"),
               parse_vec3(def, "w0_b_radps"), y);
    double yf[7];
    const auto res = star::events::propagate(f_cone, 0.0, 144.0, y, 7, yf,
                                             opt, nullptr, 0, nullptr);
    const double drift = std::fabs(std::sqrt(yf[0] * yf[0] + yf[1] * yf[1] +
                                             yf[2] * yf[2] + yf[3] * yf[3]) -
                                   1.0);
    const double bound = static_cast<double>(res.steps_accepted + 1) *
                         (rtol + atol_q);
    CAPTURE(drift);
    CAPTURE(bound);
    CAPTURE(res.steps_accepted);
    CHECK(drift <= bound);
  }
}
