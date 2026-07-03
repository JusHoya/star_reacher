// Actuator tests (FR-1, Phase 4 exit criterion 7): golden RCS pulses and
// wheel steps against the independent 60-digit mpmath references (FR-22
// layer 1), the exact MIB and saturation semantics, and the
// rigid-body-plus-wheel momentum-conservation slew (layer 2). Test IDs
// are cited by the math-library validation table (ch:actuators); do not
// rename them. Golden provenance and tolerances:
// tests/golden/actuators/manifest.toml.
//
// The conservation test integrates the coupled body+wheel dynamics
// locally (RK4, quaternion attitude, constant body inertia): the wheel
// torque enters through wheel_step and the bookkeeping through
// total_angular_momentum_Nms, so the module's sign conventions - not a
// re-derivation - are what is under test. Under zero external torque the
// inertially resolved total momentum is an exact invariant of the
// continuous dynamics (ch:actuators); the RK4 discretization error at
// omega*dt ~ 2.5e-4 is orders below the 1e-12 gate.
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "golden_io.hpp"
#include "star/models/actuators.hpp"
#include "vendor/doctest.h"

namespace {

using star::models::RcsClusterParams;
using star::models::RcsForceTorque;
using star::models::RcsImpulse;
using star::models::RcsThrusterParams;
using star::models::WheelParams;
using star::models::WheelState;
using star::models::WheelStepResult;
using star_tests::load_golden_cases;
using star_tests::parse_hex_double;

const std::string kGoldenDir = STAR_GOLDEN_DIR;

Eigen::Vector3d parse_vec3(const star_tests::GoldenCase& c,
                           const std::string& key) {
  const auto& a = c.array(key);
  REQUIRE(a.size() == 3);
  return Eigen::Vector3d(parse_hex_double(a[0]), parse_hex_double(a[1]),
                         parse_hex_double(a[2]));
}

RcsThrusterParams parse_thruster(const star_tests::GoldenCase& c) {
  RcsThrusterParams t;
  t.position_m = parse_vec3(c, "position_m");
  t.direction = parse_vec3(c, "direction");
  t.thrust_N = parse_hex_double(c.scalar("thrust_N"));
  t.mib_Ns = parse_hex_double(c.scalar("mib_Ns"));
  return t;
}

double check_vec(const Eigen::Vector3d& value, const Eigen::Vector3d& ref,
                 double tol) {
  if (ref.norm() == 0.0) {
    CHECK(value.norm() == 0.0);
    return 0.0;
  }
  const double err = (value - ref).norm() / ref.norm();
  CHECK(err <= tol);
  return err;
}

}  // namespace

TEST_CASE("ACT-RCS-MIB") {
  // Exit criterion 7: a pulse below the configured MIB delivers EXACTLY
  // zero impulse; one at 2x MIB matches the spec impulse to 1e-12
  // relative. The boundary itself (thrust * duration == MIB, built from
  // exactly-representable values 2.0 * 0.25 = 0.5) delivers in full
  // (at-or-above semantics, eq:actuators:mib).
  const auto cases = load_golden_cases(kGoldenDir + "/actuators/rcs.toml");
  bool saw_below = false;
  bool saw_two_mib = false;
  for (const auto& c : cases) {
    const std::string name = c.scalar("name");
    CAPTURE(name);
    const RcsThrusterParams t = parse_thruster(c);
    const double duration = parse_hex_double(c.scalar("duration_s"));
    const RcsImpulse pulse =
        star::models::rcs_pulse(t, duration, parse_vec3(c, "cg_m"));
    if (name == "below_mib_exact_zero") {
      saw_below = true;
      CHECK(pulse.delivered_Ns == 0.0);
      CHECK(pulse.impulse_Ns.norm() == 0.0);
      CHECK(pulse.angular_impulse_Nms.norm() == 0.0);
    }
    if (name == "at_two_mib") {
      saw_two_mib = true;
      // Spec impulse 2 * mib, independent of the golden file's value.
      const double spec = 2.0 * t.mib_Ns;
      const double err = std::fabs(pulse.delivered_Ns - spec) / spec;
      CAPTURE(err);
      CHECK(err <= 1e-12);
    }
  }
  CHECK(saw_below);
  CHECK(saw_two_mib);

  // Exact-boundary semantics with exactly representable arithmetic:
  // 2.0 N * 0.25 s = 0.5 N s == MIB delivers in full, and one ulp of
  // duration below the boundary refuses.
  RcsThrusterParams t;
  t.thrust_N = 2.0;
  t.mib_Ns = 0.5;
  t.direction = Eigen::Vector3d::UnitY();
  const RcsImpulse at_boundary =
      star::models::rcs_pulse(t, 0.25, Eigen::Vector3d::Zero());
  CHECK(at_boundary.delivered_Ns == 0.5);
  const double below = std::nextafter(0.25, 0.0);
  const RcsImpulse under =
      star::models::rcs_pulse(t, below, Eigen::Vector3d::Zero());
  CHECK(under.delivered_Ns == 0.0);
}

TEST_CASE("ACT-RCS-COUPLING-GOLDEN") {
  // Golden gate (FR-22 layer 1): delivered pulses couple force AND
  // torque about the CG (eq:actuators:rcscoupling) - linear and angular
  // impulse against the 60-digit mpmath references at norm-relative
  // 1e-12 (manifest derivation: <= ~5 roundings through the cross
  // product, floor ~1e-16).
  const auto cases = load_golden_cases(kGoldenDir + "/actuators/rcs.toml");
  REQUIRE(cases.size() == 4);
  double worst = 0.0;
  for (const auto& c : cases) {
    CAPTURE(c.scalar("name"));
    const RcsThrusterParams t = parse_thruster(c);
    const Eigen::Vector3d cg = parse_vec3(c, "cg_m");
    const double duration = parse_hex_double(c.scalar("duration_s"));
    const RcsImpulse pulse = star::models::rcs_pulse(t, duration, cg);
    worst = std::max(worst, check_vec(pulse.impulse_Ns,
                                      parse_vec3(c, "impulse_Ns"), 1e-12));
    worst = std::max(
        worst, check_vec(pulse.angular_impulse_Nms,
                         parse_vec3(c, "angular_impulse_Nms"), 1e-12));
  }
  CAPTURE(worst);
  CHECK(worst <= 1e-12);  // prints the observed margin under -s

  // Cluster composition: the two-thruster cluster sum must equal the
  // per-thruster sums bit-exactly (same operations, same fixed order),
  // and an all-off cluster is exactly zero.
  RcsThrusterParams a;
  a.position_m = Eigen::Vector3d(1.0, 0.5, -0.25);
  a.direction = Eigen::Vector3d(0.6, 0.0, 0.8);
  a.thrust_N = 12.0;
  RcsThrusterParams b;
  b.position_m = Eigen::Vector3d(-1.0, -0.5, 0.25);
  b.direction = Eigen::Vector3d::UnitZ();
  b.thrust_N = 8.0;
  RcsClusterParams cluster;
  cluster.thrusters = {a, b};
  const Eigen::Vector3d cg(0.1, 0.0, 0.05);
  const RcsForceTorque both =
      star::models::rcs_cluster_force_torque(cluster, {true, true}, cg);
  const RcsForceTorque fa = star::models::rcs_force_torque(a, cg);
  const RcsForceTorque fb = star::models::rcs_force_torque(b, cg);
  CHECK(both.force_N == fa.force_N + fb.force_N);
  CHECK(both.torque_Nm == fa.torque_Nm + fb.torque_Nm);
  const RcsForceTorque none =
      star::models::rcs_cluster_force_torque(cluster, {false, false}, cg);
  CHECK(none.force_N.norm() == 0.0);
  CHECK(none.torque_Nm.norm() == 0.0);
}

TEST_CASE("ACT-WHEEL-SATURATION") {
  // Exit criterion 7: torque and momentum clamp EXACTLY at the
  // configured saturations (eq:actuators:wheelclamp). Where the model
  // assigns literals the comparison is bit-exact: over-limit commands
  // deliver exactly +-torque_max, rail landings set exactly +-h_max, a
  // same-sign command on the rail delivers exactly zero. The remaining
  // nonzero outputs (rail-landing partial torque, body torque) compare
  // against the mpmath references at 1e-12 relative.
  const auto cases =
      load_golden_cases(kGoldenDir + "/actuators/wheels.toml");
  REQUIRE(cases.size() == 7);
  double worst = 0.0;
  for (const auto& c : cases) {
    const std::string name = c.scalar("name");
    CAPTURE(name);
    WheelParams w;
    w.axis = parse_vec3(c, "axis");
    w.torque_max_Nm = parse_hex_double(c.scalar("torque_max_Nm"));
    w.momentum_max_Nms = parse_hex_double(c.scalar("momentum_max_Nms"));
    WheelState s;
    s.momentum_Nms = parse_hex_double(c.scalar("h0_Nms"));
    const double cmd = parse_hex_double(c.scalar("torque_cmd_Nm"));
    const double dt = parse_hex_double(c.scalar("dt_s"));
    const WheelStepResult r = star::models::wheel_step(w, cmd, s, dt);

    const double tau_ref = parse_hex_double(c.scalar("torque_Nm"));
    const double h1_ref = parse_hex_double(c.scalar("h1_Nms"));
    // The golden generator mirrors the branch structure, so every
    // reference here is either a literal assignment (bit-exact) or a
    // 1-2 rounding expression; compare exactly where the model assigns
    // literals and at 1e-12 elsewhere.
    if (name == "torque_clamped") {
      CHECK(r.torque_Nm == w.torque_max_Nm);  // exact clamp
    }
    if (name == "torque_clamped_negative") {
      CHECK(r.torque_Nm == -w.torque_max_Nm);
    }
    if (name == "saturated_zero_delivery") {
      CHECK(r.torque_Nm == 0.0);  // exact zero on the rail
      CHECK(r.body_torque_Nm.norm() == 0.0);
    }
    if (name == "rail_landing_partial" || name == "negative_rail_landing" ||
        name == "saturated_zero_delivery") {
      CHECK(std::fabs(r.state.momentum_Nms) ==
            w.momentum_max_Nms);  // exact rail
    }
    if (tau_ref == 0.0) {
      CHECK(r.torque_Nm == 0.0);
    } else {
      const double err = std::fabs(r.torque_Nm - tau_ref) /
                         std::fabs(tau_ref);
      CAPTURE(err);
      CHECK(err <= 1e-12);
      worst = std::max(worst, err);
    }
    if (h1_ref == 0.0) {
      CHECK(r.state.momentum_Nms == 0.0);
    } else {
      const double err_h = std::fabs(r.state.momentum_Nms - h1_ref) /
                           std::fabs(h1_ref);
      CAPTURE(err_h);
      CHECK(err_h <= 1e-12);
      worst = std::max(worst, err_h);
    }
    worst = std::max(worst, check_vec(r.body_torque_Nm,
                                      parse_vec3(c, "body_torque_Nm"),
                                      1e-12));
  }
  CAPTURE(worst);
  CHECK(worst <= 1e-12);
}

TEST_CASE("ACT-WHEEL-MOMENTUM-CONSERVATION") {
  // Exit criterion 7: a reaction-wheel slew conserves total (body +
  // wheel) angular momentum to 1e-12 relative. Coupled dynamics with the
  // wheel axis OFF the principal axes (gyroscopic coupling active):
  //   H_body = I omega + h a          (total_angular_momentum_Nms)
  //   I omega_dot = -omega x H_body - tau a,   h_dot = tau
  //   q_dot = 1/2 q (x) (0, omega)    (body-to-inertial attitude)
  // so d/dt (R(q) H_body) = 0 exactly in continuous time; the internal
  // torque +-tau a cancels between body and wheel through the module's
  // eq:actuators:wheelreaction sign. Slew profile: accelerate the wheel
  // for 100 s, decelerate for 100 s (all within the clamps, so
  // wheel_step delivers the command and its momentum equals the RK4
  // h-integration bit-for-bit).
  const Eigen::Matrix3d inertia =
      Eigen::Vector3d(120.0, 90.0, 70.0).asDiagonal();
  const Eigen::Matrix3d inertia_inv = inertia.inverse();
  WheelParams wheel;
  wheel.axis = Eigen::Vector3d(0.6, 0.8, 0.0);  // exactly unit in binary64
  wheel.torque_max_Nm = 0.2;
  wheel.momentum_max_Nms = 30.0;

  WheelState ws;
  ws.momentum_Nms = 0.4;
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
  Eigen::Vector3d omega(0.02, -0.015, 0.01);

  const auto h_total_inertial = [&](const Eigen::Quaterniond& qq,
                                    const Eigen::Vector3d& w,
                                    const WheelState& st) {
    return qq.toRotationMatrix() *
           star::models::total_angular_momentum_Nms(inertia, w, {wheel},
                                                    {st});
  };
  const Eigen::Vector3d h0 = h_total_inertial(q, omega, ws);
  REQUIRE(h0.norm() > 0.0);

  const double dt = 0.01;
  const int n_steps = 20000;  // 200 s slew
  double worst_rel = 0.0;
  for (int k = 0; k < n_steps; ++k) {
    const double cmd = (k < n_steps / 2) ? 0.05 : -0.05;
    const WheelStepResult step = star::models::wheel_step(wheel, cmd, ws, dt);
    const double tau = step.torque_Nm;
    CHECK(tau == cmd);  // inside all clamps: command delivered verbatim

    // RK4 over y = [q, omega, h] with tau constant across the step
    // (h enters the gyroscopic term at its correct stage values).
    const auto omega_dot = [&](const Eigen::Vector3d& w, double h) {
      const Eigen::Vector3d h_body = inertia * w + h * wheel.axis;
      return (inertia_inv * (-w.cross(h_body) - tau * wheel.axis)).eval();
    };
    const auto q_dot = [](const Eigen::Quaterniond& qq,
                          const Eigen::Vector3d& w) {
      return (qq * Eigen::Quaterniond(0.0, 0.5 * w.x(), 0.5 * w.y(),
                                      0.5 * w.z()))
          .coeffs();
    };
    const double h0s = ws.momentum_Nms;
    const Eigen::Vector4d qc = q.coeffs();

    const Eigen::Vector3d k1w = omega_dot(omega, h0s);
    const Eigen::Vector4d k1q = q_dot(q, omega);
    const Eigen::Vector3d w2 = omega + 0.5 * dt * k1w;
    const Eigen::Quaterniond q2(qc + 0.5 * dt * k1q);
    const Eigen::Vector3d k2w = omega_dot(w2, h0s + 0.5 * dt * tau);
    const Eigen::Vector4d k2q = q_dot(q2, w2);
    const Eigen::Vector3d w3 = omega + 0.5 * dt * k2w;
    const Eigen::Quaterniond q3(qc + 0.5 * dt * k2q);
    const Eigen::Vector3d k3w = omega_dot(w3, h0s + 0.5 * dt * tau);
    const Eigen::Vector4d k3q = q_dot(q3, w3);
    const Eigen::Vector3d w4 = omega + dt * k3w;
    const Eigen::Quaterniond q4(qc + dt * k3q);
    const Eigen::Vector3d k4w = omega_dot(w4, h0s + dt * tau);
    const Eigen::Vector4d k4q = q_dot(q4, w4);

    omega += (dt / 6.0) * (k1w + 2.0 * k2w + 2.0 * k3w + k4w);
    q = Eigen::Quaterniond(qc + (dt / 6.0) *
                                    (k1q + 2.0 * k2q + 2.0 * k3q + k4q));
    q.normalize();
    // The module's momentum update is the same linear step the RK4
    // state would take (h is linear in t within the step).
    ws = step.state;

    if ((k + 1) % 1000 == 0) {
      const double err =
          (h_total_inertial(q, omega, ws) - h0).norm() / h0.norm();
      worst_rel = std::max(worst_rel, err);
    }
  }
  const double final_rel =
      (h_total_inertial(q, omega, ws) - h0).norm() / h0.norm();
  CAPTURE(worst_rel);
  CAPTURE(final_rel);
  CHECK(final_rel <= 1e-12);
  CHECK(worst_rel <= 1e-12);
  // The slew must have actually exchanged momentum: after 100 s at
  // +0.05 N m the wheel peaked at 0.4 + 5 N m s; at the end it is back
  // near 0.4 while the body picked up the difference transiently
  // (non-vacuous conservation).
  CHECK(std::fabs(ws.momentum_Nms - 0.4) <= 1e-9);
}

TEST_CASE("ACT-DOMAIN") {
  // Out-of-domain behavior per ch:actuators: std::domain_error.
  RcsThrusterParams t;
  t.thrust_N = 10.0;
  t.direction = Eigen::Vector3d(1.0, 1.0, 0.0);  // not unit
  CHECK_THROWS_AS(
      star::models::rcs_pulse(t, 0.1, Eigen::Vector3d::Zero()),
      std::domain_error);
  t.direction = Eigen::Vector3d::UnitX();
  t.thrust_N = -1.0;
  CHECK_THROWS_AS(
      star::models::rcs_force_torque(t, Eigen::Vector3d::Zero()),
      std::domain_error);
  t.thrust_N = 10.0;
  CHECK_THROWS_AS(
      star::models::rcs_pulse(t, -0.1, Eigen::Vector3d::Zero()),
      std::domain_error);

  RcsClusterParams cluster;
  cluster.thrusters = {t};
  CHECK_THROWS_AS(
      star::models::rcs_cluster_force_torque(cluster, {true, false},
                                             Eigen::Vector3d::Zero()),
      std::domain_error);  // flag-count mismatch

  WheelParams w;
  w.torque_max_Nm = 0.1;
  w.momentum_max_Nms = 10.0;
  WheelState s;
  CHECK_THROWS_AS(star::models::wheel_step(w, 0.05, s, 0.0),
                  std::domain_error);  // dt must be positive
  s.momentum_Nms = 11.0;  // outside the rails
  CHECK_THROWS_AS(star::models::wheel_step(w, 0.05, s, 0.1),
                  std::domain_error);
  WheelParams bad = w;
  bad.axis = Eigen::Vector3d::Zero();
  s.momentum_Nms = 0.0;
  CHECK_THROWS_AS(star::models::wheel_step(bad, 0.05, s, 0.1),
                  std::domain_error);
  CHECK_THROWS_AS(
      star::models::total_angular_momentum_Nms(
          Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), {w},
          {s, s}),
      std::domain_error);  // state-count mismatch
}
