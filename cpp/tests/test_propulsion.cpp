// Propulsion tests (FR-10, Phase 4 exit criteria 3 and 7): golden engine
// operating points against the independent 60-digit mpmath references
// (FR-22 layer 1), the USSA76 sea-level back-pressure anchor, the
// Tsiolkovsky vacuum-burn benchmark, and the exact spool/throttle/
// ignition/gimbal semantics (layer 2). Test IDs are cited by the
// math-library validation table (ch:propulsion); do not rename them.
// Golden provenance and tolerances: tests/golden/propulsion/manifest.toml.
//
// Exactness gates use power-of-two rates and steps (0.125, 0.0625,
// 0.0078125, ...) so the documented "exactly rate * dt per step" ramp
// semantics are visible bit-exactly in binary64 instead of being blurred
// by decimal rounding.
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "golden_io.hpp"
#include "star/models/atmosphere_ussa76.hpp"
#include "star/models/propulsion.hpp"
#include "vendor/doctest.h"

namespace {

using star::models::EngineCommand;
using star::models::EngineForceTorque;
using star::models::EngineParams;
using star::models::EngineState;
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

EngineParams parse_engine(const star_tests::GoldenCase& c) {
  EngineParams p;
  p.thrust_vac_N = parse_hex_double(c.scalar("thrust_vac_N"));
  p.isp_vac_s = parse_hex_double(c.scalar("isp_vac_s"));
  p.exit_area_m2 = parse_hex_double(c.scalar("exit_area_m2"));
  p.throttle_min = 0.0;  // evaluation-path tests; limits exercised below
  p.gimbal_limit_rad = 1.0;
  p.axis = parse_vec3(c, "axis");
  p.gimbal_axis_1 = parse_vec3(c, "gimbal_axis_1");
  p.gimbal_axis_2 = parse_vec3(c, "gimbal_axis_2");
  p.position_m = parse_vec3(c, "position_m");
  return p;
}

double rel(double value, double ref) {
  return std::fabs(value - ref) / std::fabs(ref);
}

}  // namespace

TEST_CASE("PROP-ENGINE-GOLDEN") {
  // Golden gate (FR-22 layer 1): thrust magnitude, mass flow, deflected
  // direction, and force/torque coupling (eq:propulsion:thrust,
  // eq:propulsion:mdot, eq:propulsion:direction,
  // eq:propulsion:forcetorque) against the 60-digit mpmath references at
  // 1e-12 relative (norm-relative for vectors; manifest derivation: a
  // handful of roundings plus 1-2 ulp libm sin/cos, floor ~1e-15). The
  // zero-throttle case must be exactly zero in every output.
  const auto cases =
      load_golden_cases(kGoldenDir + "/propulsion/engine.toml");
  REQUIRE(cases.size() == 5);
  double worst = 0.0;
  for (const auto& c : cases) {
    CAPTURE(c.scalar("name"));
    const EngineParams p = parse_engine(c);
    const double level = parse_hex_double(c.scalar("throttle_level"));
    const double p_amb = parse_hex_double(c.scalar("p_amb_Pa"));
    const auto& g = c.array("gimbal_rad");
    REQUIRE(g.size() == 2);
    EngineState s;
    s.throttle_level = level;
    s.gimbal_rad =
        Eigen::Vector2d(parse_hex_double(g[0]), parse_hex_double(g[1]));

    const double f_ref = parse_hex_double(c.scalar("thrust_N"));
    const double mdot_ref = parse_hex_double(c.scalar("mdot_kgps"));
    const Eigen::Vector3d force_ref = parse_vec3(c, "force_N");
    const Eigen::Vector3d torque_ref = parse_vec3(c, "torque_Nm");

    const double f = star::models::engine_thrust_N(p, level, p_amb);
    const double mdot = star::models::engine_mdot_kgps(p, level);
    const EngineForceTorque ft = star::models::engine_force_torque(
        p, s, p_amb, parse_vec3(c, "cg_m"));

    if (level == 0.0) {
      CHECK(f == 0.0);
      CHECK(mdot == 0.0);
      CHECK(ft.force_N.norm() == 0.0);
      CHECK(ft.torque_Nm.norm() == 0.0);
      continue;
    }
    const double err_f = rel(f, f_ref);
    const double err_mdot = rel(mdot, mdot_ref);
    const double err_force = (ft.force_N - force_ref).norm() /
                             force_ref.norm();
    CAPTURE(err_f);
    CAPTURE(err_mdot);
    CAPTURE(err_force);
    CHECK(err_f <= 1e-12);
    CHECK(err_mdot <= 1e-12);
    CHECK(err_force <= 1e-12);
    worst = std::max({worst, err_f, err_mdot, err_force});
    if (torque_ref.norm() == 0.0) {
      CHECK(ft.torque_Nm.norm() == 0.0);  // thrust line through the CG
    } else {
      const double err_torque = (ft.torque_Nm - torque_ref).norm() /
                                torque_ref.norm();
      CAPTURE(err_torque);
      CHECK(err_torque <= 1e-12);
      worst = std::max(worst, err_torque);
    }
  }
  CAPTURE(worst);
  CHECK(worst <= 1e-12);  // prints the observed margin under -s
}

TEST_CASE("PROP-BACKPRESSURE-USSA76") {
  // Exit criterion 3: sea-level thrust matches the back-pressure formula
  // against USSA76 at h = 0. The ambient pressure comes LIVE from the
  // atmosphere model (whose z = 0 pressure is the exact standard value:
  // p = P0 * pow(1, x) = 101325 Pa), and the delivered thrust must
  // reproduce both the direct formula and the committed golden value of
  // the sea_level_full_throttle case.
  const double p_amb = star::models::ussa76_state(0.0).pressure_Pa;
  CHECK(p_amb == 101325.0);

  const auto cases =
      load_golden_cases(kGoldenDir + "/propulsion/engine.toml");
  bool found = false;
  for (const auto& c : cases) {
    if (c.scalar("name") != "sea_level_full_throttle") {
      continue;
    }
    found = true;
    const EngineParams p = parse_engine(c);
    const double f = star::models::engine_thrust_N(p, 1.0, p_amb);
    // Same formula spelled independently here, and the golden value.
    const double f_formula = p.thrust_vac_N - p_amb * p.exit_area_m2;
    const double f_golden = parse_hex_double(c.scalar("thrust_N"));
    CHECK(f == f_formula);
    const double err = rel(f, f_golden);
    CAPTURE(err);
    CHECK(err <= 1e-12);
    // Physics guards: back pressure reduces delivered thrust but never
    // the mass flow - the sea-level mdot must equal the committed golden
    // value shared with the vacuum case (eq:propulsion:mdot is
    // pressure-blind), and zero ambient pressure recovers F_vac exactly.
    CHECK(f < p.thrust_vac_N);
    const double err_mdot = rel(star::models::engine_mdot_kgps(p, 1.0),
                                parse_hex_double(c.scalar("mdot_kgps")));
    CAPTURE(err_mdot);
    CHECK(err_mdot <= 1e-12);
    CHECK(star::models::engine_thrust_N(p, 1.0, 0.0) == p.thrust_vac_N);
  }
  CHECK(found);
}

TEST_CASE("PROP-TSIOLKOVSKY-BURN") {
  // Exit criterion 3: a fixed-attitude vacuum burn integrated through the
  // model's thrust and mass flow accumulates a delta-v matching
  // g0 Isp ln(m0/m1) within 0.1 %. RK4 at 4800 steps has discretization
  // error ~1e-12 relative here, so the gate is nearly pure margin: any
  // Isp/mdot bookkeeping error (sea-level Isp, g0 slip, sign error)
  // fails at its full formula-difference scale. The endpoint mass is
  // gated at 1e-12 (mass is linear in time, so RK4 tracks it to
  // accumulated rounding).
  const auto cases =
      load_golden_cases(kGoldenDir + "/propulsion/tsiolkovsky.toml");
  REQUIRE(cases.size() == 2);
  double worst_dv = 0.0;
  for (const auto& c : cases) {
    CAPTURE(c.scalar("name"));
    EngineParams p;
    p.thrust_vac_N = parse_hex_double(c.scalar("thrust_vac_N"));
    p.isp_vac_s = parse_hex_double(c.scalar("isp_vac_s"));
    p.throttle_min = 0.0;
    const double m0 = parse_hex_double(c.scalar("m0_kg"));
    const double t_burn = parse_hex_double(c.scalar("t_burn_s"));
    const double m1_ref = parse_hex_double(c.scalar("m1_kg"));
    const double dv_ref = parse_hex_double(c.scalar("dv_mps"));

    const double f = star::models::engine_thrust_N(p, 1.0, 0.0);
    const double mdot = star::models::engine_mdot_kgps(p, 1.0);
    const int n = 4800;
    const double dt = t_burn / static_cast<double>(n);
    double v = 0.0;
    double m = m0;
    for (int k = 0; k < n; ++k) {
      // RK4 for vdot = F/m, mdot = const (m enters v's stages at the
      // correct stage masses).
      const double k1 = f / m;
      const double k2 = f / (m - 0.5 * dt * mdot);
      const double k4 = f / (m - dt * mdot);
      v += dt * (k1 + 4.0 * k2 + k4) / 6.0;  // k2 == k3 for this RHS
      m -= dt * mdot;
    }
    const double err_m = rel(m, m1_ref);
    const double err_dv = rel(v, dv_ref);
    CAPTURE(err_m);
    CAPTURE(err_dv);
    CHECK(err_m <= 1e-12);
    CHECK(err_dv <= 1e-3);  // 0.1 % exit-criterion gate
    worst_dv = std::max(worst_dv, err_dv);
  }
  CAPTURE(worst_dv);
  CHECK(worst_dv <= 1e-3);
}

TEST_CASE("PROP-SPOOL-RAMP") {
  // eq:propulsion:spool exactness: with dt/t_up = 0.125 and
  // dt/t_down = 0.0625 (powers of two), every ramp step must change the
  // level by EXACTLY that increment, hit the target exactly, and hold.
  EngineParams p;
  p.thrust_vac_N = 1000.0;
  p.isp_vac_s = 300.0;
  p.throttle_min = 0.4;
  p.throttle_max = 1.0;
  p.spool_up_s = 2.0;
  p.spool_down_s = 4.0;
  p.max_ignitions = 1;
  const double dt = 0.25;

  EngineState s;
  EngineCommand up;
  up.run = true;
  up.throttle = 1.0;
  for (int k = 1; k <= 8; ++k) {
    const EngineState next = star::models::engine_advance(p, up, s, dt);
    CHECK(next.throttle_level - s.throttle_level == 0.125);  // exact
    s = next;
  }
  CHECK(s.throttle_level == 1.0);  // exactly at target after 8 steps
  s = star::models::engine_advance(p, up, s, dt);
  CHECK(s.throttle_level == 1.0);  // holds exactly

  EngineCommand off;  // shutdown: level ramps to 0 at the down rate
  off.run = false;
  for (int k = 1; k <= 16; ++k) {
    const EngineState next = star::models::engine_advance(p, off, s, dt);
    CHECK(s.throttle_level - next.throttle_level == 0.0625);  // exact
    // Thrust and mass flow follow the LEVEL through spool-down: the
    // engine keeps thrusting while the ramp decays (ch:propulsion).
    if (next.throttle_level > 0.0) {
      CHECK(star::models::engine_mdot_kgps(p, next.throttle_level) > 0.0);
    }
    s = next;
  }
  CHECK(s.throttle_level == 0.0);  // exactly zero at ramp end
  CHECK(star::models::engine_thrust_N(p, s.throttle_level, 0.0) == 0.0);
}

TEST_CASE("PROP-THROTTLE-CLAMP") {
  // Throttle commands clamp exactly at the configured limits (immediate
  // spool so the settled level IS the clamped target).
  EngineParams p;
  p.thrust_vac_N = 1000.0;
  p.isp_vac_s = 300.0;
  p.throttle_min = 0.4;
  p.throttle_max = 0.9;
  p.max_ignitions = 1;

  EngineState s;
  EngineCommand cmd;
  cmd.run = true;
  cmd.throttle = 0.1;  // below the floor
  s = star::models::engine_advance(p, cmd, s, 1.0);
  CHECK(s.throttle_level == 0.4);  // exactly the violated bound

  cmd.throttle = 2.0;  // above the ceiling
  s = star::models::engine_advance(p, cmd, s, 1.0);
  CHECK(s.throttle_level == 0.9);

  cmd.throttle = 0.75;  // in range: settles exactly on the command
  s = star::models::engine_advance(p, cmd, s, 1.0);
  CHECK(s.throttle_level == 0.75);
}

TEST_CASE("PROP-IGNITION-COUNT") {
  // Ignition budget: exactly N run transitions are accepted; the (N+1)-th
  // is refused and the engine stays off with zero level, thrust, and
  // mass flow.
  EngineParams p;
  p.thrust_vac_N = 1000.0;
  p.isp_vac_s = 300.0;
  p.throttle_min = 0.0;
  p.max_ignitions = 2;

  EngineState s;
  EngineCommand on;
  on.run = true;
  on.throttle = 1.0;
  EngineCommand off;
  off.run = false;

  s = star::models::engine_advance(p, on, s, 1.0);  // ignition 1
  CHECK(s.running);
  CHECK(s.ignitions_used == 1);
  CHECK(s.throttle_level == 1.0);
  s = star::models::engine_advance(p, off, s, 1.0);
  CHECK_FALSE(s.running);
  CHECK(s.throttle_level == 0.0);

  s = star::models::engine_advance(p, on, s, 1.0);  // ignition 2
  CHECK(s.running);
  CHECK(s.ignitions_used == 2);
  s = star::models::engine_advance(p, off, s, 1.0);

  s = star::models::engine_advance(p, on, s, 1.0);  // refused
  CHECK_FALSE(s.running);
  CHECK(s.ignitions_used == 2);  // no count consumed by the refusal
  CHECK(s.throttle_level == 0.0);
  CHECK(star::models::engine_thrust_N(p, s.throttle_level, 0.0) == 0.0);
  CHECK(star::models::engine_mdot_kgps(p, s.throttle_level) == 0.0);
}

TEST_CASE("PROP-TVC-RATE-LIMIT") {
  // Exit criterion 7: a TVC step command produces a ramp at exactly the
  // configured gimbal rate limit. Rate 2^-6 rad/s and dt 0.5 s give a
  // per-step increment of exactly 2^-7 rad; the limit 2^-5 rad is hit
  // exactly after 4 steps and the angle clamps there under the
  // over-limit command (eq:propulsion:gimbalslew).
  EngineParams p;
  p.thrust_vac_N = 1000.0;
  p.isp_vac_s = 300.0;
  p.throttle_min = 0.0;
  p.gimbal_limit_rad = 0.03125;    // 2^-5
  p.gimbal_rate_radps = 0.015625;  // 2^-6
  const double dt = 0.5;
  const double step = 0.0078125;   // rate * dt = 2^-7, exact

  EngineState s;
  EngineCommand cmd;
  cmd.run = false;
  cmd.gimbal_rad = Eigen::Vector2d(0.1, -0.1);  // far beyond the limit
  for (int k = 1; k <= 4; ++k) {
    const EngineState next = star::models::engine_advance(p, cmd, s, dt);
    CHECK(next.gimbal_rad[0] - s.gimbal_rad[0] == step);   // exact ramp
    CHECK(s.gimbal_rad[1] - next.gimbal_rad[1] == step);   // both axes
    s = next;
  }
  CHECK(s.gimbal_rad[0] == p.gimbal_limit_rad);   // exactly at the clamp
  CHECK(s.gimbal_rad[1] == -p.gimbal_limit_rad);
  s = star::models::engine_advance(p, cmd, s, dt);
  CHECK(s.gimbal_rad[0] == p.gimbal_limit_rad);   // clamped, no creep
  CHECK(s.gimbal_rad[1] == -p.gimbal_limit_rad);

  // In-range step command: ramps at the exact rate, then settles exactly
  // on the command with no overshoot.
  EngineState t;
  cmd.gimbal_rad = Eigen::Vector2d(0.015625, 0.0);  // 2 steps away
  t = star::models::engine_advance(p, cmd, t, dt);
  CHECK(t.gimbal_rad[0] == step);
  t = star::models::engine_advance(p, cmd, t, dt);
  CHECK(t.gimbal_rad[0] == 0.015625);
  t = star::models::engine_advance(p, cmd, t, dt);
  CHECK(t.gimbal_rad[0] == 0.015625);  // holds exactly
}

TEST_CASE("PROP-DOMAIN") {
  // Out-of-domain behavior per ch:propulsion: std::domain_error.
  EngineParams good;
  good.thrust_vac_N = 1000.0;
  good.isp_vac_s = 300.0;
  good.throttle_min = 0.0;

  EngineParams p = good;
  p.isp_vac_s = 0.0;
  CHECK_THROWS_AS(star::models::engine_mdot_kgps(p, 1.0),
                  std::domain_error);
  p = good;
  p.thrust_vac_N = -1.0;
  CHECK_THROWS_AS(star::models::engine_thrust_N(p, 1.0, 0.0),
                  std::domain_error);
  p = good;
  p.axis = Eigen::Vector3d(1.0, 1.0, 0.0);  // not unit
  CHECK_THROWS_AS(
      star::models::engine_thrust_direction(p, Eigen::Vector2d::Zero()),
      std::domain_error);
  p = good;
  p.gimbal_axis_1 = Eigen::Vector3d::UnitX();  // parallel to axis
  CHECK_THROWS_AS(
      star::models::engine_thrust_direction(p, Eigen::Vector2d::Zero()),
      std::domain_error);
  p = good;
  p.throttle_min = 0.8;
  p.throttle_max = 0.5;  // inverted limits
  CHECK_THROWS_AS(star::models::engine_thrust_N(p, 1.0, 0.0),
                  std::domain_error);

  CHECK_THROWS_AS(star::models::engine_thrust_N(good, 1.5, 0.0),
                  std::domain_error);  // level beyond [0, 1]
  CHECK_THROWS_AS(star::models::engine_thrust_N(good, 1.0, -1.0),
                  std::domain_error);  // negative ambient pressure
  CHECK_THROWS_AS(
      star::models::engine_advance(good, EngineCommand{}, EngineState{},
                                   -1.0),
      std::domain_error);  // negative dt
}
