// Vehicle 6DOF composition tests (Phase 4 exit criteria 2, 3, 5, 7, 10). These
// exercise the load-bearing new physics of the phase directly on constructed
// states -- the geodetic pad initial state (EC-10), the closed-form staging CG
// jump and wet-mass identity (EC-2), staging linear/angular momentum
// conservation with the FR-10 remap (EC-5), the rocket-equation burnout under a
// system-level Isp edit (EC-3), and the RCS/wheel/TVC actuator hooks (EC-7) --
// so each property holds independently of the full run path and its logging.
// Test IDs are cited by the math-library validation table (ch:vehicle6dof); do
// not rename them.
#include <cmath>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "star/constants.hpp"
#include "star/frames.hpp"
#include "star/models/actuators.hpp"
#include "star/models/aero.hpp"
#include "star/models/massprops.hpp"
#include "star/models/propulsion.hpp"
#include "star/models/vehicle6dof.hpp"
#include "star/rotation.hpp"
#include "star/time.hpp"
#include "vendor/doctest.h"

namespace {

double d2r(double deg) { return deg * (star::constants::TWO_PI / 360.0); }

star::models::BodyProps make_body(double m, const Eigen::Vector3d& cg,
                                  const Eigen::Vector3d& idiag) {
  star::models::BodyProps b;
  b.mass_kg = m;
  b.cg_m = cg;
  b.inertia_kgm2 = idiag.asDiagonal();
  return b;
}

}  // namespace

TEST_CASE("vehicle6dof_pad_release_exactness") {
  // EC-10: the geodetic launch-pad inertial velocity equals omega_earth x r to
  // 1e-12; the logged dynamic pressure before release is exactly zero (the aero
  // structural zero at |v_rel| == 0); and due-east vs due-west ascents differ
  // in along-track inertial speed by 2 omega R cos(lat) (2 omega x the distance
  // from the spin axis).
  const double lat = d2r(-39.0);
  const double lon = d2r(177.9);
  const double alt = 10.0;
  const star::time::TaiEpoch epoch{9497, 43200.0};
  const Eigen::Matrix3d c_itrf = star::frames::c_gcrf_to_itrf(epoch, 0.0);
  const star::models::PadState pad = star::models::geodetic_pad_state(
      lat, lon, alt, c_itrf, star::constants::OMEGA_EARTH_RAD_PER_S,
      star::constants::WGS84_A_M, star::constants::WGS84_INV_F);

  const Eigen::Vector3d omega_i =
      star::constants::OMEGA_EARTH_RAD_PER_S * c_itrf.row(2).transpose();
  const Eigen::Vector3d v_expected = omega_i.cross(pad.r_i_m);
  const double v_err =
      (pad.v_i_mps - v_expected).norm() / v_expected.norm();
  CAPTURE(v_err);
  CHECK(v_err <= 1e-12);

  // On-pad dynamic pressure: |v_rel| == 0 gives an exact-zero q_bar.
  star::models::AeroTables tab;
  tab.ref_area_m2 = 1.13;
  tab.ref_diameter_m = 1.2;
  tab.cmq_per_rad = 0.0;
  tab.mach = {0.0, 1.0};
  tab.ca = {0.3, 0.5};
  tab.cnalpha_per_rad = {2.0, 3.0};
  tab.xcp_m = {13.0, 12.0};
  const star::models::AeroForceTorque a = star::models::aero_force_torque(
      tab, Eigen::Vector3d::Zero(), 1.225, 340.0, 8.0,
      Eigen::Vector3d::Zero());
  CHECK(a.q_bar_Pa == 0.0);
  CHECK(a.force_N[0] == 0.0);
  CHECK(a.force_N[1] == 0.0);
  CHECK(a.force_N[2] == 0.0);

  // Along-track difference: due-east projects on +east, due-west on -east.
  const double v_east = pad.v_i_mps.dot(pad.east_i);
  const Eigen::Vector3d r_ecef = c_itrf * pad.r_i_m;
  const double dist_axis = std::hypot(r_ecef.x(), r_ecef.y());
  const double diff = v_east - (-v_east);
  const double diff_ref = 2.0 * star::constants::OMEGA_EARTH_RAD_PER_S * dist_axis;
  CAPTURE(diff);
  CAPTURE(diff_ref);
  CHECK(std::fabs(diff - diff_ref) <= 1e-9 * diff_ref);
}

TEST_CASE("vehicle6dof_staging_cg_jump") {
  // EC-2: wet mass at t0 equals dry + propellant exactly; the staging CG jump
  // matches the closed-form mass properties to 1e-12 relative, and remove_body
  // reproduces the direct composition of the retained stack.
  const star::models::BodyProps s1 =
      make_body(1000.0, Eigen::Vector3d(6.0, 0.0, 0.0),
                Eigen::Vector3d(300.0, 12000.0, 12000.0));
  const star::models::BodyProps s2 =
      make_body(250.0, Eigen::Vector3d(14.5, 0.0, 0.0),
                Eigen::Vector3d(60.0, 400.0, 400.0));
  const star::models::BodyProps slug1 =
      make_body(9500.0, Eigen::Vector3d(6.5, 0.0, 0.0),
                Eigen::Vector3d(1400.0, 9000.0, 9000.0));
  const star::models::BodyProps slug2 =
      make_body(2150.0, Eigen::Vector3d(14.0, 0.0, 0.0),
                Eigen::Vector3d(260.0, 350.0, 350.0));

  const std::vector<star::models::BodyProps> full = {s1, s2, slug1, slug2};
  const star::models::BodyProps composite = star::models::compose(full);
  // Wet-mass identity: exact same-order sum.
  CHECK(composite.mass_kg ==
        s1.mass_kg + s2.mass_kg + slug1.mass_kg + slug2.mass_kg);

  // Retained stack after separating stage 1 (its dry body and slug leave).
  const star::models::BodyProps retained =
      star::models::compose({s2, slug2});
  const double m_ret = s2.mass_kg + slug2.mass_kg;
  const Eigen::Vector3d cg_ret_cf =
      (s2.mass_kg * s2.cg_m + slug2.mass_kg * slug2.cg_m) / m_ret;
  CHECK((retained.cg_m - cg_ret_cf).norm() <= 1e-12 * cg_ret_cf.norm());

  // remove_body of the jettisoned composite reproduces the retained props.
  const star::models::BodyProps jett = star::models::compose({s1, slug1});
  const star::models::BodyProps ret2 =
      star::models::remove_body(composite, jett);
  CHECK((ret2.cg_m - retained.cg_m).norm() <= 1e-12 * retained.cg_m.norm());
  CHECK((ret2.inertia_kgm2 - retained.inertia_kgm2).norm() <=
        1e-12 * retained.inertia_kgm2.norm());
}

TEST_CASE("vehicle6dof_staging_momentum_conservation") {
  // EC-5: across a torque-free separation the retained plus jettisoned linear
  // and angular momentum (jettisoned evaluated at its own CG) equals the
  // pre-separation total to 1e-12 relative; the tracked-state remap follows
  // FR-10 (v_new = v + omega x Delta r_cg) with omega unchanged.
  const star::models::BodyProps s1 =
      make_body(1000.0, Eigen::Vector3d(6.0, 0.1, -0.2),
                Eigen::Vector3d(300.0, 12000.0, 12000.0));
  const star::models::BodyProps s2 =
      make_body(2400.0, Eigen::Vector3d(14.2, -0.05, 0.1),
                Eigen::Vector3d(320.0, 700.0, 700.0));
  const star::models::BodyProps composite = star::models::compose({s1, s2});
  const star::models::BodyProps retained = star::models::compose({s2});
  const star::models::BodyProps jett = star::models::compose({s1});

  const Eigen::Quaterniond q = star::rotation::quat_normalize(
      Eigen::Quaterniond(0.6, 0.3, -0.5, 0.4));
  const Eigen::Matrix3d c_b2i = star::rotation::dcm_from_quat(q).transpose();
  const Eigen::Vector3d omega_b(0.02, -0.015, 0.03);
  const Eigen::Vector3d omega_i = c_b2i * omega_b;
  const Eigen::Vector3d r_cg_i(7.0e6, 1.0e5, -2.0e5);
  const Eigen::Vector3d v_cg_i(120.0, -40.0, 55.0);

  const Eigen::Vector3d dr_ret_i = c_b2i * (retained.cg_m - composite.cg_m);
  const Eigen::Vector3d dr_jett_i = c_b2i * (jett.cg_m - composite.cg_m);
  const Eigen::Vector3d v_ret = v_cg_i + omega_i.cross(dr_ret_i);
  const Eigen::Vector3d v_jett = v_cg_i + omega_i.cross(dr_jett_i);

  // Linear momentum is conserved exactly (the mass-weighted CG offsets sum to
  // zero, so the v_cg terms cancel).
  const Eigen::Vector3d p_pre = composite.mass_kg * v_cg_i;
  const Eigen::Vector3d p_post =
      retained.mass_kg * v_ret + jett.mass_kg * v_jett;
  CHECK((p_post - p_pre).norm() <= 1e-12 * p_pre.norm());

  // Angular momentum about the composite CG is independent of v_cg, so it is
  // evaluated in the body frame from the rotation-only velocities (omega x dr):
  // the identity is exactly the parallel-axis theorem the composition applies.
  const Eigen::Vector3d L_pre_b = composite.inertia_kgm2 * omega_b;
  auto body_L_b = [&](const star::models::BodyProps& b) -> Eigen::Vector3d {
    const Eigen::Vector3d dr = b.cg_m - composite.cg_m;
    const Eigen::Vector3d orbital = b.mass_kg * dr.cross(omega_b.cross(dr));
    return b.inertia_kgm2 * omega_b + orbital;
  };
  const Eigen::Vector3d L_post_b = body_L_b(retained) + body_L_b(jett);
  CHECK((L_post_b - L_pre_b).norm() <= 1e-12 * L_pre_b.norm());

  // The tracked-state remap the run path applies reproduces the retained CG
  // velocity (FR-10), and omega is untouched.
  const star::models::SeparationRemap rm = star::models::separation_remap(
      composite.cg_m, retained.cg_m, r_cg_i, v_cg_i, q, omega_b);
  CHECK((rm.v_new_i_mps - v_ret).norm() <= 1e-12 * v_ret.norm());
  CHECK((rm.r_new_i_m - (r_cg_i + dr_ret_i)).norm() <=
        1e-9 * r_cg_i.norm());
}

TEST_CASE("vehicle6dof_isp_burnout_rocket_equation") {
  // EC-3: a straight-line vacuum burn integrates the engine thrust and mass
  // flow the run path uses; the burnout velocity matches Tsiolkovsky within
  // 1%, and a +10 s Isp edit moves it by the rocket-equation-predicted amount.
  const double m0 = 2400.0;
  const double prop = 2150.0;
  const double mf = m0 - prop;
  auto burnout = [&](double isp_s) {
    star::models::EngineParams p;
    p.thrust_vac_N = 26000.0;
    p.isp_vac_s = isp_s;
    p.exit_area_m2 = 0.35;
    p.throttle_min = 1.0;
    p.throttle_max = 1.0;
    p.max_ignitions = 1;
    double m = m0;
    double v = 0.0;
    double consumed = 0.0;
    const double dt = 0.02;
    while (consumed < prop) {
      const double thr = star::models::engine_thrust_N(p, 1.0, 0.0);
      v += (thr / m) * dt;
      double dm = star::models::engine_mdot_kgps(p, 1.0) * dt;
      if (consumed + dm > prop) dm = prop - consumed;
      m -= dm;
      consumed += dm;
    }
    return v;
  };
  const double g0 = star::models::STANDARD_GRAVITY_MPS2;
  const double v1 = burnout(343.0);
  const double v2 = burnout(353.0);
  const double tsiol1 = 343.0 * g0 * std::log(m0 / mf);
  const double predicted_change = 10.0 * g0 * std::log(m0 / mf);
  CAPTURE(v1);
  CAPTURE(tsiol1);
  CHECK(std::fabs(v1 - tsiol1) / tsiol1 < 0.01);
  CAPTURE(v2 - v1);
  CAPTURE(predicted_change);
  CHECK(std::fabs((v2 - v1) - predicted_change) / predicted_change < 0.01);
}

TEST_CASE("vehicle6dof_actuator_hooks") {
  // EC-7 (the actuator hooks the run path composes): RCS MIB gating, reaction-
  // wheel total-momentum conservation with a saturation clamp, and the TVC
  // gimbal ramp at exactly the configured rate limit.
  star::models::RcsThrusterParams th;
  th.position_m = Eigen::Vector3d(0.7, 0.5, 0.0);
  th.direction = Eigen::Vector3d(0.0, 1.0, 0.0);
  th.thrust_N = 10.0;
  th.mib_Ns = 0.02;
  const Eigen::Vector3d cg(0.7, 0.0, 0.0);
  const star::models::RcsImpulse below =
      star::models::rcs_pulse(th, 0.001, cg);  // 0.01 Ns < 0.02 MIB
  CHECK(below.delivered_Ns == 0.0);
  const star::models::RcsImpulse twice =
      star::models::rcs_pulse(th, 0.004, cg);  // 0.04 Ns = 2 x MIB
  CHECK(std::fabs(twice.delivered_Ns - 0.04) <= 1e-12 * 0.04);

  // Reaction wheel: a slew reacts on the body; total angular momentum
  // (I omega + h axis) is conserved through the exchange.
  star::models::WheelParams w;
  w.axis = Eigen::Vector3d(0.0, 0.0, 1.0);
  w.torque_max_Nm = 0.1;
  w.momentum_max_Nms = 1.0;
  const Eigen::Matrix3d inertia = Eigen::Vector3d(50.0, 60.0, 40.0).asDiagonal();
  Eigen::Vector3d omega(0.0, 0.0, 0.01);
  star::models::WheelState ws{0.0};
  const std::vector<star::models::WheelParams> wheels = {w};
  const double dt = 0.1;
  const Eigen::Vector3d L0 = star::models::total_angular_momentum_Nms(
      inertia, omega, wheels, {ws});
  const star::models::WheelStepResult step =
      star::models::wheel_step(w, 0.05, ws, dt);
  omega += inertia.inverse() * step.body_torque_Nm * dt;
  const Eigen::Vector3d L1 = star::models::total_angular_momentum_Nms(
      inertia, omega, wheels, {step.state});
  CHECK((L1 - L0).norm() <= 1e-12 * L0.norm());

  // Torque saturation clamps exactly at the configured maximum.
  const star::models::WheelStepResult sat =
      star::models::wheel_step(w, 1.0, star::models::WheelState{0.0}, dt);
  CHECK(std::fabs(sat.torque_Nm) <= w.torque_max_Nm + 1e-15);
  CHECK(std::fabs(std::fabs(sat.torque_Nm) - w.torque_max_Nm) <= 1e-12);

  // TVC: one gimbal step advances by exactly gimbal_rate * dt (below the
  // angle limit).
  star::models::EngineParams ep;
  ep.thrust_vac_N = 234000.0;
  ep.isp_vac_s = 303.0;
  ep.exit_area_m2 = 0.27;
  ep.throttle_min = 0.5;
  ep.throttle_max = 1.0;
  ep.max_ignitions = 1;
  ep.gimbal_limit_rad = d2r(5.0);
  ep.gimbal_rate_radps = d2r(10.0);
  star::models::EngineCommand cmd;
  cmd.run = true;
  cmd.throttle = 1.0;
  cmd.gimbal_rad = Eigen::Vector2d(d2r(4.0), 0.0);  // beyond one step
  const star::models::EngineState es0;
  const star::models::EngineState es1 =
      star::models::engine_advance(ep, cmd, es0, 0.01);
  const double expected_step = ep.gimbal_rate_radps * 0.01;
  CHECK(std::fabs(es1.gimbal_rad[0] - expected_step) <= 1e-12 * expected_step);
}
