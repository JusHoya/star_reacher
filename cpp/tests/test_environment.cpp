// Composed environment force model tests (ch:environment). The physics of
// each term is validated in its own module's golden suite; these cases pin
// the COMPOSITION: summation order and wiring, the frame/time plumbing, the
// FR-8 co-rotating v_rel definition, whole-run bit determinism through
// run_env, and the constructor's error paths.
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "star/constants.hpp"
#include "star/ephemeris.hpp"
#include "star/frames.hpp"
#include "star/models/atmosphere_hp.hpp"
#include "star/models/atmosphere_ussa76.hpp"
#include "star/models/drag.hpp"
#include "star/models/environment.hpp"
#include "star/models/gravity.hpp"
#include "star/models/srp.hpp"
#include "star/models/thirdbody.hpp"
#include "star/run.hpp"
#include "star/time.hpp"
#include "vendor/doctest.h"

using star::RunConfig;
using star::constants::GM_MOON_DE440_M3_PER_S2;
using star::constants::GM_SUN_DE440_M3_PER_S2;
using star::constants::OMEGA_EARTH_RAD_PER_S;
using star::constants::R_SUN_M;
using star::constants::WGS84_A_M;
using star::constants::WGS84_INV_F;
using star::models::EnvironmentModel;
using star::models::EnvironmentSpec;
using star::models::GravityField;
using star::models::GravityTier;
using star::models::PinesGravity;

namespace {

const std::string kGoldenDir = STAR_GOLDEN_DIR;
const std::string kEarthN20 = kGoldenDir + "/gravity/earth_egm2008_n20.srgrav";
const std::string kEphCrosstool =
    kGoldenDir + "/ephemeris/excerpt_de440s_crosstool.sreph";

// 2026-01-01T00:00:00 UTC on the TAI scale: 9497 whole TAI days since
// 2000-01-01T00:00:00.0 TAI (six leap days 2000..2024, 26*365 + 7 = 9497)
// plus TAI-UTC = 37 s per the bundled leap table. Matches
// core.utc_to_tai(2026, 1, 1, 0, 0, 0.0).
constexpr std::int64_t kEpochTaiDay = 9497;
constexpr double kEpochTaiSec = 37.0;

double tdb_s_since_j2000(const star::time::TaiEpoch& tai) {
  const star::time::TwoPartJd jd = star::time::tdb_jd(tai);
  return ((jd.jd1 - 2451545.0) + jd.jd2) * 86400.0;
}

std::vector<char> read_bytes(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  REQUIRE(in.good());
  return std::vector<char>((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());
}

}  // namespace

TEST_CASE("env_composition_matches_sum_of_terms") {
  // Full Earth stack: 8x8 harmonic gravity, Sun+Moon third bodies, SRP with
  // the central-body occulter, Harris-Priester drag. The model's total
  // acceleration must equal the sum of the individually evaluated terms
  // recomposed here with the same frame chain, ephemeris composition, and
  // summation order - a wiring error (wrong frame, wrong GM, wrong order of
  // rotation) shows as an O(1) discrepancy against the 1e-12 gate.
  EnvironmentSpec spec;
  spec.central_body = star::models::CentralBody::kEarth;
  spec.epoch_tai = {kEpochTaiDay, kEpochTaiSec};
  spec.gravity_model = "harmonic";
  spec.gravity_field_path = kEarthN20;
  spec.gravity_degree = 8;
  spec.gravity_order = 8;
  spec.third_bodies = {"sun", "moon"};
  spec.srp_enabled = true;
  spec.cr_a_over_m_m2pkg = 0.02;
  spec.srp_occulters = {"earth"};
  spec.atmosphere = star::models::AtmosphereModel::kHarrisPriester;
  spec.cd_a_over_m_m2pkg = 0.0044;
  spec.hp_exponent_n = 4.0;
  spec.ephemeris_path = kEphCrosstool;
  EnvironmentModel model(spec);

  const Eigen::Vector3d r(6878000.0, 0.0, 0.0);
  const Eigen::Vector3d v(0.0, 7350.0, 2000.0);
  const double t_s = 123.0;
  const Eigen::Vector3d a = model.acceleration(t_s, r, v);

  // --- independent recomposition -----------------------------------------
  const star::time::TaiEpoch tai =
      star::time::tai_add_seconds({kEpochTaiDay, kEpochTaiSec}, t_s);
  const double tdb_s = tdb_s_since_j2000(tai);
  const Eigen::Matrix3d c_bf = star::frames::c_gcrf_to_itrf(tai, 0.0);

  PinesGravity pines(GravityField::load_file(kEarthN20));
  const Eigen::Vector3d a_grav =
      c_bf.transpose() *
      pines.acceleration(c_bf * r, GravityTier::kFull, 8, 8);

  const star::Ephemeris eph = star::Ephemeris::load_file(kEphCrosstool);
  const Eigen::Vector3d r_earth_ssb =
      eph.state("emb", tdb_s).r_m + eph.state("earth", tdb_s).r_m;
  const Eigen::Vector3d r_sun = eph.state("sun", tdb_s).r_m - r_earth_ssb;
  const Eigen::Vector3d r_moon = eph.moon_geocentric(tdb_s).r_m;
  const Eigen::Vector3d a_tb =
      star::models::thirdbody_accel(GM_SUN_DE440_M3_PER_S2, r, r_sun) +
      star::models::thirdbody_accel(GM_MOON_DE440_M3_PER_S2, r, r_moon);

  const double nu = star::models::shadow_fraction(
      r, r_sun, R_SUN_M, Eigen::Vector3d::Zero(), WGS84_A_M);
  const Eigen::Vector3d a_srp = star::models::srp_accel(0.02, nu, r, r_sun);

  const double alt_m =
      star::models::geodetic_altitude(c_bf * r, WGS84_A_M, WGS84_INV_F);
  const Eigen::Vector3d apex =
      star::models::hp_bulge_apex(r_sun.normalized());
  const double cos_psi = r.normalized().dot(apex);
  const double rho = star::models::hp_density(alt_m, cos_psi, 4.0);
  const Eigen::Vector3d omega_vec =
      OMEGA_EARTH_RAD_PER_S * c_bf.row(2).transpose();
  const Eigen::Vector3d a_drag =
      star::models::drag_accel(rho, 0.0044, v - omega_vec.cross(r));

  const Eigen::Vector3d a_sum = a_grav + a_tb + a_srp + a_drag;
  CAPTURE(a.transpose());
  CAPTURE(a_sum.transpose());
  CHECK((a - a_sum).norm() <= 1e-12 * a.norm());

  // Guard the test's own strength: the perturbation terms must be nonzero,
  // otherwise a dropped term could not be detected. (SRP legitimately
  // vanishes in umbra, so it is guarded via nu's range instead.)
  CHECK(a_grav.norm() > 1.0);
  CHECK(a_tb.norm() > 0.0);
  CHECK(a_drag.norm() > 0.0);
  CHECK(nu >= 0.0);
  CHECK(nu <= 1.0);
}

TEST_CASE("env_vrel_corotating_definition") {
  // FR-8: drag uses v_rel = v - omega x r with omega along the body-fixed
  // z-axis resolved in GCRF. Two models differing only in drag isolate the
  // drag term; it must match the manual co-rotating evaluation and must NOT
  // match an inertial-velocity evaluation (the ~480 m/s co-rotation at LEO
  // changes the drag by far more than the comparison floor).
  EnvironmentSpec base;
  base.central_body = star::models::CentralBody::kEarth;
  base.epoch_tai = {kEpochTaiDay, kEpochTaiSec};
  base.gravity_model = "pointmass";
  EnvironmentSpec with_drag = base;
  with_drag.atmosphere = star::models::AtmosphereModel::kUssa76;
  with_drag.cd_a_over_m_m2pkg = 0.0044;
  EnvironmentModel model_base(base);
  EnvironmentModel model_drag(with_drag);

  const Eigen::Vector3d r(6678137.0, 0.0, 0.0);  // ~300 km, USSA76 domain
  const Eigen::Vector3d v(0.0, 7700.0, 0.0);
  const double t_s = 10.0;
  const Eigen::Vector3d diff =
      model_drag.acceleration(t_s, r, v) - model_base.acceleration(t_s, r, v);

  const star::time::TaiEpoch tai =
      star::time::tai_add_seconds({kEpochTaiDay, kEpochTaiSec}, t_s);
  const Eigen::Matrix3d c_bf = star::frames::c_gcrf_to_itrf(tai, 0.0);
  const double alt_m =
      star::models::geodetic_altitude(c_bf * r, WGS84_A_M, WGS84_INV_F);
  const double rho = star::models::ussa76_density(alt_m);
  const Eigen::Vector3d omega_vec =
      OMEGA_EARTH_RAD_PER_S * c_bf.row(2).transpose();
  const Eigen::Vector3d a_corot =
      star::models::drag_accel(rho, 0.0044, v - omega_vec.cross(r));
  const Eigen::Vector3d a_inertial = star::models::drag_accel(rho, 0.0044, v);

  CHECK(a_corot.norm() > 0.0);
  // (g + d) - g reconstructs d to within rounding of the dominant gravity
  // term (~1e-14 m/s^2), orders below the co-rotation signature.
  CHECK((diff - a_corot).norm() <= 1e-12);
  CHECK((diff - a_inertial).norm() > 1e-8);
}

TEST_CASE("env_run_double_run_bit_identity") {
  // D-10 through the whole run_env path: identical configs give bit-identical
  // SRLOG bytes, for both integrators. Point-mass + USSA76 drag keeps the
  // case hermetic (no ephemeris file needed).
  RunConfig cfg;
  cfg.epoch_utc = "2026-01-01T00:00:00Z";
  cfg.epoch_tai_day = kEpochTaiDay;
  cfg.epoch_tai_sec = kEpochTaiSec;
  cfg.duration_s = 120.0;
  cfg.central_body = "earth";
  cfg.r0_m = {6678137.0, 0.0, 0.0};
  cfg.v0_mps = {0.0, 7700.0, 0.0};
  cfg.mass_kg = 100.0;
  cfg.master_seed = 7;
  cfg.truth_rate_hz = 1;
  cfg.config_sha256 = std::string(64, '0');
  cfg.gravity_model = "pointmass";
  cfg.drag_enabled = true;
  cfg.atmosphere = "ussa76";
  cfg.cd_a_over_m_m2pkg = 0.0044;

  RunConfig rk4 = cfg;
  rk4.integrator = "rk4";
  rk4.dt_s = 1.0;
  RunConfig rkf = cfg;
  rkf.integrator = "rkf78";
  rkf.rtol = 1e-10;
  rkf.atol_pos_m = 1e-6;
  rkf.atol_vel_mps = 1e-9;
  rkf.h_init_s = 10.0;
  rkf.h_max_s = 10.0;

  const star::RunSummary s1 = star::run_env(rk4, "env_test_rk4_a.srlog");
  const star::RunSummary s2 = star::run_env(rk4, "env_test_rk4_b.srlog");
  CHECK(read_bytes("env_test_rk4_a.srlog") ==
        read_bytes("env_test_rk4_b.srlog"));
  CHECK(s1.steps == 120);
  CHECK(s1.truth_records == 121);  // t = 0 plus one per second
  CHECK(s2.truth_records == s1.truth_records);

  const star::RunSummary s3 = star::run_env(rkf, "env_test_rkf_a.srlog");
  const star::RunSummary s4 = star::run_env(rkf, "env_test_rkf_b.srlog");
  CHECK(read_bytes("env_test_rkf_a.srlog") ==
        read_bytes("env_test_rkf_b.srlog"));
  CHECK(s3.truth_records == 121);  // dense-output sampling on the same grid
  CHECK(s3.steps == s4.steps);

  // Both integrators propagate the same dynamics: 120 s of point-mass +
  // drag agree to well under a metre between rk4 at 1 s and rkf78 at 1e-10.
  const Eigen::Vector3d r_rk4(s1.final_r_m[0], s1.final_r_m[1],
                              s1.final_r_m[2]);
  const Eigen::Vector3d r_rkf(s3.final_r_m[0], s3.final_r_m[1],
                              s3.final_r_m[2]);
  CHECK((r_rk4 - r_rkf).norm() < 1.0);
}

TEST_CASE("env_error_paths") {
  EnvironmentSpec good;
  good.central_body = star::models::CentralBody::kEarth;
  good.epoch_tai = {kEpochTaiDay, kEpochTaiSec};

  // Unknown third-body name.
  {
    EnvironmentSpec s = good;
    s.third_bodies = {"phobos"};
    CHECK_THROWS_AS(EnvironmentModel{s}, std::invalid_argument);
  }
  // The central body cannot perturb itself.
  {
    EnvironmentSpec s = good;
    s.third_bodies = {"earth"};
    s.ephemeris_path = kEphCrosstool;
    CHECK_THROWS_AS(EnvironmentModel{s}, std::invalid_argument);
  }
  // A configuration that needs the ephemeris must name one.
  {
    EnvironmentSpec s = good;
    s.third_bodies = {"sun"};
    CHECK_THROWS_AS(EnvironmentModel{s}, std::invalid_argument);
  }
  // Earth atmospheres are refused off Earth; Mars atmosphere off Mars.
  {
    EnvironmentSpec s = good;
    s.central_body = star::models::CentralBody::kMars;
    s.atmosphere = star::models::AtmosphereModel::kHarrisPriester;
    s.cd_a_over_m_m2pkg = 0.001;
    CHECK_THROWS_AS(EnvironmentModel{s}, std::invalid_argument);
  }
  {
    EnvironmentSpec s = good;
    s.atmosphere = star::models::AtmosphereModel::kMarsExponential;
    s.cd_a_over_m_m2pkg = 0.001;
    CHECK_THROWS_AS(EnvironmentModel{s}, std::invalid_argument);
  }
  // SRP demands an occulter set and a positive Cr*A/m.
  {
    EnvironmentSpec s = good;
    s.srp_enabled = true;
    s.cr_a_over_m_m2pkg = 0.01;
    s.srp_occulters = {};
    CHECK_THROWS_AS(EnvironmentModel{s}, std::invalid_argument);
  }
  // Unknown gravity model string.
  {
    EnvironmentSpec s = good;
    s.gravity_model = "j3";
    CHECK_THROWS_AS(EnvironmentModel{s}, std::invalid_argument);
  }

  // gm() single-home wiring for the new central bodies (constants.hpp cites
  // the DE440 provenance).
  CHECK(star::gm("moon") == GM_MOON_DE440_M3_PER_S2);
  CHECK(star::gm("mars") == star::constants::GM_MARS_SYS_DE440_M3_PER_S2);
  CHECK(star::gm("earth") == star::constants::GM_EARTH_M3_PER_S2);

  // run_env defensive config checks (mis-wired caller fails fast).
  RunConfig bad;
  bad.epoch_utc = "2026-01-01T00:00:00Z";
  bad.duration_s = 60.0;
  bad.integrator = "rkf78";
  bad.config_sha256 = std::string(64, '0');
  bad.r0_m = {6678137.0, 0.0, 0.0};
  bad.v0_mps = {0.0, 7700.0, 0.0};
  // h_init/h_max/tolerances left at zero: rejected before any file I/O.
  CHECK_THROWS_AS(star::run_env(bad, "env_test_bad.srlog"),
                  std::invalid_argument);
}
