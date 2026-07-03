// Axisymmetric-aero tests (FR-9, Phase 4 exit criterion 8): golden
// coefficient lookups and force/moment reconstructions against the
// independent 60-digit mpmath references (FR-22 layer 1), the exact
// structural zeros the pad and zero-alpha semantics depend on (Phase 4
// exit criterion 10 hook), and the out-of-domain contract. Test IDs are
// cited by the math-library validation table (ch:aero); do not rename
// them. Golden provenance and tolerances:
// tests/golden/aero/manifest.toml.
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "golden_io.hpp"
#include "star/models/aero.hpp"
#include "vendor/doctest.h"

namespace {

using star::models::AeroCoefficients;
using star::models::AeroForceTorque;
using star::models::AeroTables;
using star_tests::GoldenCase;
using star_tests::load_golden_cases;
using star_tests::parse_hex_double;

const std::string kGoldenDir = STAR_GOLDEN_DIR;

Eigen::Vector3d parse_vec3(const GoldenCase& c, const std::string& key) {
  const auto& a = c.array(key);
  REQUIRE(a.size() == 3);
  return Eigen::Vector3d(parse_hex_double(a[0]), parse_hex_double(a[1]),
                         parse_hex_double(a[2]));
}

std::vector<double> parse_column(const GoldenCase& c,
                                 const std::string& key) {
  std::vector<double> out;
  for (const auto& s : c.array(key)) {
    out.push_back(parse_hex_double(s));
  }
  return out;
}

// The committed Mach-table columns (copies of the fleet CSVs plus the
// synthetic clamp table), keyed by name; ref area/diameter/cmq are
// per-case values the caller overwrites.
std::map<std::string, AeroTables> load_tables() {
  std::map<std::string, AeroTables> out;
  for (const auto& c :
       load_golden_cases(kGoldenDir + "/aero/tables.toml")) {
    AeroTables t;
    t.ref_area_m2 = 1.0;
    t.ref_diameter_m = 1.0;
    t.mach = parse_column(c, "mach");
    t.ca = parse_column(c, "ca");
    t.cnalpha_per_rad = parse_column(c, "cnalpha_per_rad");
    t.xcp_m = parse_column(c, "xcp_m");
    out[c.scalar("name")] = t;
  }
  return out;
}

// Run one force/torque golden case (breakpoints.toml / forcetorque.toml
// share the schema).
AeroForceTorque run_case(const GoldenCase& c,
                         const std::map<std::string, AeroTables>& tables) {
  AeroTables t = tables.at(c.scalar("table"));
  t.ref_area_m2 = parse_hex_double(c.scalar("ref_area_m2"));
  t.ref_diameter_m = parse_hex_double(c.scalar("ref_diameter_m"));
  t.cmq_per_rad = parse_hex_double(c.scalar("cmq_per_rad"));
  return star::models::aero_force_torque(
      t, parse_vec3(c, "v_rel_mps"),
      parse_hex_double(c.scalar("rho_kgpm3")),
      parse_hex_double(c.scalar("speed_of_sound_mps")),
      parse_hex_double(c.scalar("x_cg_m")), parse_vec3(c, "omega_radps"));
}

double check_scalar(double value, double ref, double tol) {
  if (ref == 0.0) {
    CHECK(value == 0.0);  // structural zeros are exact, not epsilon-small
    return 0.0;
  }
  const double err = std::fabs(value - ref) / std::fabs(ref);
  CHECK(err <= tol);
  return err;
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

// Compare one golden case's five outputs at the 1e-12 gate and return
// the worst relative error observed.
double check_outputs(const GoldenCase& c, const AeroForceTorque& out) {
  double worst = 0.0;
  worst = std::max(worst, check_scalar(out.mach,
                                       parse_hex_double(c.scalar("mach")),
                                       1e-12));
  worst = std::max(
      worst, check_scalar(out.q_bar_Pa,
                          parse_hex_double(c.scalar("q_bar_Pa")), 1e-12));
  worst = std::max(
      worst,
      check_scalar(out.alpha_total_rad,
                   parse_hex_double(c.scalar("alpha_total_rad")), 1e-12));
  worst = std::max(worst,
                   check_vec(out.force_N, parse_vec3(c, "force_N"), 1e-12));
  worst = std::max(
      worst, check_vec(out.torque_Nm, parse_vec3(c, "torque_Nm"), 1e-12));
  return worst;
}

}  // namespace

TEST_CASE("AERO-INTERP-GOLDEN") {
  // eq:aero:interp against the committed references: bp_* (breakpoint)
  // and clamp_* (outside the grid) cases are exact readouts of a table
  // row -- bit equality; mid_* (segment midpoint) cases carry real
  // interpolation arithmetic and compare at 1e-12 relative (manifest
  // derivation: <= 3 roundings per column, floor ~1e-15).
  const auto tables = load_tables();
  const auto cases = load_golden_cases(kGoldenDir + "/aero/interp.toml");
  REQUIRE(cases.size() == 41);
  double worst = 0.0;
  for (const auto& c : cases) {
    const std::string name = c.scalar("name");
    CAPTURE(name);
    AeroTables t = tables.at(c.scalar("table"));
    const AeroCoefficients out = star::models::aero_coefficients(
        t, parse_hex_double(c.scalar("mach")));
    const double ca_ref = parse_hex_double(c.scalar("ca"));
    const double cn_ref = parse_hex_double(c.scalar("cnalpha_per_rad"));
    const double xcp_ref = parse_hex_double(c.scalar("xcp_m"));
    if (name.rfind("bp_", 0) == 0 || name.rfind("clamp_", 0) == 0) {
      CHECK(out.ca == ca_ref);
      CHECK(out.cnalpha_per_rad == cn_ref);
      CHECK(out.xcp_m == xcp_ref);
    } else {
      worst = std::max(worst, check_scalar(out.ca, ca_ref, 1e-12));
      worst = std::max(
          worst, check_scalar(out.cnalpha_per_rad, cn_ref, 1e-12));
      worst = std::max(worst, check_scalar(out.xcp_m, xcp_ref, 1e-12));
    }
  }
  CAPTURE(worst);
  CHECK(worst <= 1e-12);  // prints the observed margin on failure
}

TEST_CASE("AERO-BREAKPOINT-GOLDEN") {
  // Exit criterion 8: aero force/moment reconstruction at each
  // CA/CNalpha/xcp Mach breakpoint of both committed fleet tables
  // matches the hand-computed (60-digit mpmath) golden values to 1e-12
  // relative -- force and torque at norm-relative 1e-12, Mach, dynamic
  // pressure, and total alpha at 1e-12 relative.
  const auto tables = load_tables();
  const auto cases =
      load_golden_cases(kGoldenDir + "/aero/breakpoints.toml");
  REQUIRE(cases.size() == 17);  // 12 full-stack + 5 upper-stack rows
  double worst = 0.0;
  for (const auto& c : cases) {
    CAPTURE(c.scalar("name"));
    worst = std::max(worst, check_outputs(c, run_case(c, tables)));
  }
  CAPTURE(worst);
  CHECK(worst <= 1e-12);
}

TEST_CASE("AERO-FORCETORQUE-GOLDEN") {
  // Formulation families (ch:aero): total-alpha sweep, crossflow roll
  // orientations, CG fore/aft of the CP, damping on/off/roll-rate-only,
  // reference scaling, off-breakpoint interpolation, above-table
  // clamping, and the pad case. Every reference recorded as zero must be
  // exactly zero in the model output (check_scalar/check_vec structural
  // rule); nonzero outputs compare at the 1e-12 gate.
  const auto tables = load_tables();
  const auto cases =
      load_golden_cases(kGoldenDir + "/aero/forcetorque.toml");
  REQUIRE(cases.size() == 19);
  double worst = 0.0;
  bool saw_pad = false;
  for (const auto& c : cases) {
    const std::string name = c.scalar("name");
    CAPTURE(name);
    const AeroForceTorque out = run_case(c, tables);
    worst = std::max(worst, check_outputs(c, out));
    if (name == "pad_static_exact_zero") {
      // Exit criterion 10 hook: on the pad the logged dynamic pressure
      // (and everything else) is exactly zero, not epsilon-small.
      saw_pad = true;
      CHECK(out.q_bar_Pa == 0.0);
      CHECK(out.mach == 0.0);
      CHECK(out.alpha_total_rad == 0.0);
      CHECK(out.force_N.norm() == 0.0);
      CHECK(out.torque_Nm.norm() == 0.0);
    }
  }
  CHECK(saw_pad);
  CAPTURE(worst);
  CHECK(worst <= 1e-12);
}

TEST_CASE("AERO-STRUCTURAL-ZEROS") {
  // The exact-zero semantics of ch:aero as bit-level branch checks, on
  // inputs independent of the golden files.
  const auto tables = load_tables();
  AeroTables t = tables.at("electron_full");
  t.ref_area_m2 = 1.13;
  t.ref_diameter_m = 1.2;
  t.cmq_per_rad = -25.0;

  // Pad before release: |v_rel| == 0 returns literal zeros in every
  // output even with rates present and damping configured.
  const AeroForceTorque pad = star::models::aero_force_torque(
      t, Eigen::Vector3d::Zero(), 1.225, 340.0, 8.0,
      Eigen::Vector3d(0.01, 0.02, 0.03));
  CHECK(pad.force_N.norm() == 0.0);
  CHECK(pad.torque_Nm.norm() == 0.0);
  CHECK(pad.mach == 0.0);
  CHECK(pad.q_bar_Pa == 0.0);
  CHECK(pad.alpha_total_rad == 0.0);

  // alpha_total == 0 (zero crossflow): exactly zero normal force and
  // static moment -- axial only -- even with transverse rates present,
  // once damping is off.
  AeroTables t0 = t;
  t0.cmq_per_rad = 0.0;
  const Eigen::Vector3d v_axial(300.0, 0.0, 0.0);
  const AeroForceTorque ax = star::models::aero_force_torque(
      t0, v_axial, 0.75, 256.0, 8.0, Eigen::Vector3d(0.1, 0.2, 0.3));
  CHECK(ax.force_N.x() < 0.0);
  CHECK(ax.force_N.y() == 0.0);
  CHECK(ax.force_N.z() == 0.0);
  CHECK(ax.torque_Nm.norm() == 0.0);
  CHECK(ax.alpha_total_rad == 0.0);

  // cmq == 0.0 disables damping exactly: bit-identical to the same
  // evaluation with zero rates (the damping block is skipped, not
  // evaluated to something small).
  const Eigen::Vector3d v_generic(280.0, 12.0, -9.0);
  const AeroForceTorque no_cmq_rates = star::models::aero_force_torque(
      t0, v_generic, 0.75, 256.0, 8.0, Eigen::Vector3d(0.1, 0.2, 0.3));
  const AeroForceTorque no_cmq_still = star::models::aero_force_torque(
      t0, v_generic, 0.75, 256.0, 8.0, Eigen::Vector3d::Zero());
  CHECK(no_cmq_rates.force_N == no_cmq_still.force_N);
  CHECK(no_cmq_rates.torque_Nm == no_cmq_still.torque_Nm);

  // A pure roll rate is never damped (no Clp in the database): with
  // damping configured, bit-identical to the rate-free evaluation.
  const AeroForceTorque roll_only = star::models::aero_force_torque(
      t, v_generic, 0.75, 256.0, 8.0, Eigen::Vector3d(0.5, 0.0, 0.0));
  const AeroForceTorque no_rate = star::models::aero_force_torque(
      t, v_generic, 0.75, 256.0, 8.0, Eigen::Vector3d::Zero());
  CHECK(roll_only.torque_Nm == no_rate.torque_Nm);

  // No aerodynamic roll torque, structurally: the torque x-component is
  // exactly zero in a generic evaluation with damping active.
  const AeroForceTorque generic = star::models::aero_force_torque(
      t, v_generic, 0.75, 256.0, 8.0, Eigen::Vector3d(0.02, -0.05, 0.03));
  CHECK(generic.torque_Nm.x() == 0.0);
  CHECK(generic.torque_Nm.norm() > 0.0);
}

TEST_CASE("AERO-DOMAIN") {
  // Out-of-domain behavior per ch:aero: std::domain_error.
  const auto tables = load_tables();
  AeroTables good = tables.at("electron_full");
  good.ref_area_m2 = 1.13;
  good.ref_diameter_m = 1.2;
  const Eigen::Vector3d v(300.0, 5.0, -2.0);
  const Eigen::Vector3d w(0.01, 0.02, 0.03);

  AeroTables bad = good;
  bad.mach[3] = bad.mach[2];  // non-increasing grid
  CHECK_THROWS_AS(star::models::aero_coefficients(bad, 1.0),
                  std::domain_error);

  bad = good;
  bad.mach.resize(1);  // fewer than two rows (and a size mismatch)
  CHECK_THROWS_AS(star::models::aero_coefficients(bad, 1.0),
                  std::domain_error);

  bad = good;
  bad.ca.pop_back();  // column-length mismatch
  CHECK_THROWS_AS(
      star::models::aero_force_torque(bad, v, 0.75, 256.0, 8.0, w),
      std::domain_error);

  bad = good;
  bad.xcp_m[5] = std::nan("");  // non-finite table entry
  CHECK_THROWS_AS(
      star::models::aero_force_torque(bad, v, 0.75, 256.0, 8.0, w),
      std::domain_error);

  bad = good;
  bad.ref_area_m2 = 0.0;
  CHECK_THROWS_AS(
      star::models::aero_force_torque(bad, v, 0.75, 256.0, 8.0, w),
      std::domain_error);

  bad = good;
  bad.cmq_per_rad = 0.4;  // damping must not pump energy in
  CHECK_THROWS_AS(
      star::models::aero_force_torque(bad, v, 0.75, 256.0, 8.0, w),
      std::domain_error);

  CHECK_THROWS_AS(star::models::aero_coefficients(good, -0.5),
                  std::domain_error);
  CHECK_THROWS_AS(
      star::models::aero_force_torque(
          good, Eigen::Vector3d(std::nan(""), 0.0, 0.0), 0.75, 256.0, 8.0,
          w),
      std::domain_error);
  CHECK_THROWS_AS(
      star::models::aero_force_torque(good, v, -0.1, 256.0, 8.0, w),
      std::domain_error);  // negative density
  CHECK_THROWS_AS(
      star::models::aero_force_torque(good, v, 0.75, 0.0, 8.0, w),
      std::domain_error);  // zero speed of sound
  CHECK_THROWS_AS(
      star::models::aero_force_torque(good, v, 0.75, 256.0,
                                      std::nan(""), w),
      std::domain_error);
}
