// Axisymmetric ascent aerodynamics (FR-9). Derivation: docs/mathlib
// chapter ch:aero. The force/moment path uses IEEE-754 basic operations
// and square roots only, in fixed order, so it is bit-portable across
// platforms under the D-10 flags; the one libm call (std::atan2 for the
// alpha_total_rad diagnostic) is never consumed by the force/moment
// path.
#include "star/models/aero.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace star {
namespace models {

namespace {

// Structural and finiteness checks only: physical plausibility (schema
// ranges, warning tiers) is the Python validator's job (D-2); the core
// refuses inputs the formulation cannot evaluate deterministically.
void check_tables(const AeroTables& t) {
  if (!std::isfinite(t.ref_area_m2) || t.ref_area_m2 <= 0.0 ||
      !std::isfinite(t.ref_diameter_m) || t.ref_diameter_m <= 0.0) {
    throw std::domain_error(
        "aero: reference area and diameter must be finite and positive");
  }
  // cmq <= 0: damping must oppose the rate (schema rule for
  // cmq_per_rad); exactly 0.0 disables the term.
  if (!std::isfinite(t.cmq_per_rad) || t.cmq_per_rad > 0.0) {
    throw std::domain_error(
        "aero: cmq_per_rad must be finite and non-positive");
  }
  const std::size_t n = t.mach.size();
  if (n < 2 || t.ca.size() != n || t.cnalpha_per_rad.size() != n ||
      t.xcp_m.size() != n) {
    throw std::domain_error(
        "aero: table needs >= 2 rows with equal-length mach, ca, "
        "cnalpha_per_rad, and xcp_m columns");
  }
  if (!std::isfinite(t.mach[0]) || t.mach[0] < 0.0) {
    throw std::domain_error(
        "aero: the mach grid must start at a finite value >= 0");
  }
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(t.mach[i]) || !std::isfinite(t.ca[i]) ||
        !std::isfinite(t.cnalpha_per_rad[i]) ||
        !std::isfinite(t.xcp_m[i])) {
      throw std::domain_error("aero: table entries must be finite");
    }
    if (i + 1 < n && !(t.mach[i] < t.mach[i + 1])) {
      throw std::domain_error(
          "aero: the mach grid must be strictly increasing");
    }
  }
}

}  // namespace

AeroCoefficients aero_coefficients(const AeroTables& tables, double mach) {
  check_tables(tables);
  if (!std::isfinite(mach) || mach < 0.0) {
    throw std::domain_error("aero: Mach must be finite and non-negative");
  }
  const std::vector<double>& m = tables.mach;
  const std::size_t n = m.size();
  AeroCoefficients out;
  // eq:aero:interp clamp branches -- the end rows are returned literally,
  // so out-of-table lookups are exact readouts, not extrapolations.
  if (mach <= m.front()) {
    out.ca = tables.ca.front();
    out.cnalpha_per_rad = tables.cnalpha_per_rad.front();
    out.xcp_m = tables.xcp_m.front();
    return out;
  }
  if (mach >= m.back()) {
    out.ca = tables.ca.back();
    out.cnalpha_per_rad = tables.cnalpha_per_rad.back();
    out.xcp_m = tables.xcp_m.back();
    return out;
  }
  // Fixed ascending scan for the segment with m[i] <= mach < m[i+1]
  // (D-10 fixed evaluation order; the grids are ~10 rows, so a scan
  // costs less than maintaining a branch-predictable bisection).
  std::size_t i = 0;
  while (i + 2 < n && m[i + 1] <= mach) {
    ++i;
  }
  // eq:aero:interp interior branch. At mach == m[i] the weight u is
  // exactly zero, so c_i + u * (c_{i+1} - c_i) reproduces the table row
  // bit for bit (exact-breakpoint readout, ch:aero).
  const double u = (mach - m[i]) / (m[i + 1] - m[i]);
  out.ca = tables.ca[i] + u * (tables.ca[i + 1] - tables.ca[i]);
  out.cnalpha_per_rad =
      tables.cnalpha_per_rad[i] +
      u * (tables.cnalpha_per_rad[i + 1] - tables.cnalpha_per_rad[i]);
  out.xcp_m = tables.xcp_m[i] + u * (tables.xcp_m[i + 1] - tables.xcp_m[i]);
  return out;
}

AeroForceTorque aero_force_torque(const AeroTables& tables,
                                  const Eigen::Vector3d& v_rel_body_mps,
                                  double rho_kgpm3,
                                  double speed_of_sound_mps, double x_cg_m,
                                  const Eigen::Vector3d& omega_b_radps) {
  check_tables(tables);
  if (!v_rel_body_mps.allFinite() || !omega_b_radps.allFinite() ||
      !std::isfinite(x_cg_m)) {
    throw std::domain_error(
        "aero: v_rel, omega, and the CG station must be finite");
  }
  if (!std::isfinite(rho_kgpm3) || rho_kgpm3 < 0.0) {
    throw std::domain_error(
        "aero: density must be finite and non-negative");
  }
  if (!std::isfinite(speed_of_sound_mps) || speed_of_sound_mps <= 0.0) {
    throw std::domain_error(
        "aero: speed of sound must be finite and positive");
  }
  AeroForceTorque out;
  const double speed = v_rel_body_mps.norm();
  if (speed == 0.0) {
    // Structural zero (ch:aero): on the pad before release every output
    // is a literal zero -- the logged dynamic pressure is exactly zero
    // (Phase 4 exit criterion 10) and no table access happens.
    return out;
  }
  out.mach = speed / speed_of_sound_mps;              // eq:aero:mach
  out.q_bar_Pa = 0.5 * rho_kgpm3 * speed * speed;     // eq:aero:qbar
  const AeroCoefficients c = aero_coefficients(tables, out.mach);
  // eq:aero:axial -- axial force along -X on the reference area.
  out.force_N.x() = -(out.q_bar_Pa * tables.ref_area_m2 * c.ca);
  const double vy = v_rel_body_mps.y();
  const double vz = v_rel_body_mps.z();
  if (vy != 0.0 || vz != 0.0) {
    // eq:aero:normal -- CN = CNalpha * sin(alpha_total) resolved
    // algebraically: sin(alpha_total) = |v_perp| / |v_rel| and the unit
    // crossflow direction is v_perp / |v_perp|, so the normal force is
    // -(q_bar S CNalpha / |v_rel|) v_perp with no trigonometry and no
    // crossflow normalization. Zero crossflow skips the whole block,
    // leaving the structural zeros (exactly zero normal force and static
    // moment at alpha_total == 0).
    const double k_n =
        out.q_bar_Pa * tables.ref_area_m2 * c.cnalpha_per_rad / speed;
    out.force_N.y() = -k_n * vy;
    out.force_N.z() = -k_n * vz;
    // eq:aero:cpmoment -- lever arm from the CG station to the CP
    // station (both on the +X axis, structural frame), crossed with the
    // normal force: (l x_hat) x F_N = l (0, -F_Nz, F_Ny). The torque
    // x-component stays untouched (no aerodynamic roll torque).
    const double lever_m = c.xcp_m - x_cg_m;
    out.torque_Nm.y() = -lever_m * out.force_N.z();
    out.torque_Nm.z() = lever_m * out.force_N.y();
  }
  if (tables.cmq_per_rad != 0.0 &&
      (omega_b_radps.y() != 0.0 || omega_b_radps.z() != 0.0)) {
    // eq:aero:damping -- pitch damping on the transverse rates with the
    // rate nondimensionalized by d_ref / (2 |v_rel|); cmq == 0.0 or a
    // pure roll rate skips the block exactly (structural zero). The
    // roll rate omega_x is never damped (axisymmetric database).
    const double k_q = out.q_bar_Pa * tables.ref_area_m2 *
                       tables.ref_diameter_m * tables.ref_diameter_m *
                       tables.cmq_per_rad / (2.0 * speed);
    out.torque_Nm.y() += k_q * omega_b_radps.y();
    out.torque_Nm.z() += k_q * omega_b_radps.z();
  }
  // eq:aero:alpha -- total angle of attack in [0, pi], diagnostic only
  // (the only libm call in the module; never consumed above).
  const double v_perp = std::sqrt(vy * vy + vz * vz);
  out.alpha_total_rad = std::atan2(v_perp, v_rel_body_mps.x());
  return out;
}

}  // namespace models
}  // namespace star
