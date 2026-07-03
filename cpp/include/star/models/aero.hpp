// Axisymmetric ascent aerodynamics (FR-9): body-frame force and moment
// about the current composite CG from a per-stack-configuration Mach
// database CA(M), CNalpha(M), xcp(M) with optional constant pitch damping
// Cmq, using the total-angle-of-attack formulation. Domain split (FR-9):
// this model covers continuum ascent flight only; orbital free-molecular
// drag is the separate cannonball model (star/models/drag.hpp).
//
// Conventions (ch:aero; FR-13 structural frame, +X toward the nose,
// origin at the aft plane of the assembled stack):
//   - xcp_m is the center-of-pressure STATION on the +X axis in that
//     frame, exactly the xcp_m column of the vehicle schema's Mach-table
//     CSV; x_cg_m is the composite-CG station in the same frame. The
//     composite CG is assumed on the symmetry axis (lateral offsets
//     neglected, ch:aero assumption 4).
//   - cnalpha_per_rad is the normal-force slope per radian with
//     CN = CNalpha(M) * sin(alpha_total).
//   - cmq_per_rad is per radian of nondimensional pitch rate
//     q_hat = omega * ref_diameter_m / (2 |v_rel|), referenced to
//     ref_area_m2 and ref_diameter_m; <= 0 opposes the transverse rate
//     and exactly 0.0 disables the term. ref_diameter_m enters the model
//     only through this damping term (the center-of-pressure moment uses
//     the dimensional lever arm directly).
//
// The caller supplies the air-relative velocity in the BODY frame,
// v_rel = v - omega_planet x r rotated by the EOM layer (the FR-8
// co-rotating-atmosphere rule, eq:drag:vrel), plus density and speed of
// sound from the atmosphere model and the CG station from the
// mass-property model. The core never parses text (D-2): AeroTables is a
// plain SI-unit value type the Python validator fills across the binding
// from the vehicle TOML and its CSV Mach table.
//
// Math-library traceability (FR-29): the derivation lives in the aero
// chapter of docs/mathlib (ch:aero); the implementation echoes its
// equation labels eq:aero:mach, eq:aero:qbar, eq:aero:alpha,
// eq:aero:interp, eq:aero:axial, eq:aero:normal, eq:aero:cpmoment, and
// eq:aero:damping at the corresponding code.
#ifndef STAR_MODELS_AERO_HPP
#define STAR_MODELS_AERO_HPP

#include <vector>

#include <Eigen/Dense>

namespace star {
namespace models {

// Axisymmetric aerodynamic database for one stack configuration (one
// FR-13 [[aero]] block plus its CSV Mach table). The four columns are
// parallel arrays on one strictly increasing Mach grid (>= 2 rows,
// mach[0] >= 0); member names follow the schema/CSV vocabulary.
struct AeroTables {
  double ref_area_m2 = 0.0;     // S_ref, force nondimensionalization
  double ref_diameter_m = 0.0;  // d_ref, Cmq rate nondimensionalization
  double cmq_per_rad = 0.0;     // <= 0; exactly 0.0 disables damping
  std::vector<double> mach;
  std::vector<double> ca;               // axial-force coefficient CA(M)
  std::vector<double> cnalpha_per_rad;  // normal-force slope CNalpha(M)
  std::vector<double> xcp_m;  // CP station, structural frame (see above)
};

// Coefficients at one Mach number (eq:aero:interp lookup result).
struct AeroCoefficients {
  double ca = 0.0;
  double cnalpha_per_rad = 0.0;
  double xcp_m = 0.0;
};

// Body-frame force [N] and moment [N m] about the supplied CG station,
// plus the air-data diagnostics the run layer logs (FR-8: Mach, dynamic
// pressure, angle of attack). All members are exactly zero when
// |v_rel| == 0 (pad before release; Phase 4 exit criterion 10).
// alpha_total_rad is reported for logging only and is never consumed by
// the force/moment path (ch:aero implementation note 4).
struct AeroForceTorque {
  Eigen::Vector3d force_N = Eigen::Vector3d::Zero();
  Eigen::Vector3d torque_Nm = Eigen::Vector3d::Zero();
  double mach = 0.0;
  double q_bar_Pa = 0.0;
  double alpha_total_rad = 0.0;
};

// Piecewise-linear Mach lookup with clamped ends (eq:aero:interp):
// exact table-row readout at every breakpoint and beyond the grid ends,
// linear in between, fixed ascending scan order. Throws
// std::domain_error for a malformed table (fewer than two rows, column
// size mismatch, non-increasing or negative Mach grid, non-finite
// entries, non-positive ref_area_m2/ref_diameter_m, positive
// cmq_per_rad) or a negative/non-finite Mach.
AeroCoefficients aero_coefficients(const AeroTables& tables, double mach);

// Total-angle-of-attack force and moment about the CG station
// (eq:aero:axial, eq:aero:normal, eq:aero:cpmoment, eq:aero:damping),
// body frame. Structural zeros, decided on exact binary64 comparisons:
// |v_rel| == 0 returns literal zeros in every output; zero crossflow
// (v_rel.y() == 0 && v_rel.z() == 0) leaves exactly zero normal force
// and static moment (axial force only); cmq_per_rad == 0.0 or a purely
// axial body rate leaves exactly zero damping moment; torque_Nm.x() is
// always exactly zero (no aerodynamic roll torque). Throws
// std::domain_error for a malformed table (see aero_coefficients),
// non-finite v_rel/omega_b/x_cg, negative or non-finite rho, or
// non-positive or non-finite speed of sound.
AeroForceTorque aero_force_torque(const AeroTables& tables,
                                  const Eigen::Vector3d& v_rel_body_mps,
                                  double rho_kgpm3,
                                  double speed_of_sound_mps, double x_cg_m,
                                  const Eigen::Vector3d& omega_b_radps);

}  // namespace models
}  // namespace star

#endif  // STAR_MODELS_AERO_HPP
