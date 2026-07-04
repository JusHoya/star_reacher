// Composed environment force model (Phase 3 integration layer): the point-
// mass translational equations of motion about one central body (Earth, Moon,
// Mars, or - since Phase 5 - the Sun) under the FR-5..FR-9 perturbations,
// summed in a FIXED, documented order per D-10:
//
//   (a) central-body gravity (point-mass, J2-only, or full harmonic tier,
//       evaluated in the body-fixed frame through the Phase 2 frame chains),
//   (b) Battin f(q) third-body differential acceleration for each enabled
//       perturber, in the canonical order sun, earth, moon, venus, mars,
//       jupiter, with positions from the Chebyshev ephemeris and the DE440
//       header GM constants (star/constants.hpp),
//   (c) cannonball SRP with conical shadow over the configured occulters,
//   (d) cannonball drag with the FR-8 co-rotating air-relative velocity
//       v_rel = v - omega_planet x r and the selected atmosphere model.
//
// This module owns only frame/time plumbing and composition; every physical
// term lives in its own chapter-tracked model module. Single-threaded, no
// heap allocation after construction, deterministic evaluation order.
//
// Math-library traceability (FR-29): the composition, the frame and time
// plumbing, the ephemeris body composition, and the documented approximations
// are described in the environment chapter of docs/mathlib (ch:environment),
// which cites the per-model chapters for the physics.
#ifndef STAR_MODELS_ENVIRONMENT_HPP
#define STAR_MODELS_ENVIRONMENT_HPP

#include <optional>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "star/ephemeris.hpp"
#include "star/models/gravity.hpp"
#include "star/time.hpp"

namespace star {
namespace models {

enum class CentralBody { kEarth, kMoon, kMars, kSun };

enum class AtmosphereModel { kNone, kUssa76, kHarrisPriester, kMarsExponential };

// Central-body gravitational parameter [m^3/s^2] for the point-mass tier and
// for Keplerian element conversions: Earth keeps the Phase 1 IERS TN36 value
// (GM_EARTH_M3_PER_S2); Moon, Mars, and the Sun use the DE440 header values,
// matching the third-body constants (single home: star/constants.hpp).
double central_body_gm(CentralBody body);

// Validated environment configuration, mirroring the [environment] and
// [spacecraft] surface the Python layer resolves (D-2: all user-facing
// validation happens there; this struct re-checks only what it needs to stay
// well-defined and throws std::invalid_argument on violations).
struct EnvironmentSpec {
  CentralBody central_body = CentralBody::kEarth;
  time::TaiEpoch epoch_tai{0, 0.0};  // mission epoch (t = 0) on the TAI scale

  // Gravity tier: "pointmass" needs no field file; "j2" and "harmonic" load
  // an SRGRAV field. degree/order select the harmonic truncation (-1 means
  // the field's full band, per PinesGravity::acceleration).
  std::string gravity_model = "pointmass";
  std::string gravity_field_path;
  int gravity_degree = -1;
  int gravity_order = -1;

  // Enabled third-body perturbers: names among sun, earth, moon, venus,
  // mars, jupiter (never the central body). Order does not matter here; the
  // model sums in the canonical order documented above.
  std::vector<std::string> third_bodies;

  bool srp_enabled = false;
  double cr_a_over_m_m2pkg = 0.0;
  // Occulter names among earth, moon, mars. Non-empty whenever SRP is on,
  // except about the Sun central body: the Sun cannot occult its own light
  // and deep-cruise planetary transits are negligible, so the heliocentric
  // regime runs SRP with an empty occulter set (nu = 1 everywhere, FR-7).
  std::vector<std::string> srp_occulters;

  AtmosphereModel atmosphere = AtmosphereModel::kNone;  // kNone: drag off
  double cd_a_over_m_m2pkg = 0.0;
  double hp_exponent_n = 4.0;  // Harris-Priester bulge exponent in [2, 6]

  std::string ephemeris_path;  // SREPH file; "" only if no model needs it
};

// One composed force model instance: loads the gravity field and ephemeris
// at construction (file I/O happens once, before the time loop), precomputes
// the perturber table in canonical order, and evaluates the right-hand side
// with no per-call allocation. acceleration() is non-const because the Pines
// evaluator writes its internal workspace; one instance serves one
// single-threaded propagation (D-10).
class EnvironmentModel {
 public:
  // Perturbing bodies in the canonical summation order (the enum value IS
  // the summation rank). Public so the composition helpers and tests can
  // name bodies without string round-trips.
  enum class Body { kSun = 0, kEarth, kMoon, kVenus, kMars, kJupiter };

  explicit EnvironmentModel(const EnvironmentSpec& spec);

  // Total perturbed acceleration [m/s^2] at mission-elapsed time t_s for the
  // GCRF-oriented, central-body-centered state (r_m, v_mps).
  Eigen::Vector3d acceleration(double t_s, const Eigen::Vector3d& r_m,
                               const Eigen::Vector3d& v_mps);

  // First-order ODE right-hand side for the shared integrators:
  // y = [r, v], ydot = [v, acceleration(t, r, v)].
  void rhs(double t_s, const double* y, double* ydot);

  bool uses_ephemeris() const { return eph_.has_value(); }

 private:
  // TDB seconds since J2000 TDB for the epoch shifted by t_s.
  double tdb_s_at(double t_s) const;

  // Position of `body` relative to the central body, ICRF/GCRF orientation.
  // r_central_ssb is the central body's SSB position at the same epoch,
  // computed once per acceleration() call and shared by every lookup.
  Eigen::Vector3d body_rel_central(Body body, double tdb_s,
                                   const Eigen::Vector3d& r_central_ssb) const;
  Eigen::Vector3d central_ssb(double tdb_s) const;

  // GCRF -> central-body-fixed rotation at the epoch shifted by t_s.
  Eigen::Matrix3d c_gcrf_to_bodyfixed(double t_s, double tdb_s) const;

  CentralBody central_;
  time::TaiEpoch epoch_;

  // Gravity: point-mass GM, or a Pines evaluator over the loaded field.
  double gm_central_ = 0.0;
  GravityTier tier_ = GravityTier::kPointMass;
  bool use_field_ = false;
  int degree_ = -1;
  int order_ = -1;
  std::optional<PinesGravity> gravity_;

  struct Perturber {
    Body body;
    double gm_m3ps2;
  };
  std::vector<Perturber> perturbers_;  // canonical order, fixed at construction

  bool srp_enabled_ = false;
  double cr_a_over_m_ = 0.0;
  struct Occulter {
    Body body;
    bool is_central;
    double radius_m;
  };
  std::vector<Occulter> occulters_;

  AtmosphereModel atmosphere_ = AtmosphereModel::kNone;
  double cd_a_over_m_ = 0.0;
  double hp_n_ = 4.0;

  std::optional<Ephemeris> eph_;
};

}  // namespace models
}  // namespace star

#endif  // STAR_MODELS_ENVIRONMENT_HPP
