// Spherical-harmonic gravity model (FR-5): Pines-formulation evaluation of a
// fully normalized gravity field in the body-fixed frame, singularity-free
// at the poles, with runtime-selectable truncation degree/order and the
// FR-5 fidelity tiers (point-mass, J2-only, full n x m).
//
// The field data (GM, reference radius, normalized C/S coefficients) comes
// either from in-memory arrays (tests) or from an SRGRAV v1 binary file
// written by `star data fetch` (docs/formats/srgrav_v1.md). Binary file I/O
// only: the core never parses text (D-2).
//
// Math-library traceability (FR-29): the derivation lives in the gravity
// chapter of docs/mathlib (ch:gravity); the implementation echoes its
// equation labels `eq:gravity:potential`, `eq:gravity:pines`,
// `eq:gravity:diag`, `eq:gravity:subdiag`, `eq:gravity:column`,
// `eq:gravity:deriv`, `eq:gravity:rmim`, `eq:gravity:sums`, and
// `eq:gravity:accel` at the corresponding code.
#ifndef STAR_MODELS_GRAVITY_HPP
#define STAR_MODELS_GRAVITY_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace star {
namespace models {

// FR-5 fidelity tiers. kPointMass evaluates only the stored degree-0 term;
// kJ2Only evaluates degree 0 plus the (2,0) zonal term and nothing else
// (regardless of what else the field stores); kFull evaluates the requested
// n x m truncation.
enum class GravityTier { kPointMass, kJ2Only, kFull };

// Permanent-tide convention of the stored C(2,0) (srgrav_v1.md section 4).
// Documentation only: the value never enters evaluation.
enum class TideSystem : std::uint32_t {
  kTideFree = 0,
  kZeroTide = 1,
  kMeanTide = 2,
  kUnknown = 3,
};

// One spherical-harmonic gravity field: GM, reference radius, and the fully
// normalized (4-pi geodesy normalization) Stokes coefficients, packed in
// the fixed (n ascending, m ascending) order of the SRGRAV format. GM and
// the reference radius always come from the same source as the coefficients
// (each field is a self-consistent triple; mixing sources changes the
// model).
struct GravityField {
  std::string name;
  double gm_m3ps2 = 0.0;
  double ref_radius_m = 0.0;
  int n_max = 0;
  int m_max = 0;
  TideSystem tide_system = TideSystem::kUnknown;
  std::vector<double> cbar;  // packed: index offset(n) + m, m <= min(n, m_max)
  std::vector<double> sbar;  // same packing as cbar

  // Load an SRGRAV v1 file (docs/formats/srgrav_v1.md). Throws
  // std::runtime_error naming the specific defect (bad magic, unsupported
  // major version, size mismatch, unknown tide code) on a malformed file.
  static GravityField load_file(const std::string& path);

  // Build a field from in-memory arrays (the test path). cbar/sbar use the
  // packed order above and must each hold exactly the entry count implied
  // by (n_max, m_max); throws std::invalid_argument otherwise.
  static GravityField from_coefficients(std::string name, double gm_m3ps2,
                                        double ref_radius_m, int n_max,
                                        int m_max, std::vector<double> cbar,
                                        std::vector<double> sbar,
                                        TideSystem tide_system);

  // Degree-0 point-mass field: C(0,0) = 1, nothing else. The reference
  // radius is irrelevant for degree 0 and is stored as 1 m.
  static GravityField point_mass(std::string name, double gm_m3ps2);

  // Number of stored (n, m) coefficient entries.
  std::size_t n_entries() const;

  // Packed index of (n, m); no bounds checking beyond the debug assert in
  // the accessors below.
  std::size_t index(int n, int m) const;

  // Coefficient accessors for tests and diagnostics. Throw
  // std::out_of_range outside the stored band.
  double cnm(int n, int m) const;
  double snm(int n, int m) const;
};

// The dynamic form factor J2 = -sqrt(5) * C-bar(2,0): the degree-2 zonal
// normalization factor is N(2,0) = sqrt(5) (eq:gravity:potential). Used by
// the J2 secular-rate acceptance test and available to analysis code.
double j2_from_field(const GravityField& field);

// Pines-formulation evaluator over one GravityField. Construction sizes and
// precomputes every workspace and recursion-coefficient table for the
// field's full degree, so acceleration() never allocates (D-10: no heap
// allocation inside the propagation loop) and runs a fixed, documented
// operation order (fixed evaluation order; deterministic).
//
// acceleration() is non-const because it writes the internal workspace: one
// evaluator instance serves one propagation thread (the core loop is
// single-threaded by design, D-10).
class PinesGravity {
 public:
  explicit PinesGravity(GravityField field);

  const GravityField& field() const { return field_; }

  // Gravitational acceleration [m/s^2] at body-fixed position r_bf [m],
  // including the central (degree-0) term. degree/order select the
  // evaluation truncation for kFull: degree < 0 means the field's full
  // n_max; order < 0 means min(degree, stored m_max). Requests beyond the
  // stored band throw std::invalid_argument (the file carries no
  // information above its band; the core never silently degrades
  // fidelity). kPointMass and kJ2Only ignore degree/order. Throws
  // std::domain_error at r = 0.
  Eigen::Vector3d acceleration(const Eigen::Vector3d& r_bf, GravityTier tier,
                               int degree = -1, int order = -1);

 private:
  std::size_t tri(int n, int m) const {
    // Full lower-triangle packing for the Helmholtz workspace (m <= n),
    // independent of the field's m_max.
    return static_cast<std::size_t>(n) * (static_cast<std::size_t>(n) + 1) / 2 +
           static_cast<std::size_t>(m);
  }

  GravityField field_;
  // Recursion coefficients, precomputed at construction (eq:gravity:diag,
  // eq:gravity:subdiag, eq:gravity:column, eq:gravity:deriv).
  std::vector<double> f_diag_;   // [m]: diagonal factor
  std::vector<double> f_sub_;    // [m]: first sub-diagonal factor sqrt(2m+3)
  std::vector<double> c1_;       // [tri(n,m)]: column recursion, n-1 term
  std::vector<double> c2_;       // [tri(n,m)]: column recursion, n-2 term
  std::vector<double> g_deriv_;  // [tri(n,m)]: derivative factor
  // Per-call workspace (sized once for n_max).
  std::vector<double> abar_;  // [tri(n,m)]: normalized Helmholtz values
  std::vector<double> rm_;    // [m]: Re[(s + i t)^m]
  std::vector<double> im_;    // [m]: Im[(s + i t)^m]
  std::vector<double> rho_;   // [n]: (GM/r) * (R/r)^n
};

}  // namespace models
}  // namespace star

#endif  // STAR_MODELS_GRAVITY_HPP
