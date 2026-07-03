// Spherical-harmonic gravity model (FR-5): SRGRAV loader and Pines
// evaluator. Derivation and label definitions: docs/mathlib/chapters/
// gravity.tex (ch:gravity); format: docs/formats/srgrav_v1.md.
#include "star/models/gravity.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace star {
namespace models {

namespace {

constexpr std::size_t kHeaderSize = 96;
constexpr std::uint16_t kVersionMajor = 1;
// "SRGRV" + NUL + CR + LF: the NUL stops C-string readers and any text-mode
// transfer mangles the CRLF, so a corrupted file fails immediately.
constexpr unsigned char kMagic[8] = {0x53, 0x52, 0x47, 0x52,
                                     0x56, 0x00, 0x0D, 0x0A};

// Little-endian field extraction by memcpy: every supported platform is
// little-endian (srgrav_v1.md section 1), so memcpy is the whole decode.
template <typename T>
T read_at(const std::vector<char>& buf, std::size_t offset) {
  T v;
  std::memcpy(&v, buf.data() + offset, sizeof(T));
  return v;
}

std::size_t entries_for(int n_max, int m_max) {
  std::size_t n_entries = 0;
  for (int n = 0; n <= n_max; ++n) {
    n_entries += static_cast<std::size_t>((n < m_max ? n : m_max) + 1);
  }
  return n_entries;
}

}  // namespace

std::size_t GravityField::n_entries() const {
  return entries_for(n_max, m_max);
}

std::size_t GravityField::index(int n, int m) const {
  // Ascending-degree packing with per-degree width min(n, m_max) + 1
  // (srgrav_v1.md section 3). Closed form below splits the sum at m_max:
  // degrees up to m_max form a triangle, the rest are full-width rows.
  std::size_t base;
  if (n <= m_max) {
    base = static_cast<std::size_t>(n) * (static_cast<std::size_t>(n) + 1) / 2;
  } else {
    const std::size_t tri_part =
        static_cast<std::size_t>(m_max + 1) * static_cast<std::size_t>(m_max + 2) / 2;
    base = tri_part + static_cast<std::size_t>(n - m_max - 1) *
                          static_cast<std::size_t>(m_max + 1);
  }
  return base + static_cast<std::size_t>(m);
}

double GravityField::cnm(int n, int m) const {
  if (n < 0 || n > n_max || m < 0 || m > n || m > m_max) {
    throw std::out_of_range("gravity: coefficient index outside stored band");
  }
  return cbar[index(n, m)];
}

double GravityField::snm(int n, int m) const {
  if (n < 0 || n > n_max || m < 0 || m > n || m > m_max) {
    throw std::out_of_range("gravity: coefficient index outside stored band");
  }
  return sbar[index(n, m)];
}

GravityField GravityField::load_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("srgrav: cannot open " + path);
  }
  std::vector<char> buf((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());
  if (buf.size() < kHeaderSize ||
      std::memcmp(buf.data(), kMagic, sizeof(kMagic)) != 0) {
    throw std::runtime_error("srgrav: " + path +
                             ": not an SRGRAV file (bad magic or truncated "
                             "header)");
  }
  const auto major = read_at<std::uint16_t>(buf, 8);
  if (major != kVersionMajor) {
    throw std::runtime_error("srgrav: " + path + ": major version " +
                             std::to_string(major) +
                             " is not supported (reader implements " +
                             std::to_string(kVersionMajor) + ")");
  }
  // Minor version (bytes 10-11) is additive-only by contract; nothing this
  // reader consumes depends on it.
  GravityField f;
  const auto n_max_u = read_at<std::uint32_t>(buf, 12);
  const auto m_max_u = read_at<std::uint32_t>(buf, 16);
  const auto tide_u = read_at<std::uint32_t>(buf, 20);
  // Degrees far beyond the FR-5 bands indicate a corrupt header long before
  // the size check could overflow; 32767 is generous and keeps int math safe.
  if (n_max_u > 32767 || m_max_u > n_max_u) {
    throw std::runtime_error("srgrav: " + path +
                             ": implausible degree/order header fields");
  }
  if (tide_u > 3) {
    // An unknown tide code is an unknown semantic for C(2,0); reject rather
    // than guess (same rule as SREPH's unknown segment kind).
    throw std::runtime_error("srgrav: " + path + ": unknown tide-system code " +
                             std::to_string(tide_u));
  }
  f.n_max = static_cast<int>(n_max_u);
  f.m_max = static_cast<int>(m_max_u);
  f.tide_system = static_cast<TideSystem>(tide_u);
  f.gm_m3ps2 = read_at<double>(buf, 28);
  f.ref_radius_m = read_at<double>(buf, 36);
  char name_raw[17] = {0};
  std::memcpy(name_raw, buf.data() + 44, 16);
  f.name.assign(name_raw);
  const std::size_t n_entries = entries_for(f.n_max, f.m_max);
  if (buf.size() != kHeaderSize + 16 * n_entries) {
    throw std::runtime_error(
        "srgrav: " + path + ": size " + std::to_string(buf.size()) +
        " does not match header (expected " +
        std::to_string(kHeaderSize + 16 * n_entries) + ")");
  }
  f.cbar.resize(n_entries);
  f.sbar.resize(n_entries);
  for (std::size_t k = 0; k < n_entries; ++k) {
    f.cbar[k] = read_at<double>(buf, kHeaderSize + 16 * k);
    f.sbar[k] = read_at<double>(buf, kHeaderSize + 16 * k + 8);
  }
  return f;
}

GravityField GravityField::from_coefficients(std::string name, double gm_m3ps2,
                                             double ref_radius_m, int n_max,
                                             int m_max,
                                             std::vector<double> cbar,
                                             std::vector<double> sbar,
                                             TideSystem tide_system) {
  if (n_max < 0 || m_max < 0 || m_max > n_max) {
    throw std::invalid_argument("gravity: invalid degree/order bounds");
  }
  const std::size_t n_entries = entries_for(n_max, m_max);
  if (cbar.size() != n_entries || sbar.size() != n_entries) {
    throw std::invalid_argument(
        "gravity: coefficient array size does not match (n_max, m_max)");
  }
  GravityField f;
  f.name = std::move(name);
  f.gm_m3ps2 = gm_m3ps2;
  f.ref_radius_m = ref_radius_m;
  f.n_max = n_max;
  f.m_max = m_max;
  f.tide_system = tide_system;
  f.cbar = std::move(cbar);
  f.sbar = std::move(sbar);
  return f;
}

GravityField GravityField::point_mass(std::string name, double gm_m3ps2) {
  return from_coefficients(std::move(name), gm_m3ps2, 1.0, 0, 0, {1.0}, {0.0},
                           TideSystem::kUnknown);
}

double j2_from_field(const GravityField& field) {
  // J2 = -N(2,0) * C-bar(2,0) with N(2,0) = sqrt(2*2+1) = sqrt(5)
  // (eq:gravity:potential normalization, m = 0 case).
  return -std::sqrt(5.0) * field.cnm(2, 0);
}

PinesGravity::PinesGravity(GravityField field) : field_(std::move(field)) {
  const int N = field_.n_max;
  const std::size_t tri_size =
      static_cast<std::size_t>(N + 1) * static_cast<std::size_t>(N + 2) / 2;
  f_diag_.assign(static_cast<std::size_t>(N) + 1, 0.0);
  f_sub_.assign(static_cast<std::size_t>(N) + 1, 0.0);
  c1_.assign(tri_size, 0.0);
  c2_.assign(tri_size, 0.0);
  g_deriv_.assign(tri_size, 0.0);
  abar_.assign(tri_size, 0.0);
  rm_.assign(static_cast<std::size_t>(N) + 1, 0.0);
  im_.assign(static_cast<std::size_t>(N) + 1, 0.0);
  rho_.assign(static_cast<std::size_t>(N) + 1, 0.0);

  // Precompute every recursion coefficient once: they depend only on (n, m),
  // and hoisting the square roots out of the evaluation keeps the per-call
  // path multiply-add only, in a fixed order (D-10).
  for (int m = 1; m <= N; ++m) {
    // eq:gravity:diag: the m = 1 step carries the extra sqrt(2) from the
    // (2 - delta_0m) factor switching on between order 0 and order 1.
    f_diag_[static_cast<std::size_t>(m)] =
        (m == 1) ? std::sqrt(3.0)
                 : std::sqrt((2.0 * m + 1.0) / (2.0 * m));
  }
  for (int m = 0; m < N; ++m) {
    // eq:gravity:subdiag
    f_sub_[static_cast<std::size_t>(m)] = std::sqrt(2.0 * m + 3.0);
  }
  for (int m = 0; m <= N; ++m) {
    for (int n = m + 2; n <= N; ++n) {
      // eq:gravity:column
      const double num1 = (2.0 * n - 1.0) * (2.0 * n + 1.0);
      const double den1 = static_cast<double>(n - m) * (n + m);
      c1_[tri(n, m)] = std::sqrt(num1 / den1);
      const double num2 =
          (2.0 * n + 1.0) * (n + m - 1.0) * (n - m - 1.0);
      const double den2 =
          (2.0 * n - 3.0) * static_cast<double>(n + m) * (n - m);
      c2_[tri(n, m)] = std::sqrt(num2 / den2);
    }
  }
  for (int n = 0; n <= N; ++n) {
    for (int m = 0; m < n; ++m) {
      // eq:gravity:deriv: the (2 - delta_0m)/2 factor is 1/2 at m = 0 and 1
      // for m >= 1.
      const double delta = (m == 0) ? 0.5 : 1.0;
      g_deriv_[tri(n, m)] =
          std::sqrt(static_cast<double>(n - m) * (n + m + 1.0) * delta);
    }
  }
}

Eigen::Vector3d PinesGravity::acceleration(const Eigen::Vector3d& r_bf,
                                           GravityTier tier, int degree,
                                           int order) {
  // Resolve the evaluation band per tier (FR-5 tiers).
  int n_eval;
  int m_eval;
  switch (tier) {
    case GravityTier::kPointMass:
      n_eval = 0;
      m_eval = 0;
      break;
    case GravityTier::kJ2Only:
      if (field_.n_max < 2) {
        throw std::invalid_argument(
            "gravity: J2-only tier requires a field of degree >= 2");
      }
      n_eval = 2;
      m_eval = 0;
      break;
    case GravityTier::kFull:
    default:
      n_eval = (degree < 0) ? field_.n_max : degree;
      if (n_eval > field_.n_max) {
        throw std::invalid_argument(
            "gravity: requested degree exceeds the stored field");
      }
      m_eval = (order < 0) ? (n_eval < field_.m_max ? n_eval : field_.m_max)
                           : order;
      if (m_eval > n_eval || m_eval > field_.m_max) {
        throw std::invalid_argument(
            "gravity: requested order exceeds the stored field or degree");
      }
      break;
  }

  const double x = r_bf.x();
  const double y = r_bf.y();
  const double z = r_bf.z();
  const double r = std::sqrt(x * x + y * y + z * z);
  if (!(r > 0.0)) {
    throw std::domain_error("gravity: evaluation at r = 0");
  }
  // eq:gravity:pines: direction cosines. The formulation never divides by
  // the projected equatorial radius, so s = t = 0 (the exact pole, u = +/-1)
  // is a regular point.
  const double inv_r = 1.0 / r;
  const double s = x * inv_r;
  const double t = y * inv_r;
  const double u = z * inv_r;

  // Normalized Helmholtz triangle A-bar(n, m)(u), computed column-first:
  // diagonal seed, first sub-diagonal, then the fixed-order column
  // recursion in ascending n for each m (eq:gravity:diag,
  // eq:gravity:subdiag, eq:gravity:column). The full triangle (m <= n) is
  // computed regardless of m_eval because the u-derivative of column m
  // reads column m + 1 (eq:gravity:deriv).
  abar_[tri(0, 0)] = 1.0;
  for (int m = 1; m <= n_eval; ++m) {
    abar_[tri(m, m)] =
        f_diag_[static_cast<std::size_t>(m)] * abar_[tri(m - 1, m - 1)];
  }
  for (int m = 0; m < n_eval; ++m) {
    abar_[tri(m + 1, m)] =
        u * f_sub_[static_cast<std::size_t>(m)] * abar_[tri(m, m)];
  }
  for (int m = 0; m <= n_eval; ++m) {
    for (int n = m + 2; n <= n_eval; ++n) {
      abar_[tri(n, m)] =
          c1_[tri(n, m)] * u * abar_[tri(n - 1, m)] -
          c2_[tri(n, m)] * abar_[tri(n - 2, m)];
    }
  }

  // eq:gravity:rmim: r_m + i*i_m = (s + i t)^m by the complex product
  // recursion -- polynomials in (s, t), no trigonometry, regular at the
  // poles where s = t = 0.
  rm_[0] = 1.0;
  im_[0] = 0.0;
  for (int m = 1; m <= m_eval; ++m) {
    rm_[static_cast<std::size_t>(m)] =
        s * rm_[static_cast<std::size_t>(m - 1)] -
        t * im_[static_cast<std::size_t>(m - 1)];
    im_[static_cast<std::size_t>(m)] =
        s * im_[static_cast<std::size_t>(m - 1)] +
        t * rm_[static_cast<std::size_t>(m - 1)];
  }

  // rho_n = (GM/r) (R/r)^n, built by one multiply per degree.
  const double rr = field_.ref_radius_m * inv_r;
  rho_[0] = field_.gm_m3ps2 * inv_r;
  for (int n = 1; n <= n_eval; ++n) {
    rho_[static_cast<std::size_t>(n)] =
        rho_[static_cast<std::size_t>(n - 1)] * rr;
  }

  // eq:gravity:sums: the four Pines sums, accumulated in a fixed ascending
  // (n outer, m inner) order (D-10 fixed evaluation order).
  double a1 = 0.0;
  double a2 = 0.0;
  double a3 = 0.0;
  double a4 = 0.0;
  for (int n = 0; n <= n_eval; ++n) {
    if (tier == GravityTier::kJ2Only && n == 1) {
      // J2-only means exactly the degree-0 and (2,0) terms: a hypothetical
      // nonzero stored degree-1 row must not leak into this tier.
      continue;
    }
    const int m_hi = (n < m_eval) ? n : m_eval;
    const double fac = rho_[static_cast<std::size_t>(n)] * inv_r;
    for (int m = 0; m <= m_hi; ++m) {
      const double cnm = field_.cbar[field_.index(n, m)];
      const double snm = field_.sbar[field_.index(n, m)];
      const double anm = abar_[tri(n, m)];
      // eq:gravity:deriv: dA-bar(n,m)/du reads A-bar(n, m+1); zero for
      // m = n since A(n, m > n) = 0.
      const double aprime =
          (m < n) ? g_deriv_[tri(n, m)] * abar_[tri(n, m + 1)] : 0.0;
      const double dnm =
          cnm * rm_[static_cast<std::size_t>(m)] +
          snm * im_[static_cast<std::size_t>(m)];
      if (m > 0) {
        const double enm =
            cnm * rm_[static_cast<std::size_t>(m - 1)] +
            snm * im_[static_cast<std::size_t>(m - 1)];
        const double fnm =
            snm * rm_[static_cast<std::size_t>(m - 1)] -
            cnm * im_[static_cast<std::size_t>(m - 1)];
        a1 += fac * static_cast<double>(m) * anm * enm;
        a2 += fac * static_cast<double>(m) * anm * fnm;
      }
      a3 += fac * aprime * dnm;
      a4 += fac * (static_cast<double>(n + m + 1) * anm + u * aprime) * dnm;
    }
  }

  // eq:gravity:accel: assemble the body-fixed acceleration; the radial sum
  // a4 carries the central term, so degree 0 alone yields -GM/r^2 r-hat.
  return Eigen::Vector3d(a1 - s * a4, a2 - t * a4, a3 - u * a4);
}

}  // namespace models
}  // namespace star
