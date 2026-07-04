// Composed environment force model (ch:environment): frame/time plumbing and
// the fixed-order force summation. The physics of every term lives in its own
// chapter-tracked module (gravity, thirdbody, srp, atmosphere_*, drag); this
// file decides only WHERE each model is evaluated (frames, ephemeris
// composition) and in WHAT order the accelerations are summed (D-10 fixed
// force-summation order).
#include "star/models/environment.hpp"

#include <cmath>
#include <stdexcept>

#include "star/constants.hpp"
#include "star/frames.hpp"
#include "star/models/atmosphere_hp.hpp"
#include "star/models/atmosphere_mars.hpp"
#include "star/models/atmosphere_ussa76.hpp"
#include "star/models/drag.hpp"
#include "star/models/srp.hpp"
#include "star/models/thirdbody.hpp"
#include "star/models/twobody.hpp"

namespace star {
namespace models {

namespace {

// J2000 on the JD scale: JD 2451545.0 TDB (eq:time:j2000). Local because the
// only consumer is the JD -> seconds-since-J2000 conversion below.
constexpr double kJ2000Jd = 2451545.0;

// Canonical summation order (ch:environment, composition section). The array
// index equals the Body enum value; iteration over this order is what makes
// the third-body sum reproducible independently of configuration file order.
constexpr int kBodyCount = 6;

EnvironmentModel::Body body_from_name(const std::string& name) {
  if (name == "sun") return EnvironmentModel::Body::kSun;
  if (name == "earth") return EnvironmentModel::Body::kEarth;
  if (name == "moon") return EnvironmentModel::Body::kMoon;
  if (name == "venus") return EnvironmentModel::Body::kVenus;
  if (name == "mars") return EnvironmentModel::Body::kMars;
  if (name == "jupiter") return EnvironmentModel::Body::kJupiter;
  throw std::invalid_argument("environment: unknown body name \"" + name +
                              "\" (allowed: sun, earth, moon, venus, mars, "
                              "jupiter)");
}

// DE440 header GM for each perturbing body (single home: constants.hpp).
double body_gm(EnvironmentModel::Body body) {
  switch (body) {
    case EnvironmentModel::Body::kSun:
      return constants::GM_SUN_DE440_M3_PER_S2;
    case EnvironmentModel::Body::kEarth:
      return constants::GM_EARTH_DE440_M3_PER_S2;
    case EnvironmentModel::Body::kMoon:
      return constants::GM_MOON_DE440_M3_PER_S2;
    case EnvironmentModel::Body::kVenus:
      return constants::GM_VENUS_DE440_M3_PER_S2;
    case EnvironmentModel::Body::kMars:
      return constants::GM_MARS_SYS_DE440_M3_PER_S2;
    case EnvironmentModel::Body::kJupiter:
      return constants::GM_JUPITER_SYS_DE440_M3_PER_S2;
  }
  throw std::logic_error("environment: unreachable body enum");
}

// Occulting-disk radius for the FR-7 shadow model (ch:environment cites the
// radius choices: WGS84 equatorial for Earth, IAU 2015 values for Moon/Mars).
double occulter_radius_m(EnvironmentModel::Body body) {
  switch (body) {
    case EnvironmentModel::Body::kEarth:
      return constants::WGS84_A_M;
    case EnvironmentModel::Body::kMoon:
      return constants::R_MOON_M;
    case EnvironmentModel::Body::kMars:
      return constants::MARS_ELLIPSOID_A_M;
    default:
      throw std::invalid_argument(
          "environment: only earth, moon, and mars can occult (FR-7)");
  }
}

EnvironmentModel::Body central_as_body(CentralBody central) {
  switch (central) {
    case CentralBody::kEarth:
      return EnvironmentModel::Body::kEarth;
    case CentralBody::kMoon:
      return EnvironmentModel::Body::kMoon;
    case CentralBody::kMars:
      return EnvironmentModel::Body::kMars;
    case CentralBody::kSun:
      return EnvironmentModel::Body::kSun;
  }
  throw std::logic_error("environment: unreachable central-body enum");
}

}  // namespace

double central_body_gm(CentralBody body) {
  switch (body) {
    case CentralBody::kEarth:
      // Phase 1 home for Earth-centered two-body dynamics (IERS TN36); see
      // the constants.hpp note on the deliberate IERS/DE440 split.
      return constants::GM_EARTH_M3_PER_S2;
    case CentralBody::kMoon:
      return constants::GM_MOON_DE440_M3_PER_S2;
    case CentralBody::kMars:
      return constants::GM_MARS_SYS_DE440_M3_PER_S2;
    case CentralBody::kSun:
      return constants::GM_SUN_DE440_M3_PER_S2;
  }
  throw std::logic_error("environment: unreachable central-body enum");
}

EnvironmentModel::EnvironmentModel(const EnvironmentSpec& spec)
    : central_(spec.central_body), epoch_(spec.epoch_tai) {
  // --- gravity tier -------------------------------------------------------
  gm_central_ = central_body_gm(central_);
  if (spec.gravity_model == "pointmass" || spec.gravity_model.empty()) {
    use_field_ = false;
  } else if (spec.gravity_model == "j2" || spec.gravity_model == "harmonic") {
    if (central_ == CentralBody::kSun) {
      // FR-5 defines harmonic tiers for Earth, Moon, and Mars only; the Sun
      // is point-mass by specification, so a field request is a mis-wiring.
      throw std::invalid_argument(
          "environment: the Sun central body is point-mass only (FR-5 "
          "defines no Sun harmonic field)");
    }
    if (spec.gravity_field_path.empty()) {
      throw std::invalid_argument(
          "environment: gravity model \"" + spec.gravity_model +
          "\" requires a gravity field file path");
    }
    use_field_ = true;
    tier_ = (spec.gravity_model == "j2") ? GravityTier::kJ2Only
                                         : GravityTier::kFull;
    degree_ = spec.gravity_degree;
    order_ = spec.gravity_order;
    gravity_.emplace(GravityField::load_file(spec.gravity_field_path));
  } else {
    throw std::invalid_argument(
        "environment: unknown gravity model \"" + spec.gravity_model +
        "\" (allowed: pointmass, j2, harmonic)");
  }

  // --- third bodies, canonical order --------------------------------------
  const Body central_body_enum = central_as_body(central_);
  bool enabled[kBodyCount] = {false, false, false, false, false, false};
  for (const std::string& name : spec.third_bodies) {
    const Body body = body_from_name(name);
    if (body == central_body_enum) {
      throw std::invalid_argument(
          "environment: the central body cannot also be a third body");
    }
    enabled[static_cast<int>(body)] = true;
  }
  for (int i = 0; i < kBodyCount; ++i) {
    if (enabled[i]) {
      const Body body = static_cast<Body>(i);
      perturbers_.push_back({body, body_gm(body)});
    }
  }

  // --- SRP -----------------------------------------------------------------
  srp_enabled_ = spec.srp_enabled;
  if (srp_enabled_) {
    if (!(spec.cr_a_over_m_m2pkg > 0.0) ||
        !std::isfinite(spec.cr_a_over_m_m2pkg)) {
      throw std::invalid_argument(
          "environment: SRP requires a positive, finite Cr*A/m");
    }
    cr_a_over_m_ = spec.cr_a_over_m_m2pkg;
    // The FR-7 "central body always occults" rule applies to the planetary
    // regimes; about the Sun there is no occulting central body and the
    // occulter set is legitimately empty (nu = 1, see EnvironmentSpec).
    if (spec.srp_occulters.empty() && central_ != CentralBody::kSun) {
      throw std::invalid_argument(
          "environment: SRP requires at least one occulter (the central "
          "body at minimum, FR-7)");
    }
    for (const std::string& name : spec.srp_occulters) {
      const Body body = body_from_name(name);
      occulters_.push_back(
          {body, body == central_body_enum, occulter_radius_m(body)});
    }
  }

  // --- drag ----------------------------------------------------------------
  atmosphere_ = spec.atmosphere;
  if (atmosphere_ != AtmosphereModel::kNone) {
    if (!(spec.cd_a_over_m_m2pkg > 0.0) ||
        !std::isfinite(spec.cd_a_over_m_m2pkg)) {
      throw std::invalid_argument(
          "environment: drag requires a positive, finite Cd*A/m");
    }
    cd_a_over_m_ = spec.cd_a_over_m_m2pkg;
    hp_n_ = spec.hp_exponent_n;
    const bool earth_model = atmosphere_ == AtmosphereModel::kUssa76 ||
                             atmosphere_ == AtmosphereModel::kHarrisPriester;
    if (earth_model && central_ != CentralBody::kEarth) {
      throw std::invalid_argument(
          "environment: USSA76 and Harris-Priester are Earth atmospheres");
    }
    if (atmosphere_ == AtmosphereModel::kMarsExponential &&
        central_ != CentralBody::kMars) {
      throw std::invalid_argument(
          "environment: the exponential Mars atmosphere requires central "
          "body mars");
    }
  }

  // --- ephemeris ------------------------------------------------------------
  // Needed by: any third body, SRP (Sun position), Harris-Priester (Sun
  // direction for the diurnal bulge), and the Moon central body (the Moon PA
  // frame interpolates the DE440 libration angles).
  const bool needs_eph =
      !perturbers_.empty() || srp_enabled_ ||
      atmosphere_ == AtmosphereModel::kHarrisPriester ||
      central_ == CentralBody::kMoon;
  if (needs_eph) {
    if (spec.ephemeris_path.empty()) {
      throw std::invalid_argument(
          "environment: this configuration needs an ephemeris file (third "
          "bodies, SRP, Harris-Priester, or a Moon central body)");
    }
    eph_.emplace(Ephemeris::load_file(spec.ephemeris_path));
  }
}

double EnvironmentModel::tdb_s_at(double t_s) const {
  const time::TaiEpoch tai = time::tai_add_seconds(epoch_, t_s);
  const time::TwoPartJd jd = time::tdb_jd(tai);
  // jd1 is a half-integer, so (jd1 - J2000) is exact; one rounding enters
  // through the fraction-of-day sum and the 86400 scale. The resulting
  // ~2.4e-7 s quantum at 2020-2060 magnitudes moves the fastest repacked
  // body (the Moon, ~1 km/s) by less than a micrometre.
  return ((jd.jd1 - kJ2000Jd) + jd.jd2) * 86400.0;
}

Eigen::Vector3d EnvironmentModel::central_ssb(double tdb_s) const {
  static const std::string kEmb = "emb";
  static const std::string kEarthSeg = "earth";
  static const std::string kMoonSeg = "moon";
  static const std::string kMarsBary = "mars_bary";
  static const std::string kSunSeg = "sun";
  switch (central_) {
    case CentralBody::kEarth:
      return eph_->state(kEmb, tdb_s).r_m + eph_->state(kEarthSeg, tdb_s).r_m;
    case CentralBody::kMoon:
      return eph_->state(kEmb, tdb_s).r_m + eph_->state(kMoonSeg, tdb_s).r_m;
    case CentralBody::kMars:
      // DE440 stores the Mars-system barycenter; the Mars body offset from
      // it is bounded by the Phobos/Deimos mass fractions (~2e-4 m) and is a
      // documented approximation (ch:environment).
      return eph_->state(kMarsBary, tdb_s).r_m;
    case CentralBody::kSun:
      // The Sun's own SREPH segment; SRP's Sun-relative-to-central position
      // then differences this against itself, so the spacecraft-Sun vector
      // reduces to exactly -r (Sun at the origin of the heliocentric frame).
      return eph_->state(kSunSeg, tdb_s).r_m;
  }
  throw std::logic_error("environment: unreachable central-body enum");
}

Eigen::Vector3d EnvironmentModel::body_rel_central(
    Body body, double tdb_s, const Eigen::Vector3d& r_central_ssb) const {
  static const std::string kSun = "sun";
  static const std::string kEmb = "emb";
  static const std::string kEarthSeg = "earth";
  static const std::string kMoonSeg = "moon";
  static const std::string kVenusBary = "venus_bary";
  static const std::string kMarsBary = "mars_bary";
  static const std::string kJupiterBary = "jupiter_bary";

  // Earth<->Moon pairs difference the EMB-relative segments directly (the
  // validated moon_geocentric composition) instead of routing through the
  // SSB, avoiding a needless ~1e11 m intermediate magnitude.
  if (central_ == CentralBody::kEarth && body == Body::kMoon) {
    return eph_->moon_geocentric(tdb_s).r_m;
  }
  if (central_ == CentralBody::kMoon && body == Body::kEarth) {
    return -eph_->moon_geocentric(tdb_s).r_m;
  }

  switch (body) {
    case Body::kSun:
      return eph_->state(kSun, tdb_s).r_m - r_central_ssb;
    case Body::kEarth:
      return eph_->state(kEmb, tdb_s).r_m + eph_->state(kEarthSeg, tdb_s).r_m -
             r_central_ssb;
    case Body::kMoon:
      return eph_->state(kEmb, tdb_s).r_m + eph_->state(kMoonSeg, tdb_s).r_m -
             r_central_ssb;
    case Body::kVenus:
      return eph_->state(kVenusBary, tdb_s).r_m - r_central_ssb;
    case Body::kMars:
      return eph_->state(kMarsBary, tdb_s).r_m - r_central_ssb;
    case Body::kJupiter:
      return eph_->state(kJupiterBary, tdb_s).r_m - r_central_ssb;
  }
  throw std::logic_error("environment: unreachable body enum");
}

Eigen::Matrix3d EnvironmentModel::c_gcrf_to_bodyfixed(double t_s,
                                                      double tdb_s) const {
  const time::TaiEpoch tai = time::tai_add_seconds(epoch_, t_s);
  switch (central_) {
    case CentralBody::kEarth:
      // Constant dUT1 = 0 per FR-3 (no EOP series is ingested); the bound is
      // documented in ch:frames and inherited by ch:environment.
      return frames::c_gcrf_to_itrf(tai, 0.0);
    case CentralBody::kMoon: {
      const LibrationAngles lib = eph_->lunar_librations(tdb_s);
      return frames::c_gcrf_to_moonpa(lib.angles_rad[0], lib.angles_rad[1],
                                      lib.angles_rad[2]);
    }
    case CentralBody::kMars:
      return frames::c_gcrf_to_marsfixed(tai);
    case CentralBody::kSun:
      // Unreachable by construction: the only body-fixed consumers are the
      // harmonic gravity tiers and the atmospheres, and the constructor
      // rejects both for the Sun central body.
      throw std::logic_error(
          "environment: no body-fixed frame is defined for the Sun");
  }
  throw std::logic_error("environment: unreachable central-body enum");
}

Eigen::Vector3d EnvironmentModel::acceleration(double t_s,
                                               const Eigen::Vector3d& r_m,
                                               const Eigen::Vector3d& v_mps) {
  // Per-call shared time quantities. The TDB epoch and the body-fixed
  // rotation are computed at most once per evaluation and reused by every
  // term that needs them, so all terms see one consistent epoch.
  const double tdb_s = eph_.has_value() ? tdb_s_at(t_s) : 0.0;
  const bool needs_bf = use_field_ ||
                        atmosphere_ != AtmosphereModel::kNone;
  Eigen::Matrix3d c_bf = Eigen::Matrix3d::Identity();
  if (needs_bf) {
    c_bf = c_gcrf_to_bodyfixed(t_s, tdb_s);
  }
  Eigen::Vector3d r_central_ssb = Eigen::Vector3d::Zero();
  if (eph_.has_value()) {
    r_central_ssb = central_ssb(tdb_s);
  }

  Eigen::Vector3d a = Eigen::Vector3d::Zero();

  // (a) central-body gravity. The harmonic tiers evaluate in the body-fixed
  // frame and rotate the result back (C is orthonormal: C^T = C^-1).
  if (use_field_) {
    const Eigen::Vector3d r_bf = c_bf * r_m;
    a += c_bf.transpose() *
         gravity_->acceleration(r_bf, tier_, degree_, order_);
  } else {
    a += twobody_accel(gm_central_, r_m);
  }

  // (b) third bodies, canonical order (perturbers_ was built in that order).
  for (const Perturber& p : perturbers_) {
    const Eigen::Vector3d r_third = body_rel_central(p.body, tdb_s,
                                                     r_central_ssb);
    a += thirdbody_accel(p.gm_m3ps2, r_m, r_third);
  }

  // (c) SRP. The combined illumination fraction over multiple occulters is
  // the product of the per-occulter fractions - exact for a single occulter
  // and whenever any fraction is 0 or 1, a documented approximation when two
  // bodies partially occult simultaneously (ch:environment).
  if (srp_enabled_) {
    const Eigen::Vector3d r_sun = body_rel_central(Body::kSun, tdb_s,
                                                   r_central_ssb);
    double nu = 1.0;
    for (const Occulter& occ : occulters_) {
      const Eigen::Vector3d r_occ =
          occ.is_central ? Eigen::Vector3d::Zero()
                         : body_rel_central(occ.body, tdb_s, r_central_ssb);
      nu *= shadow_fraction(r_m, r_sun, constants::R_SUN_M, r_occ,
                            occ.radius_m);
    }
    a += srp_accel(cr_a_over_m_, nu, r_m, r_sun);
  }

  // (d) drag with the FR-8 co-rotating air-relative velocity. The planet's
  // angular-velocity vector is omega * z_bf resolved in GCRF; z_bf in GCRF
  // coordinates is the third ROW of C_GCRF->body-fixed (eq:drag:vrel).
  if (atmosphere_ != AtmosphereModel::kNone) {
    double rho = 0.0;
    double omega = 0.0;
    const Eigen::Vector3d r_bf = c_bf * r_m;
    if (central_ == CentralBody::kEarth) {
      omega = constants::OMEGA_EARTH_RAD_PER_S;
      const double alt_m = geodetic_altitude(r_bf, constants::WGS84_A_M,
                                             constants::WGS84_INV_F);
      if (atmosphere_ == AtmosphereModel::kUssa76) {
        // The USSA76 module's domain ends at 1000 km; above it the
        // composition takes rho = 0 exactly, mirroring the Harris-Priester
        // ceiling rule so drag vanishes instead of aborting the run
        // (ch:environment, documented approximation).
        rho = (alt_m > 1000.0e3) ? 0.0 : ussa76_density(alt_m);
      } else {
        const Eigen::Vector3d u_sun =
            body_rel_central(Body::kSun, tdb_s, r_central_ssb).normalized();
        const Eigen::Vector3d apex = hp_bulge_apex(u_sun);
        const double cos_psi = r_m.normalized().dot(apex);
        rho = hp_density(alt_m, cos_psi, hp_n_);
      }
    } else {
      // Mars (the constructor rejects drag about the Moon and the Sun).
      omega = constants::OMEGA_MARS_RAD_PER_S;
      const double alt_m = geodetic_altitude(
          r_bf, constants::MARS_ELLIPSOID_A_M, constants::MARS_ELLIPSOID_INV_F);
      rho = mars_density(alt_m);
    }
    const Eigen::Vector3d omega_vec = omega * c_bf.row(2).transpose();
    const Eigen::Vector3d v_rel = v_mps - omega_vec.cross(r_m);
    a += drag_accel(rho, cd_a_over_m_, v_rel);
  }

  return a;
}

void EnvironmentModel::rhs(double t_s, const double* y, double* ydot) {
  const Eigen::Map<const Eigen::Vector3d> r(y);
  const Eigen::Map<const Eigen::Vector3d> v(y + 3);
  const Eigen::Vector3d a = acceleration(t_s, r, v);
  ydot[0] = y[3];
  ydot[1] = y[4];
  ydot[2] = y[5];
  ydot[3] = a[0];
  ydot[4] = a[1];
  ydot[5] = a[2];
}

}  // namespace models
}  // namespace star
