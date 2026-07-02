// Reference-frame implementation (FR-3). Derivation and validation
// evidence: math library chapter ch:frames. Equation labels from that
// chapter are echoed verbatim at the corresponding code (FR-29
// traceability). The Earth-orientation evaluation mirrors the ERFA 2.0.1
// reference routines (nut00b, pfw06, obl06, fw2m, bpn2xy, s06, c2ixys,
// era00) operation for operation, including summation order, so the
// ERFA-generated golden vectors pin the implementation to the model
// (D-10: series summation order fixed - smallest terms first, exactly the
// published SOFA/ERFA order).
#include "star/frames.hpp"

#include <cmath>

#include "frames_series.hpp"
#include "star/constants.hpp"
#include "star/rotation.hpp"

namespace star {
namespace frames {

namespace {

// Arcseconds to radians. Value as in SOFA/ERFA (DAS2R): 2*pi/(360*3600)
// correctly rounded; kept textually identical to the reference so the
// series conversions are bit-compatible.
constexpr double kDas2R = 4.848136811095359935899141e-6;
// Milliarcseconds to radians.
constexpr double kMas2R = kDas2R / 1e3;
// Arcseconds in one full turn.
constexpr double kTurnas = 1296000.0;
// 2*pi, single project-wide value (star/constants.hpp; identical as a
// binary64 to ERFA's D2PI).
constexpr double kTwoPi = constants::TWO_PI;
// Degrees to radians for the Mars IAU elements. TWO_PI/360 and pi/180
// round to the same binary64 (2*pi is exact scaling), so this matches the
// golden generator's math.radians(1.0).
constexpr double kDeg2Rad = constants::TWO_PI / 360.0;
// Julian date of J2000 and days per Julian century.
constexpr double kDj00 = 2451545.0;
constexpr double kDjc = 36525.0;

// Nested-polynomial evaluation in the exact parenthesization of the ERFA
// reference (c0 + t*(c1 + t*(...))): the operation order is part of the
// bit-compatibility contract with the golden vectors.
double poly5(const double (&c)[5], double t) {
  return c[0] + t * (c[1] + t * (c[2] + t * (c[3] + t * c[4])));
}

double poly6(const double (&c)[6], double t) {
  return c[0] + t * (c[1] + t * (c[2] + t * (c[3] + t * (c[4] + t * c[5]))));
}

// IAU 2006 bias-precession Fukushima-Williams angles and mean obliquity
// (eq:frames:fw); polynomials from frames_series.hpp.
void fw_angles_06(double t, double& gamb, double& phib, double& psib,
                  double& epsa) {
  gamb = poly6(series::kFwGamb, t) * kDas2R;
  phib = poly6(series::kFwPhib, t) * kDas2R;
  psib = poly6(series::kFwPsib, t) * kDas2R;
  epsa = poly6(series::kObl06, t) * kDas2R;
}

// Fukushima-Williams angles to rotation matrix:
// C = R1(-eps) R3(-psi) R1(phib) R3(gamb) (eq:frames:fw).
Eigen::Matrix3d fw2m(double gamb, double phib, double psi, double eps) {
  Eigen::Matrix3d r = rotation::r3(gamb);
  r = rotation::r1(phib) * r;
  r = rotation::r3(-psi) * r;
  r = rotation::r1(-eps) * r;
  return r;
}

// Celestial-to-intermediate matrix from CIP X, Y and the CIO locator s
// (eq:frames:c2i): C = R3(-(E+s)) R2(d) R3(E) with E = atan2(Y, X) and
// d = atan(sqrt((X^2+Y^2)/(1-X^2-Y^2))) (spherical angles of the CIP).
Eigen::Matrix3d c2ixys(double x, double y, double s) {
  const double r2 = x * x + y * y;
  const double e = (r2 > 0.0) ? std::atan2(y, x) : 0.0;
  const double d = std::atan(std::sqrt(r2 / (1.0 - r2)));
  Eigen::Matrix3d r = rotation::r3(e);
  r = rotation::r2(d) * r;
  r = rotation::r3(-(e + s)) * r;
  return r;
}

// The eight fundamental arguments of the s(X,Y) series (IERS Conventions
// 2003/2010 full forms): Delaunay l, l', F, D, Om [rad], mean longitudes
// of Venus and Earth [rad], general precession in longitude [rad].
void fundamental_args_03(double t, double (&fa)[8]) {
  fa[0] = std::fmod(poly5(series::kFaL, t), kTurnas) * kDas2R;
  fa[1] = std::fmod(poly5(series::kFaLp, t), kTurnas) * kDas2R;
  fa[2] = std::fmod(poly5(series::kFaF, t), kTurnas) * kDas2R;
  fa[3] = std::fmod(poly5(series::kFaD, t), kTurnas) * kDas2R;
  fa[4] = std::fmod(poly5(series::kFaOm, t), kTurnas) * kDas2R;
  fa[5] = std::fmod(series::kFaVe[0] + series::kFaVe[1] * t, kTwoPi);
  fa[6] = std::fmod(series::kFaE[0] + series::kFaE[1] * t, kTwoPi);
  fa[7] = (series::kFaPa[0] + series::kFaPa[1] * t) * t;
}

// CIO locator s (eq:frames:s06): the s + XY/2 series summed per t-order,
// smallest terms first within each order (the published evaluation order,
// D-10), then the Horner combination and the -XY/2 restoration.
double s06(double t, double x, double y) {
  double fa[8];
  fundamental_args_03(t, fa);
  double w0 = series::kS06Poly[0];
  double w1 = series::kS06Poly[1];
  double w2 = series::kS06Poly[2];
  double w3 = series::kS06Poly[3];
  double w4 = series::kS06Poly[4];
  const double w5 = series::kS06Poly[5];

  const auto accumulate = [&fa](const series::STerm* terms, std::size_t n,
                                double& w) {
    for (std::size_t k = n; k-- > 0;) {
      double a = 0.0;
      for (int j = 0; j < 8; ++j) {
        a += static_cast<double>(terms[k].nfa[j]) * fa[j];
      }
      w += terms[k].s * std::sin(a) + terms[k].c * std::cos(a);
    }
  };
  accumulate(series::kS06Terms0, series::kS06Terms0Count, w0);
  accumulate(series::kS06Terms1, series::kS06Terms1Count, w1);
  accumulate(series::kS06Terms2, series::kS06Terms2Count, w2);
  accumulate(series::kS06Terms3, series::kS06Terms3Count, w3);
  accumulate(series::kS06Terms4, series::kS06Terms4Count, w4);

  return (w0 + (w1 + (w2 + (w3 + (w4 + w5 * t) * t) * t) * t) * t) * kDas2R -
         x * y / 2.0;
}

// UT1 as a two-part quasi-Julian date: whole-day part exact, seconds of
// day carried at ulp(86400 s) ~ 1.5e-11 s so the Earth rotation angle
// retains the D-6 precision budget (header note). UT1 = UTC + dut1 =
// TAI - dAT + dut1, with dAT looked up for the epoch's UTC calendar date.
void ut1_jd(const time::TaiEpoch& tai, double dut1_s, double& dj1,
            double& dj2) {
  const time::UtcTime utc = time::utc_from_tai(tai);
  const int dat = time::tai_minus_utc_s(utc.year, utc.month, utc.day);
  // 2451544.5 is JD of 2000-01-01T00:00:00 TAI epoch day 0; the sum with
  // an integer day count is exact in binary64 for the whole table domain.
  dj1 = 2451544.5 + static_cast<double>(tai.day);
  dj2 = (tai.sec - static_cast<double>(dat) + dut1_s) / 86400.0;
}

}  // namespace

// IAU 2000B nutation (eq:frames:nut00b): 77 luni-solar terms on the
// simplified linear Delaunay arguments plus the fixed planetary-bias
// offsets. Summation runs smallest terms first (reverse table order),
// exactly the published SOFA/ERFA evaluation (D-10 fixed order).
void nutation_00b(double t, double& dpsi_rad, double& deps_rad) {
  const double el =
      std::fmod(series::kNutFaL[0] + series::kNutFaL[1] * t, kTurnas) * kDas2R;
  const double elp =
      std::fmod(series::kNutFaLp[0] + series::kNutFaLp[1] * t, kTurnas) *
      kDas2R;
  const double f =
      std::fmod(series::kNutFaF[0] + series::kNutFaF[1] * t, kTurnas) * kDas2R;
  const double d =
      std::fmod(series::kNutFaD[0] + series::kNutFaD[1] * t, kTurnas) * kDas2R;
  const double om =
      std::fmod(series::kNutFaOm[0] + series::kNutFaOm[1] * t, kTurnas) *
      kDas2R;

  double dp = 0.0;
  double de = 0.0;
  for (std::size_t k = series::kNut00bCount; k-- > 0;) {
    const series::NutTerm& term = series::kNut00b[k];
    const double arg =
        std::fmod(static_cast<double>(term.nl) * el +
                      static_cast<double>(term.nlp) * elp +
                      static_cast<double>(term.nf) * f +
                      static_cast<double>(term.nd) * d +
                      static_cast<double>(term.nom) * om,
                  kTwoPi);
    const double sarg = std::sin(arg);
    const double carg = std::cos(arg);
    dp += (term.ps + term.pst * t) * sarg + term.pc * carg;
    de += (term.ec + term.ect * t) * carg + term.es * sarg;
  }
  // 0.1 microarcsecond units to radians, plus the fixed offsets standing
  // in for the truncated planetary terms.
  const double u2r = kDas2R / 1e7;
  dpsi_rad = dp * u2r + series::kNut00bDpPlanMas * kMas2R;
  deps_rad = de * u2r + series::kNut00bDePlanMas * kMas2R;
}

CipCio cip_cio_06b(const time::TaiEpoch& tai) {
  const double t = time::tt_julian_centuries(tai);
  double dpsi;
  double deps;
  nutation_00b(t, dpsi, deps);
  double gamb;
  double phib;
  double psib;
  double epsa;
  fw_angles_06(t, gamb, phib, psib, epsa);
  // NPB matrix: nutation folded into the FW angles (eq:frames:xy); the CIP
  // coordinates are the bottom-row x and y components of the NPB matrix.
  const Eigen::Matrix3d rnpb = fw2m(gamb, phib, psib + dpsi, epsa + deps);
  CipCio out;
  out.x_rad = rnpb(2, 0);
  out.y_rad = rnpb(2, 1);
  out.s_rad = s06(t, out.x_rad, out.y_rad);
  return out;
}

// Earth rotation angle (eq:frames:era): ERA = 2 pi (frac(Tu) + c0 + c1 Tu)
// with Tu = UT1 Julian days since J2000. The fractional part is taken from
// the two JD parts separately before the small linear rate multiplies the
// (less precise) collapsed Tu - the construction that lets a two-part UT1
// deliver sub-1e-11-rad rotation resolution.
double era_00(const time::TaiEpoch& tai, double dut1_s) {
  double dj1;
  double dj2;
  ut1_jd(tai, dut1_s, dj1, dj2);
  double d1;
  double d2;
  if (dj1 < dj2) {
    d1 = dj1;
    d2 = dj2;
  } else {
    d1 = dj2;
    d2 = dj1;
  }
  const double t = d1 + (d2 - kDj00);
  const double f = std::fmod(d1, 1.0) + std::fmod(d2, 1.0);
  double theta =
      std::fmod(kTwoPi * (f + series::kEraC0 + series::kEraC1 * t), kTwoPi);
  if (theta < 0.0) {
    theta += kTwoPi;
  }
  return theta;
}

Eigen::Matrix3d c_gcrf_to_cirs(const time::TaiEpoch& tai) {
  const CipCio cip = cip_cio_06b(tai);
  return c2ixys(cip.x_rad, cip.y_rad, cip.s_rad);
}

// The composed chain (eq:frames:chain). Polar motion is neglected: the
// result is strictly GCRF->TIRS, documented with its ~0.3 urad bound in
// the chapter and the header.
Eigen::Matrix3d c_gcrf_to_itrf(const time::TaiEpoch& tai, double dut1_s) {
  return rotation::r3(era_00(tai, dut1_s)) * c_gcrf_to_cirs(tai);
}

// Moon principal axes from the DE 3-1-3 libration angles (eq:frames:moonpa;
// Park et al. 2021).
Eigen::Matrix3d c_gcrf_to_moonpa(double phi_rad, double theta_rad,
                                 double psi_rad) {
  return rotation::r3(psi_rad) *
         (rotation::r1(theta_rad) * rotation::r3(phi_rad));
}

// Mars IAU 2015 rotational elements (eq:frames:mars): Archinal et al.
// (2018), Cel. Mech. Dyn. Astron. 130:22, Mars entries (post-erratum
// values as distributed in NAIF pck00011.tpc). Angles in degrees, time in
// TDB days d and TDB Julian centuries T from J2000.0; each periodic
// argument is reduced mod 360 degrees before the radian conversion, in
// the same fixed order as the golden generator.
MarsElements mars_elements_iau2015(const time::TaiEpoch& tai) {
  const time::TwoPartJd tdb = time::tdb_jd(tai);
  // Two-part difference keeps the day count exact before the collapse.
  const double d_days = (tdb.jd1 - kDj00) + tdb.jd2;
  const double t = d_days / kDjc;

  // North-pole right ascension alpha0 [deg]: polynomial plus sin terms.
  double ra = 317.269202 + -0.10927547 * t;
  ra += 0.000068 * std::sin(std::fmod(198.991226 + 19139.4819985 * t, 360.0) * kDeg2Rad);
  ra += 0.000238 * std::sin(std::fmod(226.292679 + 38280.8511281 * t, 360.0) * kDeg2Rad);
  ra += 0.000052 * std::sin(std::fmod(249.663391 + 57420.7251593 * t, 360.0) * kDeg2Rad);
  ra += 0.000009 * std::sin(std::fmod(266.183510 + 76560.6367950 * t, 360.0) * kDeg2Rad);
  ra += 0.419057 * std::sin(std::fmod(79.398797 + 0.5042615 * t, 360.0) * kDeg2Rad);

  // North-pole declination delta0 [deg]: polynomial plus cos terms.
  double dec = 54.432516 + -0.05827105 * t;
  dec += 0.000051 * std::cos(std::fmod(122.433576 + 19139.9407476 * t, 360.0) * kDeg2Rad);
  dec += 0.000141 * std::cos(std::fmod(43.058401 + 38280.8753272 * t, 360.0) * kDeg2Rad);
  dec += 0.000031 * std::cos(std::fmod(57.663379 + 57420.7517205 * t, 360.0) * kDeg2Rad);
  dec += 0.000005 * std::cos(std::fmod(79.476401 + 76560.6495004 * t, 360.0) * kDeg2Rad);
  dec += 1.591274 * std::cos(std::fmod(166.325722 + 0.5042615 * t, 360.0) * kDeg2Rad);

  // Prime meridian W [deg]: linear in TDB days, plus sin terms in T.
  double w = std::fmod(176.049863 + 350.891982443297 * d_days, 360.0);
  w += 0.000145 * std::sin(std::fmod(129.071773 + 19140.0328244 * t, 360.0) * kDeg2Rad);
  w += 0.000157 * std::sin(std::fmod(36.352167 + 38281.0473591 * t, 360.0) * kDeg2Rad);
  w += 0.000040 * std::sin(std::fmod(56.668646 + 57420.9295360 * t, 360.0) * kDeg2Rad);
  w += 0.000001 * std::sin(std::fmod(67.364003 + 76560.2552215 * t, 360.0) * kDeg2Rad);
  w += 0.000001 * std::sin(std::fmod(104.792680 + 95700.4387578 * t, 360.0) * kDeg2Rad);
  w += 0.584542 * std::sin(std::fmod(95.391654 + 0.5042615 * t, 360.0) * kDeg2Rad);

  MarsElements out;
  out.alpha0_rad = ra * kDeg2Rad;
  out.delta0_rad = dec * kDeg2Rad;
  out.w_rad = w * kDeg2Rad;
  return out;
}

// Body-fixed frame from the pole and prime meridian (eq:frames:mars): the
// standard IAU construction C = R3(W) R1(pi/2 - delta0) R3(pi/2 + alpha0).
Eigen::Matrix3d c_gcrf_to_marsfixed(const time::TaiEpoch& tai) {
  const MarsElements e = mars_elements_iau2015(tai);
  const double half_pi = constants::TWO_PI / 4.0;
  return rotation::r3(e.w_rad) *
         (rotation::r1(half_pi - e.delta0_rad) *
          rotation::r3(half_pi + e.alpha0_rad));
}

}  // namespace frames
}  // namespace star
