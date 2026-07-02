// Reference frames for the star:: core (FR-3): the CIO-based GCRF<->Earth-
// fixed chain (IAU 2006/2000B), the Moon principal-axis frame from DE
// libration angles, and the Mars IAU 2015 body-fixed frame. Derivations,
// domain bounds, and validation evidence are in the math library chapter
// ch:frames (docs/mathlib/chapters/frames.tex).
//
// Model summary and documented approximations (chapter, domain of
// validity):
// - Earth orientation is the CIO-based transformation of IERS Conventions
//   (2010) Chapter 5: CIP coordinates X, Y from the IAU 2006 precession
//   (Fukushima-Williams angles) composed with the IAU 2000B 77-term
//   nutation series, CIO locator s from the s(X,Y) series, then
//   C_GCRF->ITRF = R3(ERA) * C_CIO (eq:frames:chain).
// - POLAR MOTION IS NEGLECTED: the "ITRF" produced here is strictly the
//   Terrestrial Intermediate Reference System. The omission is a rotation
//   of order 0.3 urad (~15 m displacement at the Earth's surface), bounded
//   in the chapter exactly as omitted here.
// - UT1 = UTC + dUT1 with a CONSTANT, user-suppliable dUT1 (default 0):
//   no EOP time series is ingested (PRD non-goal). |UT1-UTC| < 0.9 s by
//   construction of UTC; dUT1 = 0 therefore bounds the Earth-rotation
//   error by 0.9 s of spin, ~66 urad (~420 m at the surface), and a
//   user-supplied constant reduces it to the drift accumulated over the
//   mission window (chapter, domain of validity).
// - All matrices are DCMs in the project convention: C_A^B maps frame A
//   coordinates to frame B (star/rotation.hpp).
//
// The two-part TAI epoch (star/time.hpp) is consumed directly so UT1
// retains sub-nanosecond resolution through the Earth rotation angle: an
// ERA error of 1e-11 rad corresponds to ~1.4e-7 s of UT1, which a single
// collapsed double cannot carry across 2020-2060 (see the time chapter's
// precision analysis).
#ifndef STAR_FRAMES_HPP
#define STAR_FRAMES_HPP

#include <Eigen/Dense>

#include "star/time.hpp"

namespace star {
namespace frames {

// CIP coordinates X, Y and CIO locator s [rad] for the IAU 2006/2000B
// model at the given epoch (eq:frames:xy, eq:frames:s06). Construction:
// IAU 2006 precession (Fukushima-Williams angles, eq:frames:fw) composed
// with the IAU 2000B nutation (eq:frames:nut00b) into the NPB matrix; X, Y
// are the (2,0) and (2,1) elements; s from the s(X,Y) series (SOFA S06
// form).
struct CipCio {
  double x_rad;
  double y_rad;
  double s_rad;
};
CipCio cip_cio_06b(const time::TaiEpoch& tai);

// IAU 2000B nutation in longitude and obliquity [rad] at TT Julian
// centuries t since J2000 (eq:frames:nut00b). Exposed for validation
// against the golden vectors; the chain uses it internally.
void nutation_00b(double tt_centuries, double& dpsi_rad, double& deps_rad);

// Earth rotation angle [rad] at UT1 = UTC + dut1_s (eq:frames:era;
// Capitaine, Guinot & McCarthy 2000). dut1_s is the constant user-supplied
// UT1-UTC offset in seconds, default 0 per FR-3.
double era_00(const time::TaiEpoch& tai, double dut1_s);

// GCRS -> CIRS matrix (precession-nutation-bias plus CIO locator,
// eq:frames:c2i): the celestial-to-intermediate matrix built from X, Y, s.
Eigen::Matrix3d c_gcrf_to_cirs(const time::TaiEpoch& tai);

// The composed Earth-fixed transformation (eq:frames:chain):
// C_GCRF->ITRF = R3(ERA(UT1)) * C_GCRF->CIRS, polar motion neglected (see
// header note; the result is strictly the TIRS). dut1_s as in era_00.
Eigen::Matrix3d c_gcrf_to_itrf(const time::TaiEpoch& tai, double dut1_s);

// Moon principal-axis frame from the DE libration Euler angles
// (eq:frames:moonpa): C_GCRF->MoonPA = R3(psi) R1(theta) R3(phi), the
// 3-1-3 sequence of the DE convention (Park et al. 2021: phi, theta, psi
// are the angles bringing the ICRF axes onto the lunar principal axes).
// The angles are plain arguments: the ephemeris evaluator that
// interpolates them from DE440 Chebyshev coefficients is a separate
// module and this construction must not depend on it.
Eigen::Matrix3d c_gcrf_to_moonpa(double phi_rad, double theta_rad,
                                 double psi_rad);

// Mars IAU 2015 rotational elements (Archinal et al. 2018): north-pole
// right ascension alpha0, declination delta0, and prime meridian angle W,
// all in radians, at the epoch's TDB (eq:frames:mars). The report's time
// arguments are TDB days (W) and TDB Julian centuries (poles and periodic
// terms) since J2000.0 = JD 2451545.0 TDB, realized through the D-6
// truncated TDB series of star/time.hpp.
struct MarsElements {
  double alpha0_rad;
  double delta0_rad;
  double w_rad;
};
MarsElements mars_elements_iau2015(const time::TaiEpoch& tai);

// Mars body-fixed frame (eq:frames:mars):
// C_GCRF->MarsFixed = R3(W) R1(pi/2 - delta0) R3(pi/2 + alpha0).
Eigen::Matrix3d c_gcrf_to_marsfixed(const time::TaiEpoch& tai);

}  // namespace frames
}  // namespace star

#endif  // STAR_FRAMES_HPP
