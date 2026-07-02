// Reference-frame golden-vector and property tests (FR-3; FR-22 layers 1
// and 2). Reference values come from tests/golden/frames/ - provenance and
// tolerances in that directory's manifest.toml. Test IDs are cited by the
// math-library validation table (ch:frames); do not rename them.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "../src/frames_series.hpp"
#include "golden_io.hpp"
#include "star/frames.hpp"
#include "star/rotation.hpp"
#include "star/time.hpp"
#include "vendor/doctest.h"

namespace {

namespace frames = star::frames;
namespace series = star::frames::series;

// Arcseconds/degrees to radians, same binary64 values the core uses.
constexpr double kDas2R = 4.848136811095359935899141e-6;
constexpr double kDeg2Rad = 6.283185307179586476925286766559 / 360.0;

std::string golden_path(const char* file) {
  return std::string(STAR_GOLDEN_DIR) + "/frames/" + file;
}

// parse_hex_double is strtod underneath, so it reads the decimal strings
// of series_terms.toml and the cookbook file exactly as well.
double d(const star_tests::GoldenCase& c, const char* key) {
  return star_tests::parse_hex_double(c.scalar(key));
}

std::int64_t parse_int(const std::string& s) { return std::stoll(s); }

star::time::TaiEpoch epoch_of(const star_tests::GoldenCase& c) {
  return {parse_int(c.scalar("tai_day")), d(c, "tai_sec")};
}

Eigen::Matrix3d golden_matrix(const star_tests::GoldenCase& c,
                              const char* prefix = "c") {
  Eigen::Matrix3d m;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      const std::string key =
          std::string(prefix) + std::to_string(i) + std::to_string(j);
      m(i, j) = d(c, key.c_str());
    }
  }
  return m;
}

double max_abs_diff(const Eigen::Matrix3d& a, const Eigen::Matrix3d& b) {
  return (a - b).cwiseAbs().maxCoeff();
}

// Orthonormality at machine precision means a few ulp of rounding through
// the 3x3 compositions: the observed maximum over every produced matrix is
// ~1.1e-15, and 2e-15 admits exactly that while failing on any structural
// defect (a normalization or convention error shows at 1e-8 or worse).
void check_proper_rotation(const Eigen::Matrix3d& m) {
  CHECK((m.transpose() * m - Eigen::Matrix3d::Identity())
            .cwiseAbs()
            .maxCoeff() <= 2e-15);
  CHECK(std::fabs(m.determinant() - 1.0) <= 2e-15);
}

}  // namespace

TEST_CASE("frames_series_transcription") {
  // Transcription-equality gate: the compiled tables in frames_series.hpp
  // must equal the committed tables that were validated against ERFA at
  // generation time (manifest: exact binary64 equality; shortest-repr
  // decimal round-trips exactly through strtod).
  const auto cases =
      star_tests::load_golden_cases(golden_path("series_terms.toml"));
  std::map<std::string, const star_tests::GoldenCase*> by_name;
  std::size_t nut_terms = 0;
  std::size_t s06_terms[5] = {0, 0, 0, 0, 0};
  for (const auto& c : cases) {
    by_name[c.scalar("name")] = &c;
    const std::string name = c.scalar("name");
    if (name.rfind("nut00b_ls_", 0) == 0) {
      ++nut_terms;
    } else if (name.rfind("s06_t", 0) == 0) {
      ++s06_terms[static_cast<std::size_t>(name[5] - '0')];
    }
  }
  REQUIRE(nut_terms == series::kNut00bCount);
  REQUIRE(s06_terms[0] == series::kS06Terms0Count);
  REQUIRE(s06_terms[1] == series::kS06Terms1Count);
  REQUIRE(s06_terms[2] == series::kS06Terms2Count);
  REQUIRE(s06_terms[3] == series::kS06Terms3Count);
  REQUIRE(s06_terms[4] == series::kS06Terms4Count);

  for (std::size_t i = 0; i < series::kNut00bCount; ++i) {
    char name[32];
    std::snprintf(name, sizeof(name), "nut00b_ls_%03u",
                  static_cast<unsigned>(i + 1));
    const auto* c = by_name.at(name);
    CAPTURE(name);
    const series::NutTerm& t = series::kNut00b[i];
    CHECK(parse_int(c->scalar("nl")) == t.nl);
    CHECK(parse_int(c->scalar("nlp")) == t.nlp);
    CHECK(parse_int(c->scalar("nf")) == t.nf);
    CHECK(parse_int(c->scalar("nd")) == t.nd);
    CHECK(parse_int(c->scalar("nom")) == t.nom);
    CHECK(d(*c, "ps") == t.ps);
    CHECK(d(*c, "pst") == t.pst);
    CHECK(d(*c, "pc") == t.pc);
    CHECK(d(*c, "ec") == t.ec);
    CHECK(d(*c, "ect") == t.ect);
    CHECK(d(*c, "es") == t.es);
  }

  const series::STerm* tables[5] = {series::kS06Terms0, series::kS06Terms1,
                                    series::kS06Terms2, series::kS06Terms3,
                                    series::kS06Terms4};
  const std::size_t counts[5] = {
      series::kS06Terms0Count, series::kS06Terms1Count,
      series::kS06Terms2Count, series::kS06Terms3Count,
      series::kS06Terms4Count};
  const char* fa_labels[8] = {"l", "lp", "f", "d", "om", "ve", "e", "pa"};
  for (int order = 0; order < 5; ++order) {
    for (std::size_t i = 0; i < counts[order]; ++i) {
      char name[32];
      std::snprintf(name, sizeof(name), "s06_t%d_%02u", order,
                    static_cast<unsigned>(i + 1));
      const auto* c = by_name.at(name);
      CAPTURE(name);
      for (int j = 0; j < 8; ++j) {
        const std::string key = std::string("n") + fa_labels[j];
        CHECK(parse_int(c->scalar(key)) == tables[order][i].nfa[j]);
      }
      CHECK(d(*c, "sc") == tables[order][i].s);
      CHECK(d(*c, "cc") == tables[order][i].c);
    }
  }

  const auto check_poly = [&](const char* name, const double* coeffs,
                              std::size_t n) {
    const auto* c = by_name.at(name);
    CAPTURE(name);
    const auto& arr = c->array("coeffs");
    REQUIRE(arr.size() == n);
    for (std::size_t i = 0; i < n; ++i) {
      CHECK(star_tests::parse_hex_double(arr[i]) == coeffs[i]);
    }
  };
  check_poly("s06_poly", series::kS06Poly, 6);
  check_poly("fw_gamb", series::kFwGamb, 6);
  check_poly("fw_phib", series::kFwPhib, 6);
  check_poly("fw_psib", series::kFwPsib, 6);
  check_poly("obl06", series::kObl06, 6);
  check_poly("fa_l", series::kFaL, 5);
  check_poly("fa_lp", series::kFaLp, 5);
  check_poly("fa_f", series::kFaF, 5);
  check_poly("fa_d", series::kFaD, 5);
  check_poly("fa_om", series::kFaOm, 5);
  check_poly("fa_ve", series::kFaVe, 2);
  check_poly("fa_e", series::kFaE, 2);
  check_poly("fa_pa", series::kFaPa, 2);

  // The 2000B nutation's simplified linear fundamental arguments (their
  // truncation is part of the published 2000B model).
  check_poly("nutfa_l", series::kNutFaL, 2);
  check_poly("nutfa_lp", series::kNutFaLp, 2);
  check_poly("nutfa_f", series::kNutFaF, 2);
  check_poly("nutfa_d", series::kNutFaD, 2);
  check_poly("nutfa_om", series::kNutFaOm, 2);

  const auto* offsets = by_name.at("nut00b_planetary_offsets_mas");
  CHECK(d(*offsets, "dpplan") == series::kNut00bDpPlanMas);
  CHECK(d(*offsets, "deplan") == series::kNut00bDePlanMas);
  const auto* era = by_name.at("era00_coeffs");
  CHECK(d(*era, "c0") == series::kEraC0);
  CHECK(d(*era, "c1") == series::kEraC1);
}

TEST_CASE("frames_erfa_chain_golden") {
  // Phase 2 exit criterion 1 (frames part): the composed GCRF->ITRF chain
  // matches the ERFA-generated goldens to <= 1e-11 per matrix element at
  // every golden epoch; the chain components (nutation, CIP, CIO locator,
  // ERA) match to <= 1e-12 rad (manifest tolerances).
  const auto cases =
      star_tests::load_golden_cases(golden_path("earth_chain.toml"));
  REQUIRE(cases.size() == 14);
  double worst_component = 0.0;
  double worst_matrix = 0.0;
  for (const auto& c : cases) {
    CAPTURE(c.scalar("name"));
    star::time::UtcTime utc{};
    utc.year = static_cast<std::int32_t>(parse_int(c.scalar("year")));
    utc.month = static_cast<std::int32_t>(parse_int(c.scalar("month")));
    utc.day = static_cast<std::int32_t>(parse_int(c.scalar("day")));
    utc.hour = static_cast<std::int32_t>(parse_int(c.scalar("hour")));
    utc.minute = static_cast<std::int32_t>(parse_int(c.scalar("minute")));
    utc.second = d(c, "second");
    const star::time::TaiEpoch tai = star::time::tai_from_utc(utc);
    // The epoch plumbing must be bit-identical to the time-system goldens.
    CHECK(tai.day == parse_int(c.scalar("tai_day")));
    CHECK(tai.sec == d(c, "tai_sec"));

    const double dut1 = d(c, "dut1_s");
    const double t = star::time::tt_julian_centuries(tai);

    double dpsi;
    double deps;
    frames::nutation_00b(t, dpsi, deps);
    const frames::CipCio cip = frames::cip_cio_06b(tai);
    const double era = frames::era_00(tai, dut1);

    const double comps[6] = {
        std::fabs(dpsi - d(c, "dpsi")), std::fabs(deps - d(c, "deps")),
        std::fabs(cip.x_rad - d(c, "x")), std::fabs(cip.y_rad - d(c, "y")),
        std::fabs(cip.s_rad - d(c, "s")), std::fabs(era - d(c, "era"))};
    for (const double e : comps) {
      worst_component = std::max(worst_component, e);
      CHECK(e <= 1e-12);
    }

    const Eigen::Matrix3d cmat = frames::c_gcrf_to_itrf(tai, dut1);
    const double dmat = max_abs_diff(cmat, golden_matrix(c));
    worst_matrix = std::max(worst_matrix, dmat);
    CHECK(dmat <= 1e-11);

    check_proper_rotation(cmat);
    check_proper_rotation(frames::c_gcrf_to_cirs(tai));
  }
  CAPTURE(worst_component);
  CAPTURE(worst_matrix);
  // Evidence lines for the chapter's validation table: the observed
  // maxima are reported through doctest's CAPTURE on failure and recorded
  // in the chapter text from a passing run.
  CHECK(worst_matrix <= 1e-11);
}

TEST_CASE("frames_sofa_cookbook_crosscheck") {
  // Published-values anchor: the SOFA cookbook worked example, asserted at
  // the documented 2000B-vs-2000A model-difference bound (manifest;
  // deliberately NOT the 1e-11 ERFA gate).
  const auto cases =
      star_tests::load_golden_cases(golden_path("cookbook_2006_2000a.toml"));
  REQUIRE(cases.size() == 1);
  const auto& c = cases[0];
  const star::time::TaiEpoch tai = epoch_of(c);
  const double dut1 = d(c, "dut1_s");

  const double tol_matrix = d(c, "tol_matrix");
  const Eigen::Matrix3d ours = frames::c_gcrf_to_itrf(tai, dut1);
  const Eigen::Matrix3d pub = golden_matrix(c, "pub_c");
  CHECK(max_abs_diff(ours, pub) <= tol_matrix);

  const frames::CipCio cip = frames::cip_cio_06b(tai);
  // Published X, Y include the cookbook's dX06/dY06 CIP corrections and
  // the 2000A nutation; 1e-8 rad covers both (manifest).
  CHECK(std::fabs(cip.x_rad - d(c, "pub_x")) <= 1e-8);
  CHECK(std::fabs(cip.y_rad - d(c, "pub_y")) <= 1e-8);
  CHECK(std::fabs(cip.s_rad / kDas2R - d(c, "pub_s_arcsec")) <= 1e-6);
  CHECK(std::fabs(frames::era_00(tai, dut1) -
                  d(c, "pub_era_deg") * kDeg2Rad) <= 5e-14);
}

TEST_CASE("frames_eci_ecef_roundtrip") {
  // Phase 2 exit criterion 6b: ECI -> ECEF -> ECI position round trip at
  // LEO radius errs <= 1e-8 m, multiple epochs and directions. The
  // inverse transformation is the transpose (orthonormality), so this
  // also exercises the property the check_proper_rotation gate asserts.
  const auto cases =
      star_tests::load_golden_cases(golden_path("earth_chain.toml"));
  const double r_leo = 6778137.0;  // LEO radius per the exit criterion [m]
  // A spread of directions including axis-aligned and skew cases.
  const Eigen::Vector3d dirs[6] = {
      Eigen::Vector3d(1.0, 0.0, 0.0),   Eigen::Vector3d(0.0, 1.0, 0.0),
      Eigen::Vector3d(0.0, 0.0, 1.0),   Eigen::Vector3d(1.0, 1.0, 1.0),
      Eigen::Vector3d(-2.0, 0.5, 1.5),  Eigen::Vector3d(0.3, -0.9, -0.6)};
  double worst = 0.0;
  for (const auto& c : cases) {
    CAPTURE(c.scalar("name"));
    const star::time::TaiEpoch tai = epoch_of(c);
    const Eigen::Matrix3d m = frames::c_gcrf_to_itrf(tai, d(c, "dut1_s"));
    for (const auto& dir : dirs) {
      const Eigen::Vector3d r_eci = r_leo * dir.normalized();
      const Eigen::Vector3d r_ecef = m * r_eci;
      const Eigen::Vector3d back = m.transpose() * r_ecef;
      worst = std::max(worst, (back - r_eci).norm());
      // Rotation preserves length; the norm check catches any scale
      // defect the round trip alone would cancel.
      CHECK(std::fabs(r_ecef.norm() - r_eci.norm()) <= 1e-8);
    }
  }
  CAPTURE(worst);
  CHECK(worst <= 1e-8);
}

TEST_CASE("frames_moon_pa_golden") {
  const auto cases =
      star_tests::load_golden_cases(golden_path("moon_pa.toml"));
  REQUIRE(cases.size() == 7);
  for (const auto& c : cases) {
    CAPTURE(c.scalar("name"));
    const double phi = d(c, "phi");
    const double theta = d(c, "theta");
    const double psi = d(c, "psi");
    const Eigen::Matrix3d m = frames::c_gcrf_to_moonpa(phi, theta, psi);
    // Manifest tolerance: 1e-15 per element vs the ERFA-composed golden.
    CHECK(max_abs_diff(m, golden_matrix(c)) <= 1e-15);
    check_proper_rotation(m);
    // The Moon PA construction is by definition the 3-1-3 sequence with
    // angles applied phi, theta, psi - one code path with the rotation
    // kernel, so equality is exact. (Evaluated to a bool before CHECK:
    // doctest's expression decomposition of an Eigen operator== ICEs
    // MSVC.)
    const bool same_code_path =
        (m.array() == star::rotation::dcm_from_euler313(phi, theta, psi).array())
            .all();
    CHECK(same_code_path);
  }
}

TEST_CASE("frames_mars_iau_golden") {
  const auto cases =
      star_tests::load_golden_cases(golden_path("mars_iau.toml"));
  REQUIRE(cases.size() == 6);
  for (const auto& c : cases) {
    CAPTURE(c.scalar("name"));
    const star::time::TaiEpoch tai = epoch_of(c);
    // The TDB argument must reproduce the committed two-part TDB JD
    // bit for bit (same fold as star::time::tdb_jd).
    const star::time::TwoPartJd tdb = star::time::tdb_jd(tai);
    CHECK(tdb.jd1 == d(c, "tdb_jd1"));
    CHECK(tdb.jd2 == d(c, "tdb_jd2"));

    const frames::MarsElements e = frames::mars_elements_iau2015(tai);
    // Manifest tolerances: elements 1e-13 rad (same polynomials, same
    // fixed evaluation order; residual is libm only), matrix 1e-14.
    CHECK(std::fabs(e.alpha0_rad - d(c, "alpha0")) <= 1e-13);
    CHECK(std::fabs(e.delta0_rad - d(c, "delta0")) <= 1e-13);
    CHECK(std::fabs(e.w_rad - d(c, "w")) <= 1e-13);

    const Eigen::Matrix3d m = frames::c_gcrf_to_marsfixed(tai);
    CHECK(max_abs_diff(m, golden_matrix(c)) <= 1e-14);
    check_proper_rotation(m);

    // The third row of C is the pole direction in GCRF; it must match the
    // published (cos d cos a, cos d sin a, sin d) construction.
    const Eigen::Vector3d pole(
        std::cos(e.delta0_rad) * std::cos(e.alpha0_rad),
        std::cos(e.delta0_rad) * std::sin(e.alpha0_rad),
        std::sin(e.delta0_rad));
    CHECK((m.row(2).transpose() - pole).cwiseAbs().maxCoeff() <= 1e-15);
  }
}
