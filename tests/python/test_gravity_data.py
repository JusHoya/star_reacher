"""Gravity data pipeline tests (FR-5): parsers, SRGRAV round trip, excerpts.

Covers the ``star data fetch`` repack path without any network access: the
source-format parsers run on synthetic in-test text (the Phase 1 rule that
fixtures are synthesized where possible), and the committed excerpt fixtures
under ``tests/golden/gravity/`` anchor the byte-level contract between the
CSV and SRGRAV forms. Provenance for the committed fixtures:
``tests/golden/gravity/manifest.toml``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from star_reacher import data_fetch as df

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = REPO_ROOT / "tests" / "golden" / "gravity"

EXCERPT_STEMS = [
    "earth_egm2008_n20",
    "moon_grgm1200a_n50",
    "mars_mro120f_n20",
]


def _synthetic_field(n_max: int = 3, m_max: int = 3) -> df.GravityCoefficients:
    """A small fully populated field with values that exercise sign and scale."""
    cbar = np.zeros((n_max + 1, n_max + 1))
    sbar = np.zeros((n_max + 1, n_max + 1))
    cbar[0, 0] = 1.0
    for n in range(2, n_max + 1):
        for m in range(0, min(n, m_max) + 1):
            cbar[n, m] = (-1.0) ** n * (1.0 + n * 0.25 + m * 0.0625) * 1e-6
            if m > 0:
                sbar[n, m] = (n - m + 0.5) * -1e-7
    return df.GravityCoefficients(
        name="SYNTH",
        gm_m3ps2=3.986e14,
        ref_radius_m=6378136.3,
        n_max=n_max,
        m_max=m_max,
        tide_system="zero_tide",
        cbar=cbar,
        sbar=sbar,
        source_sha256="ab" * 32,
    )


class TestSrgravRoundTrip:
    def test_write_read_bit_exact(self, tmp_path):
        field = _synthetic_field()
        path = tmp_path / "synth.srgrav"
        df.write_srgrav(path, field)
        back = df.read_srgrav(path)
        assert back.name == "SYNTH"
        assert back.gm_m3ps2 == field.gm_m3ps2
        assert back.ref_radius_m == field.ref_radius_m
        assert back.n_max == field.n_max
        assert back.m_max == field.m_max
        assert back.tide_system == "zero_tide"
        assert back.source_sha256 == "ab" * 32
        # Bit equality, not tolerance: the container stores binary64 verbatim.
        assert np.array_equal(back.cbar, field.cbar)
        assert np.array_equal(back.sbar, field.sbar)

    def test_write_is_pure_function(self, tmp_path):
        field = _synthetic_field()
        a = tmp_path / "a.srgrav"
        b = tmp_path / "b.srgrav"
        df.write_srgrav(a, field)
        df.write_srgrav(b, field)
        assert a.read_bytes() == b.read_bytes()

    def test_rectangular_truncation_round_trip(self, tmp_path):
        field = _synthetic_field(n_max=5, m_max=5).truncated(5, 2)
        path = tmp_path / "rect.srgrav"
        df.write_srgrav(path, field)
        back = df.read_srgrav(path)
        assert (back.n_max, back.m_max) == (5, 2)
        assert np.array_equal(back.cbar, field.cbar)
        # Entry count for a rectangular band: degrees 0..2 triangular, 3..5
        # full width m_max+1 = 3.
        assert path.stat().st_size == 96 + 16 * (1 + 2 + 3 + 3 + 3 + 3)


class TestSrgravErrorPaths:
    def test_bad_magic_rejected(self, tmp_path):
        path = tmp_path / "bad.srgrav"
        field = _synthetic_field()
        df.write_srgrav(path, field)
        blob = bytearray(path.read_bytes())
        blob[0] = ord("X")
        path.write_bytes(bytes(blob))
        with pytest.raises(df.DataFetchError, match="bad magic"):
            df.read_srgrav(path)

    def test_wrong_major_version_rejected(self, tmp_path):
        path = tmp_path / "major.srgrav"
        df.write_srgrav(path, _synthetic_field())
        blob = bytearray(path.read_bytes())
        blob[8] = 2  # version_major low byte
        path.write_bytes(bytes(blob))
        with pytest.raises(df.DataFetchError, match="major version 2"):
            df.read_srgrav(path)

    def test_truncated_body_rejected(self, tmp_path):
        path = tmp_path / "short.srgrav"
        df.write_srgrav(path, _synthetic_field())
        blob = path.read_bytes()
        path.write_bytes(blob[:-8])
        with pytest.raises(df.DataFetchError, match="does not match header"):
            df.read_srgrav(path)

    def test_unknown_tide_code_rejected(self, tmp_path):
        path = tmp_path / "tide.srgrav"
        df.write_srgrav(path, _synthetic_field())
        blob = bytearray(path.read_bytes())
        blob[20] = 7  # tide_system low byte
        path.write_bytes(bytes(blob))
        with pytest.raises(df.DataFetchError, match="tide-system code 7"):
            df.read_srgrav(path)


class TestIcgemParser:
    GFC = """\
Some free-text preamble line that parsers must skip.

product_type                gravity_field
modelname                   TESTMODEL
earth_gravity_constant      0.3986004415E+15
radius                      0.63781363E+07
max_degree                  3
errors                      calibrated
norm                        fully_normalized
tide_system                 tide_free

key     L    M             C                       S
end_of_head ====================================================
gfc     0    0    1.0d0                    0.0d0
gfc     2    0   -0.484165143790815e-03    0.000000000000000e+00
gfc     2    1   -0.206615509074176e-09    0.138441389137979e-08
gfc     2    2    0.243938357328313e-05   -0.140027370385934e-05
gfc     3    0    0.957161207093473e-06    0.000000000000000e+00
gfc     3    1    0.203046201047864e-05    0.248200415856872e-06
gfc     3    2    0.904787894809528e-06   -0.619005475177618e-06
gfc     3    3    0.721321757121568e-06    0.141434926192941e-05
"""

    def test_parse_header_and_values(self, tmp_path):
        path = tmp_path / "test.gfc"
        path.write_text(self.GFC, encoding="ascii")
        field = df.parse_icgem_gfc(path, 3, "cd" * 32)
        assert field.name == "TESTMODEL"
        assert field.gm_m3ps2 == 0.3986004415e15
        assert field.ref_radius_m == 0.63781363e7
        assert field.tide_system == "tide_free"
        # 'd' exponent handled; degree 1 absent -> zero fill.
        assert field.cbar[0, 0] == 1.0
        assert field.cbar[1, 0] == 0.0 and field.cbar[1, 1] == 0.0
        assert field.cbar[2, 0] == -0.484165143790815e-03
        assert field.sbar[3, 3] == 0.141434926192941e-05

    def test_truncation_skips_high_degrees(self, tmp_path):
        path = tmp_path / "test.gfc"
        path.write_text(self.GFC, encoding="ascii")
        field = df.parse_icgem_gfc(path, 2, "cd" * 32)
        assert field.n_max == 2
        assert field.cbar[2, 2] == 0.243938357328313e-05

    def test_time_variable_rows_rejected(self, tmp_path):
        path = tmp_path / "tv.gfc"
        path.write_text(
            self.GFC + "gfct    4    0    1.0e-9    0.0    2010.0\n",
            encoding="ascii",
        )
        with pytest.raises(df.DataFetchError, match="gfct"):
            df.parse_icgem_gfc(path, 3, "cd" * 32)

    def test_unnormalized_model_rejected(self, tmp_path):
        path = tmp_path / "un.gfc"
        path.write_text(
            self.GFC.replace("fully_normalized", "unnormalized"), encoding="ascii"
        )
        with pytest.raises(df.DataFetchError, match="normalization"):
            df.parse_icgem_gfc(path, 3, "cd" * 32)

    def test_missing_end_of_head_rejected(self, tmp_path):
        path = tmp_path / "nohead.gfc"
        path.write_text("modelname X\nradius 1.0\n", encoding="ascii")
        with pytest.raises(df.DataFetchError, match="end_of_head"):
            df.parse_icgem_gfc(path, 3, "cd" * 32)


class TestShadrParser:
    SHADR = (
        " 0.3396000000000000E+04, 0.4282837566395650E+05, 0.2151084000000000E-03,"
        "    3,    3,    1, 0.0000000000000000E+00, 0.0000000000000000E+00\n"
        "    1,    0, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0, 0.0\n"
        "    1,    1, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0, 0.0\n"
        "    2,    0,-0.8750219819894000E-03, 0.0000000000000000E+00, 0.0, 0.0\n"
        "    2,    1, 0.1120944989120000E-09, 0.2515192747300000E-09, 0.0, 0.0\n"
        "    2,    2,-0.8463591398800000E-04, 0.4893448966680000E-04, 0.0, 0.0\n"
        "    3,    0,-0.1189998220000000E-04, 0.0000000000000000E+00, 0.0, 0.0\n"
        "    3,    1, 0.3803122284560000E-05, 0.2513120158700000E-04, 0.0, 0.0\n"
        "    3,    2,-0.1594791025300000E-04, 0.8365329423600000E-05, 0.0, 0.0\n"
        "    3,    3, 0.3505323977340000E-04, 0.2559419323000000E-04, 0.0, 0.0\n"
    )

    def test_parse_header_and_values(self, tmp_path):
        path = tmp_path / "test_sha.tab"
        path.write_text(self.SHADR, encoding="ascii")
        field = df.parse_pds_shadr(path, 3, "TESTMARS", "tide_free", "ef" * 32)
        # km -> m and km^3/s^2 -> m^3/s^2 by exact powers of ten.
        assert field.ref_radius_m == 0.3396e4 * 1e3
        assert field.gm_m3ps2 == 0.4282837566395650e5 * 1e9
        # Degree 0 absent from SHADR deliveries -> monopole fill C(0,0) = 1.
        assert field.cbar[0, 0] == 1.0
        assert field.cbar[2, 0] == -0.8750219819894000e-03
        assert field.sbar[3, 3] == 0.2559419323000000e-04

    def test_unnormalized_flag_rejected(self, tmp_path):
        path = tmp_path / "un_sha.tab"
        path.write_text(self.SHADR.replace(",    1,", ",    0,", 1), encoding="ascii")
        with pytest.raises(df.DataFetchError, match="normalization flag"):
            df.parse_pds_shadr(path, 3, "X", "unknown", "ef" * 32)

    def test_source_shallower_than_request_rejected(self, tmp_path):
        path = tmp_path / "shallow_sha.tab"
        path.write_text(self.SHADR, encoding="ascii")
        with pytest.raises(df.DataFetchError, match="below the requested"):
            df.parse_pds_shadr(path, 10, "X", "unknown", "ef" * 32)


class TestCommittedExcerpts:
    """The committed CSV and SRGRAV forms are the same data, byte-anchored."""

    @pytest.mark.parametrize("stem", EXCERPT_STEMS)
    def test_csv_rebuilds_committed_srgrav_bytes(self, stem, tmp_path):
        field = df.read_coeffs_csv(GOLDEN_DIR / f"{stem}.csv")
        rebuilt = tmp_path / f"{stem}.srgrav"
        df.write_srgrav(rebuilt, field)
        committed = (GOLDEN_DIR / f"{stem}.srgrav").read_bytes()
        assert rebuilt.read_bytes() == committed

    @pytest.mark.parametrize(
        "stem,name,n_max,gm,radius,tide",
        [
            ("earth_egm2008_n20", "EGM2008", 20, 398600441500000.0, 6378136.3, "tide_free"),
            ("moon_grgm1200a_n50", "GRGM1200A", 50, 4902800122445.3001, 1738000.0, "unknown"),
            ("mars_mro120f_n20", "MRO120F", 20, 42828375663956.50, 3396000.0, "tide_free"),
        ],
    )
    def test_committed_headers_match_sources(self, stem, name, n_max, gm, radius, tide):
        # GM and R must be each source's own header values (never
        # constants.hpp values): EGM2008's GM differs from the IERS constant
        # used by the two-body model, and that difference is intentional.
        field = df.read_srgrav(GOLDEN_DIR / f"{stem}.srgrav")
        assert field.name == name
        assert field.n_max == n_max and field.m_max == n_max
        assert field.gm_m3ps2 == gm
        assert field.ref_radius_m == radius
        assert field.tide_system == tide
        assert field.cbar[0, 0] == 1.0
        assert np.all(field.cbar[1, :] == 0.0)  # center-of-mass fields
        assert field.cbar[2, 0] < 0.0  # oblateness: negative C(2,0) everywhere

    @pytest.mark.parametrize("stem", EXCERPT_STEMS)
    def test_committed_source_sha_matches_manifest_pin(self, stem):
        # The excerpt's in-band source digest must equal the pinned SHA-256
        # of the source it claims to come from (the committed fetch record).
        field = df.read_srgrav(GOLDEN_DIR / f"{stem}.srgrav")
        by_name = {s.field_name: s for s in df.GRAVITY_DATASETS.values()}
        assert field.source_sha256 == by_name[field.name].source_sha256


class TestDatasetRegistry:
    def test_fr5_repack_degrees(self):
        # FR-5: Earth to 70x70, Moon to 120x120, Mars to 80x80.
        assert df.GRAVITY_DATASETS["egm2008"].n_repack == 70
        assert df.GRAVITY_DATASETS["grgm1200a"].n_repack == 120
        assert df.GRAVITY_DATASETS["mro120f"].n_repack == 80

    def test_unknown_dataset_exit_code(self, capsys):
        assert df.cli_fetch("nosuchfield", "data") == 2
        assert "unknown dataset" in capsys.readouterr().err
