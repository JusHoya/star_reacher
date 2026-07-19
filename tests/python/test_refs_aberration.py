"""Validate the independent aberration reference of ``tests/refs/aberration.py``.

Phase 6 exit criterion 9 requires the core's optical truth directions to carry
the FR-23 velocity-aberration correction and to match an independent computation
to better than 1 milliarcsecond at Earth orbital speed. This suite establishes
that the independent computation is itself correct, before it is ever pointed at
the core, by checking it against quantities that do not come from the
implementation:

* the arcsecond magnitudes Chapter ch:sensors-optical states (20.49 arcsec at
  29.78 km/s, 20.64 arcsec at 30 km/s);
* the analytic first-order law ``beta sin(theta)`` of eq:optical:abmag, over the
  full source-velocity angle range including the degenerate endpoints;
* the sign convention -- the apparent direction moves TOWARD the velocity;
* the exact relativistic form, derived independently from the Lorentz
  transformation, and its measured departure from the first-order formula, which
  the chapter bounds at ``(beta**2/4) sin(2 theta)`` and 0.52 mas at
  ``beta = 1e-4``.

These tests are pure NumPy and require no compiled core.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO_ROOT / "tests" / "refs"))
import aberration as ab  # noqa: E402

BETA_HAT = np.array([0.0, 1.0, 0.0])


def _beta_at_speed(speed_mps: float) -> np.ndarray:
    return speed_mps / ab.SPEED_OF_LIGHT_MPS * BETA_HAT


def test_speed_of_light_is_the_si_definition():
    """``c`` is exact by the SI definition of the metre, not a measured value."""
    assert ab.SPEED_OF_LIGHT_MPS == 299792458.0


@pytest.mark.parametrize(
    ("speed_mps", "expected_arcsec"),
    [(29780.0, 20.49), (30000.0, 20.64)],
)
def test_peak_deflection_matches_the_chapter_magnitudes(speed_mps, expected_arcsec):
    """Chapter ch:sensors-optical quotes both magnitudes at ``theta = pi/2``."""
    beta = _beta_at_speed(speed_mps)
    u = ab.direction_at_angle(np.pi / 2.0, BETA_HAT)[0]
    deflection = float(ab.deflection_angle(u, beta)) * ab.ARCSEC_PER_RAD
    assert deflection == pytest.approx(expected_arcsec, abs=0.005)


def test_deflection_follows_the_analytic_first_order_law():
    """``delta theta = beta sin(theta)``, equation eq:optical:abmag."""
    beta = _beta_at_speed(ab.EARTH_MEAN_ORBITAL_SPEED_MPS)
    beta_mag = float(np.linalg.norm(beta))
    theta = np.linspace(0.0, np.pi, 181)
    u = ab.direction_at_angle(theta, BETA_HAT)
    measured = ab.deflection_angle(u, beta)
    predicted = beta_mag * np.sin(theta)
    # The law is first order in beta, so the residual is O(beta**2); compare in
    # absolute radians against a bound comfortably inside that order.
    assert np.max(np.abs(measured - predicted)) < beta_mag**2


@pytest.mark.parametrize("theta", [0.0, np.pi])
def test_no_deflection_along_or_against_the_velocity(theta):
    """A source on the velocity axis has no transverse beta component."""
    beta = _beta_at_speed(ab.EARTH_MEAN_ORBITAL_SPEED_MPS)
    u = ab.direction_at_angle(theta, BETA_HAT)[0]
    assert float(ab.deflection_angle(u, beta)) * ab.MAS_PER_RAD < 1e-6


def test_apparent_direction_is_displaced_toward_the_velocity():
    """The sign convention the chapter flags as the usual defect.

    The apparent direction's component along ``beta_hat`` must INCREASE; the
    telescope tilts forward into the motion.
    """
    beta = _beta_at_speed(ab.EARTH_MEAN_ORBITAL_SPEED_MPS)
    theta = np.linspace(0.05, np.pi - 0.05, 60)
    u = ab.direction_at_angle(theta, BETA_HAT)
    for formula in (ab.aberrate_first_order, ab.aberrate_exact):
        apparent = formula(u, beta)
        along_before = u @ BETA_HAT
        along_after = apparent @ BETA_HAT
        assert np.all(along_after > along_before), formula.__name__


@pytest.mark.parametrize("formula", [ab.aberrate_first_order, ab.aberrate_exact])
def test_apparent_directions_are_unit_vectors(formula):
    """Both formulas return unit vectors; a drifting norm corrupts downstream use."""
    beta = _beta_at_speed(3.0e7)  # beta = 0.1, far outside the domain but valid maths
    theta = np.linspace(0.0, np.pi, 101)
    u = ab.direction_at_angle(theta, BETA_HAT)
    norms = np.linalg.norm(formula(u, beta), axis=-1)
    assert np.max(np.abs(norms - 1.0)) < 1e-15


def test_exact_and_first_order_converge_at_second_order_in_beta():
    """The exact form must reduce to eq:optical:aberration as ``beta -> 0``.

    This is the check that the independently derived relativistic expression is
    the same physical law as the chapter's, rather than a different one that
    merely looks similar: their difference must scale as ``beta**2``, so halving
    beta must quarter the difference.
    """
    theta = np.array([np.pi / 4.0])
    ratios = []
    for beta_mag in (1e-3, 1e-4, 1e-5, 1e-6):
        beta = beta_mag * BETA_HAT
        u = ab.direction_at_angle(theta, BETA_HAT)
        error = float(ab.first_order_error_angle(u, beta)[0])
        ratios.append(error / beta_mag**2)
    # The coefficient is (1/4) |sin 2 theta| = 1/4 at theta = pi/4.
    for ratio in ratios:
        assert ratio == pytest.approx(0.25, rel=2e-3)


def test_first_order_error_matches_the_chapter_functional_form():
    """Chapter assumption 2: the difference is ``(beta**2/4) sin(2 theta)``."""
    beta_mag = 1e-4
    beta = beta_mag * BETA_HAT
    theta = np.linspace(0.0, np.pi, 721)
    u = ab.direction_at_angle(theta, BETA_HAT)
    measured = ab.first_order_error_angle(u, beta)
    predicted = (beta_mag**2 / 4.0) * np.abs(np.sin(2.0 * theta))
    # Residual is the next order, O(beta**3); allow a generous multiple of it.
    assert np.max(np.abs(measured - predicted)) < 10.0 * beta_mag**3


def test_first_order_error_is_below_the_chapter_bound_at_beta_1e_minus_4():
    """Chapter assumption 2 bounds the truncation at 0.52 mas for beta = 1e-4."""
    beta = 1e-4 * BETA_HAT
    theta = np.linspace(0.0, np.pi, 2001)
    u = ab.direction_at_angle(theta, BETA_HAT)
    worst_mas = float(np.max(ab.first_order_error_angle(u, beta))) * ab.MAS_PER_RAD
    assert worst_mas < 0.52


def test_first_order_truncation_consumes_half_the_criterion_9_budget():
    """Record the actual margin of the 1 mas gate against formula choice.

    The gate compares two computations of the SAME first-order formula, so the
    truncation does not enter it -- but a reference that used the exact form
    instead would spend this much of the budget, and the margin is thin enough
    to be worth pinning as a regression.
    """
    beta = _beta_at_speed(ab.EARTH_MEAN_ORBITAL_SPEED_MPS)
    theta = np.linspace(0.0, np.pi, 2001)
    u = ab.direction_at_angle(theta, BETA_HAT)
    worst_mas = float(np.max(ab.first_order_error_angle(u, beta))) * ab.MAS_PER_RAD
    assert 0.45 < worst_mas < 0.55


def test_leo_observer_truncation_stays_under_one_milliarcsecond():
    """The worst in-scope beta: Earth's heliocentric speed plus LEO orbital speed.

    Equation eq:optical:beta composes the vehicle velocity with the central
    body's barycentric velocity, so the largest beta this project will see near
    Earth is roughly 29.78 + 7.8 km/s. The first-order truncation there is the
    tightest case for a 1 mas gate.
    """
    beta = _beta_at_speed(ab.EARTH_MEAN_ORBITAL_SPEED_MPS + 7800.0)
    theta = np.linspace(0.0, np.pi, 2001)
    u = ab.direction_at_angle(theta, BETA_HAT)
    worst_mas = float(np.max(ab.first_order_error_angle(u, beta))) * ab.MAS_PER_RAD
    assert worst_mas < 1.0, f"first-order truncation reaches {worst_mas:.3f} mas at LEO"


def test_beta_composition_uses_the_barycentric_velocity():
    """Equation eq:optical:beta adds the central body's barycentric velocity.

    Using the planet-relative velocity alone would omit the dominant annual
    term; the test pins the composition by checking the resulting beta against
    the direct sum.
    """
    v_sc = np.array([1000.0, -2000.0, 3000.0])
    v_body = np.array([-29000.0, 5000.0, 0.0])
    beta = ab.beta_vector(v_sc, v_body)
    assert beta == pytest.approx((v_sc + v_body) / ab.SPEED_OF_LIGHT_MPS, rel=1e-15)
    # Sanity: the barycentric term dominates, which is the whole point.
    assert np.linalg.norm(beta) > 0.5 * np.linalg.norm(v_body) / ab.SPEED_OF_LIGHT_MPS


def test_zero_direction_is_rejected_rather_than_returning_nan():
    """Abort on a missing critical input instead of emitting a plausible NaN."""
    with pytest.raises(ValueError, match="zero-length direction"):
        ab.aberrate_first_order(np.zeros(3), 1e-4 * BETA_HAT)


def test_superluminal_observer_is_rejected():
    """The exact form is undefined at or beyond ``c`` and must say so."""
    with pytest.raises(ValueError, match="below c"):
        ab.aberrate_exact(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
