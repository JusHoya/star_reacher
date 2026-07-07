"""Loader-derived osculating elements (FR-16/FR-17): analytic recovery tests.

Each test builds inertial states from known classical elements with an
independent COE -> RV implementation (Vallado, Fundamentals of Astrodynamics
and Applications, 4th ed., Algorithm 10: perifocal state rotated by
Rz(raan) Rx(i) Rz(argp)), then checks that ``osculating_elements`` recovers
the elements it was built from — the recovery path under test shares no code
with the construction path. No compiled core is required.

Singular-geometry conventions asserted here (circular argp = 0, equatorial
raan = 0, argument of latitude / true longitude in the nu slot, in-plane
angles measured in the direction of motion) are the documented contract of
``star_reacher.derived`` and ``docs/formats/derived_elements.md``.
"""

import numpy as np
import pytest

from star_reacher import _fixtures, derived, load
from star_reacher.derived import osculating_elements

# IERS Conventions (2010), TN No. 36, Table 1.1 — same provenance as
# cpp/include/star/constants.hpp GM_EARTH_M3_PER_S2; the derived-module copy
# is cross-checked against the core in test_gm_crosscheck.py.
MU_EARTH = 3.986004418e14

TWO_PI = 2.0 * np.pi

# Recovery through cross products and atan2 loses a few digits to roundoff;
# constructions here are exact to ~1e-15 relative, so 5e-10 rad / 1e-9
# relative gates catch any formulation error while never flaking.
ANGLE_ATOL = 5e-10
REL = 1e-9


def _rot_z(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_x(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def coe_to_rv(mu, a, e, i, raan, argp, nu):
    """Independent COE -> RV (Vallado Alg. 10); valid for e < 1 and e > 1."""
    p = a * (1.0 - e * e)
    assert p > 0.0, "semi-latus rectum must be positive for a valid conic"
    cn, sn = np.cos(nu), np.sin(nu)
    r_pf = (p / (1.0 + e * cn)) * np.array([cn, sn, 0.0])
    v_pf = np.sqrt(mu / p) * np.array([-sn, e + cn, 0.0])
    rot = _rot_z(raan) @ _rot_x(i) @ _rot_z(argp)
    return rot @ r_pf, rot @ v_pf


def _states(mu, a, e, i, raan, argp, nus):
    r = np.empty((len(nus), 3))
    v = np.empty((len(nus), 3))
    for k, nu in enumerate(nus):
        r[k], v[k] = coe_to_rv(mu, a, e, i, raan, argp, nu)
    return r, v


def _assert_angles_close(got, want, atol=ANGLE_ATOL):
    err = np.abs((np.asarray(got) - np.asarray(want) + np.pi) % TWO_PI - np.pi)
    assert np.all(err < atol), f"max angle error {err.max():.3e} rad"


def test_elliptical_orbit_recovered_along_propagation():
    # A synthetic truth trajectory: one Keplerian orbit sampled in true
    # anomaly. Every recovered element must be constant and equal to the
    # construction values, and nu must track the sampling.
    a, e, i, raan, argp = 8000e3, 0.15, np.radians(30), np.radians(45), np.radians(60)
    nus = np.radians(np.arange(0.0, 360.0, 10.0))
    r, v = _states(MU_EARTH, a, e, i, raan, argp, nus)
    el = osculating_elements(r, v, MU_EARTH)
    np.testing.assert_allclose(el["a_m"], a, rtol=REL)
    np.testing.assert_allclose(el["e"], e, rtol=REL)
    _assert_angles_close(el["i_rad"], i)
    _assert_angles_close(el["raan_rad"], raan)
    _assert_angles_close(el["argp_rad"], argp)
    _assert_angles_close(el["nu_rad"], nus)
    # Invariants: vis-viva energy and |h| = sqrt(mu p).
    np.testing.assert_allclose(el["energy_m2ps2"], -MU_EARTH / (2.0 * a), rtol=REL)
    np.testing.assert_allclose(
        el["hmag_m2ps"], np.sqrt(MU_EARTH * a * (1.0 - e * e)), rtol=REL
    )


def test_hyperbolic_states_recovered():
    # SOI-exit-like geometry: a < 0, e > 1, positive energy. Incoming
    # (negative) anomalies must land in (pi, 2*pi) under the [0, 2*pi)
    # convention; all samples stay inside the asymptote limit
    # arccos(-1/e) = 131.8 deg for e = 1.5.
    a, e, i, raan, argp = -15000e3, 1.5, np.radians(25), np.radians(40), np.radians(70)
    nus = np.radians(np.array([-60.0, -20.0, 0.0, 15.0, 60.0, 110.0]))
    r, v = _states(MU_EARTH, a, e, i, raan, argp, nus)
    el = osculating_elements(r, v, MU_EARTH)
    np.testing.assert_allclose(el["a_m"], a, rtol=REL)
    np.testing.assert_allclose(el["e"], e, rtol=REL)
    _assert_angles_close(el["i_rad"], i)
    _assert_angles_close(el["raan_rad"], raan)
    _assert_angles_close(el["argp_rad"], argp)
    _assert_angles_close(el["nu_rad"], np.mod(nus, TWO_PI))
    assert np.all(el["energy_m2ps2"] > 0.0)
    assert np.all(el["a_m"] < 0.0)
    # The two negative-anomaly samples sit on the incoming branch.
    assert np.all(el["nu_rad"][:2] > np.pi)


def test_circular_equatorial_conventions():
    # Circular equatorial: raan and argp are exactly 0 by convention and the
    # nu slot carries the true longitude (angle from +X to the position).
    r0 = 7000e3
    vc = np.sqrt(MU_EARTH / r0)
    lams = np.radians(np.array([0.0, 30.0, 135.0, 250.0]))
    r = np.stack([_rot_z(lam) @ np.array([r0, 0.0, 0.0]) for lam in lams])
    v = np.stack([_rot_z(lam) @ np.array([0.0, vc, 0.0]) for lam in lams])
    el = osculating_elements(r, v, MU_EARTH)
    assert np.all(el["e"] < 1e-11)
    _assert_angles_close(el["i_rad"], 0.0)
    assert np.all(el["raan_rad"] == 0.0)
    assert np.all(el["argp_rad"] == 0.0)
    _assert_angles_close(el["nu_rad"], lams)
    np.testing.assert_allclose(el["a_m"], r0, rtol=REL)


def test_circular_inclined_argument_of_latitude():
    # Circular inclined: argp is exactly 0 by convention and the nu slot
    # carries the argument of latitude, measured from the ascending node.
    a, i, raan = 7200e3, np.radians(50), np.radians(30)
    us = np.radians(np.array([0.0, 45.0, 200.0, 315.0]))
    r, v = _states(MU_EARTH, a, 0.0, i, raan, 0.0, us)
    el = osculating_elements(r, v, MU_EARTH)
    assert np.all(el["e"] < 1e-11)
    assert np.all(el["argp_rad"] == 0.0)
    _assert_angles_close(el["i_rad"], i)
    _assert_angles_close(el["raan_rad"], raan)
    _assert_angles_close(el["nu_rad"], us)


def test_retrograde_equatorial_angles_follow_motion():
    # i = 180 deg: the node is undefined and in-plane angles are measured in
    # the direction of motion (module docstring convention), so a quarter
    # orbit after +X the reported angle is pi/2, not 3*pi/2.
    r0 = 7000e3
    us = np.radians(np.array([0.0, 90.0, 210.0]))
    r, v = _states(MU_EARTH, r0, 0.0, np.pi, 0.0, 0.0, us)
    el = osculating_elements(r, v, MU_EARTH)
    _assert_angles_close(el["i_rad"], np.pi)
    assert np.all(el["raan_rad"] == 0.0)
    assert np.all(el["argp_rad"] == 0.0)
    _assert_angles_close(el["nu_rad"], us)


def test_elliptical_equatorial_longitude_of_periapsis():
    # Elliptical equatorial: raan is exactly 0 and the argp slot carries the
    # longitude of periapsis (angle from +X to periapsis).
    a, e, lonper = 9000e3, 0.3, np.radians(80)
    nus = np.radians(np.array([0.0, 50.0, 160.0, 300.0]))
    r, v = _states(MU_EARTH, a, e, 0.0, 0.0, lonper, nus)
    el = osculating_elements(r, v, MU_EARTH)
    np.testing.assert_allclose(el["e"], e, rtol=REL)
    assert np.all(el["raan_rad"] == 0.0)
    _assert_angles_close(el["argp_rad"], lonper)
    _assert_angles_close(el["nu_rad"], nus)


def test_parabolic_boundary_is_robust():
    # Escape-speed state: the energy is zero to roundoff, so a is huge (or
    # +inf at exactly zero) while e, the angles, and |h| stay finite and
    # well-defined. The gate is robustness, not a specific parabolic a.
    r0 = 7000e3
    v_esc = np.sqrt(2.0 * MU_EARTH / r0)
    r = np.array([[r0, 0.0, 0.0]])
    v = np.array([[0.0, v_esc, 0.0]])
    el = osculating_elements(r, v, MU_EARTH)
    energy = el["energy_m2ps2"][0]
    assert abs(energy) < 1e-6 * MU_EARTH / r0
    if energy == 0.0:
        assert np.isposinf(el["a_m"][0])
    else:
        assert abs(el["a_m"][0]) > 1e12
    np.testing.assert_allclose(el["e"][0], 1.0, atol=1e-8)
    for key in ("i_rad", "raan_rad", "argp_rad", "nu_rad", "hmag_m2ps"):
        assert np.isfinite(el[key][0])


def test_degenerate_states_yield_nan_not_exceptions():
    # Radial (h = 0) and zero-position samples have no defined elements;
    # they must produce nan without raising or poisoning the finite sample.
    r = np.array([[7000e3, 0.0, 0.0], [0.0, 0.0, 0.0], [7000e3, 0.0, 0.0]])
    v = np.array([[1000.0, 0.0, 0.0], [1000.0, 0.0, 0.0], [0.0, 7500.0, 0.0]])
    el = osculating_elements(r, v, MU_EARTH)
    for key in ("a_m", "e", "i_rad", "raan_rad", "argp_rad", "nu_rad"):
        assert np.isnan(el[key][0]) and np.isnan(el[key][1]), key
        assert np.isfinite(el[key][2]), key
    # Energy is still defined for the radial sample (finite r, finite v)
    # and must be computed with the true radius, not a masked placeholder.
    assert el["energy_m2ps2"][0] == 0.5 * 1000.0**2 - MU_EARTH / 7000e3
    assert np.isnan(el["energy_m2ps2"][1])


def test_single_sample_shape():
    r, v = coe_to_rv(MU_EARTH, 8000e3, 0.1, 0.5, 1.0, 2.0, 0.3)
    el = osculating_elements(r, v, MU_EARTH)
    assert all(arr.shape == (1,) for arr in el.values())


def test_input_validation():
    with pytest.raises(ValueError, match=r"shape \(n, 3\)"):
        osculating_elements(np.zeros((3, 2)), np.zeros((3, 2)), MU_EARTH)
    with pytest.raises(ValueError, match="positive"):
        osculating_elements(np.zeros((1, 3)), np.zeros((1, 3)), -1.0)


def test_central_body_gm_lookup_and_errors():
    assert derived.central_body_gm("earth") == MU_EARTH
    with pytest.raises(ValueError, match="earth, mars, moon"):
        derived.central_body_gm("jupiter")
    with pytest.raises(ValueError, match="supported central bodies"):
        derived.central_body_gm(None)


def test_run_elements_and_time_axis(tmp_path):
    # End-to-end loader path: a synthesized SRLOG truth group built from a
    # known Keplerian state must yield the same elements as calling
    # osculating_elements directly with the header body's GM.
    a, e, i, raan, argp = 8000e3, 0.15, np.radians(30), np.radians(45), np.radians(60)
    nus = np.radians(np.array([0.0, 40.0, 170.0, 300.0]))
    r, v = _states(MU_EARTH, a, e, i, raan, argp, nus)
    times = 0.1 * np.arange(len(nus))
    records = [
        _fixtures.truth_record(float(times[k]), tuple(r[k]), tuple(v[k]))
        for k in range(len(nus))
    ]
    path = tmp_path / "run.srlog"
    path.write_bytes(_fixtures.build_srlog(_fixtures.contract_header(), records))
    run = load(path)

    np.testing.assert_array_equal(run.time_s("truth"), times)
    el = run.elements()
    direct = osculating_elements(r, v, MU_EARTH)
    for key, arr in el.items():
        np.testing.assert_array_equal(arr, direct[key], err_msg=key)
    # Lazy derivation is cached per group.
    assert run.elements() is el

    with pytest.raises(KeyError, match="no channel group named 'nav.est'"):
        run.elements("nav.est")
    with pytest.raises(ValueError, match="no 'r_m' channel"):
        run.elements("events")
    with pytest.raises(KeyError, match="available groups"):
        run.time_s("forces")
