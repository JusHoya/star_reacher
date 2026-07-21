"""Phase 6 exit criterion 2, golden half: the Python PD law versus mpmath.

Exit criterion 2 asks that "a Python PD attitude controller reproduces the
built-in C++ controller's commanded torques to < 1e-9 N*m on a golden
scenario". That is a conjunction, and it is only satisfied when ONE Python
controller meets both halves of it:

* this module evaluates ``tests/refs/pd_attitude.pd_torque`` against the
  committed golden vectors of ``tests/golden/gnc/pd_attitude.toml``, whose
  ``expected_tau_nm`` is a 60-digit mpmath evaluation of the normative law on
  bit-identical inputs -- a reference produced by neither implementation, and
  the same file ``gnc_pd_attitude_golden`` drives the C++ component with;
* ``test_gnc_missions.test_pd_law_python_reimplementation_contract`` evaluates
  the same function against the torques the compiled C++ component logs on a
  closed-loop mission.

Together they close the criterion: the Python controller matches the
specification's extended-precision answer, and the C++ controller matches the
Python controller in the loop. Before this module existed the goldens were
consumed only from C++, so no Python implementation was ever evaluated against
them and the conjunction was never satisfied by any single test.

The goldens carry the branch coverage the mission-level fixture cannot supply
at a single instant: a case with ``dq0 < 0``, a case with ``dq0`` exactly
zero (pinning ``sign(0) = +1``), and a case that clamps some axes while
leaving others unclamped. ``test_golden_cases_cover_every_branch`` asserts
that coverage rather than trusting the file, so a regenerated golden set that
quietly lost a branch fails here instead of silently weakening the gate;
``test_golden_expectations_record_the_branches`` asks the same of the
recorded expectations rather than of the inputs.

Pure Python and NumPy: no compiled core is required, so this half of the
criterion is gated even on a core-less checkout.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN = REPO_ROOT / "tests" / "golden" / "gnc" / "pd_attitude.toml"

sys.path.insert(0, str(REPO_ROOT / "tests" / "refs"))
import pd_attitude as pd  # noqa: E402

# The exit criterion's own figure. The gate below is the sharper manifest
# tolerance; this constant is asserted against too, so the criterion's literal
# wording is checked rather than merely implied.
CRITERION_TOL_NM = 1e-9

# tests/golden/gnc/manifest.toml, pd_attitude.toml: per component
# |got - expected| <= max(5e-15 * |expected|, 1e-18). The Python path performs
# the same handful of binary64 roundings past the once-rounded 60-digit
# reference as the C++ path does, so it earns the same bound; saturated
# components equal the tau_max constant exactly and zero components are exact,
# which the absolute floor covers.
RELATIVE_TOL = 5e-15
ABSOLUTE_FLOOR = 1e-18


def _hex_vector(values) -> np.ndarray:
    """Parse the golden file's binary64 hex literals exactly."""
    return np.array([float.fromhex(v) for v in values], dtype=np.float64)


def _cases():
    doc = tomllib.loads(GOLDEN.read_text(encoding="utf-8"))
    return doc["case"]


@pytest.fixture(scope="module")
def cases():
    parsed = _cases()
    assert parsed, f"{GOLDEN} carries no [[case]] entries"
    return parsed


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["name"])
def test_python_pd_law_matches_the_mpmath_golden(case):
    """Exit criterion 2, golden half, one case per parametrization."""
    tau = pd.pd_torque(
        _hex_vector(case["q_cmd"]),
        _hex_vector(case["q_est"]),
        _hex_vector(case["w_cmd"]),
        _hex_vector(case["w_est"]),
        _hex_vector(case["kp"]),
        _hex_vector(case["kd"]),
        _hex_vector(case["tau_max"]),
    )
    expected = _hex_vector(case["expected_tau_nm"])
    residual = np.abs(tau - expected)
    bound = np.maximum(RELATIVE_TOL * np.abs(expected), ABSOLUTE_FLOOR)
    assert np.all(residual <= bound), (
        f"case {case['name']}: residual {residual} exceeds the golden "
        f"tolerance {bound} (got {tau}, expected {expected})"
    )
    # The criterion's literal figure, checked as well as implied.
    assert float(residual.max()) < CRITERION_TOL_NM


def test_golden_error_quaternion_matches_the_recorded_branch(cases):
    """The file's recorded ``dq0`` is the one the law actually computes.

    ``dq0`` is what decides the ``eq:gnc:sign`` branch, and the golden file
    records it so a consumer can assert which branch each case takes. If the
    reference's Hamilton product disagreed with the recorded value, the branch
    coverage asserted below would be describing a different computation than
    the one under test.
    """
    for case in cases:
        dq = pd.error_quaternion(
            _hex_vector(case["q_cmd"]), _hex_vector(case["q_est"])
        )
        recorded = float.fromhex(case["dq0"])
        assert dq[0] == pytest.approx(recorded, rel=5e-15, abs=1e-18), case["name"]


def test_golden_cases_cover_every_branch(cases):
    """The golden set exercises both sign branches, sign(0), and the clamp.

    This is the coverage the closed-loop mission fixture cannot deliver at a
    single instant, and it is what makes the criterion-2 gate sensitive to
    ``eq:gnc:sign`` and ``eq:gnc:sat`` rather than merely to the gains.
    """
    saw_negative = saw_zero = saw_positive = False
    saw_mixed_saturation = False
    saw_rotating_rate = False
    for case in cases:
        dq0 = float.fromhex(case["dq0"])
        saw_negative |= dq0 < 0.0
        saw_zero |= dq0 == 0.0
        saw_positive |= dq0 > 0.0

        args = (
            _hex_vector(case["q_cmd"]),
            _hex_vector(case["q_est"]),
            _hex_vector(case["w_cmd"]),
            _hex_vector(case["w_est"]),
            _hex_vector(case["kp"]),
            _hex_vector(case["kd"]),
        )
        tau_max = _hex_vector(case["tau_max"])
        unclamped = pd.pd_torque(*args, None)
        clamped = np.abs(unclamped) > tau_max
        if clamped.any() and not clamped.all():
            saw_mixed_saturation = True

        # eq:gnc:werr is only exercised where the commanded rate is non-zero
        # AND the error DCM is not the identity; either alone leaves the term
        # equal to a plain w_cmd subtraction.
        dq = pd.error_quaternion(args[0], args[1])
        off_identity = np.abs(pd.error_dcm(dq) - np.eye(3)).max()
        if np.abs(args[2]).max() > 0.0 and off_identity > 1e-3:
            saw_rotating_rate = True

    assert saw_negative, "no golden case takes the dq0 < 0 short-path branch"
    assert saw_zero, "no golden case pins sign(0) = +1"
    assert saw_positive, "no golden case takes the dq0 >= 0 branch"
    assert saw_mixed_saturation, (
        "no golden case clamps some axes while leaving others unclamped; "
        "eq:gnc:sat would be indistinguishable from an unsaturated law"
    )
    assert saw_rotating_rate, (
        "no golden case combines a non-zero commanded rate with a non-identity "
        "error DCM; eq:gnc:werr's rotation would be untested"
    )


# A recorded quaternion is unit to this tolerance. Measured on the committed
# file every norm is exactly 1.0 in binary64, so the tolerance is slack for a
# legitimate regeneration rather than a bit-exact change detector.
UNIT_NORM_TOL = 1e-12


def _clamped_axes(case) -> list[int]:
    """Axes whose recorded expectation sits exactly on the configured limit.

    Exact binary64 equality is used as the clamp's fingerprint rather than a
    near-miss comparison because ``eq:gnc:sat`` ASSIGNS the limit: a saturated
    component reproduces the ``tau_max`` config constant in all 53 bits.
    ``tests/golden/gnc/manifest.toml`` records the same fact from the other
    direction, citing it as the reason the golden tolerance's absolute floor
    suffices for saturated components. An unsaturated component is a
    once-rounded 60-digit mpmath evaluation, so its landing bit-for-bit on a
    round limit constant is a 2**-52 coincidence rather than a plausible
    outcome of a regeneration.
    """
    tau_max = _hex_vector(case["tau_max"])
    expected = _hex_vector(case["expected_tau_nm"])
    return [
        i for i in range(3) if tau_max[i] > 0.0 and abs(expected[i]) == tau_max[i]
    ]


def test_golden_expectations_record_the_branches(cases):
    """The committed expectations bear the branches the evidence table claims.

    WHY a structural assertion about a fixture exists at all: one branch of the
    PD law is reachable through this file and through nothing else in the
    suite, so a routine, well-intentioned regeneration could delete its
    coverage without turning any other test red. That branch is
    ``sign(0) = +1``. A closed-loop mission never lands ``dq0`` exactly on
    zero, so no scenario gate can reach ``eq:gnc:sign`` at its boundary;
    measured by mutating ``tests/refs/pd_attitude.pd_torque`` to
    ``sign(0) = -1``, the whole Python suite reports exactly one failure, the
    ``sign_zero_is_plus_one`` parametrization of the test above. The golden set
    is likewise the open-loop half of the evidence for ``eq:gnc:sat``, the
    closed-loop half being
    ``test_gnc_missions.test_pd_law_python_reimplementation_contract``.

    This test and ``test_golden_cases_cover_every_branch`` above answer
    neighbouring but different questions, which is why they are separate. That
    one re-evaluates the reference law on the golden INPUTS and asks whether
    those inputs would saturate. This one reads the committed
    ``expected_tau_nm`` -- the array the C++ ``gnc_pd_attitude_golden`` doctest
    and the parametrized test above actually compare against -- and
    asks whether the recorded answer carries the clamp's fingerprint. A
    regeneration whose expectations lost the clamp while its inputs kept it
    would satisfy the first check and fail this one.

    Every assertion below is a property of the file's own contents; the
    control law is not re-derived here.
    """
    # dq0 is the file's own descriptor of the error rotation, and the sign
    # assertions below read it alone. That is only sufficient while the
    # recorded quaternions are unit: |dq| = |q_cmd||q_est|, so unit inputs give
    # |dq_vec| = sqrt(1 - dq0**2) and a case with dq0 == 0 necessarily carries
    # a full-magnitude vector for the sign term to multiply. Without this
    # premise a dq0 == 0 case could be inert and the guard would not know.
    for case in cases:
        for key in ("q_cmd", "q_est"):
            norm = float(np.linalg.norm(_hex_vector(case[key])))
            assert abs(norm - 1.0) <= UNIT_NORM_TOL, (
                f"case {case['name']}: {key} has norm {norm!r}, so dq0 no "
                f"longer determines the error rotation's magnitude"
            )

    # eq:gnc:sat. A case must record a clamped axis AND an unclamped one. The
    # mixed requirement is what rules out the degenerate ways this could pass
    # without evidencing a per-axis clamp: a case whose every axis rails says
    # nothing about the unsaturated path, and a case with tau_max == 0 would
    # rail trivially on an all-zero expectation (excluded in _clamped_axes).
    mixed = []
    for case in cases:
        clamped = _clamped_axes(case)
        tau_max = _hex_vector(case["tau_max"])
        expected = _hex_vector(case["expected_tau_nm"])
        inside = [i for i in range(3) if abs(expected[i]) < tau_max[i]]
        if clamped and inside:
            mixed.append((case["name"], clamped))
    assert mixed, (
        "no committed expectation sits exactly on its tau_max while another "
        "axis of the same case sits strictly inside it; eq:gnc:sat leaves no "
        "trace in the goldens and a law without the clamp would reproduce "
        "every recorded torque"
    )

    # Scope, measured rather than assumed: across the committed set exactly one
    # axis of one case rails, on the positive side. Mutations that clamp only
    # axis 0, or only the positive rail, are therefore invisible here; both are
    # caught instead by test_gnc_missions.test_pd_law_python_reimplementation_
    # contract, whose scenario clamps on a substantial run of cycles. The
    # assertion above is deliberately not narrowed to the observed axis and
    # rail, which would pin the current file rather than state a property.

    # eq:gnc:sign. Each branch must be represented by a case that records a
    # non-zero response, so a branch cannot be present but inert -- a case
    # whose expectation is all zeros distinguishes no sign convention.
    responses = {}
    for case in cases:
        dq0 = float.fromhex(case["dq0"])
        branch = "negative" if dq0 < 0.0 else ("zero" if dq0 == 0.0 else "positive")
        peak = float(np.abs(_hex_vector(case["expected_tau_nm"])).max())
        responses[branch] = max(responses.get(branch, 0.0), peak)
    for branch in ("negative", "zero", "positive"):
        assert responses.get(branch, 0.0) > 0.0, (
            f"no golden case on the dq0 {branch} branch records a non-zero "
            f"torque; eq:gnc:sign's choice on that branch is unobservable in "
            f"the committed expectations"
        )
