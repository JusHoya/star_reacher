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
quietly lost a branch fails here instead of silently weakening the gate.

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
