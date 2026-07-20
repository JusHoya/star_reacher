"""FR-24 stepping API: drive a mission one control period at a time.

``Sim`` is the Python front door to the core's vehicle cycle. A batch
``star run`` and a stepped ``Sim`` traverse the *same* cycle core --
``run_vehicle()`` is literally ``while cycle.step(): pass`` over it -- so the
two produce byte-identical SRLOG files for one scenario. That equality is
Phase 6 exit criterion 4, and it is a property of the factoring rather than
of a comparison test: there is only one implementation to disagree with.

Typical use::

    from star_reacher.sim import Sim

    sim = Sim("missions/leo_gnc.toml", "out/stepped")
    obs, info = sim.reset()
    while not sim.done():
        obs = sim.step()
    print(sim.summary())

``observe()`` idempotence
------------------------

``observe()`` returns the stored snapshot of the most recently processed
cycle. Reading it runs no GNC component, draws no random number, consumes no
sensor sample, and returns freshly built dicts and lists rather than views
into core buffers. Two calls without an intervening ``step()`` therefore
return equal dictionaries -- the second clause of exit criterion 4.

``truth()`` is the privileged counterpart, deliberately a separate call: an
observation handed to an agent can never contain truth by accident.

Determinism contract for Python GNC components
----------------------------------------------

FR-25 lets a GNC component be written in Python, and such a component runs
**inside** the deterministic time loop. The core's D-10 guarantee -- same
inputs on the same binary give bit-identical outputs -- then holds only as
far as the Python does. The core cannot enforce this, so it is stated as a
contract rather than advertised as a guarantee:

* **No clock.** Do not call ``time``, ``datetime``, or anything that reads
  wall time. The core never reads the clock and the log carries no host or
  wall-clock data; a component that does breaks reproducibility and leaks
  host state into results.
* **No I/O and no network.** Do not read files, query services, or log to
  anywhere the run does not control.
* **No unseeded randomness.** ``random`` and ``numpy.random`` default to
  entropy from the OS. If a component needs random numbers, seed a private
  generator from data the run already fixes (the mission seed reaches the
  component through its configuration), and never draw from a global
  generator another component could also advance.
* **No iteration over unordered containers.** Iterating a ``set`` or a
  ``frozenset`` has an order that depends on hash values, which for strings
  vary per process unless ``PYTHONHASHSEED`` is fixed. Sort first, or use a
  list.
* **No mutable global state** shared between components or across runs, and
  no dependence on garbage-collection timing or object identity (``id()``).
* **Arithmetic is fine.** Python floats are IEEE-754 doubles, and NumPy
  operations on ``float64`` are deterministic for a fixed library version.
  Reductions over large arrays may differ between NumPy builds if the build
  changes the pairwise summation order, so a component whose output must be
  bit-stable across environments should avoid depending on the exact
  summation order of very large reductions.

What *is* guaranteed: the core calls a component exactly once per stage per
control cycle, in the fixed order nav -> guidance -> control, with inputs
that depend only on the configuration and the seed. A component that
respects the rules above is as reproducible as the C++ built-ins; one that
does not will produce runs that differ without the core being able to tell.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from star_reacher._corelink import import_core
from star_reacher.mission import (
    MissionValidationError,
    canonical_bytes,
    validate_mission_file,
)
from star_reacher.runner import RunnerError, build_run_config, mission_is_vehicle

__all__ = ["Sim", "SimError"]


class SimError(RuntimeError):
    """A stepping-API misuse that is not a mission validation error."""


class Sim:
    """Step one mission through the core vehicle cycle.

    ``mission_path`` is a mission TOML file; ``outdir`` receives ``run.srlog``
    exactly as ``star run`` would write it. The mission is validated and
    resolved on construction, and the run is opened by :meth:`reset`.
    """

    def __init__(self, mission_path, outdir, *, force=False, strict=False):
        self._mission_path = Path(mission_path)
        self._outdir = Path(outdir)
        self._force = bool(force)
        self._strict = bool(strict)
        self._core = import_core()
        self._sim = None

        resolved, errors = validate_mission_file(
            self._mission_path, strict=self._strict
        )
        if errors:
            raise MissionValidationError(errors)
        if not mission_is_vehicle(resolved):
            # The stepping API is defined over the GNC control period (D-5),
            # which only the 6DOF vehicle path has. Refusing here names the
            # reason rather than failing later inside the core.
            raise SimError(
                f"{self._mission_path}: the stepping API requires a vehicle "
                f"mission (a [vehicle] file, a [[sequence]], or a geodetic "
                f"launch state); this mission propagates a point mass, so it "
                f"has no control cycle to step"
            )
        self._resolved = resolved

    # -- lifecycle ---------------------------------------------------------

    def reset(self, seed=None, overrides=None):
        """Open a fresh run and return ``(obs, info)`` (FR-24).

        ``seed`` replaces the mission's master seed; ``overrides`` is a dict
        of resolved-mission scalar overrides applied before the configuration
        is hashed, so an overridden run carries its own ``config_sha256`` and
        is individually reproducible. Currently accepted override keys are
        ``duration_s`` and ``latency_cycles``; anything else raises, because
        an override the core silently ignores would make a sweep report
        results it never ran.

        Calling ``reset`` again starts a new run over the same output path.
        """
        resolved = _deep_copy_resolved(self._resolved)
        if seed is not None:
            resolved["run"]["seed"] = int(seed)
        for key, value in dict(overrides or {}).items():
            if key == "duration_s":
                resolved["mission"]["duration_s"] = float(value)
            elif key == "latency_cycles":
                if "gnc" not in resolved:
                    raise SimError(
                        "override 'latency_cycles' requires the mission to "
                        "configure a [gnc] block"
                    )
                resolved["gnc"]["latency_cycles"] = int(value)
            else:
                raise SimError(
                    f"unknown reset override {key!r}; accepted keys are "
                    f"'duration_s' and 'latency_cycles'"
                )

        config_bytes = canonical_bytes(resolved)
        config_sha = hashlib.sha256(config_bytes).hexdigest()
        cfg, resolved_vehicle_toml, _ = build_run_config(
            self._core, resolved, config_sha, strict=self._strict
        )

        srlog_path = self._outdir / "run.srlog"
        if srlog_path.exists() and not self._force:
            raise RunnerError(
                f"{srlog_path}: output already exists; construct the Sim with "
                f"force=True to overwrite, or choose another directory"
            )
        self._outdir.mkdir(parents=True, exist_ok=True)
        # The same sidecars a batch run writes, so a stepped run's output
        # directory is interchangeable with a batch one for every downstream
        # tool (plot, view, export, consistency).
        (self._outdir / "resolved_config.json").write_bytes(config_bytes)
        if resolved_vehicle_toml is not None:
            (self._outdir / "resolved_vehicle.toml").write_text(
                resolved_vehicle_toml, encoding="utf-8"
            )

        # Dropping the previous Sim closes its log before the new one opens
        # the same path.
        self._sim = None
        self._sim = self._core.Sim(cfg, str(srlog_path))
        info = {
            "config_sha256": config_sha,
            "srlog_path": str(srlog_path),
            "seed": resolved["run"]["seed"],
            "duration_s": resolved["mission"]["duration_s"],
            "has_external_command": self._sim.has_external_command(),
        }
        return self.observe(), info

    # -- stepping ----------------------------------------------------------

    def step(self, commands=None):
        """Advance exactly one control period and return the observation.

        ``commands`` is an optional dict with keys ``torque_b_nm``,
        ``omega_b_radps``, ``q_i2b`` (scalar-first ``(w, x, y, z)``), and
        ``valid``. Supplied keys replace the held command; missing keys hold
        it (D-5 zero-order hold) and are logged as the held value. An unknown
        key raises. Commanding requires the mission to configure an
        ``external`` guidance or control component.
        """
        return self._active().step(commands)

    def observe(self):
        """The non-privileged observation of the most recent cycle.

        Pure: two calls without an intervening :meth:`step` return equal
        dictionaries (Phase 6 exit criterion 4).
        """
        return self._active().observe()

    def truth(self):
        """PRIVILEGED true state at the instant :meth:`observe` describes.

        Never reaches a GNC component through this path; a component sees
        truth only via ``GncInput.oracle``, and only under ``oracle = true``.
        """
        return self._active().truth()

    def time(self):
        """Current cycle time [s] -- the next cycle to be processed."""
        return self._active().time()

    def cycle(self):
        """Current cycle index (0-based), the next cycle to be processed."""
        return self._active().cycle()

    def done(self):
        """True once the run has ended and the log is complete and closed."""
        return self._active().done()

    def summary(self):
        """Run summary dict; valid once :meth:`done` is true."""
        return self._active().summary()

    def run_to_completion(self):
        """Step until the run ends and return the summary.

        The stepped equivalent of a batch ``star run`` of the same mission,
        and the driver the criterion-4 hash comparison uses.
        """
        sim = self._active()
        while not sim.done():
            sim.step()
        return sim.summary()

    def _active(self):
        if self._sim is None:
            raise SimError("call reset() before stepping the simulation")
        return self._sim


def _deep_copy_resolved(resolved):
    """Copy a resolved mission deeply enough for override application.

    ``copy.deepcopy`` would work, but the resolved config is plain JSON-able
    data by construction (it is hashed through ``canonical_bytes``), so a
    round trip through the same canonical representation is both sufficient
    and a check that the assumption still holds.
    """
    import json

    return json.loads(canonical_bytes(resolved).decode("utf-8"))
