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
from star_reacher.plugin import check_plugin_selections, load_plugins
from star_reacher.runner import RunnerError, build_run_config, mission_is_vehicle

__all__ = [
    "GncComponentCfg",
    "GncOutput",
    "IGncComponent",
    "InnovationSample",
    "Sim",
    "SimError",
]

# The core names a GNC plugin author needs, re-exported so a plugin file
# imports from a public module rather than reaching into the private
# extension. Resolved lazily (PEP 562) because importing this module must stay
# possible without a compiled core -- the mission validator and the docs build
# both import it on core-less checkouts -- while a plugin that actually uses
# one of these names gets the actionable "build and install it" error from
# import_core() rather than an AttributeError.
_CORE_REEXPORTS = (
    "GncComponentCfg",
    "GncOutput",
    "IGncComponent",
    "InnovationSample",
)


def __getattr__(name):
    if name in _CORE_REEXPORTS:
        return getattr(import_core(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class SimError(RuntimeError):
    """A stepping-API misuse that is not a mission validation error."""


class Sim:
    """Step one mission through the core vehicle cycle.

    ``mission_path`` is a mission TOML file; ``outdir`` receives ``run.srlog``
    exactly as ``star run`` would write it. The mission is validated and
    resolved on construction, and the run is opened by :meth:`reset`.
    """

    def __init__(
        self, mission_path, outdir, *, force=False, strict=False, gnc_plugins=None
    ):
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
        # FR-25 parity with `star run --gnc-plugin`: a mission naming a
        # "python:" component must be steppable too, and must fail the same
        # way when the plugin declaring it was not supplied.
        self._plugins = load_plugins(gnc_plugins, self._core)
        check_plugin_selections(resolved, self._plugins)
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

        ``seed`` replaces the mission's master seed. ``overrides`` is a dict of
        numeric overrides applied to the resolved mission before the
        configuration is hashed, so an overridden run carries its own
        ``config_sha256`` and is individually reproducible -- the property
        FR-27 relies on when it records a per-run ``overrides`` entry in a
        Monte Carlo manifest.

        A key is a dotted path into the resolved mission, with an integer
        segment indexing a list::

            sim.reset(overrides={
                "mission.duration_s": 120.0,
                "gnc.control.kp_nm_per_rad": [0.5, 0.5, 0.5],
                "sequence.0.t_s": 3.0,
            })

        ``duration_s`` and ``latency_cycles`` remain accepted as shorthands
        for the two paths a stepping driver reaches for most.

        What is refused, and why. The path must already exist: inventing a key
        would produce a configuration the validator never saw. The existing
        value must be a number or an array of numbers, and the new value must
        match it in kind and length -- an integer leaf takes an integer, so a
        control rate cannot silently become fractional. Strings and whole
        tables are not overridable, because they select *structure* (a
        component name, a frame, a file path) whose consequences the mission
        validator checked and this path cannot recheck.

        Numeric RANGE is not rechecked here; the core's defensive
        construction checks are the backstop and raise with a named reason.
        An override is therefore capable of failing the run loudly, never of
        producing a run whose configuration was never validated at all.

        Calling ``reset`` again starts a new run over the same output path.
        """
        resolved = _deep_copy_resolved(self._resolved)
        if seed is not None:
            resolved["run"]["seed"] = int(seed)
        for key, value in dict(overrides or {}).items():
            _apply_override(resolved, _OVERRIDE_ALIASES.get(key, key), value)

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


# The two paths a stepping driver overrides most, kept as bare names because
# they were the whole accepted vocabulary before dotted paths existed and are
# still the two a hand-written driver reaches for.
_OVERRIDE_ALIASES = {
    "duration_s": "mission.duration_s",
    "latency_cycles": "gnc.latency_cycles",
}


def _is_number(value) -> bool:
    # TOML/JSON booleans are ints in Python; they are never a numeric leaf.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _override_target(resolved, path):
    """Walk a dotted override path to its ``(container, key)`` slot.

    Raises :class:`SimError` naming the failing segment rather than the whole
    path, because on a long path the segment is the actionable part.
    """
    segments = path.split(".")
    node = resolved
    for depth, segment in enumerate(segments[:-1]):
        walked = ".".join(segments[: depth + 1])
        if isinstance(node, list):
            if not segment.isdigit() or int(segment) >= len(node):
                raise SimError(
                    f"override path {path!r}: {walked!r} does not index the "
                    f"list at {'.'.join(segments[:depth]) or 'the mission'} "
                    f"(length {len(node)})"
                )
            node = node[int(segment)]
        elif isinstance(node, dict):
            if segment not in node:
                raise SimError(
                    f"override path {path!r}: the resolved mission has no "
                    f"{walked!r}"
                )
            node = node[segment]
        else:
            raise SimError(
                f"override path {path!r}: {walked!r} is not a table or an "
                f"array, so it has no members to override"
            )
    leaf = segments[-1]
    if isinstance(node, list):
        if not leaf.isdigit() or int(leaf) >= len(node):
            raise SimError(
                f"override path {path!r}: {leaf!r} does not index a list of "
                f"length {len(node)}"
            )
        return node, int(leaf)
    if not isinstance(node, dict) or leaf not in node:
        raise SimError(
            f"override path {path!r}: the resolved mission has no such key; "
            f"an override may only change a value the mission already sets, "
            f"because a key the validator never saw would not have been "
            f"checked"
        )
    return node, leaf


def _apply_override(resolved, path, value):
    """Apply one numeric override in place, preserving the leaf's kind."""
    container, key = _override_target(resolved, path)
    current = container[key]

    if _is_number(current):
        if not _is_number(value):
            raise SimError(
                f"override {path!r}: expected a number to replace "
                f"{current!r}, got {type(value).__name__}"
            )
        # An integer leaf keeps its type: control_rate_hz, latency_cycles and
        # seed are counts, and letting one become 10.0 would change the
        # canonical config bytes - and so the config hash - without changing
        # the run.
        container[key] = int(value) if isinstance(current, int) else float(value)
        return

    if isinstance(current, list) and current and all(_is_number(v) for v in current):
        if not isinstance(value, (list, tuple)) or len(value) != len(current):
            raise SimError(
                f"override {path!r}: expected an array of {len(current)} "
                f"number(s) to replace {current!r}, got {value!r}"
            )
        if not all(_is_number(v) for v in value):
            raise SimError(
                f"override {path!r}: every element must be a number, got "
                f"{value!r}"
            )
        container[key] = [
            int(v) if isinstance(c, int) else float(v)
            for c, v in zip(current, value)
        ]
        return

    raise SimError(
        f"override {path!r}: only numbers and arrays of numbers are "
        f"overridable, and this key currently holds {current!r}. A string or "
        f"a table selects structure the mission validator checked, and "
        f"changing it here would skip that check"
    )


def _deep_copy_resolved(resolved):
    """Copy a resolved mission deeply enough for override application.

    ``copy.deepcopy`` would work, but the resolved config is plain JSON-able
    data by construction (it is hashed through ``canonical_bytes``), so a
    round trip through the same canonical representation is both sufficient
    and a check that the assumption still holds.
    """
    import json

    return json.loads(canonical_bytes(resolved).decode("utf-8"))
