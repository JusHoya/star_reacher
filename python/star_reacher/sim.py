"""FR-24 stepping API: drive a mission one control period at a time.

``Sim`` is the Python front door to the core's vehicle cycle. A batch
``star run`` and a stepped ``Sim`` traverse the *same* cycle core --
``run_vehicle()`` is literally ``while cycle.step(): pass`` over it -- so the
two produce byte-identical SRLOG files for one scenario. That equality is
Phase 6 exit criterion 4.

Sharing the cycle core is necessary for that equality but is *not*
sufficient, and the criterion is therefore gated by a real comparison rather
than asserted from the factoring. The two paths also share one configuration
builder, ``runner.build_run_config``, but they hand it the resolved mission
in different orders -- the validator's canonical order for ``star run``, the
``sort_keys=True`` alphabetical order for ``reset()``, which round-trips
through ``canonical_bytes``. A builder that inherited its input's order would
therefore configure the two runs differently while both still stepped the
identical core. ``mission.canonical_sensor_items`` is what closes that gap,
and the criterion-4 fixture is multi-sensor so the gate can see it reopen.

Typical use::

    from star_reacher.sim import Sim

    with Sim("missions/leo_gnc.toml", "out/stepped") as sim:
        obs, info = sim.reset()
        while not sim.done():
            obs = sim.step()
        print(sim.summary())

The context manager is the recommended form because it releases the log
handle on the way out whether the run ended or an exception escaped
``step()``; see :meth:`Sim.close`.

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
from star_reacher.overrides import (
    OVERRIDE_ALIASES as _OVERRIDE_ALIASES,
    OverrideError,
    apply_override as _apply_override,
    deep_copy_resolved as _deep_copy_resolved,
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

    Usable as a context manager, which is the recommended form: ``__exit__``
    calls :meth:`close`, so the log handle is released even when an exception
    escapes :meth:`step`.
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
        # True once this Sim has opened its output path itself, which is what
        # makes a later reset() over the same path its own run rather than
        # someone else's file.
        self._opened = False

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

        Calling ``reset`` again starts a new run over the same output path,
        which is the FR-24 episode loop -- ``for ep in range(N):
        sim.reset()`` -- and works at the default ``force=False``. The
        ``force`` guard exists to protect an output this ``Sim`` did not
        write; once it has opened that path itself, a later ``reset``
        overwrites its own previous run. Each episode therefore leaves only
        the last episode's ``run.srlog`` behind: a driver that needs one log
        per episode constructs a ``Sim`` per output directory.
        """
        resolved = _deep_copy_resolved(self._resolved)
        try:
            if seed is not None:
                # Through the same path as any other integer override, so a
                # fractional seed is refused here for the reason it is refused
                # there rather than silently truncated.
                _apply_override(resolved, "run.seed", seed)
            for key, value in dict(overrides or {}).items():
                _apply_override(resolved, _OVERRIDE_ALIASES.get(key, key), value)
        except OverrideError as exc:
            # The stepping API's public error type is SimError; the shared
            # override module raises OverrideError, so translate it here while
            # preserving the message verbatim (the same one the tests match).
            raise SimError(str(exc)) from exc

        config_bytes = canonical_bytes(resolved)
        config_sha = hashlib.sha256(config_bytes).hexdigest()
        cfg, resolved_vehicle_toml, _ = build_run_config(
            self._core, resolved, config_sha, strict=self._strict
        )

        srlog_path = self._outdir / "run.srlog"
        # The guard is against clobbering a log this Sim did not write. A
        # second reset() over a path this Sim already opened is the
        # documented episode loop, not an accident, so it is not refused --
        # otherwise the default-constructed Gym-style driver FR-24 describes
        # fails on its second episode.
        if srlog_path.exists() and not self._force and not self._opened:
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

        # Close explicitly rather than relying on the drop below to do it:
        # refcounting releases the previous run's handle only if nothing else
        # still references it, and a traceback pinning a frame is enough to
        # keep it alive past the point the new run needs the path.
        self.close()
        self._sim = self._core.Sim(cfg, str(srlog_path))
        self._opened = True
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

    def close(self):
        """Release the open run's log handle. Idempotent.

        A run abandoned part way -- a driver that stops early, or any
        exception escaping :meth:`step` -- holds ``run.srlog`` open for as
        long as the core ``Sim`` lives, and how long that is depends on
        refcount timing and on whether a traceback still pins the frame
        holding it. On Windows the open handle then makes a later unlink of
        the directory, or a reopen of the same path, fail with
        ``PermissionError: [WinError 32]``; on Linux the unlink silently
        succeeds, so this reproduces under MSVC only. Closing explicitly
        makes the file's lifetime something a driver states rather than
        something it infers.

        The log of a run closed before it ended is a valid PREFIX, not a
        complete run: it carries no ``run_end`` event, and stepping after
        close raises rather than resuming.
        """
        sim, self._sim = self._sim, None
        if sim is not None:
            sim.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def _active(self):
        if self._sim is None:
            if self._opened:
                raise SimError(
                    "the run was closed; call reset() to start a new one"
                )
            raise SimError("call reset() before stepping the simulation")
        return self._sim
