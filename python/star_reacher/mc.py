"""Monte Carlo sweep execution and manifest writing (FR-27).

``star mc <sweep.toml>`` expands a sweep spec (``star_reacher.sweep``) into N
independent single-run cases, runs each in its own process, and writes a
``manifest.json`` that makes every run individually reproducible:

* per-run seed = ``SplitMix64(master_seed)[index]`` -- element ``index`` of the
  SplitMix64 stream the core owns (D-9), so the seed derivation has exactly one
  implementation and a run's seed is a function of the master seed and its
  index alone;
* per-run overrides = the sweep case's ``{dotted.path: value}`` dict, in the
  FR-24/FR-27 vocabulary (``star_reacher.overrides``);
* per-run ``config_sha256`` and ``log_sha256`` = exactly what ``star run
  --seed <seed> --set <path>=<value> ...`` of the base mission produces, so
  re-executing any manifest entry reproduces its logged hash. That equality is
  Phase 7 exit criterion 1, and ``test_mc.py`` proves it.

Parallelism is process-level only (D-10): the core time loop stays
single-threaded, each worker runs one mission in a fresh process, and no
mutable state is shared, so a run is bit-identical whether it ran in the pool
or standalone.

The NPZ export path (``star export --npz``) is the training-data pipeline for
a completed sweep; ``star mc`` deliberately writes no bespoke dataset format.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from star_reacher.sweep import SweepError, load_sweep_spec

__all__ = [
    "McError",
    "run_sweep",
    "splitmix64_stream",
]

MANIFEST_SCHEMA_VERSION = 1

_U64_MASK = (1 << 64) - 1

# SplitMix64 constants (Vigna reference splitmix64.c), kept here only for the
# build-free fallback below; the compiled core's binding is the source of
# truth, and test_mc_seed.py asserts the two agree bit-for-bit.
_SM64_INCREMENT = 0x9E3779B97F4A7C15
_SM64_MIX_1 = 0xBF58476D1CE4E5B9
_SM64_MIX_2 = 0x94D049BB133111EB


class McError(Exception):
    """A Monte Carlo orchestration failure (bad spec, or a failed sweep)."""


def _splitmix64_stream(seed: int, n: int) -> list[int]:
    """Pure-Python mirror of the core ``splitmix64_stream`` binding.

    A build-free cross-check and fallback, NOT the source of truth: the C++
    binding in ``cpp/src/rng.cpp`` (``SplitMix64::next``) owns the algorithm
    (D-9), and ``test_mc_seed.py`` asserts this mirror equals it bit-for-bit
    and equals the committed golden vectors. The fallback exists so ``star mc``
    still derives correct per-run seeds on a checkout whose compiled core
    predates the binding (it is re-added by the same phase that adds this
    module); once the core carries the binding, the binding is used.
    """
    state = seed & _U64_MASK
    out = []
    for _ in range(n):
        state = (state + _SM64_INCREMENT) & _U64_MASK
        z = state
        z = ((z ^ (z >> 30)) * _SM64_MIX_1) & _U64_MASK
        z = ((z ^ (z >> 27)) * _SM64_MIX_2) & _U64_MASK
        out.append(z ^ (z >> 31))
    return out


def splitmix64_stream(seed: int, n: int) -> list[int]:
    """The first ``n`` SplitMix64 outputs for ``seed``, from the core binding.

    Uses the compiled core's ``splitmix64_stream`` when present -- the one
    SplitMix64 the project owns -- and falls back to the tested pure-Python
    mirror only when the core is absent or predates the binding, so a sweep's
    per-run seeds are correct either way.
    """
    try:
        from star_reacher._corelink import import_core

        core = import_core()
        binding = getattr(core, "splitmix64_stream", None)
        if binding is not None:
            return list(binding(seed, n))
    except Exception:
        # A core-less checkout (CoreMissingError) or a core without the
        # binding falls through to the mirror; the mirror is exact.
        pass
    return _splitmix64_stream(seed, n)


def per_run_seeds(master_seed: int, n_runs: int) -> list[int]:
    """Per-run seeds for a sweep: element ``i`` is run ``i``'s master seed."""
    return splitmix64_stream(master_seed, n_runs)


def _core_binary_sha256() -> tuple[str, str, str]:
    """(core_version, core_git_hash, binary_sha256) of the compiled core.

    ``binary_sha256`` is the SHA-256 of the ``_core`` extension file on disk,
    located via its ``__file__``: two runs are only reproducible on the same
    binary, so the manifest pins which one produced them.
    """
    from star_reacher._corelink import import_core

    core = import_core()
    binary_path = Path(core.__file__)
    binary_sha = hashlib.sha256(binary_path.read_bytes()).hexdigest()
    return core.core_version(), core.git_hash(), binary_sha


def _sweep_spec_sha256(spec_path) -> str:
    """SHA-256 of the sweep spec's raw bytes, recorded in the manifest.

    The spec file's bytes rather than a canonicalized form: the spec is the
    experiment's definition, and pinning its exact bytes lets a reader confirm
    a manifest was produced from the spec they hold.
    """
    return hashlib.sha256(Path(spec_path).read_bytes()).hexdigest()


def _run_one_case(args):
    """Execute one sweep case; a module-level function so it is picklable.

    Returns a manifest ``runs`` entry. Catches every exception so one failed
    run is recorded as ``status: failed`` with its error rather than aborting
    the whole sweep -- the manifest is then a complete record of what ran.
    """
    index, mission_path, seed, overrides, outdir = args
    entry = {
        "index": index,
        "seed": seed,
        "overrides": dict(overrides),
        "outdir": Path(outdir).name,
    }
    try:
        from star_reacher.runner import run_mission

        result = run_mission(
            mission_path,
            outdir=outdir,
            force=True,
            seed=seed,
            overrides=overrides,
        )
        entry["config_sha256"] = result.config_sha256
        entry["log_sha256"] = result.srlog_sha256
        entry["status"] = "success"
    except Exception as exc:  # noqa: BLE001 -- a failed run is data, not a crash
        entry["config_sha256"] = None
        entry["log_sha256"] = None
        entry["status"] = "failed"
        entry["error"] = f"{type(exc).__name__}: {exc}"
    return entry


def run_sweep(spec_path, *, workers=8, outdir=None, force=False):
    """Run a sweep and write ``<outdir>/manifest.json``. Return the manifest.

    ``workers`` is the process-pool size (default 8, per exit criterion 1's
    8-worker sweep). Raises :class:`SweepError` for a bad spec (before any run)
    and :class:`McError` when the output directory is occupied and ``force``
    is not set. A failed run does not raise here; it is recorded in the
    manifest and reflected in the caller's exit code.
    """
    spec = load_sweep_spec(spec_path)

    out = Path(outdir) if outdir is not None else Path("out") / "mc"
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and not force:
        raise McError(
            f"{manifest_path}: a manifest already exists; pass --force to "
            f"overwrite, or choose another directory with -o"
        )
    out.mkdir(parents=True, exist_ok=True)

    seeds = per_run_seeds(spec.master_seed, spec.n_runs)
    mission_path = str(spec.mission_path)
    tasks = [
        (i, mission_path, seeds[i], spec.cases[i], str(out / f"run_{i:04d}"))
        for i in range(spec.n_runs)
    ]

    runs = _dispatch(tasks, workers)
    runs.sort(key=lambda e: e["index"])

    core_version, core_git_hash, binary_sha = _core_binary_sha256()
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "sweep": {
            "mission": spec.mission,
            "master_seed": spec.master_seed,
            "method": spec.method,
            "n_runs": spec.n_runs,
            "parameters": spec.parameters,
            "sweep_spec_sha256": _sweep_spec_sha256(spec_path),
        },
        "binary": {
            "core_version": core_version,
            "core_git_hash": core_git_hash,
            "binary_sha256": binary_sha,
        },
        "runs": runs,
    }
    # Same style as runner.py meta.json: sorted keys, two-space indent, a
    # trailing newline, so the manifest is a stable, diffable artifact.
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _dispatch(tasks, workers):
    """Run every task, in a process pool when workers > 1, else in-process.

    A single worker runs the tasks in the calling process: it keeps a 1-worker
    sweep debuggable and free of the pool's pickling round trip, and the result
    is identical because each task is already self-contained.
    """
    if workers <= 1 or len(tasks) <= 1:
        return [_run_one_case(t) for t in tasks]
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_run_one_case, tasks))


def cli_mc(spec_path, *, workers=8, outdir=None, force=False) -> int:
    """``star mc`` entry point: run the sweep, print a summary, return a code.

    Exit 0 when every run succeeded, 1 when any failed or a runtime error
    occurred, 2 for a spec validation error.
    """
    try:
        manifest = run_sweep(spec_path, workers=workers, outdir=outdir,
                             force=force)
    except SweepError as exc:
        for line in exc.errors:
            print(line, file=sys.stderr)
        print(f"star mc: {len(exc.errors)} sweep-spec error(s) in "
              f"{spec_path}; all are listed above.", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"star mc: {exc}", file=sys.stderr)
        return 1
    except McError as exc:
        print(f"star mc: {exc}", file=sys.stderr)
        return 1

    runs = manifest["runs"]
    failed = [r for r in runs if r["status"] != "success"]
    out = Path(outdir) if outdir is not None else Path("out") / "mc"
    print(f"mc: {manifest['sweep']['method']} sweep of "
          f"{manifest['sweep']['mission']}")
    print(f"runs: {len(runs) - len(failed)}/{len(runs)} success")
    print(f"manifest: {out / 'manifest.json'}")
    for r in failed:
        print(f"  run {r['index']:04d} FAILED: {r.get('error', 'unknown')}",
              file=sys.stderr)
    return 0 if not failed else 1
