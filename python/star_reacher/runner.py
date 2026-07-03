"""Mission execution: the shared implementation behind ``star run``.

The verify V001 determinism check calls this same function, so the acceptance
gate exercises exactly the code path users run, not a parallel one. Order of
operations follows the Phase 1 contract: validate, resolve and hash, then
lazily import the core (so a machine without the compiled core still gets the
full validation report before the actionable core-missing error).
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from star_reacher import __version__
from star_reacher._corelink import import_core
from star_reacher.mission import (
    MissionValidationError,
    canonical_bytes,
    keplerian_to_cartesian,
    validate_mission_file,
)


class RunnerError(Exception):
    """Runtime (non-validation) failure while executing a mission."""


@dataclass
class RunResult:
    mission_name: str
    outdir: Path
    srlog_path: Path
    srlog_sha256: str
    config_sha256: str
    summary: dict


def run_mission(mission_path, outdir=None, force=False, command_line=None) -> RunResult:
    """Validate, resolve, hash, propagate, and write the run artifacts.

    Raises ``MissionValidationError`` (exit 2 at the CLI) for config errors,
    ``CoreMissingError`` or ``RunnerError`` (exit 1) for runtime failures.
    """
    start_wall = time.monotonic()
    start_utc = datetime.now(timezone.utc).isoformat()

    resolved, errors = validate_mission_file(mission_path)
    if errors:
        raise MissionValidationError(errors)

    config_bytes = canonical_bytes(resolved)
    config_sha = hashlib.sha256(config_bytes).hexdigest()

    name = resolved["mission"]["name"]
    out = Path(outdir) if outdir is not None else Path("out") / name
    srlog_path = out / "run.srlog"
    if srlog_path.exists() and not force:
        raise RunnerError(
            f"{srlog_path}: output already exists; pass --force to overwrite, "
            f"or choose another directory with -o"
        )

    core = import_core()

    env = resolved["environment"]
    integ = resolved["integrator"]

    initial = resolved["initial_state"]
    if "cartesian" in initial:
        r0 = tuple(initial["cartesian"]["r_m"])
        v0 = tuple(initial["cartesian"]["v_mps"])
    else:
        # gm comes from the core so the gravitational parameter has exactly
        # one home (contract section 3); the conversion is pure NumPy.
        r_vec, v_vec = keplerian_to_cartesian(
            initial["keplerian"], core.gm(env["central_body"])
        )
        r0 = tuple(float(x) for x in r_vec)
        v0 = tuple(float(x) for x in v_vec)

    cfg = core.RunConfig()
    cfg.epoch_utc = resolved["mission"]["epoch_utc"]
    cfg.duration_s = resolved["mission"]["duration_s"]
    cfg.integrator = integ["type"]
    cfg.central_body = env["central_body"]
    cfg.r0_m = r0
    cfg.v0_mps = v0
    cfg.mass_kg = resolved["spacecraft"]["mass_kg"]
    cfg.master_seed = resolved["run"]["seed"]
    cfg.truth_rate_hz = resolved["logging"]["truth_rate_hz"]
    cfg.config_sha256 = config_sha
    cfg.oracle = False

    # Path selection: a mission that uses none of the Phase 3 surface takes
    # the byte-frozen Phase 1 two-body path, whose output is pinned by the
    # committed determinism record (tests/golden/determinism/
    # cross_platform.toml); anything else takes the composed-environment path.
    env_features = ("gravity", "third_bodies", "srp", "drag", "ephemeris")
    new_path = (
        integ["type"] != "rk4"
        or env["central_body"] != "earth"
        or any(key in env for key in env_features)
    )

    if new_path:
        # The core never parses text (D-2): the ISO epoch is converted here,
        # through the bound leap-table conversion, into the two-part TAI
        # epoch the environment model propagates from.
        moment = datetime.fromisoformat(
            resolved["mission"]["epoch_utc"]
        ).astimezone(timezone.utc)
        tai_day, tai_sec = core.utc_to_tai(
            moment.year,
            moment.month,
            moment.day,
            moment.hour,
            moment.minute,
            moment.second + moment.microsecond * 1e-6,
        )
        cfg.epoch_tai_day = tai_day
        cfg.epoch_tai_sec = tai_sec
        if integ["type"] == "rk4":
            cfg.dt_s = integ["dt_s"]
        else:
            cfg.rtol = integ["rtol"]
            cfg.atol_pos_m = integ["atol_pos_m"]
            cfg.atol_vel_mps = integ["atol_vel_mps"]
            cfg.h_init_s = integ["h_init_s"]
            cfg.h_max_s = integ["h_max_s"]
        gravity = env.get("gravity", {"model": "pointmass"})
        cfg.gravity_model = gravity["model"]
        cfg.gravity_field_path = gravity.get("field", "")
        cfg.gravity_degree = gravity.get("degree", -1)
        cfg.gravity_order = gravity.get("order", -1)
        cfg.third_bodies = env.get("third_bodies", [])
        if "srp" in env:
            cfg.srp_enabled = True
            cfg.cr_a_over_m_m2pkg = resolved["spacecraft"]["cr_a_over_m_m2pkg"]
            cfg.srp_occulters = env["srp"]["occulters"]
        if "drag" in env:
            cfg.drag_enabled = True
            cfg.atmosphere = env["drag"]["atmosphere"]
            cfg.cd_a_over_m_m2pkg = resolved["spacecraft"]["cd_a_over_m_m2pkg"]
            cfg.hp_exponent_n = env["drag"].get("hp_exponent_n", 4.0)
        cfg.ephemeris_path = env.get("ephemeris", "")
    else:
        cfg.dt_s = integ["dt_s"]

    out.mkdir(parents=True, exist_ok=True)
    # Exactly the hashed bytes, so the file re-hashes to config_sha256.
    (out / "resolved_config.json").write_bytes(config_bytes)

    summary = core.run_env(cfg, str(srlog_path)) if new_path else core.run(
        cfg, str(srlog_path)
    )

    srlog_sha = hashlib.sha256(srlog_path.read_bytes()).hexdigest()

    # Wall-clock, host identity, and tool versions live only in this sidecar:
    # the log itself must stay free of them so reruns are bit-identical (D-11).
    meta = {
        "command_line": list(command_line) if command_line else [],
        "config_sha256": config_sha,
        "srlog_sha256": srlog_sha,
        "host": {
            "node": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "star_reacher": __version__,
            "core": core.core_version(),
            "core_git_hash": core.git_hash(),
        },
        "wall_clock": {
            "start_utc": start_utc,
            "end_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": time.monotonic() - start_wall,
        },
    }
    (out / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return RunResult(
        mission_name=name,
        outdir=out,
        srlog_path=srlog_path,
        srlog_sha256=srlog_sha,
        config_sha256=config_sha,
        summary=dict(summary),
    )
