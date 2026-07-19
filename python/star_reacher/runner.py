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


def _flatten_inertia(m) -> list:
    """Row-major flatten of a 3x3 inertia list (the core carries 9 doubles)."""
    return [float(m[i][j]) for i in range(3) for j in range(3)]


def _read_aero_csv(path):
    """Parse a validated Mach-table CSV into parallel column lists.

    The FR-9 CSV structure (header ``mach,ca,cnalpha_per_rad,xcp_m``, ``#``
    comments, strictly increasing Mach) is already checked by the vehicle
    validator; this reads the columns the core aero model needs.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    content = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    mach, ca, cnalpha, xcp = [], [], [], []
    for ln in content[1:]:  # content[0] is the validated header row
        parts = [c.strip() for c in ln.split(",")]
        mach.append(float(parts[0]))
        ca.append(float(parts[1]))
        cnalpha.append(float(parts[2]))
        xcp.append(float(parts[3]))
    return mach, ca, cnalpha, xcp


def _build_vehicle_config(core, vres: dict):
    """Translate a resolved vehicle dict into the bound VehicleConfig (D-2)."""
    vc = core.VehicleConfig()
    stages = []
    for st in vres["stage"]:
        sc = core.StageCfg()
        sc.name = st["name"]
        sc.dry_mass_kg = float(st["dry_mass_kg"])
        sc.dry_cg_m = tuple(float(x) for x in st["dry_cg_m"])
        sc.dry_inertia_kgm2 = _flatten_inertia(st["dry_inertia_kgm2"])
        tank_names = [t["name"] for t in st.get("tank", [])]
        tanks = []
        for t in st.get("tank", []):
            tc = core.TankCfg()
            tc.radius_m = float(t["radius_m"])
            tc.length_m = float(t["length_m"])
            tc.position_m = tuple(float(x) for x in t["position_m"])
            tc.propellant_mass_kg = float(t["propellant_mass_kg"])
            tc.density_kgpm3 = float(t["density_kgpm3"])
            tanks.append(tc)
        sc.tanks = tanks
        engines = []
        for e in st.get("engine", []):
            ec = core.EngineCfg()
            ec.name = e["name"]
            ec.feeds_tank_index = (
                tank_names.index(e["feeds_tank"]) if e["feeds_tank"] in tank_names else -1
            )
            ec.thrust_vac_N = float(e["thrust_vac_N"])
            ec.isp_vac_s = float(e["isp_vac_s"])
            ec.exit_area_m2 = float(e["exit_area_m2"])
            ec.position_m = tuple(float(x) for x in e["position_m"])
            ec.axis = tuple(float(x) for x in e["axis"])
            ec.gimbal_max_deg = float(e["gimbal_max_deg"])
            ec.gimbal_rate_dps = float(e["gimbal_rate_dps"])
            ec.throttle_min = float(e["throttle_min"])
            ec.throttle_max = float(e["throttle_max"])
            ec.spool_time_s = float(e["spool_time_s"])
            ec.ignitions = int(e["ignitions"])
            engines.append(ec)
        sc.engines = engines
        rcs = []
        for r in st.get("rcs", []):
            rc = core.RcsCfg()
            rc.name = r["name"]
            rc.thrust_N = float(r["thrust_N"])
            rc.min_impulse_bit_Ns = float(r["min_impulse_bit_Ns"])
            rc.thruster_positions_m = [tuple(float(x) for x in p) for p in r["thruster_positions_m"]]
            rc.thruster_directions = [tuple(float(x) for x in d) for d in r["thruster_directions"]]
            rcs.append(rc)
        sc.rcs = rcs
        wheels = []
        for w in st.get("wheel", []):
            wc = core.WheelCfg()
            wc.name = w["name"]
            wc.axis = tuple(float(x) for x in w["axis"])
            wc.max_torque_Nm = float(w["max_torque_Nm"])
            wc.max_momentum_Nms = float(w["max_momentum_Nms"])
            wheels.append(wc)
        sc.wheels = wheels
        jett = []
        for j in st.get("jettison", []):
            jc = core.JettisonCfg()
            jc.name = j["name"]
            jc.mass_kg = float(j["mass_kg"])
            jc.cg_m = tuple(float(x) for x in j["cg_m"])
            jc.inertia_kgm2 = _flatten_inertia(j["inertia_kgm2"])
            jett.append(jc)
        sc.jettison = jett
        stages.append(sc)
    vc.stages = stages
    aero = []
    for a in vres.get("aero", []):
        ac = core.AeroCfg()
        ac.config = a["config"]
        ac.ref_area_m2 = float(a["ref_area_m2"])
        ac.ref_diameter_m = float(a["ref_diameter_m"])
        ac.cmq_per_rad = float(a.get("cmq_per_rad", 0.0))
        mach, ca, cnalpha, xcp = _read_aero_csv(a["mach_table_csv"])
        ac.mach = mach
        ac.ca = ca
        ac.cnalpha_per_rad = cnalpha
        ac.xcp_m = xcp
        aero.append(ac)
    vc.aero = aero
    return vc


def _build_sequence(core, seq: list):
    """Translate the resolved [[sequence]] entries into bound SequenceEntry."""
    out = []
    for e in seq:
        se = core.SequenceEntry()
        se.name = e["name"]
        se.trigger = e["trigger"]
        se.action = e["action"]
        if e["trigger"] == "elapsed":
            se.t_s = float(e["t_s"])
        elif e["trigger"] == "after_event":
            se.event = e["event"]
            se.offset_s = float(e["offset_s"])
        elif e["trigger"] == "condition":
            se.condition = e["condition"]
            if "altitude_m" in e:
                se.altitude_m = float(e["altitude_m"])
            if "perigee_alt_m" in e:
                se.perigee_alt_m = float(e["perigee_alt_m"])
            if "body" in e:
                se.body = e["body"]
        if "stage" in e:
            se.stage = e["stage"]
        if "engine" in e:
            se.engine = e["engine"]
        if "item" in e:
            se.item = e["item"]
        if "azimuth_deg" in e:
            se.azimuth_deg = float(e["azimuth_deg"])
        if "pitch_t_s" in e:
            se.pitch_t_s = [float(x) for x in e["pitch_t_s"]]
        if "pitch_deg" in e:
            se.pitch_deg = [float(x) for x in e["pitch_deg"]]
        if "frame" in e:
            se.frame = e["frame"]
        if "omega_dps" in e:
            se.omega_dps = tuple(float(x) for x in e["omega_dps"])
        out.append(se)
    return out


def _build_gnc_component(core, spec: dict):
    """Translate one resolved [gnc.*] slot into the bound GncComponentCfg.

    Every key besides ``component`` is a parameter: numbers ride in the
    scalar map, arrays in the vector map (the plain-data composition rule of
    star/gnc/config.hpp), so new component parameters need no runner change.
    """
    cc = core.GncComponentCfg()
    cc.component = spec["component"]
    scalars = {}
    vectors = {}
    for key, value in spec.items():
        if key == "component":
            continue
        if isinstance(value, list):
            vectors[key] = [float(x) for x in value]
        else:
            scalars[key] = float(value)
    cc.scalars = scalars
    cc.vectors = vectors
    return cc


def run_mission(mission_path, outdir=None, force=False, command_line=None, strict=False) -> RunResult:
    """Validate, resolve, hash, propagate, and write the run artifacts.

    Raises ``MissionValidationError`` (exit 2 at the CLI) for config errors,
    ``CoreMissingError`` or ``RunnerError`` (exit 1) for runtime failures.
    ``strict`` promotes validation warnings (the vehicle plausibility tier)
    to errors (FR-15).
    """
    start_wall = time.monotonic()
    start_utc = datetime.now(timezone.utc).isoformat()

    resolved, errors = validate_mission_file(mission_path, strict=strict)
    if errors:
        raise MissionValidationError(errors)

    # Path selection. A mission with a vehicle file, an event [[sequence]], or a
    # geodetic launch state takes the Phase 4 6DOF path (run_vehicle); a mission
    # that uses any Phase 3 environment surface takes the composed-environment
    # path (run_env); anything else takes the byte-frozen Phase 1 two-body path
    # (run), whose output is pinned by the committed determinism record.
    is_vehicle = (
        "vehicle" in resolved
        or "sequence" in resolved
        or "geodetic" in resolved["initial_state"]
    )

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

    cfg = core.RunConfig()
    cfg.epoch_utc = resolved["mission"]["epoch_utc"]
    cfg.duration_s = resolved["mission"]["duration_s"]
    cfg.integrator = integ["type"]
    cfg.central_body = env["central_body"]
    cfg.mass_kg = resolved["spacecraft"]["mass_kg"]
    cfg.master_seed = resolved["run"]["seed"]
    cfg.truth_rate_hz = resolved["logging"]["truth_rate_hz"]
    cfg.config_sha256 = config_sha
    cfg.oracle = False

    if "cartesian" in initial:
        cfg.r0_m = tuple(float(x) for x in initial["cartesian"]["r_m"])
        cfg.v0_mps = tuple(float(x) for x in initial["cartesian"]["v_mps"])
        cfg.initial_form = "cartesian"
    elif "keplerian" in initial:
        # gm comes from the core so the gravitational parameter has exactly
        # one home (contract section 3); the conversion is pure NumPy.
        r_vec, v_vec = keplerian_to_cartesian(
            initial["keplerian"], core.gm(env["central_body"])
        )
        cfg.r0_m = tuple(float(x) for x in r_vec)
        cfg.v0_mps = tuple(float(x) for x in v_vec)
        cfg.initial_form = "keplerian"
    else:  # geodetic launch (FR-14): the core builds the pad state
        geo = initial["geodetic"]
        cfg.initial_form = "geodetic"
        cfg.launch_lat_deg = float(geo["lat_deg"])
        cfg.launch_lon_deg = float(geo["lon_deg"])
        cfg.launch_alt_m = float(geo["alt_m"])

    env_features = ("gravity", "third_bodies", "srp", "drag", "ephemeris")
    new_path = (
        integ["type"] != "rk4"
        or env["central_body"] != "earth"
        or any(key in env for key in env_features)
    )

    if is_vehicle or new_path:
        # The core never parses text (D-2): the ISO epoch is converted here,
        # through the bound leap-table conversion, into the two-part TAI epoch
        # the environment/vehicle models propagate from.
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

    resolved_vehicle_toml = None
    if is_vehicle:
        # Re-validate the vehicle file to recover its resolved nested dict (the
        # mission resolved config carries only the vehicle path + hash). The
        # vehicle SHA-256 is already folded into config_sha256, so the vehicle
        # is covered without double-hashing.
        from star_reacher.vehicle import canonical_vehicle_toml, validate_vehicle_file

        vres, verrs, _vwarns = validate_vehicle_file(
            resolved["vehicle"]["path"], strict=strict
        )
        if vres is None:
            # Unreachable in practice: mission validation already validated the
            # vehicle. Surfaced as a runtime failure rather than silently.
            raise RunnerError(
                f"{resolved['vehicle']['path']}: vehicle re-validation failed: "
                + "; ".join(verrs)
            )
        cfg.vehicle = _build_vehicle_config(core, vres)
        cfg.sequence = _build_sequence(core, resolved.get("sequence", []))
        # Vehicle channel groups at 1 Hz by default (FR-16), so depletion,
        # staging jumps, and the per-source force breakdown are inspectable; a
        # mission [logging] entry can lower a rate or set 0 to disable a group
        # (e.g. a multi-day coast that would otherwise write a very large log).
        log_cfg = resolved["logging"]
        cfg.forces_rate_hz = log_cfg.get("forces_rate_hz", 1)
        cfg.mass_rate_hz = log_cfg.get("mass_rate_hz", 1)
        cfg.env_rate_hz = log_cfg.get("env_rate_hz", 1)
        resolved_vehicle_toml = canonical_vehicle_toml(vres)

        # Phase 6 GNC chain (FR-23/FR-25). The oracle flag comes from [gnc]
        # and is stamped into the log header by the core; sensors ride in
        # canonical kind order (only "imu" exists this phase).
        if "gnc" in resolved:
            g = resolved["gnc"]
            gc = core.GncConfig()
            gc.enabled = True
            gc.control_rate_hz = g["control_rate_hz"]
            gc.latency_cycles = g["latency_cycles"]
            gc.nav = _build_gnc_component(core, g["nav"])
            gc.guidance = _build_gnc_component(core, g["guidance"])
            gc.control = _build_gnc_component(core, g["control"])
            sensor_cfgs = []
            for kind, spec in resolved["sensors"].items():
                sc = core.GncSensorCfg()
                sc.kind = kind
                sc.sample_rate_hz = spec["sample_rate_hz"]
                # Every other resolved key is a kind-specific model
                # parameter, forwarded verbatim on the flat scalar/vector
                # maps; the sensor module owns its own key vocabulary, so a
                # new FR-23 error term needs no change here.
                scalars = {}
                vectors = {}
                for key, value in spec.items():
                    if key == "sample_rate_hz":
                        continue
                    if isinstance(value, list):
                        vectors[key] = [float(v) for v in value]
                    else:
                        scalars[key] = float(value)
                sc.scalars = scalars
                sc.vectors = vectors
                sensor_cfgs.append(sc)
            gc.sensors = sensor_cfgs
            cfg.gnc = gc
            cfg.oracle = g["oracle"]

    out.mkdir(parents=True, exist_ok=True)
    # Exactly the hashed bytes, so the file re-hashes to config_sha256.
    (out / "resolved_config.json").write_bytes(config_bytes)
    if resolved_vehicle_toml is not None:
        (out / "resolved_vehicle.toml").write_text(
            resolved_vehicle_toml, encoding="utf-8"
        )

    if is_vehicle:
        summary = core.run_vehicle(cfg, str(srlog_path))
    elif new_path:
        summary = core.run_env(cfg, str(srlog_path))
    else:
        summary = core.run(cfg, str(srlog_path))

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
