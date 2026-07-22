"""FR-25 ``star run --gnc-plugin``: a Python component named by a mission.

FR-25 requires that ``star run mission.toml --gnc-plugin my_nav.py`` swap a
component with zero recompilation. Two halves have to hold for that to be
true rather than merely available:

* a mission file can NAME a Python component -- the ``python:`` namespace,
  validated core-less by ``star_reacher.mission`` and proved real against the
  loaded plugin by ``star_reacher.plugin``;
* naming one and flying it produces the same run a built-in would, which is
  checked here against the committed reference mission whose answer is
  already pinned by the Phase 6 suite.

The validator half runs without a compiled core, deliberately: strictness
must not depend on the core being present. The run half fails cleanly, never
skips, when the core is absent (the project's agent-honesty gate).
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import star_reacher
from star_reacher.mission import validate_mission_file

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILTIN_MISSION = REPO_ROOT / "missions" / "leo_attitude_gnc.toml"
PLUGIN_MISSION = REPO_ROOT / "missions" / "leo_attitude_gnc_plugin.toml"
EXAMPLE_PLUGIN = REPO_ROOT / "examples" / "gnc_plugins" / "pd_attitude.py"

_CORE_MISSING_MESSAGE = (
    "star_reacher._core is not built in this environment. These integration "
    "tests require the compiled core: build and install it with 'pip install .' "
    "from the repository root. This failure is expected on a core-less checkout "
    "and must be green at integration/CI."
)


def _core_or_fail():
    try:
        from star_reacher import _core
    except ImportError:
        pytest.fail(_CORE_MISSING_MESSAGE)
    return _core


def _run_cli(*args, cwd=None):
    env = os.environ.copy()
    # Point the subprocess at the same package this test process imported
    # (source tree or installed wheel), so both see identical code.
    pkg_parent = Path(star_reacher.__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(pkg_parent) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "star_reacher", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
    )


def _validate(tmp_path, text, monkeypatch):
    # Vehicle paths resolve against the working directory; validation of the
    # referenced starter vehicle needs the repository root.
    monkeypatch.chdir(REPO_ROOT)
    path = tmp_path / "mission.toml"
    path.write_text(text, encoding="utf-8")
    return validate_mission_file(path)


def _plugin_mission_text():
    return PLUGIN_MISSION.read_text(encoding="utf-8")


# --- the python: namespace, validated core-less ----------------------------


@pytest.mark.parametrize(
    "anchor",
    [
        'component = "dead_reckoning"',
        'component = "attitude_hold"',
        'component = "python:pd_attitude"',
    ],
)
def test_every_chain_slot_accepts_a_plugin_component(tmp_path, monkeypatch, anchor):
    """FR-25 defines one base for all three roles, so all three accept one.

    Restricting the flag to a single slot would be an invention of the loader
    rather than a property of the interface: the registry is slot-agnostic and
    ``IGncComponent`` is the same base in every role.
    """
    text = _plugin_mission_text().replace(anchor, 'component = "python:swapped"')
    resolved, errors = _validate(tmp_path, text, monkeypatch)
    assert not errors, errors
    assert resolved is not None


def test_a_typod_builtin_name_is_still_a_hard_error(tmp_path, monkeypatch):
    """The namespace must not have loosened the vocabulary it sits beside.

    This is the mutation that would prove the design wrong: if an unknown
    name were waved through as "probably a plugin", a misspelt built-in would
    reach the core instead of the user. It must still fail, naming the key.
    """
    text = _plugin_mission_text().replace(
        'component = "dead_reckoning"', 'component = "dead_reckonning"'
    )
    resolved, errors = _validate(tmp_path, text, monkeypatch)
    assert resolved is None
    assert any(
        "[gnc.nav] component: unknown component 'dead_reckonning'" in e
        for e in errors
    ), errors
    assert all(e.endswith("No default applied; run aborted.") for e in errors), errors


@pytest.mark.parametrize("name", ["python:", "python:2bad", "python:has-dash"])
def test_malformed_plugin_names_are_rejected(tmp_path, monkeypatch, name):
    """The grammar is checked core-less, so the prefix is not a free pass."""
    text = _plugin_mission_text().replace(
        'component = "python:pd_attitude"', f'component = "{name}"'
    )
    resolved, errors = _validate(tmp_path, text, monkeypatch)
    assert resolved is None
    assert any("malformed plugin component name" in e for e in errors), errors


@pytest.mark.parametrize(
    "param, fragment",
    [
        ('mode = "fast"', "expected a number or an array of numbers, got str"),
        ("enabled = true", "expected a number or an array of numbers, got bool"),
        ("gains = []", "expected a non-empty array of numbers"),
        ('gains = ["a"]', "expected a non-empty array of numbers"),
    ],
)
def test_plugin_parameter_values_obey_the_plain_data_rule(
    tmp_path, monkeypatch, param, fragment
):
    """Parameter KEYS belong to the plugin; VALUES still belong to the core.

    ``GncComponentCfg`` carries a scalar map and a vector map and nothing
    else, so a string parameter could not reach the component whatever the
    plugin expected. Rejecting it here names the key instead of failing
    later, or worse, silently dropping it.
    """
    text = _plugin_mission_text().replace(
        'component = "python:pd_attitude"',
        f'component = "python:pd_attitude"\n{param}',
    )
    resolved, errors = _validate(tmp_path, text, monkeypatch)
    assert resolved is None
    assert any(fragment in e for e in errors), errors


def test_plugin_parameters_survive_into_the_resolved_config(tmp_path, monkeypatch):
    """A plugin's numeric parameters reach the resolved config unchanged."""
    text = _plugin_mission_text().replace(
        'component = "python:pd_attitude"',
        'component = "python:pd_attitude"\nextra_gain = 2.5',
    )
    resolved, errors = _validate(tmp_path, text, monkeypatch)
    assert not errors, errors
    assert resolved["gnc"]["control"]["extra_gain"] == 2.5


def test_the_committed_plugin_mission_validates(tmp_path, monkeypatch):
    """The shipped example mission is valid without the plugin being present.

    Validation is a core-less, plugin-less operation by design: a user can
    check a mission file on a machine that cannot run it.
    """
    monkeypatch.chdir(REPO_ROOT)
    resolved, errors = validate_mission_file(PLUGIN_MISSION)
    assert not errors, errors
    assert resolved["gnc"]["control"]["component"] == "python:pd_attitude"


# --- the loader ------------------------------------------------------------


def _write_plugin(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


_TRIVIAL_PLUGIN = '''\
from star_reacher.sim import GncOutput, IGncComponent


class Trivial(IGncComponent):
    def __init__(self, cfg):
        super().__init__()

    def init(self, ctx):
        pass

    def update(self, inp):
        return GncOutput()


STAR_GNC_COMPONENTS = {"trivial": Trivial}
'''


def test_missing_plugin_file_is_named(tmp_path):
    _core_or_fail()
    from star_reacher.plugin import PluginError, load_plugins
    from star_reacher._corelink import import_core

    with pytest.raises(PluginError, match="not found"):
        load_plugins([tmp_path / "absent.py"], import_core())


def test_plugin_without_a_declaration_is_refused(tmp_path):
    """A file that registers nothing is a mistake, not an empty success."""
    _core_or_fail()
    from star_reacher.plugin import PluginError, load_plugins
    from star_reacher._corelink import import_core

    path = _write_plugin(tmp_path, "bare.py", "x = 1\n")
    with pytest.raises(PluginError, match="STAR_GNC_COMPONENTS"):
        load_plugins([path], import_core())


def test_plugin_raising_at_import_is_reported_with_its_path(tmp_path):
    _core_or_fail()
    from star_reacher.plugin import PluginError, load_plugins
    from star_reacher._corelink import import_core

    path = _write_plugin(tmp_path, "boom.py", "raise ValueError('nope')\n")
    with pytest.raises(PluginError, match="raised while being imported"):
        load_plugins([path], import_core())


def test_two_files_declaring_one_name_are_refused(tmp_path):
    """Shadowing would make the flown component depend on flag order."""
    _core_or_fail()
    from star_reacher.plugin import PluginError, load_plugins
    from star_reacher._corelink import import_core

    body = _TRIVIAL_PLUGIN.replace("trivial", "collider")
    first = _write_plugin(tmp_path, "first.py", body)
    second = _write_plugin(tmp_path, "second.py", body)
    core = import_core()
    load_plugins([first], core)
    with pytest.raises(PluginError, match="already provided by"):
        load_plugins([second], core)


def test_loading_one_file_twice_is_idempotent(tmp_path):
    """Re-loading must not trip the core's duplicate-name refusal.

    One process may run several missions off one plugin (a sweep, a test
    session). The second load returns the first module's classes rather than
    re-executing the file, so which class object answers to a name stays
    fixed for the life of the process.
    """
    _core_or_fail()
    from star_reacher.plugin import load_plugins
    from star_reacher._corelink import import_core

    path = _write_plugin(tmp_path, "twice.py", _TRIVIAL_PLUGIN.replace("trivial", "twicer"))
    core = import_core()
    first = load_plugins([path], core)
    second = load_plugins([path], core)
    assert first == second == ["python:twicer"]


def test_a_plugin_cannot_shadow_a_builtin(tmp_path):
    """The example plugin reuses 'pd_attitude' and both remain selectable."""
    core = _core_or_fail()
    from star_reacher.plugin import load_plugins

    names = load_plugins([EXAMPLE_PLUGIN], core)
    assert names == ["python:pd_attitude"]
    registry = core.gnc_component_names()
    assert "pd_attitude" in registry, "the built-in was displaced"
    assert "python:pd_attitude" in registry


def test_unregistered_plugin_selection_names_the_slot():
    """The second stage of the two-stage check, in isolation."""
    from star_reacher.plugin import PluginError, check_plugin_selections

    resolved = {"gnc": {"control": {"component": "python:absent"}}}
    with pytest.raises(PluginError) as excinfo:
        check_plugin_selections(resolved, [])
    message = str(excinfo.value)
    assert "[gnc.control] component" in message
    assert "python:absent" in message
    assert "--gnc-plugin" in message


# --- end to end through the CLI --------------------------------------------


def test_cli_plugin_run_matches_the_builtin_run(tmp_path):
    """The FR-25 headline: a plugin flies a real mission, and it is right.

    The example plugin reimplements the built-in ``pd_attitude`` law, so the
    plugin mission and the reference mission are the same experiment run two
    ways. Comparing the commanded torque channel makes this a check on the
    plugin seam rather than a demonstration that some Python ran.
    """
    _core_or_fail()
    from star_reacher import load

    plugin_out = tmp_path / "plugin"
    builtin_out = tmp_path / "builtin"
    got = _run_cli(
        "run", str(PLUGIN_MISSION), "-o", str(plugin_out),
        "--gnc-plugin", str(EXAMPLE_PLUGIN), cwd=str(REPO_ROOT),
    )
    assert got.returncode == 0, got.stderr
    ref = _run_cli(
        "run", str(BUILTIN_MISSION), "-o", str(builtin_out), cwd=str(REPO_ROOT)
    )
    assert ref.returncode == 0, ref.stderr

    plugin_log = load(plugin_out / "run.srlog")
    builtin_log = load(builtin_out / "run.srlog")
    tau_plugin = plugin_log.groups["gnc.cmd"]["tau_b_nm"]
    tau_builtin = builtin_log.groups["gnc.cmd"]["tau_b_nm"]
    assert tau_plugin.shape == tau_builtin.shape
    # The mission opens with a 10-degree attitude error, so the commanded
    # torque is genuinely exercised rather than identically zero.
    assert abs(tau_builtin).max() > 1e-3
    worst = float(abs(tau_plugin - tau_builtin).max())
    assert worst == 0.0, (
        f"the plugin-flown control law departs from the built-in by "
        f"{worst:.3e} N*m; the two implement the same documented equations"
    )
    for field in builtin_log.groups["truth"].dtype.names:
        delta = abs(
            builtin_log.groups["truth"][field] - plugin_log.groups["truth"][field]
        ).max()
        assert delta == 0.0, f"truth channel {field} differs by {delta}"


def test_cli_without_the_flag_names_the_slot(tmp_path):
    """A mission needing a plugin must not run silently without one.

    The gate proved capable of failing: the same mission that passes above
    exits nonzero here, naming the chain slot and the flag that fixes it.
    """
    _core_or_fail()
    got = _run_cli(
        "run", str(PLUGIN_MISSION), "-o", str(tmp_path / "out"), cwd=str(REPO_ROOT)
    )
    assert got.returncode == 1
    assert "[gnc.control] component" in got.stderr
    assert "python:pd_attitude" in got.stderr
    assert "--gnc-plugin" in got.stderr


def test_cli_rejects_a_name_the_plugin_does_not_declare(tmp_path):
    """A well-formed name that no loaded plugin provides is still an error."""
    _core_or_fail()
    mission = tmp_path / "typo.toml"
    mission.write_text(
        _plugin_mission_text().replace(
            'component = "python:pd_attitude"', 'component = "python:pd_attitud"'
        ),
        encoding="utf-8",
    )
    got = _run_cli(
        "run", str(mission), "-o", str(tmp_path / "out"),
        "--gnc-plugin", str(EXAMPLE_PLUGIN), cwd=str(REPO_ROOT),
    )
    assert got.returncode == 1
    assert "python:pd_attitud" in got.stderr
    assert "loaded plugin components: python:pd_attitude" in got.stderr


def test_cli_prints_the_determinism_and_trust_notice(tmp_path):
    """The contract reaches a shell user, who may never read a docstring."""
    _core_or_fail()
    got = _run_cli(
        "run", str(PLUGIN_MISSION), "-o", str(tmp_path / "out"),
        "--gnc-plugin", str(EXAMPLE_PLUGIN), cwd=str(REPO_ROOT),
    )
    assert got.returncode == 0, got.stderr
    assert "deterministic time loop" in got.stderr
    assert "unseeded RNG" in got.stderr
    assert "privileges" in got.stderr


def test_cli_help_documents_the_trust_boundary():
    """`star run --help` states what loading a plugin actually does."""
    got = _run_cli("run", "--help")
    assert got.returncode == 0
    text = re.sub(r"\s+", " ", got.stdout)
    assert "--gnc-plugin" in text
    assert "SECURITY" in text
    assert "never fetched over a network" in text
    assert "DETERMINISM" in text


def test_meta_json_records_plugin_provenance(tmp_path):
    """config_sha256 does not cover plugin source, so meta.json must.

    Two revisions of a plugin flying one mission are different experiments
    that would otherwise be indistinguishable in the run artifacts.
    """
    _core_or_fail()
    import hashlib

    out = tmp_path / "prov"
    got = _run_cli(
        "run", str(PLUGIN_MISSION), "-o", str(out),
        "--gnc-plugin", str(EXAMPLE_PLUGIN), cwd=str(REPO_ROOT),
    )
    assert got.returncode == 0, got.stderr
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert len(meta["gnc_plugins"]) == 1
    entry = meta["gnc_plugins"][0]
    assert entry["path"].endswith("pd_attitude.py")
    assert entry["sha256"] == hashlib.sha256(EXAMPLE_PLUGIN.read_bytes()).hexdigest()


def test_a_run_without_plugins_records_an_empty_provenance_list(tmp_path):
    _core_or_fail()
    out = tmp_path / "noplug"
    got = _run_cli(
        "run", str(BUILTIN_MISSION), "-o", str(out), cwd=str(REPO_ROOT)
    )
    assert got.returncode == 0, got.stderr
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["gnc_plugins"] == []


# --- the external component under a batch run ------------------------------


def test_batch_cli_run_of_an_external_mission_flies_the_initial_hold(tmp_path):
    """`star run` on an 'external' mission: nobody commands, so it holds.

    The external component is the stepping-API seam, and a batch run has no
    driver to supply commands. The honest reading of that is the initial
    hold, and this asserts it on the CLI path rather than only in the C++
    tests: the run completes normally, commands zero torque with the hold
    flag clear for every cycle, and the attitude never moves.
    """
    _core_or_fail()
    from star_reacher import load

    text = BUILTIN_MISSION.read_text(encoding="utf-8")
    head, sep, tail = text.partition("[gnc.control]")
    assert sep, "reference mission no longer has a [gnc.control] table"
    rest = ""
    lines = tail.split("\n")
    for index, line in enumerate(lines):
        if line.startswith("[") and index > 0:
            rest = "\n".join(lines[index:])
            break
    mission = tmp_path / "external.toml"
    mission.write_text(
        head + '[gnc.control]\ncomponent = "external"\n\n' + rest, encoding="utf-8"
    )

    out = tmp_path / "ext"
    got = _run_cli("run", str(mission), "-o", str(out), cwd=str(REPO_ROOT))
    assert got.returncode == 0, got.stderr

    log = load(out / "run.srlog")
    tau = log.groups["gnc.cmd"]["tau_b_nm"]
    valid = log.groups["gnc.cmd"]["valid"]
    assert abs(tau).max() == 0.0, "an uncommanded external slot applied torque"
    assert set(valid.tolist()) == {0}, "a hold was logged as a live command"
    q = log.groups["truth"]["q_i2b"]
    assert abs(q - q[0]).max() == 0.0, "the attitude moved with no torque applied"
