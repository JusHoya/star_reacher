"""FR-25 ``--gnc-plugin``: load a Python-authored GNC component from a file.

FR-25 requires that ``star run mission.toml --gnc-plugin my_nav.py`` swap a
component into the chain with zero recompilation. The pieces that makes
possible are the pybind11 trampoline over ``IGncComponent`` and the core's
name-keyed component registry (``star/gnc/component.hpp``); this module is the
loader that joins a file path on the command line to a component name in a
mission file.

Naming: the ``python:`` namespace
---------------------------------

A mission selects a plugin component the same way it selects a built-in --
by name -- but in a reserved namespace::

    [gnc.control]
    component = "python:my_pd"
    kp_nm_per_rad = 40.0

The prefix is load-bearing three times over:

* **The validator stays strict.** ``star_reacher.mission`` must validate a
  mission without a compiled core and without the plugin, so it cannot check a
  plugin name against a registry. It checks the *grammar* instead: a name
  without the prefix is still matched against the built-in vocabulary, so
  ``dead_reckonning`` remains a hard error rather than being waved through as
  "probably a plugin". The plugin's own names are then checked against what
  the plugin actually declared, here, once it is loaded.
* **A plugin cannot shadow a built-in.** ``python:pd_attitude`` and
  ``pd_attitude`` are different registry keys, so no plugin can quietly
  displace the C++ component a mission meant to select -- the shadowing
  hazard ``register_component`` already refuses for built-ins.
* **The mission file states its own dependency.** A reader of the TOML can
  see that the run needs a plugin, and a run that omits ``--gnc-plugin``
  fails naming the slot rather than reporting "unknown component".

Any of the three chain slots may be a plugin -- ``[gnc.nav]``,
``[gnc.guidance]``, ``[gnc.control]``. FR-25 defines exactly one abstract
base for all three roles, and the registry is slot-agnostic, so restricting
the flag to one slot would be an invention of this loader rather than a
property of the interface.

The plugin file's contract
--------------------------

A plugin module declares a module-level ``STAR_GNC_COMPONENTS`` mapping bare
names to factories::

    from star_reacher.sim import IGncComponent, GncOutput

    class MyPd(IGncComponent):
        def __init__(self, cfg):
            self.kp = cfg.scalars["kp_nm_per_rad"]
        def init(self, ctx): ...
        def update(self, inp): ...

    STAR_GNC_COMPONENTS = {"my_pd": MyPd}

A factory is called with the slot's ``GncComponentCfg`` and must return an
``IGncComponent`` subclass instance; a class object with a one-argument
``__init__`` is such a callable. The declaration is explicit rather than
discovered by scanning the module for subclasses: which names a file exports
is then a decision its author wrote down, not an artifact of what happened to
be imported into it.

Trust boundary
--------------

**Loading a plugin executes the file's code with the full privileges of the
process running ``star``.** That is inherent in the requirement -- a plugin is
a program, not data -- so the loader does not pretend to sandbox it. What the
loader does guarantee is that this is the *only* code path by which a run
executes a file the mission did not already name: plugins are loaded only
from paths given explicitly on the command line, never fetched over a
network, and never discovered by scanning the working directory. A mission
TOML alone can therefore never cause code execution; it takes a second,
explicit act by the person running the command.

Determinism
-----------

A plugin component runs *inside* the deterministic time loop, so D-10 holds
only as far as the plugin's Python does. The full contract is in
``star_reacher.sim``; :data:`DETERMINISM_NOTICE` restates its rules in the
form the CLI prints, because someone loading a plugin from a shell may never
read a docstring.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

__all__ = [
    "DETERMINISM_NOTICE",
    "PLUGIN_PREFIX",
    "PluginError",
    "load_plugins",
    "plugin_selections",
    "check_plugin_selections",
]

# The reserved mission-file namespace for plugin components. Mirrored by
# star_reacher.mission, which validates the grammar core-less; the two are
# tied together by test_gnc_plugin.py.
PLUGIN_PREFIX = "python:"

# The bare name after the prefix. Deliberately identifier-shaped: the name is
# a registry key, appears in error messages, and is round-tripped through the
# resolved config's canonical JSON, so anything requiring quoting or escaping
# would buy nothing and cost clarity.
_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# The module attribute a plugin file declares.
_DECL_ATTR = "STAR_GNC_COMPONENTS"

DETERMINISM_NOTICE = """\
A GNC plugin runs inside the deterministic time loop, so this run's
reproducibility (D-10) holds only as far as the plugin's own code does. A
plugin component must not read the clock, perform file or network I/O, draw
from an unseeded RNG, iterate a set or frozenset, or carry mutable global
state between cycles or runs. The core cannot enforce this; it is a contract.
Loading a plugin also executes its code with this process's privileges."""

# Modules already loaded in this process, keyed by resolved path. Loading the
# same plugin twice must not re-register its names (the core refuses
# duplicates by design), and re-executing the file would additionally leave
# two distinct class objects answering to one name. Caching makes a repeated
# load a no-op with the first module's classes still bound - the only
# behaviour that keeps one process's runs consistent with each other.
_loaded_modules: dict[str, object] = {}
# Registered plugin component name -> the resolved path that declared it.
_registered: dict[str, str] = {}


class PluginError(RuntimeError):
    """A GNC plugin could not be loaded, or does not satisfy its contract."""


def _module_name(path: Path) -> str:
    """A unique, import-system-safe module name for a plugin file.

    Plugins are loaded by path and are not importable packages, so the name
    exists only to key ``sys.modules``. Deriving it from the resolved path
    keeps two same-named files in different directories distinct.
    """
    digest = "".join(ch if ch.isalnum() else "_" for ch in str(path))
    return f"star_reacher._gnc_plugin.{digest}"


def _load_module(path: Path):
    resolved = str(path.resolve())
    cached = _loaded_modules.get(resolved)
    if cached is not None:
        return cached

    if not path.is_file():
        raise PluginError(f"{path}: GNC plugin file not found")

    spec = importlib.util.spec_from_file_location(_module_name(path), path)
    if spec is None or spec.loader is None:
        raise PluginError(
            f"{path}: not an importable Python module (a GNC plugin must be a "
            f".py source file)"
        )
    module = importlib.util.module_from_spec(spec)
    # Registered before execution so a plugin that imports itself, or that a
    # traceback must name, resolves; removed again if execution fails so a
    # half-initialized module is never reachable.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException as exc:
        sys.modules.pop(spec.name, None)
        raise PluginError(
            f"{path}: GNC plugin raised while being imported: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    _loaded_modules[resolved] = module
    return module


def _declared_components(module, path: Path) -> dict:
    decl = getattr(module, _DECL_ATTR, None)
    if decl is None:
        raise PluginError(
            f"{path}: GNC plugin declares no {_DECL_ATTR}; add a module-level "
            f'{_DECL_ATTR} = {{"my_component": MyComponent}} mapping each name '
            f"a mission may select (as \"{PLUGIN_PREFIX}my_component\") to a "
            f"factory returning an IGncComponent subclass"
        )
    if not isinstance(decl, dict):
        raise PluginError(
            f"{path}: {_DECL_ATTR} must be a dict of name -> factory, got "
            f"{type(decl).__name__}"
        )
    if not decl:
        raise PluginError(
            f"{path}: {_DECL_ATTR} is empty; a GNC plugin must declare at "
            f"least one component"
        )
    for name, factory in decl.items():
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
            raise PluginError(
                f"{path}: {_DECL_ATTR} key {name!r} is not a valid component "
                f"name; names must match {_NAME_RE.pattern} and are selected "
                f'in a mission as "{PLUGIN_PREFIX}<name>"'
            )
        if not callable(factory):
            raise PluginError(
                f"{path}: {_DECL_ATTR}[{name!r}] is not callable; it must be "
                f"callable as factory(cfg) -> IGncComponent (a class is such "
                f"a callable)"
            )
    return dict(decl)


def load_plugins(paths, core) -> list[str]:
    """Load GNC plugin files and register their components with ``core``.

    Returns the registered names in ``python:``-prefixed form, sorted. Raises
    :class:`PluginError` naming the file for every contract violation, and for
    two different files declaring the same name -- a silent shadowing there
    would make which component flew depend on flag order.
    """
    registered: list[str] = []
    for raw in paths or ():
        path = Path(raw)
        module = _load_module(path)
        resolved_path = str(path.resolve())
        for name, factory in _declared_components(module, path).items():
            full = f"{PLUGIN_PREFIX}{name}"
            owner = _registered.get(full)
            if owner is not None:
                if owner != resolved_path:
                    raise PluginError(
                        f"{path}: GNC component '{full}' is already provided "
                        f"by {owner}; two plugins declaring one name would "
                        f"make the flown component depend on flag order"
                    )
                # Same file loaded twice in one process: already registered.
                registered.append(full)
                continue
            core.register_python_component(full, factory)
            _registered[full] = resolved_path
            registered.append(full)
    return sorted(set(registered))


def plugin_selections(resolved: dict) -> list[tuple[str, str]]:
    """The ``python:`` components a resolved mission selects.

    Returns ``(slot_path, component_name)`` pairs in chain order, e.g.
    ``[("gnc.control", "python:my_pd")]``. Empty for a mission with no
    ``[gnc]`` table or none that names a plugin.
    """
    gnc = resolved.get("gnc")
    if not isinstance(gnc, dict):
        return []
    out: list[tuple[str, str]] = []
    for slot in ("nav", "guidance", "control"):
        spec = gnc.get(slot)
        if not isinstance(spec, dict):
            continue
        name = spec.get("component")
        if isinstance(name, str) and name.startswith(PLUGIN_PREFIX):
            out.append((f"gnc.{slot}", name))
    return out


def check_plugin_selections(resolved: dict, registered) -> None:
    """Fail unless every plugin the mission selects was actually loaded.

    This is the second half of the two-stage check the ``python:`` namespace
    buys: the validator proved the name well-formed core-less, and this proves
    it real. Raises :class:`PluginError` naming the slot and the name, because
    the useful question for the reader is which chain slot has no component,
    not which registry key is missing.
    """
    have = set(registered)
    for slot, name in plugin_selections(resolved):
        if name in have:
            continue
        if have:
            offer = "loaded plugin components: " + ", ".join(sorted(have))
        else:
            offer = "no GNC plugin was loaded"
        raise PluginError(
            f"[{slot}] component: '{name}' was not provided by any loaded "
            f"plugin; pass the file declaring it with "
            f"--gnc-plugin <path.py> ({offer})"
        )
