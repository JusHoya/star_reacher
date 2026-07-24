"""Numeric override application over a resolved mission (FR-24/FR-27).

An override replaces a numeric leaf of an already validated, resolved mission
with a new value, in a deep copy, *before* the configuration is hashed, so an
overridden run carries its own ``config_sha256`` and is individually
reproducible. Two callers share this vocabulary:

* the FR-24 stepping API (``Sim.reset(seed, overrides)``), where a driver
  reaches for the two aliases below most; and
* the FR-27 Monte Carlo engine (``star mc``, ``star run --seed/--set``),
  where each sweep case is one ``{dotted.path: value}`` dict recorded verbatim
  in the run manifest.

Housing the mechanism here rather than in either caller is what keeps a
manifest's per-run ``overrides`` entry and a ``star run --set`` re-execution
of it the *same* transformation of the *same* resolved mission, which is the
property Phase 7 exit criterion 1 rests on: re-running a manifest entry with
its recorded seed and overrides must reproduce the entry's logged SHA-256.

A key is a dotted path into the resolved mission, with an integer segment
indexing a list::

    apply_override(resolved, "mission.duration_s", 120.0)
    apply_override(resolved, "gnc.control.kp_nm_per_rad", [0.5, 0.5, 0.5])
    apply_override(resolved, "sequence.0.t_s", 3.0)

What is refused, and why. The path must already exist: inventing a key would
produce a configuration the validator never saw. The existing value must be a
number or an array of numbers, and the new value must match it in kind and
length -- an integer leaf takes an integer, so a control rate cannot silently
become fractional. Strings and whole tables are not overridable, because they
select *structure* (a component name, a frame, a file path) whose consequences
the mission validator checked and this path cannot recheck.

Numeric RANGE is not rechecked here; the core's defensive construction checks
are the backstop and raise with a named reason. An override is therefore
capable of failing the run loudly, never of producing a run whose
configuration was never validated at all.
"""

from __future__ import annotations

from star_reacher.mission import canonical_bytes

__all__ = [
    "OVERRIDE_ALIASES",
    "OverrideError",
    "apply_override",
    "deep_copy_resolved",
    "override_target",
]


class OverrideError(ValueError):
    """An override that names no existing numeric leaf, or the wrong kind.

    A ``ValueError`` subclass because a bad override is a validation-shaped
    input error: it is caught at the CLI boundary and reported with exit code
    2, the same code the mission validator uses. ``Sim.reset`` re-raises it as
    ``SimError`` so the stepping API's public error type is unchanged.
    """


# The two paths a stepping driver overrides most, kept as bare names because
# they were the whole accepted vocabulary before dotted paths existed and are
# still the two a hand-written driver reaches for.
OVERRIDE_ALIASES = {
    "duration_s": "mission.duration_s",
    "latency_cycles": "gnc.latency_cycles",
}


def _is_number(value) -> bool:
    # TOML/JSON booleans are ints in Python; they are never a numeric leaf.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def override_target(resolved, path):
    """Walk a dotted override path to its ``(container, key)`` slot.

    Raises :class:`OverrideError` naming the failing segment rather than the
    whole path, because on a long path the segment is the actionable part.
    """
    segments = path.split(".")
    node = resolved
    for depth, segment in enumerate(segments[:-1]):
        walked = ".".join(segments[: depth + 1])
        if isinstance(node, list):
            if not segment.isdigit() or int(segment) >= len(node):
                raise OverrideError(
                    f"override path {path!r}: {walked!r} does not index the "
                    f"list at {'.'.join(segments[:depth]) or 'the mission'} "
                    f"(length {len(node)})"
                )
            node = node[int(segment)]
        elif isinstance(node, dict):
            if segment not in node:
                raise OverrideError(
                    f"override path {path!r}: the resolved mission has no "
                    f"{walked!r}"
                )
            node = node[segment]
        else:
            raise OverrideError(
                f"override path {path!r}: {walked!r} is not a table or an "
                f"array, so it has no members to override"
            )
    leaf = segments[-1]
    if isinstance(node, list):
        if not leaf.isdigit() or int(leaf) >= len(node):
            raise OverrideError(
                f"override path {path!r}: {leaf!r} does not index a list of "
                f"length {len(node)}"
            )
        return node, int(leaf)
    if not isinstance(node, dict) or leaf not in node:
        raise OverrideError(
            f"override path {path!r}: the resolved mission has no such key; "
            f"an override may only change a value the mission already sets, "
            f"because a key the validator never saw would not have been "
            f"checked"
        )
    return node, leaf


def apply_override(resolved, path, value):
    """Apply one numeric override in place, preserving the leaf's kind."""
    container, key = override_target(resolved, path)
    current = container[key]

    if _is_number(current):
        if not _is_number(value):
            raise OverrideError(
                f"override {path!r}: expected a number to replace "
                f"{current!r}, got {type(value).__name__}"
            )
        # An integer leaf keeps its type: control_rate_hz, latency_cycles and
        # seed are counts, and letting one become 10.0 would change the
        # canonical config bytes - and so the config hash - without changing
        # the run. A fractional value is REFUSED rather than truncated: the
        # truncated run is perfectly reproducible and is not the run the
        # driver asked for, which is the worst of both.
        if isinstance(current, int):
            # A Python int is integral by definition, so it never truncates -
            # and it must be admitted WITHOUT the float() round trip below,
            # which would lose precision on a full-width u64 (the per-run seed
            # SplitMix64 produces) and misreport an exact integer as
            # fractional. Only a float value can carry a fractional part.
            if isinstance(value, float) and not value.is_integer():
                raise OverrideError(
                    f"override {path!r}: {current!r} is an integer count, so "
                    f"it takes an integer; {value!r} would be truncated to "
                    f"{int(value)}"
                )
            container[key] = int(value)
        else:
            container[key] = float(value)
        return

    if isinstance(current, list) and current and all(_is_number(v) for v in current):
        if not isinstance(value, (list, tuple)) or len(value) != len(current):
            raise OverrideError(
                f"override {path!r}: expected an array of {len(current)} "
                f"number(s) to replace {current!r}, got {value!r}"
            )
        if not all(_is_number(v) for v in value):
            raise OverrideError(
                f"override {path!r}: every element must be a number, got "
                f"{value!r}"
            )
        # Same rule as the scalar branch, element by element: only a float
        # element with a fractional part truncates; an int element is already
        # integral and must not pass through float() (precision loss on a
        # full-width u64 would misreport it as fractional).
        fractional = [
            v for c, v in zip(current, value)
            if isinstance(c, int) and isinstance(v, float) and not v.is_integer()
        ]
        if fractional:
            raise OverrideError(
                f"override {path!r}: {current!r} is an array of integer "
                f"counts, so every element takes an integer; {fractional!r} "
                f"would be truncated"
            )
        container[key] = [
            int(v) if isinstance(c, int) else float(v)
            for c, v in zip(current, value)
        ]
        return

    raise OverrideError(
        f"override {path!r}: only numbers and arrays of numbers are "
        f"overridable, and this key currently holds {current!r}. A string or "
        f"a table selects structure the mission validator checked, and "
        f"changing it here would skip that check"
    )


def deep_copy_resolved(resolved):
    """Copy a resolved mission deeply enough for override application.

    ``copy.deepcopy`` would work, but the resolved config is plain JSON-able
    data by construction (it is hashed through ``canonical_bytes``), so a
    round trip through the same canonical representation is both sufficient
    and a check that the assumption still holds.
    """
    import json

    return json.loads(canonical_bytes(resolved).decode("utf-8"))


def apply_overrides(resolved, overrides, *, aliases=True):
    """Apply a dict of overrides to a copy of ``resolved`` and return it.

    The single call the seed/override plumbing in ``runner`` and ``mc`` uses:
    it deep-copies first so the caller's resolved mission is never mutated,
    resolves the two aliases when ``aliases`` is set (the stepping-API
    shorthands, which the ``--set`` and sweep vocabularies also honour), and
    applies each entry in dict order. Order is immaterial to the result -- an
    override targets one leaf and two overrides of the same leaf are a caller
    error, not a documented last-wins -- but is deterministic regardless.
    """
    out = deep_copy_resolved(resolved)
    for key, value in dict(overrides or {}).items():
        path = OVERRIDE_ALIASES.get(key, key) if aliases else key
        apply_override(out, path, value)
    return out
