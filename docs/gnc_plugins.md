# GNC in the loop: the `Sim` stepping API and `--gnc-plugin`

How to drive a mission one control period at a time (FR-24) and how to fly a
guidance, navigation, or control component you wrote in Python (FR-25).

Both surfaces sit on the same seam. The core's vehicle loop is a cycle object
advanced one control period per call, and a batch `star run` is literally a
loop over it — so a stepped run and a batch run of one scenario produce
byte-identical logs, and a Python component is called by the same registry the
built-in C++ components come from.

- [1. The stepping API](#1-the-stepping-api)
- [2. Writing a GNC plugin](#2-writing-a-gnc-plugin)
- [3. Naming a plugin from a mission file](#3-naming-a-plugin-from-a-mission-file)
- [4. Running it](#4-running-it)
- [5. The determinism contract](#5-the-determinism-contract)
- [6. The trust boundary](#6-the-trust-boundary)
- [7. The privileged-truth boundary](#7-the-privileged-truth-boundary)

## 1. The stepping API

```python
from star_reacher.sim import Sim

with Sim("missions/leo_attitude_gnc.toml", "out/stepped") as sim:
    obs, info = sim.reset()
    while not sim.done():
        obs = sim.step()
    print(sim.summary())
```

| Call | Returns | Notes |
| --- | --- | --- |
| `reset(seed=None, overrides=None)` | `(obs, info)` | Opens a fresh run and writes the log header. `info` carries `config_sha256`, `srlog_path`, `seed`, `duration_s`, `has_external_command`. |
| `step(commands=None)` | `obs` | Advances exactly one control period (D-5). |
| `observe()` | `obs` | The stored snapshot of the most recent cycle. Pure. |
| `truth()` | `dict` | **Privileged.** The true state at the instant `observe()` describes. |
| `time()` / `cycle()` | `float` / `int` | The next cycle to be processed. |
| `done()` | `bool` | True once the run ended and the log is closed. |
| `summary()` | `dict` | Final state and record tallies; valid once `done()`. |
| `run_to_completion()` | `dict` | Steps to the end and returns the summary. |
| `close()` | `None` | Releases the log handle. Idempotent, and called by `__exit__`. |

`observe()` is idempotent: it runs no component, draws no random number,
consumes no sensor sample, and returns fresh dicts rather than views into core
buffers, so two calls without an intervening `step()` are equal. `truth()` is
a deliberately separate call, so an observation handed to an agent can never
contain truth by accident.

### Lifetime

Use the context-manager form. A run that ends normally closes its own log,
but a run abandoned part way — a driver that stops early, or any exception
escaping `step()` — holds `run.srlog` open for as long as the underlying core
object lives, and how long that is depends on refcount timing and on whether
a traceback still pins the frame holding it. On Windows the open handle then
makes a later unlink of the output directory, or a reopen of the same path,
fail with `PermissionError: [WinError 32]`; on Linux the unlink silently
succeeds. `close()` makes the file's lifetime something the driver states
rather than something it infers. Stepping after `close()` raises; the log of a
run closed early is a valid prefix, carrying no `run_end` event.

`reset()` may be called repeatedly on one `Sim` — the FR-24 episode loop —
at the default `force=False`. The `force` guard protects an output this `Sim`
did not write; once it has opened that path itself, a later `reset()`
overwrites its own previous run, so each episode leaves only the last
episode's `run.srlog`. A driver that needs one log per episode constructs a
`Sim` per output directory.

### Commanding a run

`step(commands)` addresses the `external` component, which a mission selects
in its guidance or control slot:

```toml
[gnc.control]
component = "external"
```

```python
sim.step({"torque_b_nm": [0.0, 0.0, 0.02]})   # command
sim.step()                                     # hold: the command persists
```

`commands` accepts `torque_b_nm`, `omega_b_radps`, `q_i2b` (scalar-first
`(w, x, y, z)`), and `valid`. **Unknown keys raise** — a silently dropped
command is indistinguishable from a vehicle that refused to manoeuvre.
**Missing keys hold** (D-5 zero-order hold) **and are logged**: `gnc.cmd`
carries the command as applied on every control cycle, so a held field is
written out again rather than leaving a gap the reader must guess at.

Commanding a mission whose slots are all autonomous raises. A batch
`star run` of an `external` mission is legal and flies the initial hold —
zero torque, the activation attitude, the valid flag clear — which is the
honest reading of "nobody is commanding".

### `reset(overrides=...)`

An override key is a dotted path into the resolved mission; an integer
segment indexes a list:

```python
sim.reset(overrides={
    "mission.duration_s": 120.0,
    "gnc.control.kp_nm_per_rad": [0.5, 0.5, 0.5],
    "sequence.0.t_s": 3.0,
})
```

`duration_s` and `latency_cycles` remain accepted as bare shorthands.
Overrides are applied **before** the configuration is hashed, so an overridden
run carries its own `config_sha256` and is individually reproducible.

What is refused, and why:

- **A path the mission does not already set.** Inventing a key would produce a
  configuration the validator never saw.
- **Strings and whole tables.** Those select *structure* — a component name, a
  frame, a file path — whose consequences the mission validator checked and
  this path cannot recheck.
- **A value that does not match the leaf in kind and length.** An integer leaf
  takes an integer, so a count cannot silently become fractional and change
  the canonical config bytes without changing the run.

Numeric **range** is not rechecked. The core's construction checks are the
backstop and fail with a named reason, so an override can produce a loud
failure but never a run whose configuration was never validated at all.

## 2. Writing a GNC plugin

A plugin is a Python file declaring a module-level `STAR_GNC_COMPONENTS`
mapping bare names to factories. A factory is called with the slot's
`GncComponentCfg` and returns an `IGncComponent` subclass instance — a class
with a one-argument `__init__` is such a callable.

```python
from star_reacher.sim import GncOutput, IGncComponent


class MyControl(IGncComponent):
    def __init__(self, cfg):
        super().__init__()               # required by the pybind11 trampoline
        self.kp = [float(x) for x in cfg.vectors["kp_nm_per_rad"]]

    def init(self, ctx):
        """One-time setup. ctx carries t0, q0, the pad basis when the mission
        is a geodetic launch, the central-body constants, and the run's
        configured sensor model."""

    def update(self, inp):
        """Called once per control cycle, in the fixed order nav -> guidance
        -> control. Return a GncOutput; valid = False means hold."""
        out = GncOutput()
        out.valid = True
        out.torque_b_nm = [0.0, 0.0, 0.0]
        return out


STAR_GNC_COMPONENTS = {"my_control": MyControl}
```

An estimator additionally overrides `state_dim()` and `state()`, and may
override `cov_dim()` / `covariance_upper()`, `innov_max_dim()` /
`innovations()`, and `error_layout()`. Those hooks are what let the loop log
`nav.est`, `nav.err`, and `nav.innov` generically for any estimator
dimension. A wrong-length return raises rather than being silently truncated.

`state_dim()`, `cov_dim()`, and `innov_max_dim()` are **constants of a run**.
Each is queried once, at GNC activation; the loop sizes its fixed log buffers
from what it got, and the log header records the same values as the file's
fixed record strides. Nothing resizes either afterwards, so an estimator that
means to augment its state declares the augmented dimension up front and
zero-pads until it is populated. This is enforced rather than requested: the
first value each method returns is pinned, and a later divergence raises. So
does a negative dimension, an `InnovationSample` whose `y` is longer than the
declared `innov_max_dim()`, and an `s_upper` that is not exactly
`m(m+1)/2` entries for its own `y`. Every one of those was a heap write or
read past the end of a buffer sized at construction, reachable from pure
Python with no unsafe API.

`error_layout()` is how an estimator earns a `nav.err` channel. It returns a
list of `ErrorBlock`, each naming a truth quantity, the form of the
difference, and the block's first slot in the state vector:

```python
def error_layout(self):
    return [
        ErrorBlock(ErrorQuantity.ATTITUDE, ErrorForm.QUAT_ERROR_LOCAL, 0),
        ErrorBlock(ErrorQuantity.VELOCITY, ErrorForm.DIFFERENCE, 4),
        ErrorBlock(ErrorQuantity.POSITION, ErrorForm.DIFFERENCE, 7),
    ]
```

The blocks must tile `[0, state_dim())` exactly — no gaps, no overlaps —
because a gap would be logged as zero, and zero in an error channel reads as
"no error" rather than "not known". A partial layout raises at construction;
declaring none at all is fine and simply means the run writes no `nav.err`.

`ErrorForm` exists for attitude. An attitude error is a rotation difference
rather than a subtraction, and which side it is composed on and how it is
parameterized are the estimator's own convention, so the descriptor carries
`QUAT_ERROR_LOCAL` (`dq = conj(q_est) ⊗ q_true`, resolved in the estimated
body frame), `QUAT_ERROR_GLOBAL` (`dq = q_true ⊗ conj(q_est)`, resolved in
the inertial frame), and `QUAT_DIFFERENCE_ALIGNED` for an estimator that
treats the four quaternion components as ordinary state entries.
Quaternion forms are sign-canonicalized to the `+w` hemisphere so the double
cover cannot flip the logged error between neighbouring epochs. Quaternions
are scalar-first (D-7).

Every attitude form occupies **four** slots, so a block's error width always
equals its state width. That is what lets one number — `error_block_size` —
both tile the state vector during validation and size the write into
`nav.err`.

### At `state_dim() == cov_dim() + 1` the attitude block must come first

If your estimator declares a state one slot wider than its covariance — the
shape a four-slot quaternion state with a three-slot attitude error produces —
then the block at offset 0 must be the attitude block. `validate_error_layout`
refuses any other arrangement at run construction, naming the two dimensions
and the block that holds offset 0.

The rule exists because of what happens downstream. `star consistency` pairs
an `n`-dimensional `nav.err` with the `m`-dimensional covariance `nav.est`
reports, and at `n == m + 1` (with `n >= 4`) it collapses slots 0..3 as a
scalar-first error quaternion, `dtheta = 2 sgn(dq_w) dq_v`. The SRLOG header
records only whether a layout is present, not the layout itself, so the
consumer cannot verify that slots 0..3 really are a quaternion. A layout such
as `[VELOCITY(3), ATTITUDE(4)]` against `m == 6` reaches that collapse with
three velocity-error components and one quaternion component in those slots;
they are reduced as though they were a rotation, and the resulting NEES is
positive, order-unity, and wrong. Refusing the layout at construction is what
keeps such a log from being written in the first place.

Two ways out if your state is genuinely not quaternion-led. Reorder the layout
so the attitude block leads, which costs nothing but the declaration order.
Or declare no layout at all: a component that returns an empty `error_layout()`
writes no `nav.err` channel, so nothing is reduced and `star consistency`
simply has no error state to evaluate. The rule does not apply when
`n == m`, which is the shape the built-in `dead_reckoning` navigator presents.

### Three-parameter attitude states are not supported

An estimator whose state carries a three-parameter attitude directly — MRP,
Gibbs/Rodrigues, or a rotation-vector error state — has no admissible form
here, and this is a real gap rather than an oversight in the enumeration.
Every attitude form reads `q_est` as four consecutive state slots at the
block's offset, and a three-parameter state does not publish one. Serving
that case requires a way for a component to supply its estimated quaternion
independently of its state layout, which is a change to the descriptor
rather than an added enumerator.

A pair of three-slot `ROTATION_VECTOR_LOCAL` / `ROTATION_VECTOR_GLOBAL`
forms was removed for appearing to serve this case while not doing so. They
declared three slots but were read as four, so an estimator that placed its
attitude block last could pass `validate_error_layout` and then have its
error state read one `double` past the state buffer. The reduction itself is
not lost: the consistency evaluator already applies
`dtheta = 2 sgn(dq_w) dq_v` downstream, to exactly the `n = 16` / `m = 15`
case the built-in EKF presents ([`docs/formats/srlog_v1.md`](formats/srlog_v1.md)).
If you need a three-parameter attitude state, the descriptor must grow —
re-adding the enumerators would restore the out-of-bounds read without
serving the case.

A worked example — the built-in `pd_attitude` law reimplemented in Python and
validated against it — is at
[`examples/gnc_plugins/pd_attitude.py`](../examples/gnc_plugins/pd_attitude.py).

## 3. Naming a plugin from a mission file

A mission selects a plugin component in the reserved `python:` namespace, in
any of the three chain slots:

```toml
[gnc.control]
component = "python:my_control"
kp_nm_per_rad = [0.4, 0.4, 0.4]
```

The prefix earns its place three times over:

- **The validator stays strict.** Mission validation works without a compiled
  core and without the plugin, so it cannot check a plugin name against a
  registry. It checks the grammar instead — and because an unprefixed name is
  still matched against the built-in vocabulary, a misspelt built-in like
  `dead_reckonning` remains a hard error rather than being waved through as
  "probably a plugin". That the name matches something the plugin really
  declares is checked once the file is loaded, naming the slot if it does not.
- **A plugin cannot shadow a built-in.** `python:pd_attitude` and
  `pd_attitude` are different registry keys. Which one flies is decided by the
  mission file, visibly, and never by load order.
- **The mission states its own dependency.** A reader can see the run needs a
  plugin, and running without one fails naming the slot.

Parameter **keys** in a plugin slot are the plugin's contract, not the
validator's — it cannot know them. Their **values** are still held to the rule
`GncComponentCfg` imposes: a finite number, or a non-empty array of finite
numbers. A parameter the plugin does not recognize is the plugin's to reject.

## 4. Running it

```console
$ star run missions/leo_attitude_gnc_plugin.toml \
      --gnc-plugin examples/gnc_plugins/pd_attitude.py
```

`--gnc-plugin` is repeatable, so a navigation plugin and a control plugin may
come from separate files. Two files declaring one name is an error, because
the flown component would otherwise depend on flag order.

`meta.json` records each plugin file's path and SHA-256. The resolved-config
hash covers the mission, not the plugin source, so without that record two
runs of one mission against two revisions of a plugin would be
indistinguishable in their artifacts.

The same mission is steppable:

```python
Sim(mission, outdir, gnc_plugins=["examples/gnc_plugins/pd_attitude.py"])
```

## 5. The determinism contract

A plugin component runs **inside** the deterministic time loop. The core's
guarantee — same inputs on the same binary give bit-identical outputs — then
holds only as far as the plugin's Python does. The core cannot enforce this,
so it is a contract, restated by the CLI whenever `--gnc-plugin` is used:

- **No clock.** The core never reads wall time and the log carries no host
  data; a component that does breaks reproducibility and leaks host state.
- **No I/O and no network.**
- **No unseeded randomness.** Seed a private generator from data the run
  already fixes; never draw from a global generator another component could
  also advance.
- **No iteration over `set` or `frozenset`.** String hash values vary per
  process unless `PYTHONHASHSEED` is fixed. Sort first, or use a list.
- **No mutable global state** between components or across runs, and no
  dependence on garbage-collection timing or `id()`.

Arithmetic is fine: Python floats are IEEE-754 doubles and NumPy `float64`
operations are deterministic for a fixed library version. Reductions over very
large arrays may differ between NumPy builds that change the pairwise
summation order.

What *is* guaranteed regardless: the core calls a component exactly once per
stage per control cycle, in the fixed order nav → guidance → control, with
inputs that depend only on the configuration and the seed.

## 6. The trust boundary

**Loading a plugin executes that file's code with the full privileges of the
process running `star`.** That is inherent in the requirement — a plugin is a
program, not data — and nothing here sandboxes it.

What is guaranteed is that this is the *only* path by which a run executes a
file the mission did not already name:

- plugins load only from paths given explicitly on the command line;
- they are never fetched over a network;
- they are never discovered by scanning the working directory.

A mission TOML alone can therefore never cause code execution. Naming
`python:something` in a mission produces an *error* without the flag, never an
implicit search. It takes a second, explicit act by the person running the
command.

## 7. The privileged-truth boundary

Truth does not appear in `GncInput`. A component sees the true state only
through `GncInput.oracle`, and only when the scenario sets `oracle = true` —
a debug flag stamped into the log header, so a run that used it cannot be
mistaken for one that did not. `Sim.truth()` is a separate, privileged
accessor available to a stepping driver and to no component.

The guarantee is **structural**: no method of `IGncComponent` takes the truth
state as an argument, so there is no override through which a plugin can
receive it. That is worth stating precisely, because it was not always so.
The interface used to expose `error_state(truth, e)`, which the loop called
for every estimator on every cycle regardless of the oracle flag so the
estimator could compute `nav.err` in its own state convention. An
implementation was told not to retain the argument and nothing enforced it,
which made FR-24's "never visible to GNC plugins" a promise rather than a
property.

`nav.err` is now computed by the loop. A component declares the layout of its
state vector through `error_layout()` (section 2) and the loop differences
that layout against truth itself, using the state vector the component
already publishes through `state()`. What crosses the plugin boundary is a
description of the state vector, which carries no information about the
world, and `nav.err` survives for plugin estimators — which is why this route
was taken rather than gating the old call on the oracle flag.

What holds this claim up is an adversarial test rather than a reading of the
loop:
[`tests/python/test_gnc_python_component.py::test_truth_is_unreachable_from_a_python_component`](../tests/python/test_gnc_python_component.py)
drives a component that scrapes every float reachable through the factory
argument, the init context, itself, its garbage-collector referrers, the
per-cycle input, and every argument of every virtual it can override, and
requires the harvest to contain none of the evolving true state read from
`Sim.truth()` on the same cycles.

### What this does not cover

A plugin runs in-process with the full privileges of the process (section 6),
so what is guaranteed is that the *interface* hands over no truth — not that
arbitrary Python cannot reach it by other means. Nothing here sandboxes a
plugin, and the determinism contract has the same character. One such route
is concrete enough to name rather than leave to the imagination:

> **Stack walking under a Python stepping driver.** When a run is driven from
> Python through `Sim`, the driver's own `Sim` handle is a local on the
> interpreter stack. A component that walks frames with `sys._getframe` finds
> it and can call `Sim.truth()`, obtaining the full evolving true state
> regardless of the oracle flag. Under the batch path (`star run`) the loop is
> owned by the core, no `Sim` is on the Python stack, and the route finds
> nothing.

This differs in kind from the `error_state(truth)` channel described above.
That one was unconditional — the loop pushed truth to every estimator on
every cycle of every run, batch or stepped, with no action by the plugin
author, so an ordinary implementation held truth whether or not it wanted
it. This one takes a Python stepping driver plus a plugin that deliberately
introspects the interpreter. Closing it would take sandboxing, which this
project does not attempt. It is pinned by
`test_stack_walking_reaches_truth_under_a_python_stepping_driver` so the
current behaviour is a recorded fact rather than an assumption.

Finally, a plugin estimator remains free to be dishonest about its own state;
an implausibly good `nav.err` is what a reviewer should look at.
