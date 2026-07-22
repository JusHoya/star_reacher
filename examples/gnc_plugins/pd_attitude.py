"""A GNC plugin: the built-in ``pd_attitude`` control law, written in Python.

Run the committed reference attitude mission with this file swapped into the
control slot::

    star run missions/leo_attitude_gnc_plugin.toml \\
        --gnc-plugin examples/gnc_plugins/pd_attitude.py

and compare it against the C++ built-in flying the same scenario::

    star run missions/leo_attitude_gnc.toml

The two runs command identical torques. That is the point of the example: it
is a control law you can read, edit, and re-fly with no compiler in the loop
(FR-25), validated against a reference whose answer is already known.

Naming. The mission selects this component as ``python:pd_attitude`` -- the
same bare name as the C++ built-in, deliberately. The ``python:`` namespace is
disjoint from the built-in registry, so a plugin can reuse a built-in's name
without any risk of displacing it; which one flies is decided by the mission
file, visibly, and never by load order.

Arithmetic. The sequence below is the normative one documented in
``cpp/include/star/gnc/builtin.hpp`` (eq:gnc:deltaq, eq:gnc:sign, eq:gnc:werr,
eq:gnc:pd, eq:gnc:sat), evaluated per axis exactly as written with no
renormalization of the error quaternion. The quaternion product and the
direction-cosine matrix come from the core's own bound rotation kernel, so
what this file contributes is the control law and nothing else -- an
independent quaternion implementation here would make any comparison against
the built-in a test of two things at once.

Determinism. Every rule of the ``star_reacher.sim`` contract is visible in
what this file does *not* do: it reads no clock, opens no file, draws no
random number, iterates no set, and keeps no state between cycles beyond the
gains it was configured with. The law is memoryless, so reproducibility here
is structural rather than merely intended.
"""

from star_reacher.sim import GncOutput, IGncComponent
from star_reacher._corelink import import_core

_core = import_core()


class PyPdAttitude(IGncComponent):
    """Proportional-derivative attitude control on the estimated state."""

    def __init__(self, cfg):
        # pybind11 trampoline requirement: the C++ base must be constructed
        # before the Python subclass touches anything the core will call.
        super().__init__()
        self.kp = [float(x) for x in cfg.vectors["kp_nm_per_rad"]]
        self.kd = [float(x) for x in cfg.vectors["kd_nm_per_radps"]]
        self.tau_max = [float(x) for x in cfg.vectors["tau_max_nm"]]

    def init(self, ctx):
        # Nothing to capture: the gains are configuration and the law carries
        # no state across cycles.
        pass

    def update(self, inp):
        out = GncOutput()
        est = inp.nav_est
        cmd = inp.att_cmd
        if not est.valid or not cmd.valid:
            # Hold, exactly as the built-in does: an invalid estimate or an
            # absent command is not an instruction to apply zero torque.
            return out

        qc = cmd.q_i2b
        qe = est.q_i2b
        # dq = q_cmd^* (x) q_est   (eq:gnc:deltaq)
        dq = _core.quat_multiply(
            qc[0], -qc[1], -qc[2], -qc[3], qe[0], qe[1], qe[2], qe[3]
        )
        sign = 1.0 if dq[0] >= 0.0 else -1.0  # eq:gnc:sign, sign(0) = +1
        dcm = _core.quat_to_dcm(dq[0], dq[1], dq[2], dq[3])
        wc = cmd.omega_b_radps
        we = est.omega_b_radps
        # w_err = w_est - C(dq) w_cmd   (eq:gnc:werr)
        rotated = [
            dcm[0] * wc[0] + dcm[1] * wc[1] + dcm[2] * wc[2],
            dcm[3] * wc[0] + dcm[4] * wc[1] + dcm[5] * wc[2],
            dcm[6] * wc[0] + dcm[7] * wc[1] + dcm[8] * wc[2],
        ]
        torque = []
        for i in range(3):
            w_err = we[i] - rotated[i]
            # eq:gnc:pd then eq:gnc:sat, left-associated as written
            t = -self.kp[i] * sign * dq[i + 1] - self.kd[i] * w_err
            t = max(-self.tau_max[i], min(self.tau_max[i], t))
            torque.append(t)

        out.valid = True
        out.q_i2b = cmd.q_i2b
        out.omega_b_radps = cmd.omega_b_radps
        out.torque_b_nm = torque
        return out


# The names this file offers a mission, each selected as "python:<name>".
STAR_GNC_COMPONENTS = {"pd_attitude": PyPdAttitude}
