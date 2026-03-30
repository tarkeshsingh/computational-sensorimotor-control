"""
16D muscle plant interface for iLQG (Week 13).
===============================================

Wraps the Hill-type muscle model into a clean state-space interface
for iterative linear-quadratic control.  The 16D state vector is:

    x = [θ₁, θ₂, θ̇₁, θ̇₂, γ₁..γ₆, l_CE₁..l_CE₆]

    x[0:2]   — joint angles (rad)
    x[2:4]   — joint velocities (rad/s)
    x[4:10]  — muscle activations γ (Hatze calcium, dimensionless)
    x[10:16] — contractile element lengths l_CE (m)

The 6D control vector is:

    u = [stim₁, stim₂, stim₃, stim₄, stim₅, stim₆]   ∈ [0, 1]

Usage
-----
>>> from smc import make_x0, make_target, hill_step, forward_rollout
>>> from smc import Arm, Q_REF, make_hill_muscles
>>>
>>> arm = Arm()
>>> q_target = arm.inverse_kinematics(*(arm.forward_kinematics(Q_REF) + [0.12, 0]))
>>> x0 = make_x0(Q_REF)
>>> xs = make_target(q_target)
>>> u_seq = np.zeros((100, 6))           # 100 control steps × 6 muscles
>>> x_traj = forward_rollout(x0, u_seq)  # → (101, 16) array
"""

import numpy as np
from .hill_muscle import HillMuscle, HILL_MUSCLE_DEFS, make_hill_muscles
from .dynamics import joint_accelerations
from .params import Q_REF

__all__ = [
    "NX", "NU", "MUSCLE_NAMES",
    "DT_SIM", "DT_CTRL", "N_SUBSTEPS",
    "make_x0", "make_target",
    "set_muscle_state", "hill_step", "forward_rollout",
]

# ── Constants ──────────────────────────────────────────────────

NX = 16               # state dimension
NU = 6                # control dimension

DT_SIM = 0.001        # simulation timestep (1 ms)
DT_CTRL = 0.005       # control timestep (5 ms)
N_SUBSTEPS = int(DT_CTRL / DT_SIM)   # = 5

MUSCLE_NAMES = ["Pec", "Bic_l", "Bic_s", "Delt", "Tri_l", "Tri_lg"]


# ── State construction ─────────────────────────────────────────

def make_x0(q, gamma_rest=0.001):
    """Create a 16D initial state from joint angles.

    Parameters
    ----------
    q : (2,) array-like
        Joint angles [θ₁, θ₂] in radians.
    gamma_rest : float
        Resting calcium activation level (default 0.001).

    Returns
    -------
    x0 : (16,) ndarray
        State vector [θ₁, θ₂, 0, 0, γ₁..γ₆, l_CE₁..l_CE₆].
        Velocities are zero. Each γ is set to gamma_rest.
        Each l_CE is computed from geometry: l_MTC(q) − l_SE_0.
    """
    q = np.asarray(q, dtype=float)
    x0 = np.zeros(NX)
    x0[0:2] = q
    # velocities = 0
    muscles = make_hill_muscles()
    for j, m in enumerate(muscles):
        x0[4 + j] = gamma_rest
        x0[10 + j] = m.mtc_length(q) - m.l_SE_0
    return x0


def make_target(q_target, gamma_rest=0.001):
    """Create a 16D target state from target joint angles.

    Identical to make_x0 — at the target, velocities are zero
    and muscles are at resting activation with CE lengths
    determined by the target posture geometry.

    Parameters
    ----------
    q_target : (2,) array-like
        Target joint angles [θ₁, θ₂] in radians.

    Returns
    -------
    xs : (16,) ndarray
    """
    return make_x0(q_target, gamma_rest=gamma_rest)


# ── Muscle state management ───────────────────────────────────

def set_muscle_state(muscles, x):
    """Set each muscle's γ and l_CE from the 16D state vector.

    This is essential for Jacobian computation: you create fresh
    muscle objects with make_hill_muscles(), then call this function
    to initialise them to the state x̄_t before perturbing.

    Parameters
    ----------
    muscles : list of HillMuscle (length 6)
    x : (16,) ndarray
        State vector.
    """
    for j, m in enumerate(muscles):
        m.gamma = max(x[4 + j], 1e-4)
        m.l_CE = max(x[10 + j], 0.01)


# ── Forward dynamics ──────────────────────────────────────────

def hill_step(x, u, muscles, dt_sim=DT_SIM, n_substeps=N_SUBSTEPS):
    """Advance the 16D state by one control step.

    Runs n_substeps simulation steps at dt_sim with constant control u.
    The muscle objects are modified in place (their γ and l_CE are
    updated).  The returned state vector reflects the final values.

    Parameters
    ----------
    x : (16,) ndarray
        Current state.
    u : (6,) ndarray
        Muscle stimulations ∈ [0, 1].
    muscles : list of HillMuscle (length 6)
        Muscle objects whose γ and l_CE match x[4:10] and x[10:16].
        **Modified in place.**
    dt_sim : float
        Simulation timestep (default 0.001 s).
    n_substeps : int
        Number of simulation steps per control step (default 5).

    Returns
    -------
    x_next : (16,) ndarray
        State after one control step.
    """
    xc = x.copy()
    uc = np.clip(u, 0.0, 1.0)

    for _ in range(n_substeps):
        q = xc[0:2].copy()
        qd = xc[2:4].copy()
        tau = np.zeros(2)

        for j, m in enumerate(muscles):
            F, _, _ = m.step(uc[j], q, qd, dt_sim)
            tau[0] += m.r_sh * F
            tau[1] += m.r_el * F

        qdd = joint_accelerations(q, qd, tau, B=0.0)
        xc[0:2] = q + qd * dt_sim
        xc[2:4] = qd + qdd * dt_sim

        for j, m in enumerate(muscles):
            xc[4 + j] = m.gamma
            xc[10 + j] = m.l_CE

    return xc


# ── Trajectory rollout ────────────────────────────────────────

def forward_rollout(x0, u_seq, dt_sim=DT_SIM, n_substeps=N_SUBSTEPS):
    """Simulate a full trajectory on the nonlinear Hill plant.

    Parameters
    ----------
    x0 : (16,) ndarray
        Initial state.
    u_seq : (N, 6) ndarray
        Control sequence (N control steps).
    dt_sim : float
        Simulation timestep (default 0.001 s).
    n_substeps : int
        Simulation steps per control step (default 5).

    Returns
    -------
    x_traj : (N+1, 16) ndarray
        State trajectory.  x_traj[0] = x0.
    """
    N = len(u_seq)
    muscles = make_hill_muscles()
    set_muscle_state(muscles, x0)

    x_traj = np.zeros((N + 1, NX))
    x_traj[0] = x0

    for t in range(N):
        x_traj[t + 1] = hill_step(x_traj[t], u_seq[t], muscles,
                                   dt_sim=dt_sim, n_substeps=n_substeps)

    return x_traj
