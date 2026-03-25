"""
Week 11 addition to smc/dynamics.py: simulate_hill()

Drive Hill-type muscles (CE + SE + PE, Kistemaker 2006) using the Gribble
lambda activation law as a STIM source. This is the 'hybrid' approach:
threshold control (feedforward) + tendon compliance (plant realism).

Add this function to dynamics.py and export from __init__.py.
"""

def simulate_hill(lam_fn, T=1.0, dt=0.0001, q0=None, perturb_fn=None,
                  stim_gain=0.020, damping=1.0):
    """Simulate arm movement with Hill-type muscles driven by λ-derived STIM.

    The Gribble threshold law A = [l − λ + μ·dl/dt]⁺ is computed at each
    timestep, then converted to a STIM signal for the Hill muscles via
    STIM_j = stim_gain × A_j(mm).  Optional velocity damping adds STIM
    proportional to muscle lengthening velocity.

    Parameters
    ----------
    lam_fn : callable
        lam_fn(t) → ndarray of shape (6,), threshold values λ (m).
        Same interface as simulate_lambda.
    T : float
        Simulation duration (s).
    dt : float
        Integration timestep (s). Default: 0.1 ms.
    q0 : array-like or None
        Initial joint angles (rad). Defaults to Q_REF.
    perturb_fn : callable or None
        perturb_fn(t) → ndarray of shape (2,), external torque (N·m).
    stim_gain : float
        Scaling from threshold displacement (mm) to STIM [0,1].
        Default 0.020 produces physiological reaching speeds.
    damping : float
        Velocity feedback gain — adds STIM proportional to lengthening
        velocity, improving settling. Default 1.0.

    Returns
    -------
    t : ndarray, shape (N,)
    states : ndarray, shape (N, 4)
    hand : ndarray, shape (N, 2)
    stim_traces : ndarray, shape (N, 6)
        STIM values applied to each Hill muscle at each timestep.
    """
    import numpy as np
    from .arm import Arm
    from .muscle import make_muscles
    from .hill_muscle import make_hill_muscles
    from .params import Q_REF, MU_LAMBDA

    arm = Arm()
    muscles_hill = make_hill_muscles()
    muscles_gribble = make_muscles()  # for length/velocity queries only
    q_init = Q_REF.copy() if q0 is None else np.asarray(q0, dtype=float)

    # Pre-equilibrate Hill muscles at near-zero STIM
    for m in muscles_hill:
        m.reset()
    for _ in range(1000):
        for m in muscles_hill:
            m.step(0.005, q_init, np.zeros(2), dt)

    state = np.concatenate([q_init, [0.0, 0.0]])
    t_vec = np.arange(0, T, dt)
    n = len(t_vec)
    states = np.zeros((n, 4))
    hand = np.zeros((n, 2))
    stim_traces = np.zeros((n, 6))

    for i in range(n):
        t = t_vec[i]
        q, qd = state[:2], state[2:]
        states[i] = state
        hand[i] = arm.forward_kinematics(q)
        lams = np.asarray(lam_fn(t))

        # Compute Gribble-style threshold displacement → STIM
        stim = np.zeros(6)
        for j in range(6):
            gm = muscles_gribble[j]
            ml = gm.length(q)
            mv = gm.velocity(qd)
            A_m = max(0.0, ml - lams[j] + MU_LAMBDA * mv)  # meters
            stim[j] = stim_gain * A_m * 1000.0              # mm → STIM

            # Velocity damping: resist lengthening
            stim[j] += damping * max(0.0, mv)

        stim = np.clip(stim, 0.0, 1.0)
        stim_traces[i] = stim

        # Compute torques from Hill muscles
        tau = np.zeros(2)
        for j, m in enumerate(muscles_hill):
            F, _, _ = m.step(stim[j], q, qd, dt)
            tau[0] += m.r_sh * F
            tau[1] += m.r_el * F

        tau_ext = perturb_fn(t) if perturb_fn is not None else np.zeros(2)

        state = rk4_step(
            state,
            lambda s: np.array([s[2], s[3],
                                *joint_accelerations(s[:2], s[2:], tau + tau_ext, 0.0)]),
            dt,
        )

    return t_vec, states, hand, stim_traces
