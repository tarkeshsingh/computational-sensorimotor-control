"""
Arm dynamics: mass matrix, Coriolis terms, RK4 integrator, and simulation loops.

Provides two simulation modes:
    - simulate_direct(): direct activation a(t) ∈ [0,1] (Weeks 2-3)
    - simulate_lambda(): threshold control λ(t) (Week 4 onward)
"""
import numpy as np
from .params import A1, A2, A3, Q_REF
from .muscle import make_muscles


def mass_matrix(q):
    """2×2 inertia matrix M(q).

    Parameters
    ----------
    q : array-like, shape (2,)
        Joint angles (θ₁, θ₂) in radians.

    Returns
    -------
    M : ndarray, shape (2, 2)
        Symmetric positive-definite mass matrix.
    """
    c2 = np.cos(q[1])
    return np.array([
        [A1 + 2 * A3 * c2, A2 + A3 * c2],
        [A2 + A3 * c2,      A2],
    ])


def coriolis(q, qd):
    """Coriolis and centripetal torque vector C(q, q̇).

    Parameters
    ----------
    q : array-like, shape (2,)
        Joint angles (rad).
    qd : array-like, shape (2,)
        Joint angular velocities (rad/s).

    Returns
    -------
    c : ndarray, shape (2,)
        Coriolis/centripetal torque vector.
    """
    h = -A3 * np.sin(q[1])
    return np.array([
        h * qd[1]**2 + 2 * h * qd[0] * qd[1],
        -h * qd[0]**2,
    ])


def joint_accelerations(q, qd, tau, B=0.0):
    """Solve forward dynamics: M·q̈ = τ - C(q,q̇) - B·q̇.

    Parameters
    ----------
    q : array-like, shape (2,)
        Joint angles (rad).
    qd : array-like, shape (2,)
        Joint angular velocities (rad/s).
    tau : array-like, shape (2,)
        Applied joint torques (N·m).
    B : float
        Joint viscous damping coefficient (N·m·s/rad).

    Returns
    -------
    qdd : ndarray, shape (2,)
        Joint angular accelerations (rad/s²).
    """
    return np.linalg.solve(
        mass_matrix(q),
        np.asarray(tau) - coriolis(q, qd) - B * np.asarray(qd),
    )


def rk4_step(state, deriv_fn, dt):
    """Single step of the 4th-order Runge-Kutta integrator.

    Parameters
    ----------
    state : ndarray
        Current state vector.
    deriv_fn : callable
        deriv_fn(state) → time derivative of state.
    dt : float
        Timestep (s).

    Returns
    -------
    new_state : ndarray
        State after one timestep.
    """
    k1 = dt * deriv_fn(state)
    k2 = dt * deriv_fn(state + k1 / 2)
    k3 = dt * deriv_fn(state + k2 / 2)
    k4 = dt * deriv_fn(state + k3)
    return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def compute_torques_direct(muscles, q, qd, activations, dt):
    """Compute joint torques from direct activation vector.

    Parameters
    ----------
    muscles : list of Muscle
        The six arm muscles.
    q : array-like, shape (2,)
        Joint angles (rad).
    qd : array-like, shape (2,)
        Joint angular velocities (rad/s).
    activations : array-like, shape (6,)
        Activation levels in [0, 1] for each muscle.
    dt : float
        Timestep (s).

    Returns
    -------
    tau : ndarray, shape (2,)
        Joint torques [shoulder, elbow] (N·m).
    """
    tau = np.zeros(2)
    for i, m in enumerate(muscles):
        f = m.compute_force_direct(activations[i], q, qd, dt)
        tau[0] += m.r_sh * f
        tau[1] += m.r_el * f
    return tau


def compute_torques_lambda(muscles, q, qd, lambdas, dt):
    """Compute joint torques from λ threshold vector.

    Parameters
    ----------
    muscles : list of Muscle
        The six arm muscles.
    q : array-like, shape (2,)
        Joint angles (rad).
    qd : array-like, shape (2,)
        Joint angular velocities (rad/s).
    lambdas : array-like, shape (6,)
        Threshold values λ (m) for each muscle.
    dt : float
        Timestep (s).

    Returns
    -------
    tau : ndarray, shape (2,)
        Joint torques [shoulder, elbow] (N·m).
    activations : ndarray, shape (6,)
        Threshold displacements A = [l - λ + μ·dl/dt]⁺ (m).
    """
    tau = np.zeros(2)
    activations = np.zeros(6)
    for i, m in enumerate(muscles):
        f, a_val = m.compute_force_lambda(lambdas[i], q, qd, dt)
        activations[i] = a_val
        tau[0] += m.r_sh * f
        tau[1] += m.r_el * f
    return tau, activations


def simulate_direct(act_fn, T=1.0, dt=0.0001, B=0.0, q0=None):
    """Simulate arm movement with direct activation control.

    Parameters
    ----------
    act_fn : callable
        act_fn(t) → ndarray of shape (6,), activations in [0, 1].
    T : float
        Simulation duration (s).
    dt : float
        Integration timestep (s). Default: 0.1 ms.
    B : float
        Joint viscous damping (N·m·s/rad). B=3.0 for Model A, B=0 for Model B.
    q0 : array-like or None
        Initial joint angles (rad). Defaults to Q_REF = (55°, 75°).

    Returns
    -------
    t : ndarray, shape (N,)
        Time vector (s).
    states : ndarray, shape (N, 4)
        [θ₁, θ₂, θ̇₁, θ̇₂] at each timestep.
    hand : ndarray, shape (N, 2)
        Hand position (x, y) in meters.
    activations : ndarray, shape (N, 6)
        Activation levels used at each timestep.
    """
    from .arm import Arm
    arm = Arm()
    muscles = make_muscles()
    q_init = Q_REF.copy() if q0 is None else np.asarray(q0, dtype=float)
    state = np.concatenate([q_init, [0.0, 0.0]])
    t = np.arange(0, T, dt)
    n = len(t)
    states = np.zeros((n, 4))
    hand = np.zeros((n, 2))
    acts = np.zeros((n, 6))

    for i in range(n):
        q, qd = state[:2], state[2:]
        states[i] = state
        hand[i] = arm.forward_kinematics(q)
        a = np.asarray(act_fn(t[i]))
        acts[i] = a
        tau = compute_torques_direct(muscles, q, qd, a, dt)
        state = rk4_step(
            state,
            lambda s: np.array([s[2], s[3],
                                *joint_accelerations(s[:2], s[2:], tau, B)]),
            dt,
        )
    return t, states, hand, acts


def simulate_lambda(lam_fn, T=1.0, dt=0.0001, q0=None, perturb_fn=None):
    """Simulate arm movement with λ threshold control.

    Parameters
    ----------
    lam_fn : callable
        lam_fn(t) → ndarray of shape (6,), threshold values λ (m).
    T : float
        Simulation duration (s).
    dt : float
        Integration timestep (s). Default: 0.1 ms.
    q0 : array-like or None
        Initial joint angles (rad). Defaults to Q_REF = (55°, 75°).
    perturb_fn : callable or None
        perturb_fn(t) → ndarray of shape (2,), external torque perturbation (N·m).

    Returns
    -------
    t : ndarray, shape (N,)
        Time vector (s).
    states : ndarray, shape (N, 4)
        [θ₁, θ₂, θ̇₁, θ̇₂] at each timestep.
    hand : ndarray, shape (N, 2)
        Hand position (x, y) in meters.
    activations : ndarray, shape (N, 6)
        Threshold displacements A = [l - λ + μ·dl/dt]⁺ (m) at each timestep.
    """
    from .arm import Arm
    arm = Arm()
    muscles = make_muscles()
    q_init = Q_REF.copy() if q0 is None else np.asarray(q0, dtype=float)
    state = np.concatenate([q_init, [0.0, 0.0]])
    t = np.arange(0, T, dt)
    n = len(t)
    states = np.zeros((n, 4))
    hand = np.zeros((n, 2))
    acts = np.zeros((n, 6))

    for i in range(n):
        q, qd = state[:2], state[2:]
        states[i] = state
        hand[i] = arm.forward_kinematics(q)
        lams = np.asarray(lam_fn(t[i]))
        tau, a = compute_torques_lambda(muscles, q, qd, lams, dt)
        acts[i] = a
        tau_ext = perturb_fn(t[i]) if perturb_fn is not None else np.zeros(2)
        state = rk4_step(
            state,
            lambda s: np.array([s[2], s[3],
                                *joint_accelerations(s[:2], s[2:], tau + tau_ext, 0.0)]),
            dt,
        )
    return t, states, hand, acts

def simulate_hill(lam_fn, T=1.0, dt=0.0001, q0=None, perturb_fn=None,
                  stim_gain=0.020, damping=3.0):
    """Simulate arm movement with Hill-type muscles driven by R-C lambda.

    The Gribble threshold law A = [l - lambda + mu*dl/dt]+ is computed
    at each timestep, then converted to a STIM signal for Hill muscles.
    This isolates the effect of tendon compliance: same R-C control
    architecture, different muscle plant.

    Parameters
    ----------
    lam_fn : callable
        lam_fn(t) -> ndarray of shape (6,), threshold values (m).
    T : float
        Simulation duration (s).
    dt : float
        Integration timestep (s).
    q0 : array-like or None
        Initial joint angles (rad). Defaults to Q_REF.
    perturb_fn : callable or None
        perturb_fn(t) -> ndarray of shape (2,), external torque (N*m).
    stim_gain : float
        Scaling from threshold displacement (mm) to STIM [0,1].
    damping : float
        Velocity feedback gain for lengthening muscles.

    Returns
    -------
    t : ndarray, shape (N,)
    states : ndarray, shape (N, 4)
    hand : ndarray, shape (N, 2)
    stim_traces : ndarray, shape (N, 6)
    """
    from .arm import Arm
    from .muscle import make_muscles as make_gribble_muscles
    from .hill_muscle import make_hill_muscles
    from .params import Q_REF, MU_LAMBDA

    arm = Arm()
    muscles_hill = make_hill_muscles()
    muscles_gribble = make_gribble_muscles()
    q_init = Q_REF.copy() if q0 is None else np.asarray(q0, dtype=float)

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

        stim = np.zeros(6)
        for j in range(6):
            gm = muscles_gribble[j]
            ml = gm.length(q)
            mv = gm.velocity(qd)
            A_m = max(0.0, ml - lams[j] + MU_LAMBDA * mv)
            stim[j] = stim_gain * A_m * 1000.0 + damping * max(0.0, mv)
        stim = np.clip(stim, 0.0, 1.0)
        stim_traces[i] = stim

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


def simulate_kmhm(lam_fn, q_target, T=1.0, dt=0.0001, q0=None,
                  perturb_fn=None, stim_gain=0.020, damping=3.0,
                  kd_ce=0.30, delay=0.025, t_start=0.05, ramp_dur=0.50):
    """Simulate with KMHM: R-C base STIM + CE velocity reference feedback.

    Extends simulate_hill with a feedback loop that tracks a desired CE
    velocity profile (derived from a min-jerk trajectory). Inspired by
    Kistemaker et al. (2006), who showed that a velocity reference signal
    improved the performance of their hybrid EP controller.

    Parameters
    ----------
    lam_fn : callable
        lam_fn(t) -> ndarray of shape (6,), threshold values (m).
    q_target : array-like, shape (2,)
        Target joint angles (rad) for the CE velocity reference.
    T, dt, q0, perturb_fn, stim_gain, damping :
        Same as simulate_hill.
    kd_ce : float
        CE velocity feedback gain.
    delay : float
        Feedback delay (s). Default: 25 ms.
    t_start : float
        Movement onset time (s).
    ramp_dur : float
        Duration of the lambda ramp (s).

    Returns
    -------
    t, states, hand, stim_traces : same as simulate_hill.
    """
    from .arm import Arm
    from .muscle import make_muscles as make_gribble_muscles
    from .hill_muscle import make_hill_muscles
    from .params import Q_REF, MU_LAMBDA

    arm = Arm()
    muscles_hill = make_hill_muscles()
    muscles_gribble = make_gribble_muscles()
    q_init = Q_REF.copy() if q0 is None else np.asarray(q0, dtype=float)
    q_target = np.asarray(q_target, dtype=float)

    for m in muscles_hill:
        m.reset()
    for _ in range(1000):
        for m in muscles_hill:
            m.step(0.005, q_init, np.zeros(2), dt)

    ce_start = np.array([m.l_CE for m in muscles_hill])
    ce_target = np.array([m.mtc_length(q_target) - m.l_SE_0
                          for m in muscles_hill])

    state = np.concatenate([q_init, [0.0, 0.0]])
    t_vec = np.arange(0, T, dt)
    n = len(t_vec)
    states = np.zeros((n, 4))
    hand = np.zeros((n, 2))
    stim_traces = np.zeros((n, 6))

    delay_steps = int(delay / dt)
    vce_buf = np.zeros((delay_steps + 1, 6))
    bi = 0

    for i in range(n):
        t = t_vec[i]
        q, qd = state[:2], state[2:]
        states[i] = state
        hand[i] = arm.forward_kinematics(q)
        lams = np.asarray(lam_fn(t))

        # Base STIM from R-C (same as simulate_hill)
        stim_base = np.zeros(6)
        for j in range(6):
            gm = muscles_gribble[j]
            ml = gm.length(q)
            mv = gm.velocity(qd)
            A_m = max(0.0, ml - lams[j] + MU_LAMBDA * mv)
            stim_base[j] = stim_gain * A_m * 1000.0 + damping * max(0.0, mv)

        # Desired CE velocity from min-jerk profile
        if t < t_start:
            vce_des = np.zeros(6)
        elif t < t_start + ramp_dur:
            tau_mj = (t - t_start) / ramp_dur
            sd = (30 * tau_mj**2 - 60 * tau_mj**3 + 30 * tau_mj**4) / ramp_dur
            vce_des = sd * (ce_target - ce_start)
        else:
            vce_des = np.zeros(6)

        # CE velocity feedback
        di = (bi - delay_steps) % (delay_steps + 1)
        fb = np.array([kd_ce * (vce_des[j] - vce_buf[di, j]) for j in range(6)])

        stim = np.clip(stim_base + fb, 0.0, 1.0)
        stim_traces[i] = stim

        tau = np.zeros(2)
        cur_vce = np.zeros(6)
        for j, m in enumerate(muscles_hill):
            F, lce, vce = m.step(stim[j], q, qd, dt)
            tau[0] += m.r_sh * F
            tau[1] += m.r_el * F
            cur_vce[j] = vce

        bi = (bi + 1) % (delay_steps + 1)
        vce_buf[bi] = cur_vce

        tau_ext = perturb_fn(t) if perturb_fn is not None else np.zeros(2)
        state = rk4_step(
            state,
            lambda s: np.array([s[2], s[3],
                                *joint_accelerations(s[:2], s[2:], tau + tau_ext, 0.0)]),
            dt,
        )

    return t_vec, states, hand, stim_traces
