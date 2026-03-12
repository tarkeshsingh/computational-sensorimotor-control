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
