"""
Gribble et al. (1998) muscle model.

Each muscle has:
    - Exponential force-length relationship
    - Second-order calcium dynamics (low-pass filter)
    - Sigmoidal force-velocity relationship
    - Passive spring stiffness
    - Two control modes: direct activation a(t) ∈ [0,1] or λ threshold

The six muscles span two joints via moment arms (r_sh, r_el).
"""
import numpy as np
from .params import (
    C_EXP, G_DIRECT, TAU_CA, MU_LAMBDA,
    FV_F1, FV_F2, FV_F3, FV_F4,
    Q_REF, MUSCLE_DEFS,
)


def force_velocity_multiplier(dl):
    """Sigmoidal force-velocity relationship.

    Parameters
    ----------
    dl : float
        Muscle velocity (m/s). Positive = lengthening, negative = shortening.

    Returns
    -------
    fv : float
        Multiplicative factor on calcium-filtered force.
    """
    return FV_F1 + FV_F2 * np.arctan(FV_F3 + FV_F4 * dl)


def calcium_dynamics(state, mt):
    """Second-order calcium filter derivative.

    Parameters
    ----------
    state : ndarray, shape (2,)
        [M, Mdot] — calcium-filtered force and its derivative.
    mt : float
        Target force from the exponential force-length curve.

    Returns
    -------
    dstate : ndarray, shape (2,)
        Time derivatives [Mdot, Mddot].
    """
    m, md = state
    return np.array([md, (mt - m - 2 * TAU_CA * md) / TAU_CA**2])


class Muscle:
    """Single muscle with Gribble et al. (1998) properties.

    Parameters
    ----------
    name : str
        Muscle identifier (e.g., 'pec', 'delt').
    rho : float
        Force scaling parameter (N).
    k : float
        Passive spring stiffness (N/m).
    r_sh : float
        Shoulder moment arm (m). Positive = flexor.
    r_el : float
        Elbow moment arm (m). Positive = flexor.
    rest_length : float
        Rest length at the reference posture Q_REF (m).
    """

    def __init__(self, name, rho, k, r_sh, r_el, rest_length):
        self.name = name
        self.rho = rho
        self.k = k
        self.r_sh = r_sh
        self.r_el = r_el
        self.rl = rest_length
        self.ca = np.array([0.0, 0.0])  # calcium state [M, Mdot]

    def reset(self):
        """Reset calcium dynamics to zero."""
        self.ca = np.array([0.0, 0.0])

    def length(self, q):
        """Muscle length given joint angles.

        Uses linear approximation around Q_REF via moment arms.

        Parameters
        ----------
        q : array-like, shape (2,)
            Joint angles (θ₁, θ₂) in radians.

        Returns
        -------
        l : float
            Muscle length (m).
        """
        return self.rl - self.r_sh * (q[0] - Q_REF[0]) - self.r_el * (q[1] - Q_REF[1])

    def velocity(self, qd):
        """Muscle velocity given joint angular velocities.

        Parameters
        ----------
        qd : array-like, shape (2,)
            Joint angular velocities (θ̇₁, θ̇₂) in rad/s.

        Returns
        -------
        v : float
            Muscle velocity (m/s). Positive = lengthening.
        """
        return -self.r_sh * qd[0] - self.r_el * qd[1]

    def compute_force_direct(self, activation, q, qd, dt):
        """Compute muscle force using direct activation.

        This is the control mode used in Weeks 2-3, where the activation
        a(t) ∈ [0, 1] is specified directly by the controller.

        Parameters
        ----------
        activation : float
            Neural activation in [0, 1].
        q : array-like, shape (2,)
            Joint angles (rad).
        qd : array-like, shape (2,)
            Joint angular velocities (rad/s).
        dt : float
            Integration timestep (s).

        Returns
        -------
        force : float
            Total muscle force (N): calcium-filtered active + passive spring.
        """
        a = np.clip(activation, 0, 1)
        mt = self.rho * (np.exp(C_EXP * G_DIRECT * a) - 1)
        self.ca += dt * calcium_dynamics(self.ca, mt)
        self.ca[0] = max(self.ca[0], 0)
        ml = self.length(q)
        mv = self.velocity(qd)
        return self.ca[0] * force_velocity_multiplier(mv) + self.k * (ml - self.rl)

    def compute_force_lambda(self, lam, q, qd, dt, mu=None):
        """Compute muscle force using λ threshold control.

        This is the control mode introduced in Week 4 (EPH / λ model).
        Activation A = [l - λ + μ·dl/dt]⁺ is computed from the muscle's
        own length and velocity relative to the threshold λ.

        Parameters
        ----------
        lam : float
            Threshold length λ (m).
        q : array-like, shape (2,)
            Joint angles (rad).
        qd : array-like, shape (2,)
            Joint angular velocities (rad/s).
        dt : float
            Integration timestep (s).
        mu : float or None
            Velocity sensitivity μ (s). Defaults to MU_LAMBDA = 0.06 s.
            Pass an explicit value to study how damping depends on μ
            (e.g. HW04 Part 2) without modifying the library.

        Returns
        -------
        force : float
            Total muscle force (N).
        activation : float
            Threshold displacement A = [l - λ + μ·dl/dt]⁺ (m).
        """
        if mu is None:
            mu = MU_LAMBDA
        ml = self.length(q)
        mv = self.velocity(qd)
        a_m = max(0.0, ml - lam + mu * mv)  # meters
        a_mm = a_m * 1000.0  # convert to mm for exponential
        mt = self.rho * (np.exp(C_EXP * a_mm) - 1)
        self.ca += dt * calcium_dynamics(self.ca, mt)
        self.ca[0] = max(self.ca[0], 0)
        force = self.ca[0] * force_velocity_multiplier(mv) + self.k * (ml - self.rl)
        return force, a_m


def make_muscles():
    """Create the six Gribble et al. (1998) muscles.

    Returns
    -------
    muscles : list of Muscle
        [pec, bic_l, bic_s, delt, tri_l, tri_lg]
    """
    return [Muscle(*params) for params in MUSCLE_DEFS]

def lambda_for_posture(q, C=0.25):
    """Assign λ values *relative to* the muscle lengths at posture q.

    Each λ is set so that the threshold displacement A = l − λ equals
    C × (|r_sh| + |r_el|), producing a baseline co-contraction level C.

    .. warning::
       This does **not** hold the arm at q. λ sets a *threshold*, not a
       target: the posture the limb actually adopts is the one where all
       six muscle torques cancel, and that equilibrium must be solved for.
       With the default C = 0.25 the resulting equilibrium sits about
       2.4° from q at the shoulder, and the offset grows with C, because
       the bi-articular muscles act on the shoulder with unequal moment
       arms (bic_s +0.025 m, tri_lg −0.040 m).

       Use :func:`lambda_for_equilibrium` when you need the arm to come to
       rest at a specific posture.

    Parameters
    ----------
    q : array-like, shape (2,)
        Joint angles (rad) whose muscle lengths define the thresholds.
    C : float
        Co-contraction level (rad). Default: 0.25.

    Returns
    -------
    lam : ndarray, shape (6,)
        Threshold values (m) for the six muscles.

    See Also
    --------
    lambda_for_equilibrium : Solve for λ whose equilibrium *is* q.
    """
    muscles = make_muscles()
    return np.array([m.length(q) - (abs(m.r_sh) + abs(m.r_el)) * C
                     for m in muscles])


def equilibrium_posture(lam, guess=None, n_settle=3000, dt=1e-4):
    """Find the posture at which the net muscle torque vanishes.

    Under threshold control the resting posture is emergent: it is wherever
    the six muscle torques happen to cancel, not the posture used to assign λ.

    Parameters
    ----------
    lam : array-like, shape (6,)
        Threshold values (m).
    guess : array-like or None
        Starting guess for the root find (rad). Defaults to Q_REF.
    n_settle : int
        Iterations used to let the calcium filter reach steady state.
    dt : float
        Timestep for that settling (s).

    Returns
    -------
    q_eq : ndarray, shape (2,)
        Equilibrium joint angles (rad).
    """
    from scipy.optimize import fsolve
    from .dynamics import compute_torques_lambda

    def _net_torque(q):
        muscles = make_muscles()
        for _ in range(n_settle):
            tau, _ = compute_torques_lambda(muscles, q, np.zeros(2), lam, dt)
        return tau

    q0 = np.array(Q_REF, dtype=float) if guess is None else np.asarray(guess, float)
    return fsolve(_net_torque, q0)


def lambda_for_equilibrium(q_target, C=0.25):
    """Solve for λ values whose *equilibrium* is q_target.

    Inverts the emergent-equilibrium map. `lambda_for_posture(q)` assigns
    thresholds relative to q but settles elsewhere; this routine finds the
    "virtual" posture whose assigned thresholds settle at q_target.

    Parameters
    ----------
    q_target : array-like, shape (2,)
        Joint angles (rad) at which the arm should come to rest.
    C : float
        Co-contraction level (rad). Default: 0.25.

    Returns
    -------
    lam : ndarray, shape (6,)
        Threshold values (m) whose equilibrium is q_target.

    Notes
    -----
    Self-checking: ``equilibrium_posture(lambda_for_equilibrium(q))`` should
    return ``q`` to solver tolerance.
    """
    from scipy.optimize import fsolve
    q_target = np.asarray(q_target, float)

    def _residual(q_lam):
        return equilibrium_posture(lambda_for_posture(q_lam, C),
                                   guess=q_target) - q_target

    q_lam = fsolve(_residual, q_target)
    return lambda_for_posture(q_lam, C)


def make_ramp(lam_init, lam_final, t_start=0.05, duration=0.35):
    """Create a constant-rate λ ramp function.

    Returns a callable lam_fn(t) that linearly interpolates from
    lam_init to lam_final over [t_start, t_start + duration].

    Parameters
    ----------
    lam_init : ndarray, shape (6,)
        Initial λ values (m).
    lam_final : ndarray, shape (6,)
        Final λ values (m).
    t_start : float
        Ramp onset time (s). Default: 0.05.
    duration : float
        Ramp duration (s). Default: 0.35.

    Returns
    -------
    lam_fn : callable
        lam_fn(t) -> ndarray of shape (6,).
    """
    lam_init = np.asarray(lam_init, dtype=float)
    lam_final = np.asarray(lam_final, dtype=float)

    def fn(t):
        if t < t_start:
            return lam_init.copy()
        elif t < t_start + duration:
            return lam_init + (t - t_start) / duration * (lam_final - lam_init)
        else:
            return lam_final.copy()
    return fn
