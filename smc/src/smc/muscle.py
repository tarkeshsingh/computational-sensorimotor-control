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

    def compute_force_lambda(self, lam, q, qd, dt):
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

        Returns
        -------
        force : float
            Total muscle force (N).
        activation : float
            Threshold displacement A = [l - λ + μ·dl/dt]⁺ (m).
        """
        ml = self.length(q)
        mv = self.velocity(qd)
        a_m = max(0.0, ml - lam + MU_LAMBDA * mv)  # meters
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
