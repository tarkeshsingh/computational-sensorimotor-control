"""
2-link planar arm: forward kinematics, inverse kinematics, Jacobian.

Conventions (from Week 1):
    θ₁ (shoulder): 0° = arm abducted to side, positive = horizontal flexion
    θ₂ (elbow):    0° = fully extended, positive = flexion
    Joint limits:  θ₁ ∈ [0°, 140°], θ₂ ∈ [0°, 145°]
"""
import numpy as np
from .params import L1, L2, Q1_LIMITS, Q2_LIMITS


class Arm:
    """2-link planar arm with forward/inverse kinematics and Jacobian.

    Parameters
    ----------
    l1 : float
        Upper arm length (m). Default: 0.34 m (Gribble et al., 1998).
    l2 : float
        Forearm + hand length (m). Default: 0.46 m.
    """

    def __init__(self, l1=L1, l2=L2):
        self.l1, self.l2 = l1, l2

    def forward_kinematics(self, q):
        """Compute hand position from joint angles.

        Parameters
        ----------
        q : array-like, shape (2,)
            Joint angles (θ₁, θ₂) in radians.

        Returns
        -------
        p : ndarray, shape (2,)
            Hand position (x, y) in meters.
        """
        q = np.asarray(q)
        x = self.l1 * np.cos(q[0]) + self.l2 * np.cos(q[0] + q[1])
        y = self.l1 * np.sin(q[0]) + self.l2 * np.sin(q[0] + q[1])
        return np.array([x, y])

    def elbow_position(self, q):
        """Compute elbow position from joint angles.

        Parameters
        ----------
        q : array-like, shape (2,)
            Joint angles in radians.

        Returns
        -------
        p : ndarray, shape (2,)
            Elbow position (x, y) in meters.
        """
        q = np.asarray(q)
        return np.array([self.l1 * np.cos(q[0]),
                         self.l1 * np.sin(q[0])])

    def inverse_kinematics(self, x, y):
        """Compute joint angles from hand position (elbow-up solution).

        Parameters
        ----------
        x, y : float
            Target hand position in meters.

        Returns
        -------
        q : ndarray, shape (2,)
            Joint angles (θ₁, θ₂) in radians. Uses the elbow-up (θ₂ > 0)
            solution, which is the only physiologically valid configuration
            given our joint limits.

        Raises
        ------
        ValueError
            If the target is outside the reachable workspace.
        """
        r2 = x**2 + y**2
        c2 = (r2 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
        if abs(c2) > 1.0:
            raise ValueError(
                f"Target ({x:.3f}, {y:.3f}) is outside the reachable workspace. "
                f"|cos(θ₂)| = {abs(c2):.4f} > 1."
            )
        s2 = np.sqrt(1 - c2**2)
        q2 = np.arctan2(s2, c2)
        k1 = self.l1 + self.l2 * c2
        k2 = self.l2 * s2
        q1 = np.arctan2(y, x) - np.arctan2(k2, k1)
        return np.array([q1, q2])

    def jacobian(self, q):
        """Compute the 2×2 Jacobian matrix J = ∂p/∂q.

        Maps joint velocities to hand velocities: ṗ = J · q̇.

        Parameters
        ----------
        q : array-like, shape (2,)
            Joint angles in radians.

        Returns
        -------
        J : ndarray, shape (2, 2)
            Jacobian matrix.
        """
        q = np.asarray(q)
        s1 = np.sin(q[0]); c1 = np.cos(q[0])
        s12 = np.sin(q[0] + q[1]); c12 = np.cos(q[0] + q[1])
        return np.array([
            [-self.l1 * s1 - self.l2 * s12, -self.l2 * s12],
            [ self.l1 * c1 + self.l2 * c12,  self.l2 * c12],
        ])

    def manipulability(self, q):
        """Yoshikawa's manipulability index: w = |det(J)| = l₁·l₂·|sin(θ₂)|.

        Parameters
        ----------
        q : array-like, shape (2,)
            Joint angles in radians.

        Returns
        -------
        w : float
            Manipulability index. Zero at singularity (θ₂ = 0).
        """
        return self.l1 * self.l2 * abs(np.sin(q[1]))

    def in_workspace(self, x, y):
        """Check if a target position is within the reachable workspace."""
        r = np.sqrt(x**2 + y**2)
        return abs(self.l1 - self.l2) <= r <= self.l1 + self.l2

    def in_joint_limits(self, q):
        """Check if joint angles are within physiological limits."""
        return (Q1_LIMITS[0] <= q[0] <= Q1_LIMITS[1] and
                Q2_LIMITS[0] <= q[1] <= Q2_LIMITS[1])
