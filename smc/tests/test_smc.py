"""Tests for the smc package — verifies the plant matches HW01-03 solutions."""
import numpy as np
import pytest


def test_forward_kinematics_at_reference():
    """FK at Q_REF should return the known hand position."""
    from smc import Arm, Q_REF
    arm = Arm()
    p = arm.forward_kinematics(Q_REF)
    # Known values: (-10.1, 63.1) cm
    assert abs(p[0] * 100 - (-10.1)) < 0.1
    assert abs(p[1] * 100 - 63.1) < 0.1


def test_inverse_kinematics_roundtrip():
    """IK(FK(q)) should recover q."""
    from smc import Arm, Q_REF
    arm = Arm()
    p = arm.forward_kinematics(Q_REF)
    q_recovered = arm.inverse_kinematics(p[0], p[1])
    np.testing.assert_allclose(q_recovered, Q_REF, atol=1e-10)


def test_ik_out_of_workspace():
    """IK should raise ValueError for unreachable targets."""
    from smc import Arm
    arm = Arm()
    with pytest.raises(ValueError):
        arm.inverse_kinematics(10.0, 0.0)  # way too far


def test_jacobian_determinant_at_reference():
    """det(J) = l1*l2*sin(θ₂), should be nonzero at Q_REF."""
    from smc import Arm, Q_REF, L1, L2
    arm = Arm()
    J = arm.jacobian(Q_REF)
    expected = L1 * L2 * np.sin(Q_REF[1])
    assert abs(np.linalg.det(J) - expected) < 1e-10


def test_six_muscles_created():
    """make_muscles() should return 6 muscles with correct names."""
    from smc import make_muscles
    muscles = make_muscles()
    assert len(muscles) == 6
    names = [m.name for m in muscles]
    assert names == ["pec", "bic_l", "bic_s", "delt", "tri_l", "tri_lg"]


def test_muscle_length_at_reference():
    """At Q_REF, each muscle length should equal its rest length."""
    from smc import make_muscles, Q_REF
    muscles = make_muscles()
    for m in muscles:
        assert abs(m.length(Q_REF) - m.rl) < 1e-12, f"{m.name} length != rest length at Q_REF"


def test_mass_matrix_symmetric_positive_definite():
    """Mass matrix should be SPD at Q_REF."""
    from smc import mass_matrix, Q_REF
    M = mass_matrix(Q_REF)
    # Symmetric
    np.testing.assert_allclose(M, M.T, atol=1e-15)
    # Positive definite (both eigenvalues > 0)
    eigvals = np.linalg.eigvalsh(M)
    assert all(eigvals > 0), f"Eigenvalues: {eigvals}"


def test_simulate_direct_model_a_settles():
    """Model A (B=3.0) with triphasic should settle (velocity < 2 cm/s)."""
    from smc import simulate_direct

    def triphasic(t):
        a = np.zeros(6)
        if 0.02 <= t < 0.20: a[0] = 1.0; a[2] = 0.4
        if 0.18 <= t < 0.30: a[3] = 0.8; a[5] = 0.3
        if t >= 0.28: a[0] = 0.25; a[3] = 0.25
        return a

    t, states, hand, _ = simulate_direct(triphasic, T=1.0, B=3.0)
    # Final velocity should be low
    final_vel = np.linalg.norm(np.diff(hand[-10:], axis=0) / 0.0001, axis=1).mean() * 100
    assert final_vel < 2.0, f"Final velocity = {final_vel:.1f} cm/s (should be < 2)"


def test_simulate_direct_model_b_drifts():
    """Model B (B=0) with triphasic should NOT settle."""
    from smc import simulate_direct

    def triphasic(t):
        a = np.zeros(6)
        if 0.02 <= t < 0.20: a[0] = 1.0; a[2] = 0.4
        if 0.18 <= t < 0.30: a[3] = 0.8; a[5] = 0.3
        if t >= 0.28: a[0] = 0.25; a[3] = 0.25
        return a

    t, states, hand, _ = simulate_direct(triphasic, T=1.0, B=0.0)
    # Final velocity should still be significant
    final_vel = np.linalg.norm(np.diff(hand[-10:], axis=0) / 0.0001, axis=1).mean() * 100
    assert final_vel > 2.0, f"Final velocity = {final_vel:.1f} cm/s (should be > 2)"


def test_joint_limits():
    """Reference posture should be within limits; extreme values outside."""
    from smc import Arm, Q_REF
    arm = Arm()
    assert arm.in_joint_limits(Q_REF)
    assert not arm.in_joint_limits(np.array([3.0, 0.5]))  # θ₁ > 140°


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
