"""
SMC: Computational Sensorimotor Control — Course Plant Library
==============================================================

A 2-link planar arm model with six Gribble et al. (1998) muscles,
built incrementally across Weeks 1-3 of the course. From Week 4
onward, students import this library and focus on controllers.

From Week 13, the plant16d module provides a 16D state-space
interface for iterative LQG (iLQG) on the Hill-type muscle plant.

Quick start
-----------
>>> from smc import Arm, make_muscles, simulate_lambda, Q_REF
>>> import numpy as np
>>>
>>> arm = Arm()
>>> print(arm.forward_kinematics(Q_REF))  # hand position at reference posture

iLQG quick start (Week 13)
--------------------------
>>> from smc import Arm, Q_REF, make_x0, make_target, hill_step, forward_rollout
>>> arm = Arm()
>>> q_tgt = arm.inverse_kinematics(*(arm.forward_kinematics(Q_REF) + [0.12, 0]))
>>> x0 = make_x0(Q_REF)
>>> xs = make_target(q_tgt)
"""

from .arm import Arm
from .muscle import Muscle, make_muscles, force_velocity_multiplier, lambda_for_posture, make_ramp
from .hill_muscle import HillMuscle, make_hill_muscles
from .dynamics import (
    mass_matrix,
    coriolis,
    joint_accelerations,
    rk4_step,
    simulate_direct,
    simulate_lambda,
    simulate_hill,
    simulate_kmhm,
)
from .params import (
    Q_REF, L1, L2, M1, M2, S1, S2, I1, I2,
    MUSCLE_DEFS, Q1_LIMITS, Q2_LIMITS,
    C_EXP, G_DIRECT, TAU_CA, MU_LAMBDA,
)
from .sensor import Sensor
from .plant16d import (
    NX, NU, MUSCLE_NAMES,
    DT_SIM, DT_CTRL, N_SUBSTEPS,
    make_x0, make_target,
    set_muscle_state, hill_step, forward_rollout,
)

__version__ = "0.2.0"
