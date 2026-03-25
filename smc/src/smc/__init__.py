"""
SMC: Computational Sensorimotor Control — Course Plant Library
==============================================================

A 2-link planar arm model with six Gribble et al. (1998) muscles,
built incrementally across Weeks 1-3 of the course. From Week 4
onward, students import this library and focus on controllers.

Quick start
-----------
>>> from smc import Arm, make_muscles, simulate_lambda, Q_REF
>>> import numpy as np
>>>
>>> arm = Arm()
>>> print(arm.forward_kinematics(Q_REF))  # hand position at reference posture
"""

from .arm import Arm
from .muscle import Muscle, make_muscles, force_velocity_multiplier
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

__version__ = "0.1.0"
