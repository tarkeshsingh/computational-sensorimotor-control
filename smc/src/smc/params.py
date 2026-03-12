"""
Physical parameters for the 2-link planar arm.

All constants from Gribble, Ostry, Sanguineti & Laboissière (1998),
"Are complex control signals required for human arm movement?"
Journal of Neurophysiology, 79(3), 1409-1424.

This module is the single source of truth for the course plant.
Every other module imports from here.
"""
import numpy as np

# ── Link geometry ──
L1 = 0.34          # upper arm length (m)
L2 = 0.46          # forearm + hand length (m)

# ── Inertial parameters ──
M1, M2 = 2.10, 1.65        # segment masses (kg)
S1, S2 = 0.11, 0.16        # COM distances from proximal joint (m)
I1, I2 = 0.015, 0.022      # moments of inertia about COM (kg·m²)

# ── Derived inertia constants (for mass matrix) ──
A1 = I1 + I2 + M1 * S1**2 + M2 * (L1**2 + S2**2)
A2 = I2 + M2 * S2**2
A3 = M2 * L1 * S2

# ── Muscle model constants ──
C_EXP = 0.112       # exponential force-length curvature (1/mm)
G_DIRECT = 20.0     # neural drive gain for direct activation mode (mm)
TAU_CA = 0.015      # calcium dynamics time constant (s)
MU_LAMBDA = 0.06    # velocity sensitivity for λ threshold law (s)

# ── Force-velocity parameters ──
FV_F1, FV_F2, FV_F3, FV_F4 = 0.82, 0.50, 0.43, 58.0

# ── Reference posture (used for muscle rest lengths) ──
Q_REF = np.array([np.radians(55), np.radians(75)])

# ── Joint limits (Week 1 anatomical conventions) ──
#    θ₁: 0° = arm abducted, positive = horizontal flexion
#    θ₂: 0° = fully extended, positive = elbow flexion
Q1_LIMITS = (0.0, np.radians(140))   # shoulder: 0°–140°
Q2_LIMITS = (0.0, np.radians(145))   # elbow: 0°–145°

# ── Muscle definitions ──
# Each tuple: (name, rho [N], k [N/m], r_sh [m], r_el [m], rest_length [m])
MUSCLE_DEFS = [
    ("pec",    14.9, 258.5,  0.04,  0.00, 0.26),
    ("bic_l",  11.0, 150.0,  0.00,  0.03, 0.26),
    ("bic_s",   2.1, 100.0,  0.025, 0.03, 0.29),
    ("delt",   14.9, 258.5, -0.04,  0.00, 0.26),
    ("tri_l",  12.1, 200.0,  0.00, -0.02, 0.26),
    ("tri_lg",  6.7, 100.0, -0.04, -0.02, 0.32),
]
