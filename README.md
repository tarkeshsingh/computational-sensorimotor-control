# Computational Sensorimotor Control

A 15-week graduate seminar on how the brain controls movement — from muscle mechanics to sensorimotor integration.

**Instructor:** Tarkesh Singh · Penn State University · Fall 2026

## Course Overview

This course builds a complete computational model of human arm reaching, one layer at a time:

| Module | Weeks | Topic |
|--------|-------|-------|
| **1. The Plant** | 1–3 | Kinematics, muscles, dynamics — build the arm from scratch |
| **2. Controllers** | 4–7 | EPH/λ model, VITE, inverse dynamics, noise |
| **3. Feedback** | 8–11 | Sensory processing, delays, state estimation |
| **4. Optimal Control** | 12–15 | Cost functions, OFC, motor learning |

### Key design principle

Students build the biological arm model (2-link, 6 muscles, Gribble et al. 1998) by hand in Weeks 1–3. From Week 4 onward, they install the `smc` library and focus entirely on controllers and theory — the plant never changes, only the control signal does.

## Repository Structure

```
├── smc/                    # Pip-installable plant library
│   ├── src/smc/
│   │   ├── arm.py          # FK, IK, Jacobian (Week 1)
│   │   ├── muscle.py       # Gribble muscle model (Week 2)
│   │   ├── dynamics.py     # Mass matrix, Coriolis, RK4 (Week 3)
│   │   └── params.py       # All physical parameters
│   └── tests/
├── lectures/               # Weekly lecture notes (.docx)
├── homework/               # Assignments + student notebooks
└── labs/                   # Jupyter lab notebooks
```

## Installing the Plant Library

From Week 4 onward, students install `smc` with a single command:

```bash
pip install git+https://github.com/YOUR_USERNAME/computational-sensorimotor-control.git#subdirectory=smc
```

Then in any notebook:

```python
from smc import Arm, make_muscles, simulate_lambda, Q_REF
import numpy as np

# The same arm you built in Weeks 1-3, ready to use
arm = Arm()
print(arm.forward_kinematics(Q_REF))  # hand position at (55°, 75°)
```

## The Plant Model

The arm model follows Gribble, Ostry, Sanguineti & Laboissière (1998):

- **Geometry**: 2-link planar arm (l₁ = 0.34 m, l₂ = 0.46 m)
- **Muscles**: 6 muscles (pec, bic_l, bic_s, delt, tri_l, tri_lg) with exponential force-length, sigmoidal force-velocity, second-order calcium dynamics, and passive springs
- **Dynamics**: Full nonlinear 2-DOF dynamics with mass matrix, Coriolis terms, and RK4 integration
- **Two control modes**:
  - `simulate_direct(act_fn, B=3.0)` — direct activation a(t) ∈ [0,1] (Weeks 2–3)
  - `simulate_lambda(lam_fn)` — threshold control λ(t) in meters (Week 4+)

### Reference posture

All muscle rest lengths are defined relative to Q_REF = (55°, 75°), which places the hand at approximately (−10, 63) cm in the horizontal plane.

### Joint conventions (Week 1)

- θ₁ (shoulder): 0° = arm abducted to side, positive = horizontal flexion. Range: [0°, 140°]
- θ₂ (elbow): 0° = fully extended, positive = flexion. Range: [0°, 145°]

## Quick Reference: What Students Build vs. Import

| Week | Students build by hand | Students import from `smc` |
|------|----------------------|--------------------------|
| 1 | FK, IK, Jacobian | — |
| 2 | Muscle class, force-length, FV, calcium | — |
| 3 | Mass matrix, Coriolis, RK4, full simulation | — |
| 4+ | λ controller, R/C commands, analysis | `Arm`, `make_muscles`, `simulate_lambda`, ... |

## Running Tests

```bash
cd smc/
pip install -e ".[dev]"
pytest tests/ -v
```

## References

- Gribble, P. L., Ostry, D. J., Sanguineti, V., & Laboissière, R. (1998). Are complex control signals required for human arm movement? *Journal of Neurophysiology*, 79(3), 1409–1424.
- Feldman, A. G. (1966). Functional tuning of the nervous system with control of movement or maintenance of a steady posture. *Biophysics*, 11(3), 565–578.
- Flash, T., & Hogan, N. (1985). The coordination of arm movements: An experimentally confirmed mathematical model. *Journal of Neuroscience*, 5(7), 1688–1703.

## License

MIT
