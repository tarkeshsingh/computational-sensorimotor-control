# Computational Sensorimotor Control

A 15-week graduate seminar on how the brain controls movement — from muscle mechanics to optimal feedback control.

**Instructor:** Tarkesh Singh · Penn State University · Fall 2026

## Course Overview

This course builds a complete computational model of human arm reaching, one layer at a time:

| Module | Weeks | Topic |
|--------|-------|-------|
| **1. The Plant** | 1–3 | Kinematics, muscles, dynamics — build the arm from scratch |
| **2. Controllers** | 4–7 | EPH/λ model, VITE, inverse dynamics, noise |
| **3. Feedback** | 8–10 | Vision, proprioception, Kalman filter, interception |
| **4. Optimal Control** | 11–15 | Muscle mechanics, OFC, iLQG, cerebellar ataxia, presentations |

### Key design principle

Students build the biological arm model (2-link, 6 muscles, Gribble et al. 1998) by hand in Weeks 1–3. From Week 4 onward, they install the `smc` library and focus entirely on controllers and theory — the plant never changes, only the control signal does. In Week 11, the plant is extended with Hill-type muscles (CE + SE + PE, Hatze activation) to explore what tendon compliance costs and buys. In Week 13, the `plant16d` module provides a 16D state-space interface for iLQG on the Hill-type plant.

## Repository Structure

```
├── smc/                        # Pip-installable plant library (v0.2.0)
│   ├── src/smc/
│   │   ├── arm.py              # FK, IK, Jacobian (Week 1)
│   │   ├── muscle.py           # Gribble muscle model + lambda_for_posture, make_ramp (Week 2, extended Week 11)
│   │   ├── hill_muscle.py      # Hill-type muscle: CE+SE+PE, Hatze activation (Week 11)
│   │   ├── dynamics.py         # Mass matrix, Coriolis, RK4, simulate_lambda/hill/kmhm (Week 3, extended Week 11)
│   │   ├── plant16d.py         # 16D state-space interface for iLQG (Week 13)
│   │   ├── params.py           # All physical parameters
│   │   ├── sensor.py           # Foveal, peripheral, proprioceptive sensors (Week 8)
│   │   └── polar_kf.py         # Polar Kalman filter for target tracking (Week 10)
│   └── tests/
├── lectures/                   # Weekly lecture notes (.docx)
│   ├── Week01_Lecture.docx     # Kinematics & workspace geometry
│   ├── Week02_Lecture.docx     # Muscle mechanics (Gribble model)
│   ├── Week03_Lecture.docx     # Forward dynamics & simulation
│   ├── Week04_Lecture.docx     # EPH / λ model
│   ├── Week05_Lecture.docx     # VITE & movement invariants
│   ├── Week06_Lecture.docx     # Inverse dynamics
│   ├── Week07_Lecture.docx     # Motor noise & variability
│   ├── Week08_Lecture.docx     # Vision & sensory delay
│   ├── Week09_Lecture.docx     # Kalman filter / proprioception
│   ├── Week10_Lecture.docx     # From reaching to interception
│   ├── Week11_Lecture.docx     # When motor commands meet muscle (Hill-type)
│   ├── Week12_Lecture.docx     # OFC: LQR + KF = LQG, minimum intervention
│   └── Week13_Lecture.docx     # iLQG: re-linearization, 16D state, augmented delays
├── labs/                       # Jupyter lab notebooks (student + solutions)
│   ├── Lab01_Kinematics.ipynb
│   ├── Lab02_Muscles.ipynb
│   ├── Lab03_Dynamics.ipynb
│   ├── Lab04_EPH.ipynb
│   ├── Lab05_VITE.ipynb
│   ├── Lab06_InverseDynamics.ipynb
│   ├── Lab07_Noise.ipynb
│   ├── Lab08_Vision.ipynb
│   ├── Lab09_KalmanFilter.ipynb
│   ├── Lab10_Interception.ipynb
│   ├── Lab11_EPH_Muscle.ipynb
│   ├── Lab12_OFC.ipynb
│   ├── Lab12_OFC_Solutions.ipynb
│   ├── Lab13_iLQG.ipynb
│   └── Lab13_iLQG_Solutions.ipynb
├── homework/                   # Assignments (.docx) + student/solutions notebooks
│   ├── HW01_Reaching_Predictions.ipynb
│   ├── HW02_MuscleSpeedLimit.ipynb
│   ├── HW03_TriphasicReach.ipynb
│   ├── HW04_EPH_Experiments.ipynb
│   ├── HW05_Workspace_ViaPoint.ipynb
│   ├── HW06_InverseDynamics.ipynb
│   ├── HW07_Noise.ipynb
│   ├── HW08_Vision.ipynb
│   ├── HW09_WhenDoesVisionHelp.ipynb
│   ├── HW10_Bayesian_Interception.ipynb
│   ├── HW11_EPH_Muscle.ipynb
│   ├── HW12_OFC.docx
│   ├── HW12_OFC.ipynb
│   ├── HW12_OFC_Solutions.ipynb
│   ├── HW13_iLQG.docx
│   ├── HW13_iLQG.ipynb
│   └── HW13_iLQG_Solutions.ipynb
└── Course_Outline_v12.docx     # Full course outline
```

## Installing the Plant Library

From Week 4 onward, students install `smc` with a single command:

```bash
pip install git+https://github.com/tarkeshsingh/computational-sensorimotor-control.git#subdirectory=smc
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
- **Hill-type muscles** (Week 11): Same 6 muscles with contractile element (CE), series elastic element (SE, tendon), parallel elastic element (PE), and Hatze length-dependent activation dynamics
- **16D plant interface** (Week 13): State vector [θ₁, θ₂, θ̇₁, θ̇₂, γ₁..γ₆, l_CE₁..l_CE₆] for iLQG
- **Dynamics**: Full nonlinear 2-DOF dynamics with mass matrix, Coriolis terms, and RK4 integration

### Control modes

| Function | Plant | Control | Week |
|----------|-------|---------|------|
| `simulate_direct(act_fn, B)` | Gribble muscles | Direct activation a(t) ∈ [0,1] | 2–3 |
| `simulate_lambda(lam_fn)` | Gribble muscles | Threshold control λ(t) | 4+ |
| `simulate_hill(lam_fn)` | Hill muscles (CE+SE+PE) | R-C λ → STIM | 11 |
| `simulate_kmhm(lam_fn, q_target)` | Hill muscles (CE+SE+PE) | R-C + CE velocity feedback | 11 |
| `forward_rollout(x0, u_seq)` | Hill muscles (16D state) | Stimulation u(t) ∈ [0,1]⁶ | 13+ |

### OFC control (Week 12+)

Week 12 introduces optimal feedback control on a torque-driven (linearized) arm — no muscle simulation functions needed. Students build the LQR/LQG pipeline from scratch:

| Function | What it computes | Built in |
|----------|-----------------|----------|
| `linearize_arm(q_lin, dt)` | A (4×4), B (4×2) state-space matrices | Lab 12 |
| `riccati_backward(A, B, Q, R, N)` | Time-varying gains L_t, cost-to-go S_t | Lab 12 |
| `kalman_step(xh, P, u, y, A, B, H, W, V)` | KF predict-update for hand state | Lab 12 |
| `simulate_lqg(...)` | Full closed-loop LQG with sensory noise | Lab 12 |

### iLQG control (Week 13)

Week 13 extends OFC to the nonlinear Hill-type muscle plant using iterative LQG:

| Function | What it computes | Built in |
|----------|-----------------|----------|
| `compute_jacobians(x_bar, u_bar)` | A_t (16×16), B_t (16×6) via central finite differences | Lab 13 |
| `riccati_backward_ilqg(...)` | L_t, L_v, L_u gains + v_t target-miss vector | Lab 13 |
| `forward_pass(...)` | Three-term control law with line search | Lab 13 |
| `run_ilqg(...)` | Full iLQG loop: linearize → backward → forward → iterate | Lab 13 |

### 16D plant interface (Week 13, `plant16d` module)

```python
from smc import (
    NX, NU, MUSCLE_NAMES,           # 16, 6, ['Pec', 'Bic_l', ...]
    DT_SIM, DT_CTRL, N_SUBSTEPS,    # 0.001, 0.005, 5
    make_x0, make_target,            # joint angles → 16D state
    set_muscle_state,                # sync muscle objects to state vector
    hill_step, forward_rollout,      # forward dynamics
)
```

The 16D state vector:
- `x[0:2]` — joint angles θ₁, θ₂ (rad)
- `x[2:4]` — joint velocities θ̇₁, θ̇₂ (rad/s)
- `x[4:10]` — muscle activations γ₁..γ₆ (Hatze calcium)
- `x[10:16]` — contractile element lengths l_CE₁..l_CE₆ (m)

### Helper functions

```python
from smc import lambda_for_posture, make_ramp

li = lambda_for_posture(Q_REF, C=0.25)      # λ values for a posture
lf = lambda_for_posture(q_target, C=0.25)    # λ values for the target
lam_fn = make_ramp(li, lf, t_start=0.05, duration=0.50)  # constant-rate λ ramp
```

### Sensors (Week 8+)

| Sensor | Signal | Noise (σ) | Delay |
|--------|--------|-----------|-------|
| `FovealSensor` | Target position (x,y) | 0.2° visual angle | 100 ms |
| `PeripheralSensor` | Hand position (x,y) | 2–5° (eccentricity-dependent) | 100 ms |
| `ProprioceptiveSensor` | Joint angles + velocities | ~1° position, ~5°/s velocity | 40 ms |

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
| 4 | λ controller, R/C commands, equifinality | `Arm`, `make_muscles`, `simulate_lambda` |
| 5 | VITE model, min-jerk trajectories | same as Week 4 |
| 6 | Inverse dynamics, static optimization | + `mass_matrix`, `coriolis`, `joint_accelerations` |
| 7 | Motor noise analysis, signal-dependent noise | same as Week 6 |
| 8 | Vision sensors, delay compensation | + `FovealSensor`, `PeripheralSensor`, `ProprioceptiveSensor` |
| 9 | Kalman filter, sensor fusion | same as Week 8 |
| 10 | Polar KF, pursuit efference copy, interception | + `PolarKF` |
| 11 | Hill muscle comparison, perturbation analysis | + `HillMuscle`, `make_hill_muscles`, `simulate_hill`, `simulate_kmhm`, `lambda_for_posture`, `make_ramp` |
| 12 | LQR (Riccati backward), KF (predict-update), LQG simulation, point-vs-bar MIP | same as Week 11 |
| 13 | iLQG iteration (Jacobians, extended Riccati, three-term forward pass), augmented-state delay matrices | + `make_x0`, `make_target`, `set_muscle_state`, `hill_step`, `forward_rollout` |
| 14 | Cerebellar ataxia: degraded forward model, increased process noise, impaired adaptation | same as Week 13 |
| 15 | Student project presentations | same as Week 13 |

## Week-by-Week Materials

| Wk | Topic | Lab and HW | Reading |
|----|-------|-----------|---------|
| 1 | Kinematics & Workspace | Lab01_Kinematics.ipynb, HW01_Reaching_Predictions.ipynb | Morasso (1981) |
| 2 | Muscle Mechanics | Lab02_Muscles.ipynb, HW02_MuscleSpeedLimit.ipynb | Zajac (1989) + Gribble et al. (1998) |
| 3 | Forward Dynamics | Lab03_Dynamics.ipynb, HW03_TriphasicReach.ipynb | Hollerbach & Flash (1982) |
| 4 | EPH / λ Model | Lab04_EPH.ipynb, HW04_EPH_Experiments.ipynb | Gribble et al. (1998) |
| 5 | VITE & Movement Invariants | Lab05_VITE.ipynb, HW05_Workspace_ViaPoint.ipynb | Flash & Hogan (1985) |
| 6 | Inverse Dynamics | Lab06_InverseDynamics.ipynb, HW06_InverseDynamics.ipynb | Crowninshield & Brand (1981) |
| 7 | Motor Noise & Variability | Lab07_Noise.ipynb, HW07_Noise.ipynb | Harris & Wolpert (1998) |
| 8 | Vision & Sensory Delay | Lab08_Vision.ipynb, HW08_Vision.ipynb | Keele & Posner (1968); Elliott et al. (2017) |
| 9 | Kalman Filter / Proprioception | Lab09_KalmanFilter.ipynb, HW09_WhenDoesVisionHelp.ipynb | Wolpert et al. (1995) |
| 10 | From Reaching to Interception | Lab10_Interception.ipynb, HW10_Bayesian_Interception.ipynb | Fooken et al. (2021) |
| 11 | When Motor Commands Meet Muscle | Lab11_EPH_Muscle.ipynb, HW11_EPH_Muscle.ipynb | Gribble et al. (1998) + Kistemaker et al. (2006) |
| 12 | Optimal Feedback Control | Lab12_OFC.ipynb, HW12_OFC.ipynb | Todorov & Jordan (2002) |
| 13 | Linearizing the Plant (iLQG) | Lab13_iLQG.ipynb, HW13_iLQG.ipynb | Todorov (2005) |
| 14 | Cerebellar Ataxia | Lab14_CerebellarAtaxia.ipynb | Kakei et al. (2026) |
| 15 | Student Presentations | Project slides + code | — |

## Running Tests

```bash
cd smc/
pip install -e ".[dev]"
pytest tests/ -v
```

## References

- Gribble, P. L., Ostry, D. J., Sanguineti, V., & Laboissière, R. (1998). Are complex control signals required for human arm movement? *Journal of Neurophysiology*, 79(3), 1409–1424.
- Kakei, S., Bostan, A. C., Ebner, T. J., Fakharian, M. A., Gomi, H., Guell, X., Hemelt, M., Hoang, H., Hull, C., Inoue, M., Ishikawa, T., Kameda, M., Kawato, M., Kitazawa, S., Manto, M., Medina, J. F., Mitoma, H., Ohmae, K., Ohmae, S., … Yamazaki, T. (2026). Consensus Paper: Models of Cerebellar Functions. *The Cerebellum*, 25(1), 15.
- Kistemaker, D. A., Van Soest, A. J., & Bobbert, M. F. (2006). Is equilibrium point control feasible for fast goal-directed single-joint movements? *Journal of Neurophysiology*, 95(5), 2898–2912.
- Feldman, A. G. (1966). Functional tuning of the nervous system with control of movement or maintenance of a steady posture. *Biophysics*, 11(3), 565–578.
- Flash, T., & Hogan, N. (1985). The coordination of arm movements: An experimentally confirmed mathematical model. *Journal of Neuroscience*, 5(7), 1688–1703.
- Li, W., & Todorov, E. (2004). Iterative linear quadratic regulator design for nonlinear biological movement systems. *ICINCO*, 1, 222–229.
- Scott, S. H. (2004). Optimal feedback control and the neural basis of volitional motor control. *Nature Reviews Neuroscience*, 5(7), 532–546.
- Todorov, E. (2005). Stochastic optimal control and estimation methods adapted to the noise characteristics of the sensorimotor system. *Neural Computation*, 17(5), 1084–1108.
- Todorov, E., & Jordan, M. I. (2002). Optimal feedback control as a theory of motor coordination. *Nature Neuroscience*, 5(11), 1226–1235.

## License

MIT
