"""
Hill-type muscle model with contractile element (CE), series elastic
element (SE), and parallel elastic element (PE).

Based on Kistemaker, Van Soest & Bobbert (2006), "Is equilibrium point
control feasible for fast goal-directed single-joint movements?"
J. Neurophysiol. 95: 2898-2912.

Key difference from the Gribble muscle (muscle.py): tendon compliance.
In the Gribble model, muscle length = MTC length (no tendon).
Here, l_MTC = l_CE + l_SE, and l_SE depends on force.
Muscle spindles sense l_CE, not l_MTC.
"""
import numpy as np
from .params import Q_REF

# ── Kistemaker nonspecific parameters (Table A1) ──
HATZE_M = 11.30       # activation dynamics rate (1/s)
HATZE_C = 1.37e-4     # rho shape constant
HATZE_ETA = 5.27e4    # rho shape constant
HATZE_Q0 = 5.00e-3    # baseline active state
HATZE_K = 2.90        # rho shape constant
WIDTH = 0.56          # force-length width (in l_CE_opt units)
A_REL = 0.41          # Hill equation normalised a
B_REL = 5.20          # Hill equation normalised b (1/s)
Q_CRIT = 0.03         # activation threshold for velocity scaling
SE_STRAIN_AT_FMAX = 0.04   # 4 % strain at F_MAX

# ── Muscle-specific parameters for the 6-muscle smc arm ──
# (name, F_max [N], l_CE_opt [m], l_SE_0 [m], r_sh, r_el, rl)
# l_CE_opt + l_SE_0 = rl (rest MTC length at Q_REF)
HILL_MUSCLE_DEFS = [
    ("pec",    450.0, 0.100, 0.160,  0.04,  0.00, 0.26),
    ("bic_l",  350.0, 0.092, 0.168,  0.00,  0.03, 0.26),
    ("bic_s",  120.0, 0.110, 0.180,  0.025, 0.03, 0.29),
    ("delt",   450.0, 0.100, 0.160, -0.04,  0.00, 0.26),
    ("tri_l",  400.0, 0.093, 0.167,  0.00, -0.02, 0.26),
    ("tri_lg", 200.0, 0.115, 0.205, -0.04, -0.02, 0.32),
]


class HillMuscle:
    """Hill-type muscle with CE + SE + PE and Hatze activation dynamics.

    State variables
    ---------------
    gamma : float   — relative free Ca²⁺ (0 to 1)
    l_CE  : float   — contractile element length (m)
    """

    def __init__(self, name, F_max, l_CE_opt, l_SE_0, r_sh, r_el, rl):
        self.name = name
        self.F_max = F_max
        self.l_CE_opt = l_CE_opt
        self.l_SE_0 = l_SE_0
        self.r_sh = r_sh
        self.r_el = r_el
        self.rl = rl          # MTC rest length at Q_REF

        # Derived — force-length parabola coefficient
        self.a_fl = 1.0 / WIDTH ** 2

        # SE stiffness: F_max at SE_STRAIN_AT_FMAX strain
        se_stretch_at_fmax = SE_STRAIN_AT_FMAX * self.l_SE_0
        self.k_SE = F_max / se_stretch_at_fmax ** 2

        # PE: slack at l_CE_rel = 1.4, F_PE = 0.5·F_max at l_CE_rel = 1+WIDTH
        self.l_PE_0_rel = 1.4
        pe_stretch = (1.0 + WIDTH - self.l_PE_0_rel) * self.l_CE_opt
        if pe_stretch > 0:
            self.k_PE = 0.5 * F_max / pe_stretch ** 2
        else:
            self.k_PE = 0.0

        # State
        self.gamma = 0.005    # tiny baseline Ca²⁺
        self.l_CE = rl - l_SE_0   # CE at rest = MTC − tendon slack

    def reset(self):
        self.gamma = 0.005
        self.l_CE = self.rl - self.l_SE_0

    # ── geometry (identical to Gribble muscle) ──
    def mtc_length(self, q):
        return self.rl - self.r_sh * (q[0] - Q_REF[0]) - self.r_el * (q[1] - Q_REF[1])

    def mtc_velocity(self, qd):
        return -self.r_sh * qd[0] - self.r_el * qd[1]

    # ── SE / PE forces ──
    def se_force(self, l_SE):
        stretch = l_SE - self.l_SE_0
        return self.k_SE * stretch ** 2 if stretch > 0 else 0.0

    def pe_force_from_lCE(self, l_CE):
        l_rel = l_CE / self.l_CE_opt
        stretch = l_rel - self.l_PE_0_rel
        if stretch <= 0:
            return 0.0
        return self.k_PE * (stretch * self.l_CE_opt) ** 2

    # ── Hatze activation ──
    def active_state(self, l_CE_rel):
        """Convert gamma (free Ca²⁺) → q (active state), with length
        dependent Ca²⁺ sensitivity (Hatze 1981)."""
        if l_CE_rel <= 0.01 or l_CE_rel >= HATZE_K - 0.01:
            return HATZE_Q0
        rho = HATZE_C * HATZE_ETA * (HATZE_K - 1.0) / (
            (HATZE_K - l_CE_rel) * l_CE_rel)
        rg = rho * self.gamma
        return (HATZE_Q0 + rg ** 3) / (1.0 + rg ** 3)

    # ── isometric force-length ──
    def f_isom_norm(self, l_CE_rel):
        """Normalised isometric force (parabolic, 0 at 1±width)."""
        a = self.a_fl
        f = -a * l_CE_rel ** 2 + 2.0 * a * l_CE_rel - a + 1.0
        return max(f, 0.0)

    # ── Hill force-velocity → CE velocity ──
    def ce_velocity(self, q_act, l_CE_rel, F_CE_rel):
        """Return v_CE in m/s (positive = shortening)."""
        F_isom_n = self.f_isom_norm(l_CE_rel)
        if F_isom_n < 1e-8:
            F_isom_n = 1e-8
        if q_act < 1e-6:
            q_act = 1e-6

        # Normalised Hill parameters
        a_star = A_REL * F_isom_n if l_CE_rel > 1.0 else A_REL
        b_star = B_REL
        if q_act < Q_CRIT:
            frac = (q_act - Q_CRIT) / (HATZE_Q0 - Q_CRIT)
            b_star = B_REL * (1.0 - 0.9 * frac ** 2)
            b_star = max(b_star, 0.1)

        qF = q_act * F_isom_n

        if F_CE_rel <= qF:
            # ── concentric (shortening): classic Hill ──
            denom = F_CE_rel + q_act * a_star
            if abs(denom) < 1e-10:
                v_rel = 0.0
            else:
                v_rel = b_star * (F_CE_rel - qF) / denom
            # v_rel is negative (shortening) when F_CE < qF
        else:
            # ── eccentric (lengthening): linear with 2× slope at v=0 ──
            slope_conc = b_star * (qF + a_star) / (q_act * a_star) ** 2
            slope_ecc = slope_conc / 2.0      # Katz (1939): eccentric slope = ½ concentric
            v_rel = (F_CE_rel - qF) * slope_ecc
            # cap eccentric velocity
            v_rel = min(v_rel, 2.0)

        return -v_rel * self.l_CE_opt    # m/s, positive = shortening

    # ── main integration step ──
    def step(self, stim, q, qd, dt):
        """Advance one timestep.

        Parameters
        ----------
        stim : float    Muscle stimulation in [0, 1].
        q    : (2,)     Joint angles (rad).
        qd   : (2,)     Joint velocities (rad/s).
        dt   : float    Timestep (s).

        Returns
        -------
        force : float   Total muscle force (N) = SE force.
        l_CE  : float   Current CE length (m).
        v_CE  : float   CE velocity (m/s, positive = shortening).
        """
        stim = np.clip(stim, 0.0, 1.0)

        # 1. Activation dynamics: STIM → gamma
        self.gamma += dt * HATZE_M * (stim - self.gamma)
        self.gamma = np.clip(self.gamma, 0.0, 1.0)

        # 2. MTC length
        l_MTC = self.mtc_length(q)

        # 3. SE length and force
        l_SE = l_MTC - self.l_CE
        F_SE = self.se_force(l_SE)

        # 4. PE force
        F_PE = self.pe_force_from_lCE(self.l_CE)

        # 5. CE must produce: F_CE = F_SE - F_PE
        F_CE = max(F_SE - F_PE, 0.0)
        F_CE_rel = F_CE / self.F_max

        # 6. Active state
        l_CE_rel = self.l_CE / self.l_CE_opt
        q_act = self.active_state(l_CE_rel)

        # 7. CE velocity from Hill equation
        v_CE = self.ce_velocity(q_act, l_CE_rel, F_CE_rel)

        # 8. Update CE length (shortening decreases l_CE)
        self.l_CE -= v_CE * dt

        # Clamp CE length to prevent blow-up
        self.l_CE = np.clip(self.l_CE, 0.3 * self.l_CE_opt,
                            1.8 * self.l_CE_opt)

        return F_SE, self.l_CE, v_CE


def make_hill_muscles():
    """Create six Hill-type muscles matching the smc arm topology."""
    return [HillMuscle(*p) for p in HILL_MUSCLE_DEFS]
