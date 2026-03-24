"""
Week 10: Polar Kalman filter for target state estimation on circular arc.
Three conditions: pursuit (foveal + efference copy), fixation + MT velocity,
fixation position-only.

Classes:
    ArcTarget     - target moving on circular arc at constant ω
    Eye           - oculomotor model: fixation → saccade → post-saccade → pursuit
    PolarTgtKF    - Kalman filter on [θ, ω] with position and velocity observations

Functions:
    run_diag()     - run single trial, returns diagnostics dict
    run_diag_MT()  - fixation with MT/MST velocity channel (10 ms sampling)
"""
import numpy as np
from smc import Arm, Q_REF

# ─── Constants ───
arm = Arm()
fk = arm.forward_kinematics
dt = 0.001  # s
start_hand = np.array(fk(Q_REF))


class ArcTarget:
    """Target moving CCW on a circular arc at constant angular velocity."""
    def __init__(self, center, R=0.15, omega_deg=60.0):
        self.c = np.array(center)
        self.R = R                    # arc radius (m)
        self.w = omega_deg            # angular velocity (°/s)

    def theta(self, t):
        """Angular position on arc (degrees) at time t."""
        return self.w * t

    def omega(self):
        """Angular velocity (°/s), constant."""
        return self.w

    def pos(self, t):
        """Cartesian position (m) at time t."""
        th = np.radians(self.theta(t))
        return self.c + self.R * np.array([np.cos(th), np.sin(th)])

    def vel(self, t):
        """Cartesian velocity (m/s) at time t."""
        th = np.radians(self.theta(t))
        return self.R * np.radians(self.w) * np.array([-np.sin(th), np.cos(th)])


class Eye:
    """
    Oculomotor model with four phases:
        fixation     (0 – 200 ms)    : eyes at arc center
        saccade      (200 – 220 ms)  : ballistic jump to target (20 ms)
        post_saccade (220 – 250 ms)  : settling + foveal acquisition (30 ms)
        pursuit      (250 ms +)      : smooth pursuit, eev ramps up

    Parameters:
        mode       : 'pursuit' or 'fixation'
        sacc_onset : saccade trigger time (s), default 0.200
        sacc_dur   : saccade duration (s), default 0.020
        post_sacc  : post-saccadic settling (s), default 0.030
        gain       : pursuit gain, default 0.9
        tau        : eye plant time constant (s), default 0.200
    """
    def __init__(self, mode='pursuit', sacc_onset=0.200, sacc_dur=0.020,
                 post_sacc=0.030, gain=0.9, tau=0.200):
        self.mode = mode
        self.sacc_onset = sacc_onset
        self.sacc_dur = sacc_dur
        self.post_sacc = post_sacc
        self.gain = gain
        self.tau = tau

    def reset(self, center):
        self.gaze_theta = 0.0       # gaze angle on arc (°)
        self.gaze_omega = 0.0       # gaze angular velocity (°/s)
        self.eev_omega = 0.0        # efference copy angular velocity (°/s)
        self.phase = 'fixation'
        self._t = 0
        self._sacc_start_theta = None
        self._sacc_target_theta = None

    def step(self, true_theta, kf_omega_est, dt):
        """Advance one timestep. Returns (gaze_theta, eev_omega, phase)."""
        self._t += dt
        if self.mode == 'fixation':
            self.eev_omega = 0.0
            return self.gaze_theta, self.eev_omega, 'fixation'

        sacc_land = self.sacc_onset + self.sacc_dur
        pursuit_start = sacc_land + self.post_sacc

        if self._t < self.sacc_onset:
            self.phase = 'fixation'
            self.eev_omega = 0.0
        elif self._t < sacc_land:
            if self._sacc_start_theta is None:
                self._sacc_start_theta = self.gaze_theta
                self._sacc_target_theta = true_theta + kf_omega_est * self.sacc_dur
            f = min((self._t - self.sacc_onset) / self.sacc_dur, 1.0)
            self.gaze_theta = self._sacc_start_theta + f * (self._sacc_target_theta - self._sacc_start_theta)
            self.gaze_omega = 0.0
            self.eev_omega = 0.0
            self.phase = 'saccade'
        elif self._t < pursuit_start:
            self.gaze_theta += (true_theta - self.gaze_theta) * 0.3 * dt
            self.gaze_omega = 0.0
            self.eev_omega = 0.0
            self.phase = 'post_saccade'
        else:
            retinal_slip = true_theta - self.gaze_theta
            vel_cmd = retinal_slip * 20.0
            self.gaze_omega += (vel_cmd - self.gaze_omega) / self.tau * dt
            self.gaze_theta += self.gaze_omega * dt
            self.eev_omega = self.gaze_omega
            self.phase = 'pursuit'

        return self.gaze_theta, self.eev_omega, self.phase


class PolarTgtKF:
    """
    Kalman filter on target state in polar arc coordinates.

    State: x = [θ, ω]ᵀ
        θ : angular position on arc (degrees)
        ω : angular velocity (°/s)

    Process model: θ += ω·dt, ω constant (exact for circular arc).

    Parameters:
        sigma_fov_deg  : foveal angular position noise (°), default 0.38
        sigma_per_deg  : peripheral angular position noise (°), default 1.91
        sigma_eev_deg  : efference copy velocity noise (°/s), default 5.0
        sample_interval: peripheral sampling interval (s), default 0.050
    """
    def __init__(self, sigma_fov_deg=0.38, sigma_per_deg=1.91,
                 sigma_eev_deg=5.0, sample_interval=0.050):
        self.sf = sigma_fov_deg
        self.sp = sigma_per_deg
        self.se = sigma_eev_deg
        self.si = sample_interval

        self.x = np.zeros(2)
        self.P = np.array([[5.0**2, 0], [0, 30.0**2]])

        self.A = np.array([[1.0, dt], [0.0, 1.0]])
        self.Q = np.array([[1e-6, 0], [0, 1e-6]])

        self.H_pos = np.array([[1.0, 0.0]])
        self.H_vel = np.array([[0.0, 1.0]])

        self._lpu = -1.0
        self._t = 0

    def reset(self, theta_est, omega_est=0.0):
        self.x = np.array([theta_est, omega_est])
        self.P = np.array([[5.0**2, 0], [0, 30.0**2]])
        self._lpu = -1.0
        self._t = 0

    def step(self, gaze_theta, true_theta, eev_omega, phase, rng, dt,
             slip_rate=None):
        """One predict-update cycle.
        
        Args:
            slip_rate: retinal slip rate (°/s), computed externally as
                       d(true_theta - gaze_theta)/dt. During pursuit, the
                       composite observation = eev + slip_rate ≈ target velocity
                       at all times (even during pursuit ramp-up).
        
        Returns (x_hat, K_norm, sigma_theta, sigma_omega).
        """
        self._t += dt

        # Predict
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

        # Update
        Kn = 0
        if phase in ('pursuit', 'post_saccade'):
            theta_obs = true_theta + rng.normal(0, self.sf)
            Kn = self._update_pos(theta_obs, self.sf)
            # Composite velocity: eev + retinal slip rate = target velocity
            if phase == 'pursuit' and abs(eev_omega) > 1.0:
                sr = slip_rate if slip_rate is not None else 0.0
                omega_composite = eev_omega + sr
                self._update_vel(omega_composite, self.se)
        elif phase == 'fixation':
            if self._t - self._lpu >= self.si:
                theta_obs = true_theta + rng.normal(0, self.sp)
                Kn = self._update_pos(theta_obs, self.sp)
                self._lpu = self._t

        return (self.x.copy(), Kn,
                np.sqrt(self.P[0, 0]), np.sqrt(self.P[1, 1]))

    def _update_pos(self, y, sigma):
        R = np.array([[sigma**2]])
        inn = y - self.H_pos @ self.x
        S = self.H_pos @ self.P @ self.H_pos.T + R
        K = self.P @ self.H_pos.T / S[0, 0]
        self.x = self.x + K.flatten() * inn
        self.P = (np.eye(2) - K @ self.H_pos) @ self.P
        return abs(K[0, 0])

    def _update_vel(self, y_omega, sigma):
        R = np.array([[sigma**2]])
        inn = y_omega - self.H_vel @ self.x
        S = self.H_vel @ self.P @ self.H_vel.T + R
        K = self.P @ self.H_vel.T / S[0, 0]
        self.x = self.x + K.flatten() * inn
        self.P = (np.eye(2) - K @ self.H_vel) @ self.P


# ═══════════════════════════════════════════════════════════
# Diagnostic runners
# ═══════════════════════════════════════════════════════════

def run_diag(gaze_mode, seed=42, T_total=0.850, sigma_omega_MT=12.0, mt_interval=0.010,
             mt_bias=1.12, mt_onset=0.060, pos_onset=0.080):
    """Run pursuit or fixation (position-only) trial.
    
    During pre-pursuit phases (fixation, saccade), MT velocity observations
    are included at mt_interval — both conditions use peripheral MT before saccade.
    During pursuit, composite velocity observation = eev + retinal slip rate.
    
    Parameters:
        mt_onset   : MT/MST velocity signal onset delay (s), default 0.060
        pos_onset  : peripheral position signal onset delay (s), default 0.080
    
    Returns diagnostics dict.
    """
    rng_k = np.random.default_rng(seed * 1000 + 1)
    tgt = ArcTarget(start_hand, omega_deg=60)
    n = int(T_total / dt) + 1

    eye = Eye(mode=gaze_mode)
    eye.reset(start_hand)
    kf = PolarTgtKF()
    kf.reset(theta_est=tgt.theta(0) + rng_k.normal(0, 2.0), omega_est=0.0)

    d = dict(true_theta=np.zeros(n), true_omega=np.zeros(n),
             kf_theta=np.zeros(n), kf_omega=np.zeros(n),
             sigma_theta=np.zeros(n), sigma_omega=np.zeros(n),
             Kn=np.zeros(n), gaze_theta=np.zeros(n), phase=[])

    prev_slip = 0.0   # previous retinal slip for finite difference
    last_mt = -1.0     # MT velocity observation timer

    # Set initial _lpu so first position observation fires at ~pos_onset
    kf._lpu = pos_onset - kf.si

    for i in range(n):
        t = i * dt
        true_th = tgt.theta(t)
        true_om = tgt.omega()

        gz_th, eev_om, ph = eye.step(true_th, kf.x[1], dt)
        kph = ph if gaze_mode == 'pursuit' else 'fixation'

        # Compute retinal slip rate for pursuit composite
        curr_slip = true_th - gz_th
        slip_rate = (curr_slip - prev_slip) / dt if i > 0 else 0.0
        prev_slip = curr_slip

        xh, Kn, sig_th, sig_om = kf.step(
            gz_th, true_th, eev_om, kph, rng_k, dt, slip_rate=slip_rate)

        # MT velocity during pre-pursuit phases (onset at mt_onset)
        if kph in ('fixation', 'saccade'):
            if t >= mt_onset and kf._t - last_mt >= mt_interval:
                omega_obs = true_om * mt_bias + rng_k.normal(0, sigma_omega_MT)
                kf._update_vel(omega_obs, sigma_omega_MT)
                last_mt = kf._t
                sig_th = np.sqrt(kf.P[0, 0])
                sig_om = np.sqrt(kf.P[1, 1])
                xh = kf.x.copy()

        d['true_theta'][i] = true_th
        d['true_omega'][i] = true_om
        d['kf_theta'][i] = xh[0]
        d['kf_omega'][i] = xh[1]
        d['sigma_theta'][i] = sig_th
        d['sigma_omega'][i] = sig_om
        d['Kn'][i] = Kn
        d['gaze_theta'][i] = gz_th
        d['phase'].append(ph)

    d['n'] = n
    return d


def run_diag_MT(seed=42, T_total=0.850, sigma_omega_MT=12.0, mt_interval=0.010,
               mt_bias=1.12, mt_onset=0.060, pos_onset=0.080):
    """Fixation with MT/MST direct velocity observation (every 10 ms)
    and peripheral position observation (every 50 ms).

    Parameters:
        sigma_omega_MT : MT velocity noise (°/s), default 12.0
        mt_interval    : MT temporal integration window (s), default 0.010
        mt_bias        : Aubert-Fleischl multiplicative bias, default 1.12
        mt_onset       : MT signal onset delay (s), default 0.060
        pos_onset      : position signal onset delay (s), default 0.080
    """
    rng_k = np.random.default_rng(seed * 1000 + 1)
    tgt = ArcTarget(start_hand, omega_deg=60)
    n = int(T_total / dt) + 1

    kf = PolarTgtKF()
    kf.reset(theta_est=tgt.theta(0) + rng_k.normal(0, 2.0), omega_est=0.0)

    d = dict(true_theta=np.zeros(n), true_omega=np.zeros(n),
             kf_theta=np.zeros(n), kf_omega=np.zeros(n),
             sigma_theta=np.zeros(n), sigma_omega=np.zeros(n))

    last_mt = -1.0

    # Set initial _lpu so first position observation fires at ~pos_onset
    kf._lpu = pos_onset - kf.si

    for i in range(n):
        t = i * dt
        true_th = tgt.theta(t)
        true_om = tgt.omega()

        # Predict
        kf.x = kf.A @ kf.x
        kf.P = kf.A @ kf.P @ kf.A.T + kf.Q
        kf._t += dt

        # Peripheral position: every 50 ms (onset at pos_onset)
        if t >= pos_onset and kf._t - kf._lpu >= kf.si:
            theta_obs = true_th + rng_k.normal(0, kf.sp)
            kf._update_pos(theta_obs, kf.sp)
            kf._lpu = kf._t

        # MT/MST velocity: every 10 ms (onset at mt_onset, biased)
        if t >= mt_onset and kf._t - last_mt >= mt_interval:
            omega_obs = true_om * mt_bias + rng_k.normal(0, sigma_omega_MT)
            kf._update_vel(omega_obs, sigma_omega_MT)
            last_mt = kf._t

        d['true_theta'][i] = true_th
        d['true_omega'][i] = true_om
        d['kf_theta'][i] = kf.x[0]
        d['kf_omega'][i] = kf.x[1]
        d['sigma_theta'][i] = np.sqrt(kf.P[0, 0])
        d['sigma_omega'][i] = np.sqrt(kf.P[1, 1])

    d['n'] = n
    return d


# ═══════════════════════════════════════════════════════════
# §5: Feedforward interception plan
# ═══════════════════════════════════════════════════════════

def feedforward_plan(seed=42, T_obs=0.200, T_cross=0.550, sigma_T_move=0.050,
                     T_launch=0.245, sigma_T_launch=0.008, v_ratio_cross=0.70):
    """
    Compute feedforward interception plan from Fixation+MT estimates.

    Parameters:
        T_obs          : observation duration (s), estimates taken at this time
        T_cross        : planned time from launch to arc crossing (s)
        sigma_T_move   : movement duration noise (s)
        T_launch       : launch time from target onset (s)
        sigma_T_launch : launch time noise (s)
        v_ratio_cross  : fraction of peak velocity at arc crossing (0.65-0.75)

    Returns dict with plan parameters and uncertainty budget.
    """
    from scipy.optimize import brentq

    # Get MT estimates at observation time
    dm = run_diag_MT(seed=seed)
    t_idx = int(T_obs / dt)
    theta_hat = dm['kf_theta'][t_idx]
    omega_hat = dm['kf_omega'][t_idx]
    sig_th = dm['sigma_theta'][t_idx]
    sig_om = dm['sigma_omega'][t_idx]

    # Min-jerk crossing fraction
    def v_rat(x): return x**2 * (1-x)**2 / 0.0625
    x_cross = brentq(lambda x: v_rat(x) - v_ratio_cross, 0.5, 0.95)

    # Movement parameterization
    def mj_pos(x): return 10*x**3 - 15*x**4 + 6*x**5
    T_total_mj = T_cross / x_cross
    A = 0.15 / mj_pos(x_cross)
    v_peak = 1.875 * A / T_total_mj
    v_cross = v_ratio_cross * v_peak

    # Interception angle
    theta_intercept = theta_hat + omega_hat * T_cross
    true_intercept = 60.0 * (T_launch + T_cross)
    target_at_launch = 60.0 * T_launch
    lead_angle = theta_intercept - target_at_launch
    ff_error = true_intercept - theta_intercept

    # Uncertainty budget
    sig_T = np.sqrt(sigma_T_move**2 + sigma_T_launch**2)
    sig_from_pos = sig_th
    sig_from_vel = T_cross * sig_om
    sig_from_timing = omega_hat * sig_T
    sig_intercept = np.sqrt(sig_from_pos**2 + sig_from_vel**2 + sig_from_timing**2)

    return dict(
        # Estimates at plan time
        theta_hat=theta_hat, omega_hat=omega_hat,
        sig_th=sig_th, sig_om=sig_om,
        # Plan
        theta_intercept=theta_intercept, lead_angle=lead_angle,
        true_intercept=true_intercept, ff_error=ff_error,
        # Movement
        T_cross=T_cross, T_total_mj=T_total_mj,
        A=A, v_peak=v_peak, v_cross=v_cross,
        x_cross=x_cross,
        # Uncertainty
        sig_intercept=sig_intercept,
        sig_from_pos=sig_from_pos,
        sig_from_vel=sig_from_vel,
        sig_from_timing=sig_from_timing,
        sig_T=sig_T,
    )
