"""
Sensor: delayed, noisy measurement of a state vector.

Introduced in Week 8 (Vision & Sensory Delay) and used throughout
Module 3 (Weeks 8–11). The Sensor class models a sensory channel as:

    y(t) = x(t - Δ) + ε(t),   ε(t) ~ N(0, R)

where Δ is the sensory delay, R is the noise covariance matrix,
and x is the true state.

Parameters
----------
delay : float
    Sensory delay in seconds (e.g. 0.150 for central vision, 0.080 for
    peripheral vision, 0.030 for proprioception).
R : np.ndarray, shape (n, n)
    Noise covariance matrix. For isotropic noise with standard deviation σ,
    pass ``np.eye(n) * σ**2``. For anisotropic noise (Week 10+), pass the
    full covariance matrix.
dt : float, optional
    Simulation timestep in seconds. Default is 0.001 (1 kHz). This value
    is fixed throughout the course and should only be changed after
    consulting with the instructors.
sample_interval : float or None, optional
    Minimum interval (in seconds) between successive measurements. If None
    (default), the sensor returns a new measurement at every timestep.
    If set (e.g. 0.033 for ~30 Hz), the sensor returns the most recent
    measurement between samples, modeling an intermittent controller.

    Note: it is not known whether the nervous system samples sensory
    information continuously or at discrete intervals for the purpose of
    online control. Some researchers favor the intermittent control
    hypothesis (Loram et al., 2011; Gawthrop et al., 2011), which posits
    that the CNS updates its motor commands at a fixed or variable sampling
    rate rather than continuously. Others argue that the feedback loop is
    effectively continuous given the high firing rates of sensory neurons.
    The sample_interval parameter lets you explore both regimes.

Usage
-----
>>> import numpy as np
>>> from smc import Sensor
>>>
>>> # Central vision: 150ms delay, 1mm isotropic noise
>>> central = Sensor(delay=0.150, R=np.eye(2) * 0.001**2)
>>>
>>> # Peripheral vision: 80ms delay, 5mm isotropic noise
>>> peripheral = Sensor(delay=0.080, R=np.eye(2) * 0.005**2)
>>>
>>> # In the simulation loop:
>>> rng = np.random.default_rng(42)
>>> central.reset()
>>> y = central.sense(true_hand_position, rng)
"""

import numpy as np
from math import ceil


class Sensor:
    """Delayed, noisy sensory channel.

    Stores a ring buffer of recent true states and returns a delayed,
    noise-corrupted measurement on each call to ``sense()``.
    """

    def __init__(self, delay, R, dt=0.001, sample_interval=None):
        self.delay = delay
        self.R = np.asarray(R, dtype=float)
        self.dt = dt
        self.buf_len = max(1, ceil(delay / dt))
        self.sample_interval = sample_interval
        self._sample_steps = (
            max(1, round(sample_interval / dt))
            if sample_interval is not None
            else 1
        )
        # Precompute Cholesky for efficient noise generation
        self._L = np.linalg.cholesky(self.R)
        self.reset()

    def reset(self):
        """Clear the delay buffer and step counter."""
        self._buf = []
        self._step = 0
        self._last_measurement = None

    def sense(self, true_state, rng):
        """Return a delayed, noisy measurement.

        Parameters
        ----------
        true_state : array-like, shape (n,)
            The true state at the current timestep.
        rng : np.random.Generator
            Random number generator for reproducibility.

        Returns
        -------
        y : np.ndarray, shape (n,)
            The delayed, noisy measurement. If sample_interval is set and
            the current timestep is between samples, returns the most
            recent measurement (no new noise draw).
        """
        true_state = np.asarray(true_state, dtype=float)
        self._buf.append(true_state.copy())

        # Read delayed state from buffer
        idx = max(0, len(self._buf) - 1 - self.buf_len)
        delayed_state = self._buf[idx]

        # Check if this is a sampling timestep
        if self._step % self._sample_steps == 0 or self._last_measurement is None:
            # Draw noise: ε = L @ z, where z ~ N(0, I)
            z = rng.standard_normal(len(true_state))
            noise = self._L @ z
            self._last_measurement = delayed_state + noise

        self._step += 1
        return self._last_measurement.copy()

    def __repr__(self):
        sigma_diag = np.sqrt(np.diag(self.R))
        return (
            f"Sensor(delay={self.delay*1000:.0f}ms, "
            f"σ={sigma_diag}, "
            f"dt={self.dt}, "
            f"sample_interval={self.sample_interval})"
        )
