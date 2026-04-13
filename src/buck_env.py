"""Gymnasium env wrapping an averaged-model synchronous buck converter.

State: [vout_error, iL, vout]
Action: duty cycle in [0, 1]
Reward: -(vout_error^2 + 1e-4 * d(duty)^2)

Averaged model — ignores switching ripple, which is fine for control
policy development. HIL rig will see the real waveform.
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class BuckEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, vin: float = 12.0, vref: float = 3.3, L: float = 47e-6,
                 C: float = 100e-6, R: float = 5.0, dt: float = 1e-6, horizon: int = 2000):
        super().__init__()
        self.vin = vin; self.vref = vref
        self.L = L; self.C = C; self.R = R
        self.dt = dt; self.horizon = horizon
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.iL = 0.0
        self.vout = 0.0
        self.prev_d = 0.0
        self.t = 0
        if options and "load_step_at" in options:
            self.load_step_at = options["load_step_at"]
        else:
            self.load_step_at = self.horizon // 2
        return self._obs(), {}

    def _obs(self):
        return np.array([self.vref - self.vout, self.iL, self.vout], dtype=np.float32)

    def step(self, action):
        d = float(np.clip(action[0], 0.0, 1.0))
        R = self.R * (0.5 if self.t >= self.load_step_at else 1.0)  # load step
        diL = (d * self.vin - self.vout) / self.L
        dvout = (self.iL - self.vout / R) / self.C
        self.iL += diL * self.dt
        self.vout += dvout * self.dt
        err = self.vref - self.vout
        reward = -(err ** 2) - 1e-4 * (d - self.prev_d) ** 2
        self.prev_d = d
        self.t += 1
        done = self.t >= self.horizon
        return self._obs(), float(reward), done, False, {}
