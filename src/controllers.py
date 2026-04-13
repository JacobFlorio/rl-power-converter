"""Controllers: PID, MLP neural policy, and adapters for rollout().

Every controller is a `(state: [B, 4]) -> duty: [B, 1]` callable so they
plug into buck_sim.rollout unchanged.
"""
from __future__ import annotations
import torch
import torch.nn as nn


STATE_DIM = 4
ACTION_DIM = 1


class PIDController:
    """Standard PID with anti-windup via integral clamping.

    state[:, 0] is v_err, which is exactly the PID error signal. Derivative
    is computed from the change in error across steps. Integral is
    clamped to [-i_max, i_max].
    """
    def __init__(self, Kp: float, Ki: float, Kd: float,
                 dt: float, i_max: float = 10.0, batch: int = 1,
                 device: str = "cpu"):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.i_max = i_max
        self.device = device
        self.reset(batch)

    def reset(self, batch: int):
        self.integral = torch.zeros(batch, device=self.device)
        self.prev_err = torch.zeros(batch, device=self.device)

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        err = state[:, 0]  # v_err
        self.integral = torch.clamp(self.integral + err * self.dt, -self.i_max, self.i_max)
        d_err = (err - self.prev_err) / self.dt
        self.prev_err = err
        u = self.Kp * err + self.Ki * self.integral + self.Kd * d_err
        # PID output is not a duty cycle — it's a correction. The steady-state
        # duty for V_ref = 3.3, V_in = 12 is 0.275, so we add that as a bias.
        bias = 3.3 / 12.0
        duty = torch.clamp(bias + u, 0.0, 1.0)
        return duty.unsqueeze(-1)


class MLPPolicy(nn.Module):
    """Tiny MLP: [v_err, i_L, v_out, prev_duty] -> duty.

    Input is normalized by hand-picked scales so the network doesn't
    need to learn the magnitudes of volts vs amps.
    """
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.scale = nn.Buffer(torch.tensor([3.3, 5.0, 5.0, 1.0]))
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, ACTION_DIM),
            nn.Sigmoid(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = state / self.scale
        return self.net(x)
