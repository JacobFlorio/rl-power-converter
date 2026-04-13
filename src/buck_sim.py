"""Differentiable averaged-model buck converter simulator.

Pure torch: one forward step is a function of (state, duty) that
returns the next state. The whole rollout is a sequence of such steps,
so gradients flow from the terminal loss back through every duty
command — exactly what you need to train a neural policy by BPTT.

Continuous-time state-space (averaged over the switching period):
    dI_L / dt  = (d · V_in − V_out) / L
    dV_out / dt = (I_L − V_out / R_load) / C

Euler discretization with a small dt. For typical power-electronics
values (L=47 µH, C=100 µF, dt=1 µs) this is stable and accurate enough
for control policy development. Not a replacement for SPICE, not a
substitute for hardware-in-the-loop — but fine for the sim-only RL
study the README asks for.

State vector for the policy is [v_err, i_L, v_out, prev_duty], which
gives the controller the signal, its measured current, and the last
command (so it can learn to avoid command jitter).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import torch


STATE_DIM = 4
ACTION_DIM = 1


@dataclass
class BuckParams:
    vin: float = 12.0            # input voltage [V]
    vref: float = 3.3            # regulation target [V]
    L: float = 47e-6             # inductor [H]
    C: float = 100e-6            # output cap [F]
    R_nominal: float = 5.0       # load resistance [Ω]
    dt: float = 1e-6             # simulation timestep [s]
    horizon: int = 2000          # total simulation steps
    load_step_at: int = 1000     # step index at which load halves
    load_step_factor: float = 0.5
    duty_rate_penalty: float = 1e-4

    def total_time_ms(self) -> float:
        return self.horizon * self.dt * 1000.0


def init_state(params: BuckParams, batch: int, device: str = "cpu") -> torch.Tensor:
    """Zero initial state [B, 4]."""
    return torch.zeros(batch, STATE_DIM, device=device)


def step(state: torch.Tensor, duty: torch.Tensor, R_load: torch.Tensor,
         params: BuckParams) -> torch.Tensor:
    """One Euler step of the averaged buck converter dynamics.

    state: [B, 4] = [v_err, i_L, v_out, prev_duty]  (v_err is dropped on update)
    duty:  [B, 1] duty cycle in [0, 1]  (saturated by caller)
    R_load: [B] current load resistance
    """
    i_L = state[:, 1]
    v_out = state[:, 2]

    d = duty.squeeze(-1).clamp(0.0, 1.0)
    di_L = (d * params.vin - v_out) / params.L
    dv_out = (i_L - v_out / R_load) / params.C

    i_L_new = i_L + di_L * params.dt
    v_out_new = v_out + dv_out * params.dt
    v_err_new = params.vref - v_out_new

    new_state = torch.stack([v_err_new, i_L_new, v_out_new, d], dim=-1)
    return new_state


def rollout(policy: Callable[[torch.Tensor], torch.Tensor],
            params: BuckParams,
            batch: int = 1,
            R_nominal: torch.Tensor | None = None,
            device: str = "cpu",
            seed: int | None = None) -> dict:
    """Run a full rollout of `params.horizon` steps and return the
    full state trajectory + duty commands + losses.

    `policy` is any callable that maps a [B, STATE_DIM] tensor to a
    [B, ACTION_DIM] tensor. A PID controller is wrapped in a callable
    the same way, so both neural and classical policies use this.

    `R_nominal` is an optional [B]-shaped tensor of load resistances;
    defaults to params.R_nominal for every run. A load step to
    `R_nominal * params.load_step_factor` happens at step `params.load_step_at`.
    """
    if R_nominal is None:
        R0 = torch.full((batch,), params.R_nominal, device=device)
    else:
        R0 = R_nominal.to(device)

    state = init_state(params, batch, device=device)
    state[:, 0] = params.vref  # initial v_err = vref (v_out starts at 0)

    states, duties = [], []
    total_loss = torch.zeros(batch, device=device)
    prev_duty = torch.zeros(batch, 1, device=device)

    for t in range(params.horizon):
        R_t = torch.where(torch.tensor(t >= params.load_step_at, device=device),
                           R0 * params.load_step_factor,
                           R0)
        duty = policy(state)                       # [B, 1]
        duty = duty.clamp(0.0, 1.0)
        new_state = step(state, duty, R_t, params)

        err = params.vref - new_state[:, 2]
        duty_delta = (duty - prev_duty).squeeze(-1)
        loss_t = err ** 2 + params.duty_rate_penalty * duty_delta ** 2
        total_loss = total_loss + loss_t

        states.append(new_state)
        duties.append(duty)
        state = new_state
        prev_duty = duty

    states = torch.stack(states, dim=1)   # [B, T, 4]
    duties = torch.stack(duties, dim=1)   # [B, T, 1]
    return {
        "states": states,
        "duties": duties,
        "total_loss": total_loss,
        "mean_loss": total_loss.mean(),
    }
