"""Benchmark all three controllers on the same load-step transient.

Metrics:
  - steady-state error (V)
  - overshoot (V above setpoint)
  - settling time to 2% (ms from load step to recovery)
  - integral of squared error over the full rollout
  - max duty-cycle rate (for wear/filter sanity)

Also reports robustness at four out-of-distribution loads and produces
the comparison plot.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .buck_sim import BuckParams, rollout
from .controllers import PIDController, MLPPolicy


def load_policy(path: Path, device: str) -> MLPPolicy:
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = blob.get("config", {"hidden": 32})
    p = MLPPolicy(**cfg).to(device)
    p.load_state_dict(blob["state_dict"])
    p.eval()
    return p


def metrics_from_rollout(out: dict, params: BuckParams) -> dict:
    vout = out["states"][0, :, 2].cpu().numpy()
    duty = out["duties"][0, :, 0].cpu().numpy()
    t_arr = np.arange(params.horizon) * params.dt * 1000.0  # ms

    steady_end = vout[-50:].mean()
    ss_err = params.vref - steady_end
    overshoot = max(0.0, vout.max() - params.vref)

    # Settling time to 2% band AFTER the load step
    tol = 0.02 * params.vref
    step_idx = params.load_step_at
    in_band = np.abs(vout[step_idx:] - params.vref) <= tol
    settle_idx = None
    for i in range(len(in_band)):
        if in_band[i:].all():
            settle_idx = i
            break
    settling_ms = (settle_idx if settle_idx is not None else params.horizon - step_idx) * params.dt * 1000.0

    ise = float(((vout - params.vref) ** 2).sum())
    max_duty_rate = float(np.abs(np.diff(duty)).max())

    return {
        "steady_state_error_V": float(ss_err),
        "overshoot_V": float(overshoot),
        "settling_ms": float(settling_ms),
        "ise": ise,
        "max_duty_rate": max_duty_rate,
        "vout_trace": vout.tolist(),
        "duty_trace": duty.tolist(),
        "t_ms": t_arr.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", default="results/pid_gains.json")
    ap.add_argument("--bptt", default="results/policy_bptt.pt")
    ap.add_argument("--cem", default="results/policy_cem.pt")
    ap.add_argument("--out", default="results/benchmark.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    params = BuckParams()

    # Build controllers
    pid_gains = json.loads(Path(args.pid).read_text())
    pid = PIDController(Kp=pid_gains["Kp"], Ki=pid_gains["Ki"], Kd=pid_gains["Kd"],
                        dt=params.dt, batch=1, device=args.device)
    bptt = load_policy(Path(args.bptt), args.device)
    cem = load_policy(Path(args.cem), args.device) if Path(args.cem).exists() else None

    controllers = [("PID (tuned)", pid), ("BPTT NN", bptt)]
    if cem is not None:
        controllers.append(("CEM NN", cem))

    # 1. Headline: nominal load-step response (R=5, step to R=2.5)
    print("=== nominal load step (R=5 → 2.5Ω at t=1ms) ===")
    results = {"gains": pid_gains, "nominal": {}, "robustness": {}}
    for name, ctrl in controllers:
        if hasattr(ctrl, "reset"):
            ctrl.reset(1)
        with torch.no_grad():
            out = rollout(ctrl, params, batch=1, device=args.device)
        m = metrics_from_rollout(out, params)
        results["nominal"][name] = m
        print(f"  {name:15s}  ISE {m['ise']:>8.1f}  "
              f"settle {m['settling_ms']:>5.2f} ms  "
              f"overshoot {m['overshoot_V']:+.3f} V  "
              f"ss_err {m['steady_state_error_V']:+.4f} V")

    # 2. Robustness: 4 out-of-distribution loads
    print("\n=== robustness to out-of-distribution loads ===")
    oof_loads = [2.0, 10.0, 15.0, 20.0]
    for R in oof_loads:
        print(f"  load R={R}Ω")
        results["robustness"][str(R)] = {}
        for name, ctrl in controllers:
            if hasattr(ctrl, "reset"):
                ctrl.reset(1)
            with torch.no_grad():
                out = rollout(ctrl, params, batch=1,
                              R_nominal=torch.tensor([R], device=args.device),
                              device=args.device)
            m = metrics_from_rollout(out, params)
            # Drop the traces for robustness rows to keep JSON small
            m = {k: v for k, v in m.items()
                 if k not in ("vout_trace", "duty_trace", "t_ms")}
            results["robustness"][str(R)][name] = m
            print(f"    {name:15s}  ISE {m['ise']:>8.1f}  "
                  f"settle {m['settling_ms']:>5.2f} ms  "
                  f"ss_err {m['steady_state_error_V']:+.4f} V")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved → {out_path}")


if __name__ == "__main__":
    main()
