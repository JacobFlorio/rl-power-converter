"""Grid-search tune the PID gains on the same rollout objective the
neural policy sees. This gives us a tuned-PID baseline the RL policy
has to actually beat, rather than a weak strawman.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .buck_sim import BuckParams, rollout
from .controllers import PIDController


def score_pid(Kp: float, Ki: float, Kd: float, params: BuckParams, device: str) -> float:
    pid = PIDController(Kp=Kp, Ki=Ki, Kd=Kd, dt=params.dt, batch=1, device=device)
    with torch.no_grad():
        out = rollout(pid, params, batch=1, device=device)
    return float(out["mean_loss"].item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/pid_gains.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    params = BuckParams()

    # Coarse grid — log-spaced since the right answer can be orders of
    # magnitude apart for converters like this.
    Kps = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    Kis = [10, 50, 100, 500, 1000, 5000, 10000]
    Kds = [0.0, 1e-6, 1e-5, 1e-4, 1e-3]

    best = None
    n = 0
    total = len(Kps) * len(Kis) * len(Kds)
    print(f"searching {total} gain combinations...")
    for Kp in Kps:
        for Ki in Kis:
            for Kd in Kds:
                n += 1
                try:
                    loss = score_pid(Kp, Ki, Kd, params, args.device)
                except Exception:
                    continue
                if best is None or loss < best["loss"]:
                    best = {"Kp": Kp, "Ki": Ki, "Kd": Kd, "loss": loss}
                    print(f"  [{n}/{total}] new best Kp={Kp} Ki={Ki} Kd={Kd}  loss {loss:.2f}")

    # Fine local search around the best
    print(f"\nfine local search near Kp={best['Kp']} Ki={best['Ki']} Kd={best['Kd']}")
    center = best
    coarse = best
    for _ in range(3):
        Kps = [center["Kp"] * f for f in [0.5, 0.7, 1.0, 1.4, 2.0]]
        Kis = [center["Ki"] * f for f in [0.5, 0.7, 1.0, 1.4, 2.0]]
        Kds = [center["Kd"] * f for f in [0.5, 1.0, 2.0]] if center["Kd"] > 0 else [0.0, 1e-5, 1e-4]
        improved = None
        for Kp in Kps:
            for Ki in Kis:
                for Kd in Kds:
                    loss = score_pid(Kp, Ki, Kd, params, args.device)
                    if improved is None or loss < improved["loss"]:
                        improved = {"Kp": Kp, "Ki": Ki, "Kd": Kd, "loss": loss}
        if improved["loss"] < center["loss"]:
            center = improved
            print(f"  refined → Kp={center['Kp']:.4g} Ki={center['Ki']:.4g} Kd={center['Kd']:.4g} loss {center['loss']:.2f}")
        else:
            break

    best = center
    print(f"\nbest PID gains: Kp={best['Kp']:.4g}  Ki={best['Ki']:.4g}  Kd={best['Kd']:.4g}")
    print(f"best loss: {best['loss']:.3f}  (coarse start: {coarse['loss']:.3f})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(best, indent=2))
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
