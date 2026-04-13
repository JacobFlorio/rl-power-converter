"""Train the neural MLP policy by backpropagation through the
differentiable buck converter simulation.

Each training step is one full rollout, and the gradient of the
rollout loss flows back through every duty command to the policy's
weights. This is much more sample-efficient than model-free RL
when the dynamics are known and differentiable — which is exactly
the case for an averaged-model converter.

Training with domain randomization over load resistance so the policy
doesn't overfit to a single operating point. Each rollout batches
several loads in parallel.

Run:
    python -m src.train_bptt
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .buck_sim import BuckParams, rollout
from .controllers import MLPPolicy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/policy_bptt.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    params = BuckParams()
    policy = MLPPolicy(hidden=args.hidden).to(args.device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    history = []
    print(f"BPTT training: {args.epochs} epochs × batch {args.batch} rollouts of {params.horizon} steps")
    for epoch in range(args.epochs):
        # Domain randomization over load resistance so we can't overfit
        R_sample = torch.empty(args.batch, device=args.device).uniform_(3.0, 8.0)
        out = rollout(policy, params, batch=args.batch,
                      R_nominal=R_sample, device=args.device)
        loss = out["mean_loss"]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
        opt.step()
        history.append({"epoch": epoch, "loss": float(loss.item())})
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:4d}  loss {loss.item():.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": policy.state_dict(),
        "config": {"hidden": args.hidden},
        "history": history,
    }, out)
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
