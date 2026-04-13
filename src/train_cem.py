"""Train the MLP policy with the cross-entropy method (CEM) —
a model-free RL algorithm. Included as a robustness check: the BPTT
result relies on having a differentiable simulator; CEM does not
(it only evaluates the policy) so it's a fairer comparison when the
question is "does an RL-style search outperform hand-tuned PID."

CEM is the simplest thing that can reasonably be called RL here:
  1. Flatten the policy weights into a single parameter vector.
  2. Sample N candidates from a Gaussian around the current mean.
  3. Score each candidate by rolling out the policy and summing reward.
  4. Update the mean to the weighted average of the top-k candidates.
  5. Repeat.

No gradients through the sim. Only evaluation. ~50 lines.

Run:
    python -m src.train_cem
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .buck_sim import BuckParams, rollout
from .controllers import MLPPolicy


def get_flat_params(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def set_flat_params(model: torch.nn.Module, flat: torch.Tensor):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx:idx + n].view_as(p))
        idx += n


@torch.no_grad()
def score_params(model: torch.nn.Module, flat_theta: torch.Tensor,
                  params: BuckParams, R_batch: torch.Tensor,
                  device: str) -> float:
    set_flat_params(model, flat_theta)
    out = rollout(model, params, batch=R_batch.shape[0],
                  R_nominal=R_batch, device=device)
    return float(out["mean_loss"].item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--population", type=int, default=32)
    ap.add_argument("--elite-frac", type=float, default=0.2)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--sigma-decay", type=float, default=0.95)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/policy_cem.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    params = BuckParams()
    model = MLPPolicy(hidden=args.hidden).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"CEM: {args.population} candidates × {args.generations} generations, "
          f"{n_params} parameters")

    mean = get_flat_params(model).cpu().numpy()
    sigma = args.sigma
    n_elite = max(1, int(args.elite_frac * args.population))

    best_ever_loss = float("inf")
    history = []

    for gen in range(args.generations):
        R_batch = torch.tensor(rng.uniform(3.0, 8.0, size=args.batch),
                                dtype=torch.float32, device=args.device)

        # Sample population
        noise = rng.normal(size=(args.population, n_params)).astype(np.float32)
        candidates = mean[None, :] + sigma * noise
        losses = []
        for c in candidates:
            theta = torch.tensor(c, dtype=torch.float32, device=args.device)
            losses.append(score_params(model, theta, params, R_batch, args.device))
        losses = np.array(losses)

        # Select elites
        elite_idx = np.argsort(losses)[:n_elite]
        elites = candidates[elite_idx]
        mean = elites.mean(axis=0)
        # Shrink-toward-elite std
        sigma = sigma * args.sigma_decay

        gen_best = float(losses[elite_idx[0]])
        if gen_best < best_ever_loss:
            best_ever_loss = gen_best
        history.append({"generation": gen, "best": gen_best, "mean": float(losses.mean())})
        print(f"  gen {gen:3d}  best {gen_best:8.2f}  mean {losses.mean():8.2f}  sigma {sigma:.4f}")

    # Commit the final mean to the policy
    final = torch.tensor(mean, dtype=torch.float32, device=args.device)
    set_flat_params(model, final)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": {"hidden": args.hidden},
        "history": history,
        "best_ever_loss": best_ever_loss,
    }, out)
    print(f"\nbest loss seen: {best_ever_loss:.3f}")
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
