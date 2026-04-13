"""Headline plots for the rl-power-converter study.

Produces:
  - nominal_response.png   vout + duty for the nominal load step, three
                           controllers overlaid.
  - robustness.png         settling ISE vs out-of-distribution load R,
                           three controllers as lines.
  - steady_state_error.png ss_err vs load R, showing the integrator gap.
  - training_curves.png    BPTT loss per epoch + CEM best-per-generation.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


COLORS = {
    "PID (tuned)": "#1f77b4",
    "BPTT NN": "#2ca02c",
    "CEM NN": "#d62728",
}
STYLES = {
    "PID (tuned)": "-",
    "BPTT NN": "-",
    "CEM NN": "--",
}


def nominal_plot(data: dict, out: Path):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax_v, ax_d = axes

    for name, m in data["nominal"].items():
        t = np.array(m["t_ms"])
        v = np.array(m["vout_trace"])
        d = np.array(m["duty_trace"])
        ax_v.plot(t, v, color=COLORS[name], ls=STYLES[name], lw=2, label=name)
        ax_d.plot(t, d, color=COLORS[name], ls=STYLES[name], lw=1.5,
                  alpha=0.9, label=name)

    ax_v.axhline(3.3, color="black", ls=":", lw=1, alpha=0.5, label="V_ref = 3.3 V")
    ax_v.axvline(1.0, color="grey", ls=":", lw=1, alpha=0.5)
    ax_v.text(1.02, 2.6, "load step\n5Ω → 2.5Ω", fontsize=8, color="grey")
    ax_v.set_ylabel("V_out [V]")
    ax_v.set_ylim(0, 4.0)
    ax_v.set_title("Load-step transient response (R = 5Ω → 2.5Ω at t = 1 ms)")
    ax_v.grid(alpha=0.3)
    ax_v.legend(loc="lower right", fontsize=9)

    ax_d.axvline(1.0, color="grey", ls=":", lw=1, alpha=0.5)
    ax_d.set_xlabel("time [ms]")
    ax_d.set_ylabel("duty cycle")
    ax_d.set_ylim(0, 1)
    ax_d.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved → {out}")


def robustness_plot(data: dict, out: Path):
    loads = sorted(float(k) for k in data["robustness"].keys())
    controllers = list(data["nominal"].keys())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_ise, ax_ss = axes

    for name in controllers:
        ise = [data["robustness"][str(R)][name]["ise"] for R in loads]
        sse = [data["robustness"][str(R)][name]["steady_state_error_V"] for R in loads]
        ax_ise.plot(loads, ise, "o-", color=COLORS[name], ls=STYLES[name],
                    lw=2, markersize=7, label=name)
        ax_ss.plot(loads, sse, "o-", color=COLORS[name], ls=STYLES[name],
                   lw=2, markersize=7, label=name)

    for ax in axes:
        ax.set_xlabel("load resistance R [Ω]")
        ax.grid(alpha=0.3)
    ax_ise.set_ylabel("integrated squared error")
    ax_ise.set_title("Transient ISE vs load (step to R/2)")
    ax_ise.legend(fontsize=9)

    ax_ss.axhline(0, color="black", ls=":", lw=1, alpha=0.5)
    ax_ss.set_ylabel("steady-state error [V]")
    ax_ss.set_title("Steady-state error — PID has integrator, NNs don't")
    ax_ss.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved → {out}")


def training_curves(out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax_b, ax_c = axes

    bptt_blob = torch.load("results/policy_bptt.pt", map_location="cpu", weights_only=False)
    hist = bptt_blob["history"]
    xs = [r["epoch"] for r in hist]
    ys = [r["loss"] for r in hist]
    ax_b.plot(xs, ys, color=COLORS["BPTT NN"], lw=2)
    ax_b.set_yscale("log")
    ax_b.set_xlabel("epoch")
    ax_b.set_ylabel("training loss (log)")
    ax_b.set_title("BPTT — gradient through the sim")
    ax_b.grid(which="both", alpha=0.3)

    cem_blob = torch.load("results/policy_cem.pt", map_location="cpu", weights_only=False)
    cem_hist = cem_blob["history"]
    xs = [r["generation"] for r in cem_hist]
    ys = [r["best"] for r in cem_hist]
    means = [r["mean"] for r in cem_hist]
    ax_c.plot(xs, means, color="#ccc", lw=1, label="population mean")
    ax_c.plot(xs, ys, color=COLORS["CEM NN"], lw=2, label="best")
    ax_c.set_yscale("log")
    ax_c.set_xlabel("generation")
    ax_c.set_ylabel("training loss (log)")
    ax_c.set_title("CEM — model-free search")
    ax_c.grid(which="both", alpha=0.3)
    ax_c.legend(fontsize=9)

    fig.suptitle("Training curves — BPTT converges smoothly, CEM explores stochastically")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved → {out}")


def main():
    data = json.loads(Path("results/benchmark.json").read_text())
    nominal_plot(data, Path("results/nominal_response.png"))
    robustness_plot(data, Path("results/robustness.png"))
    training_curves(Path("results/training_curves.png"))


if __name__ == "__main__":
    main()
