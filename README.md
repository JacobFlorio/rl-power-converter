# RL for DC-DC Buck Converter Control — Honest Comparison to PID

> **Headline: on a linear buck converter with a known model, a grid-search-tuned PID beats both BPTT-trained and CEM-trained neural network policies on every metric, at every tested operating point. The interesting finding is *why*: the neural policies have no integrator, so they inherit no steady-state-error guarantee — and it shows up most sharply on out-of-distribution loads (13–21% ss_err for the NN policies; ~0.03% for PID).** This is the opposite of the loose "does RL beat PID?" framing I started with, and it's a more textbook-aligned and honest answer.

Independent research by [Jacob Florio](https://github.com/JacobFlorio). Runs end to end in about 15 minutes (mostly BPTT training).

---

## Headline numbers

### Training-objective scoreboard (5Ω → 2.5Ω load step, 2 ms rollout)

| controller | training loss (lower is better) |
|---|---:|
| **Tuned PID** (Kp=8, Ki=1.25, Kd=4e-4) | **343.22** |
| BPTT MLP (32-unit, backprop through the sim) | 348.35 |
| CEM MLP (same architecture, model-free) | 383.83 |

### Nominal load-step response
![Nominal response](results/nominal_response.png)

Top: V_out traces. Bottom: duty cycle. Load steps from 5 Ω to 2.5 Ω at t = 1 ms. PID recovers cleanly with a ~350 mV overshoot and essentially zero steady-state error. BPTT is very close but carries a 60 mV static offset. CEM has a 750 mV overshoot and a 150 mV steady-state offset.

### Robustness — the integrator gap is the story
![Robustness](results/robustness.png)

Left: transient ISE vs load. Right: **steady-state error vs load** — this panel is the punchline.

| load R | PID ss_err | BPTT ss_err | CEM ss_err |
|---:|---:|---:|---:|
| 2 Ω | −0.001 V | **+0.433 V (13%)** | **+0.688 V (21%)** |
| 10 Ω | −0.001 V | +0.003 V | −0.069 V |
| 15 Ω | −0.001 V | −0.013 V | −0.149 V |
| 20 Ω | −0.001 V | −0.020 V | −0.190 V |

PID holds essentially zero steady-state error at every tested load. Both neural policies develop significant static offset out-of-distribution — a textbook symptom of missing integral action. **PID's integrator gives it zero-ss-err-under-constant-disturbance for free; the neural controllers would have to learn that structural property from the loss, and they don't fully manage it with the aggregate MSE objective I used.**

### Training curves
![Training curves](results/training_curves.png)

BPTT converges smoothly over ~400 epochs from an initial loss of ~25,000 down to ~348. CEM finds a comparable-but-worse region stochastically, bottoming out around 384 after 30 generations. Both are in the same ballpark as PID but neither crosses it.

## Why this is an interesting result, not a failure

The loose framing "does deep RL beat PID for transient control?" is answered in many published papers with a confident "yes." But those papers almost always involve either (a) **non-linear converter topologies** where PID's linearity assumption breaks (PFC boost converters at the DCM/CCM boundary, multi-level converters) or (b) **combined objectives** PID can't optimize directly (efficiency + transient + thermal + EMI). On a linear buck at a single operating point with a pure voltage-tracking objective, the **right structural prior is already baked into the integrator**, and a well-tuned PID is very hard to beat structurally.

The BPTT policy got within 1% of PID on the aggregate training loss, which is remarkable given it had to rediscover integral action from MSE alone. It didn't *fully* rediscover it — you can see the residual offset in the robustness plot — but it came surprisingly close. The model-free CEM policy is meaningfully worse, which is the cost you pay for throwing away the gradient when you have a differentiable simulator.

## Technical approach

- **Buck dynamics as a pure-torch module.** `src/buck_sim.py` implements the averaged buck converter state-space (I_L, V_out) as a one-step differentiable function. A 2000-step rollout is a sequence of these calls with gradients flowing through every one. See `docs/report.md` for the dynamics, parameters, and rollout details.
- **PID tuned by grid search + local refinement.** Coarse search over 7×7×5 = 245 gain combinations, then 3 rounds of 5-point local refinement. Final gains: Kp=8.0, Ki=1.25, Kd=4e-4. Uses a V_ref / V_in feedforward bias to avoid wasting integral wind-up on the DC operating point.
- **BPTT neural policy.** Two-hidden-layer MLP (32 units, Tanh, sigmoid output), trained for 400 full-rollout epochs with domain randomization over load resistance (R ∈ [3, 8] Ω). Gradient flows through every one of the 2000 simulation steps.
- **CEM model-free.** Same MLP trained by cross-entropy method (30 generations × 24 candidates, top-20% elitism, σ-decay 0.95). Included as a control: if the BPTT result were somehow a differentiable-sim artifact, CEM wouldn't see it.

## Reproduce it yourself

```bash
pip install -r requirements.txt          # just torch + numpy + matplotlib

python -m src.tune_pid                   # grid-search PID gains
python -m src.train_bptt                 # BPTT through the differentiable sim
python -m src.train_cem                  # CEM model-free baseline
python -m src.benchmark                  # nominal + robustness comparison
python -m src.plots                      # headline figures
```

About 15 minutes on an RTX 5080, mostly BPTT training.

## Full technical writeup

See [`docs/report.md`](docs/report.md) for the dynamics, the exact training configurations, the honest caveats (averaged model not switched, single operating point, fixed MLP architecture, no hardware-in-the-loop yet), and a roadmap toward the HIL rig + non-linear converter topologies where the RL-vs-PID comparison actually starts being interesting.

## Part of [AI-and-EE-Research](https://github.com/JacobFlorio/AI-and-EE-Research)

The broader index of my independent EE × AI research projects. Companion to:
- [mech-interp-tiny-transformer](https://github.com/JacobFlorio/mech-interp-tiny-transformer) — grokking + SAE recovery + causal ablation
- [edge-llm-eval-harness](https://github.com/JacobFlorio/edge-llm-eval-harness) — hardware-aware quantized-LLM eval
- [sae-rf-classifier](https://github.com/JacobFlorio/sae-rf-classifier) — SAE rediscovers classical modulation features
- [fpga-transformer-accel](https://github.com/JacobFlorio/fpga-transformer-accel) — bit-accurate systolic accelerator simulator
