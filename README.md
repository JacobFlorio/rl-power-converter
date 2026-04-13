# RL-Controlled DC-DC Buck Converter

Reinforcement learning for transient response control of a synchronous buck converter, benchmarked against a tuned PID baseline.

## Research question
Does a deep RL controller (PPO / SAC) provide measurably better load-step transient response than a well-tuned PID for a 12V→3.3V buck converter, once the sim-to-real gap is honestly characterized?

## Approach
1. Build an averaged-model buck converter simulator in Python.
2. Train PPO with domain randomization on inductor/ESR/load.
3. Implement PID baseline with Ziegler-Nichols + manual tuning.
4. HIL test on a Texas Instruments LAUNCHXL-F28379D.
5. Quantify settling time, overshoot, and steady-state error.

## Deliverables
- Sim environment in `src/sim/`
- Trained policies in `src/policies/`
- C2000 firmware in `src/firmware/`
- Comparative plots in `results/`
