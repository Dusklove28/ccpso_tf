---
name: rl-tensorflow-introduction
description: Use when working in this RL_Tensorflow PSO/DDPG project, especially to understand the current RLCCPSO mainline, Q-centered CCPSO theory, and experiment constraints.
---

# RL TensorFlow Project Memory

Use these notes to avoid reviving old experimental branches by accident.

## Current Code Status

The runnable project has been narrowed to the mainline:

```text
PSO baseline
RLPSO: NormalEnv + binary reward
RLCCPSO: second-order CCPSO + delayed progress prior + continuous reward
```

`task/all_tasks_generate.py` no longer reads a mode-selection environment
variable. Do not suggest old ablation, noise-sweep, pbest-reset,
evolutionary-state-control, or center-trust commands unless the user explicitly
asks to restore those branches.

Run the current experiment with:

```bash
CUDA_VISIBLE_DEVICES=0 python -u main.py
```

## Theory

RLPSO controls ordinary PSO coefficients such as `w`, `c1`, `c2`, and mutation.
RLCCPSO controls the Q-centered convergence strength through `Conv_a`.

The current CCPSO is second-order DualC:

```text
Q = (c1*r1*pbest + c2*r2*gbest) / (c1*r1 + c2*r2)
X(t+1)=C*a1*X(t)+C*a2*X(t-1)+[1-C*(a1+a2)]Q
```

Before boundary clipping, the coefficients sum to one, so the update satisfies
the unified affine form. `Conv_a` scales movement around `Q`; it does not move
`Q`.

F1 stagnation diagnosis:

```text
early pbest/gbest collapse -> Q collapse -> Conv_a only scales motion around
the wrong Q
```

## Current Conv_a Prior

The current mainline replaces the old linear `1.5 -> 0.2` prior with:

```python
progress = fe / max_fe

if progress <= 0.6:
    conv_a_base = 1.5 - 0.48 * (progress / 0.6) ** 1.2
else:
    conv_a_base = 1.0 - 0.8 * ((progress - 0.6) / 0.4) ** 0.7
```

Final value:

```text
Conv_a = conv_a_base + actor_residual + stagnation_boost
actor_residual = raw_action * conv_a_delta_scale
```

Default config:

```text
conv_a_delta_scale = 0.2
conv_a_clip_min = 0.05
conv_a_clip_max = 1.8
stagnation_boost_max = 0.25
stagnation_boost_fe_ratio = 0.2
```

Interpretation: keep exploration/convergence radius larger until about 60% FE,
then decrease faster in the last 40% FE.

## Reward

RLPSO uses binary reward:

```text
gbest improves -> +1
otherwise      -> -1
```

RLCCPSO uses continuous reward:

```text
reward =
  8.0 * normalized_gbest_improvement
+ 2.0 * normalized_mean_improvement
+ 0.5 * diversity_term
- 0.3 * instability_penalty
```

with default clip `[-2.0, 2.0]`.

## Plotting And Diagnosis

CEC2013 F1 is shifted sphere with optimum `-1400`. Use optimality gap:

```text
gap = fitness - f_opt
```

Do not take log of raw F1 fitness.

Plot:

```bash
python -u plot_final_battle.py <top_task_md5> final_battle_plots/current_mainline
```

Q-collapse diagnosis:

```bash
CUDA_VISIBLE_DEVICES=0 python -u evaluate/q_collapse_diagnosis.py \
  --runs 5 \
  --dim 30 \
  --max-fe 10000 \
  --n-part 100 \
  --group 1 \
  --model "<actor_model.h5>" \
  --output-dir data/diagnosis/RLCCPSO_F1
```

## Guardrails

- Do not change learning rates or gamma unless the user explicitly asks.
- Do not run full training unless the user explicitly asks.
- Do not delete experiment data, plots, or diagnosis CSVs unless explicitly asked.
- If more than two consecutive modify-test attempts fail, stop and explain the failure cause.
