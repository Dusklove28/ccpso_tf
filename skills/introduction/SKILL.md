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
RLCCPSO: second-order CCPSO + 10-D evolution-state CcPSOEnv + direct Conv_a action + gap-progress reward
```

`task/all_tasks_generate.py` no longer reads a mode-selection environment
variable. RLPSO and RLEPSO are currently removed from the runnable task
generator. Do not suggest old ablation, noise-sweep, pbest-reset, hard
progress-prior, or center-trust commands unless the user explicitly asks to
restore those branches.

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

## Current Conv_a Action

The current mainline removes the hard progress prior. The actor directly maps
one continuous action to `Conv_a`:

```text
raw_action in [-1, 1]
Conv_a = Conv_a_min + (raw_action + 1) / 2 * (Conv_a_max - Conv_a_min)
```

Default range:

```text
Conv_a_min = 0.0
Conv_a_max = 2.0
```

Target behavior: Conv_a should be globally larger early and smaller late, but
local sharp rises/falls are allowed. Do not reintroduce a smooth hand-written
progress-prior curve unless explicitly asked.

## Evolution State

RLCCPSO uses `CcPSOEnv` with 10 raw, clipped state features:

```text
FE progress
recent gbest improvement
recent mean improvement
swarm diversity
pbest diversity
Q diversity
mean distance(x, Q)
mean distance(Q, gbest)
current Conv_a
stagnation length
```

Do not apply the old `sin_encode` expansion to this state.

## Reward

RLCCPSO uses gap-progress reward:

```text
reward =
  1.0  * gbest_gap_progress
+ 0.25 * mean_gap_progress
- 0.2  * instability_penalty
```

where `gap = fitness - f_opt` and progress is the log ratio between old and new
gap. Do not add positive diversity reward in this version; diversity and
Q-collapse metrics belong in state/diagnosis.

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
