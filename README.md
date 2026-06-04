# RL_Tensorflow Research Notes

This project studies PSO, RLPSO, CCPSO, and DDPG-controlled CCPSO on CEC2013.
The current code has been narrowed back to the main research line:

```text
PSO baseline
RLPSO: original NormalEnv binary reward
RLCCPSO: second-order CCPSO + delayed progress prior + continuous reward
```

The old ablation switching, noise sweep, pbest reset, evolutionary-state control,
and center-trust branches are not part of the current runnable pipeline.

## Core Idea

RLPSO and RLCCPSO control different objects.

```text
RLPSO controls ordinary PSO coefficients such as w/c1/c2/mutation.
RLCCPSO controls Q-centered convergence strength through Conv_a.
```

The active CCPSO update is the second-order DualC form:

```text
Q = (c1*r1*pbest + c2*r2*gbest) / (c1*r1 + c2*r2)
X(t+1) = C*a1*X(t) + C*a2*X(t-1) + [1 - C*(a1+a2)]Q
```

The coefficients sum to one before boundary clipping, so this is still a
Q-centered affine convergence form. `Conv_a` changes the movement radius around
`Q`; it does not move `Q` itself.

## Current Conv_a Prior

The previous linear prior `1.5 -> 0.2` crossed `Conv_a = 1` too early. The
current mainline uses a delayed progress prior:

```python
progress = fe / max_fe

if progress <= 0.6:
    conv_a_base = 1.5 - 0.48 * (progress / 0.6) ** 1.2
else:
    conv_a_base = 1.0 - 0.8 * ((progress - 0.6) / 0.4) ** 0.7
```

Final `Conv_a` is:

```text
Conv_a = conv_a_base + actor_residual + stagnation_boost
```

where:

```text
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

Interpretation: keep `Conv_a` above about 1 for the first 60% FE to delay
premature contraction, then decrease faster in the final 40% FE to restore
exploitation.

## Reward

RLPSO still uses the original binary reward:

```text
gbest improves -> +1
otherwise      -> -1
```

RLCCPSO uses the CCPSO continuous reward:

```text
reward =
  8.0 * normalized_gbest_improvement
+ 2.0 * normalized_mean_improvement
+ 0.5 * diversity_term
- 0.3 * instability_penalty
```

The default reward clip is `[-2.0, 2.0]`.

## Running

The task generator no longer reads any mode-selection environment variable. Run
the current mainline with:

```bash
CUDA_VISIBLE_DEVICES=0 python -u main.py
```

PowerShell:

```powershell
$env:CUDA_VISIBLE_DEVICES="0"
python -u main.py
```

The generated task contains PSO, RLPSO, and `RLCCPSO`.

## Plotting

```bash
python -u plot_final_battle.py <top_task_md5> final_battle_plots/current_mainline
```

Use optimality gap for CEC2013 plots:

```text
gap = fitness - f_opt
```

CEC2013 F1 is shifted sphere with optimum `-1400`, so do not take log of raw
fitness directly.

## Diagnosis

F1 Q-collapse diagnosis:

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

Without `--model`, the script runs the same CCPSO dynamics with zero actor
residual. The diagnosis records `Q`, `pbest`, `gbest`, diversity, distance to
`Q`, `Conv_a`, `conv_a_base`, actor residual, and stagnation boost.

## Research Caution

If F1 still stagnates, the likely mechanism remains:

```text
early pbest/gbest collapse -> Q collapse -> Conv_a can only scale movement
around the wrong Q
```

Changing the `Conv_a` prior can delay contraction and may reduce early collapse,
but it does not directly change `Q`, `pbest`, or `gbest`.

Do not change learning rates or gamma in the same experiment round. If more than
two consecutive modify-test attempts fail, stop and analyze the failure cause
instead of blindly adding mechanisms.
