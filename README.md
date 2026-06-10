# RL_Tensorflow Research Notes

This project studies PSO, RLPSO, CCPSO, and DDPG-controlled CCPSO on CEC2013.
The current code has been narrowed back to the main research line:

```text
PSO baseline
RLCCPSO: second-order CCPSO + 10-D evolution-state CcPSOEnv + direct Conv_a action + gap-progress reward
```

RLPSO and RLEPSO are no longer included in the current runnable task generator.
They are kept as historical code, but the active experiment first asks a simpler
question:

```text
Can RLCCPSO learn a useful Conv_a control law and outperform the PSO baseline?
```

The old ablation switching, noise sweep, pbest reset, hard progress-prior
branch, and center-trust branch are not part of the current runnable pipeline.

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

## Current Conv_a Action

The current mainline removes the hard progress prior. The actor directly
outputs one continuous action and maps it to `Conv_a`:

```text
raw_action in [-1, 1]
Conv_a = Conv_a_min + (raw_action + 1) / 2 * (Conv_a_max - Conv_a_min)
```

Default range:

```text
Conv_a_min = 0.0
Conv_a_max = 2.0
```

The research goal is not monotonic control. The desired learned behavior is:

```text
global tendency: larger Conv_a early, smaller Conv_a late
local behavior: Conv_a may rise or fall sharply when the evolutionary state
requires it
```

This replaces the previous hand-written curve:

```text
Conv_a = progress_prior + actor_residual + anti_collapse_boost
```

That previous form is now treated as a failed/intermediate lesson rather than
the active method, because it makes it hard to prove that RL learned the control
law.

## Evolution State

RLCCPSO now uses `CcPSOEnv` with 10 raw, clipped state features:

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

The state is not passed through the old `sin_encode` expansion.

## Reward

The old RLPSO binary reward is only historical context:

```text
gbest improves -> +1
otherwise      -> -1
```

The active RLCCPSO reward has been simplified. It no longer gives a positive
reward for large diversity. The current reward is based on optimality-gap
progress:

```text
reward =
  alpha  * gbest_progress
+ beta   * mean_progress
- lambda * instability_penalty
```

where:

```text
gbest_progress = log((old_gbest_gap + eps) / (new_gbest_gap + eps))
mean_progress  = log((old_mean_gap  + eps) / (new_mean_gap  + eps))
gap = fitness - f_opt
```

Default weights:

```text
alpha = 1.0
beta = 0.25
lambda = 0.2
reward clip = [-2.0, 2.0]
```

The purpose is to align reward directly with optimization progress:

```text
gbest improves -> reward
mean fitness improves -> smaller reward
boundary/velocity clipping instability -> penalty
```

Diversity, Q diversity, stagnation, and distances remain in the state/diagnosis
trace, but they are not directly rewarded in this version.

## Failed Attempts And Lessons

This section records the negative results and research lessons that should not
be forgotten when designing the next RLCCPSO variant.

### A/B/C/D Ablation Lessons

The earlier ablation plan used two factors:

```text
reward mode: binary vs continuous
Conv_a schedule: direct vs progress prior
```

The four combinations gave the following lessons:

```text
A: direct + binary reward
   Not useful for CCPSO. The binary reward only tells whether gbest improved,
   but CCPSO's action channel changes Conv_a, which mainly scales movement
   around Q. The sparse binary signal is poorly aligned with this control
   object.

B: direct + continuous reward
   The actor can learn a very rough large-to-small Conv_a tendency, but the
   learned curve is unstable. On F1 it still stagnates easily; on F11 it is not
   consistently better than PSO/RLPSO.

C: progress prior + binary reward
   The progress prior is the useful factor. It improves F11 and can outperform
   PSO/RLPSO on multimodal cases, but F1 still stagnates. This shows that a
   convergence schedule helps, but binary reward still does not explain CCPSO's
   Q-centered dynamics well.

D: progress prior + continuous reward
   This is the strongest old setting, especially on F11. However, its
   explanatory power is limited because the Conv_a trend is largely imposed by
   the hand-written prior rather than fully learned by RL.
```

### Why Fixed Or Smooth Conv_a Priors Are Insufficient

The previous progress prior was introduced to keep `Conv_a` large early and
small late. It did delay contraction and partially reduced early stagnation,
especially after changing the curve so that `Conv_a > 1` for roughly the first
60% FE. But it did not solve the F1 stagnation problem.

The important correction from later discussion is:

```text
Conv_a should not be forced to follow a smooth monotonic decay.
The desired behavior is only global: larger in the early stage and smaller in
the late stage.
Locally, Conv_a should be allowed to rise or fall sharply according to the
evolutionary state.
```

Therefore, a hard progress prior should not be treated as the final answer. At
most, it can be used as a baseline, weak scaffold, or ablation reference. The
new research objective is to let RL learn the large-early/small-late tendency
from state and reward, while permitting middle-stage oscillations.

### Q Collapse Is The Core Failure Mode

The repeated F1 failures are best explained by Q collapse:

```text
early pbest/gbest collapse -> Q collapse -> particles move around a wrong
center -> increasing Conv_a only enlarges oscillation around that center ->
decreasing Conv_a accelerates convergence to that wrong center
```

This explains why controlling only one scalar `Conv_a` is much harder than
RLPSO controlling several ordinary PSO parameters:

```text
RLPSO actions can affect inertia, pbest attraction, gbest attraction, and
mutation-like exploration.

Old RLCCPSO mainly adjusted the movement radius around Q, but did not directly
change how Q is constructed.
```

Thus, if `pbest` and `gbest` already collapsed into a poor region, `Conv_a`
alone has limited ability to create new search directions.

### Negative Mechanism Attempts

Several repair attempts were explored or discussed and should not be mixed into
the next main experiment without clear attribution:

```text
Noise sensitivity:
  Changing DDPG exploration sigma affected Conv_a variance but did not provide
  a reliable solution. Sigma alone cannot control CCPSO convergence.

Q/pbest reset:
  Resetting pbest or repairing Q can directly attack collapse, but it changes
  the optimizer mechanism and may damage the convergence-strategy narrative.
  Earlier Q-reset-style tests either weakened F11 or made F1 recovery too slow.

Middle-stage guard parameters:
  Adding many hand thresholds for diversity, improvement windows, floors, and
  guards makes the algorithm complicated and hard to defend. It risks becoming
  parameter stacking rather than a clean RL method.

Hard delayed progress prior:
  Keeping Conv_a high until about 60% FE can reduce early collapse, but it
  still imposes the trend manually and does not prove that RL learned the
  control law.
```

### New Research Turn

The new goal is no longer to keep RLCCPSO artificially comparable with the old
RLPSO codebase. The old source paper has limited reference value for the new
algorithm. The new target is:

```text
Let RL learn that Conv_a should be generally large early and small late, while
allowing strong local oscillations when the evolutionary state requires it.
```

This means the next design can adjust:

```text
state representation
action mapping
reward details
DDPG learning rates and noise
possibly the RL algorithm in a later, separated comparison
```

The first clean direction should be:

```text
Actor directly controls Conv_a or its normalized value, rather than only a small
residual around a hard progress-prior curve.

The state must expose whether the swarm is exploring, converging, stagnating,
or collapsing around Q.
```

Useful state candidates:

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
previous Delta Conv_a
stagnation length
boundary hit ratio
```

The learned Conv_a behavior should be evaluated by trend statistics, not by
visual monotonicity:

```text
mean(Conv_a in early FE) > mean(Conv_a in late FE)
moving average has a decreasing global tendency
middle-stage local variance is allowed
F1 stagnation is reduced
F11 advantage is preserved
```

## 2026-06-11 Active Experiment

The current experiment isolates RLCCPSO before comparing against RLPSO/RLEPSO.

Runnable task:

```text
PSO baseline
RLCCPSO only
```

Training budget:

```text
train_max_episode = 100
train_max_steps = 10000
```

Current action:

```text
one actor output directly maps to Conv_a in [0, 2]
no progress prior
no anti-collapse action
```

Current reward:

```text
reward =
  1.0  * gbest_gap_progress
+ 0.25 * mean_gap_progress
- 0.2  * instability_penalty
```

Key question:

```text
Can direct-C RLCCPSO learn a globally large-early/small-late Conv_a tendency
with local oscillations, and can it beat PSO?
```

Tracked variables now include more than `Conv_a`:

```text
Conv_a
raw actor action
normalized Conv_a
swarm diversity
pbest diversity
Q diversity
mean distance(x, Q)
mean distance(Q, gbest)
collapse risk
recent gbest improvement
recent mean improvement
boundary ratio
velocity clipping ratio
instability penalty
```

`plot_final_battle.py` still draws the Conv_a mean/variance figure, and now also
generates a control-diagnostics figure for these tracked variables.

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

The generated task currently contains only PSO and `RLCCPSO`.

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
residual and zero anti-collapse strength. The diagnosis records `Q`, `pbest`,
`gbest`, diversity, distance to `Q`, `Conv_a`, `conv_a_base`, actor residual,
collapse risk, guard strength, and anti-collapse boost.

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

This guardrail applied to the old comparison-focused phase. In the new
algorithm-design phase, learning rates, state, and action mapping may be changed
deliberately, but each change must have a clear hypothesis and must be tested in
a separately named experiment.
