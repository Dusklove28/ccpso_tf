# RL_Tensorflow Research Notes

This repository is used for experiments on PSO, RLPSO, CCPSO, and RL-controlled
CCPSO. The current research focus is not a generic engineering benchmark. It is
about whether reinforcement learning can control the convergence dynamics of a
Q-centered CCPSO update more effectively than directly controlling ordinary PSO
parameters.

## 1. Research Motivation

Traditional RLPSO in this project uses DDPG to control PSO parameters such as:

- inertia weight `w`
- acceleration factors `c1` and `c2`
- mutation-related parameters

The CCPSO direction is different. It controls the convergence strength around an
equivalent gravity center `Q`, mainly through `Conv_a` / `C_eff`.

The key theoretical distinction is:

```text
RLPSO controls PSO coefficients.
RL-CCPSO controls Q-centered convergence dynamics.
```

Therefore, RLPSO and RL-CCPSO should not be forced to share the same reward
interpretation without analysis.

## 2. Current CCPSO Update Interpretation

The active CCPSO implementation is a second-order DualC form. It uses both
`x(t)` and `x(t-1)`, not a first-order random update.

The equivalent gravity center is:

```text
Q = (c1*r1*pbest + c2*r2*gbest) / (c1*r1 + c2*r2)
```

The update can be expanded into:

```text
X(t+1) = C*a1*X(t) + C*a2*X(t-1) + [1 - C*(a1+a2)]Q
```

The coefficients sum to one, so without boundary clipping the update satisfies
the unified affine form. This supports viewing CCPSO as a Q-centered convergence
control mechanism.

Important implication:

```text
Conv_a changes the movement radius around Q.
It does not move Q itself.
```

If `pbest` and `gbest` collapse into a wrong region early, then `Q` also stays in
that region. Increasing or decreasing `Conv_a` only expands or shrinks motion
around the wrong center.

## 3. Reward Conflict

The original binary reward is inherited from RLPSO:

```text
gbest improves -> +1
otherwise      -> -1
```

This reward is meaningful for coarse PSO parameter adaptation, but it is weak for
CCPSO convergence control because:

- `Conv_a` has delayed effects.
- `Conv_a` affects the whole swarm through Q-centered dynamics.
- Binary reward gives no magnitude information.
- Binary reward cannot distinguish stable convergence from accidental short-term improvement.

For CCPSO, the current continuous reward is:

```text
reward =
  8.0 * normalized_gbest_improvement
+ 2.0 * normalized_mean_improvement
+ 0.5 * diversity_term
- 0.3 * instability_penalty
```

The default reward clip is:

```text
[-2.0, 2.0]
```

This continuous reward is not a Gaussian or normal distribution. It is a dense
reward shaped from normalized fitness improvement, diversity, and instability.

## 4. Four Main Ablation Modes

The main controlled ablation has two variables:

- `reward_mode`
- `conv_a_schedule`

The four base combinations are:

```text
A: direct         + binary reward
B: direct         + continuous reward
C: progress_prior + binary reward
D: progress_prior + continuous reward
```

Commands:

```bash
CCPSO_MODE=direct_binary CUDA_VISIBLE_DEVICES=0 python -u main.py
CCPSO_MODE=direct_continuous CUDA_VISIBLE_DEVICES=0 python -u main.py
CCPSO_MODE=progress_binary CUDA_VISIBLE_DEVICES=0 python -u main.py
CCPSO_MODE=progress_continuous CUDA_VISIBLE_DEVICES=0 python -u main.py
```

Multiple modes:

```bash
CCPSO_MODE=direct_continuous,progress_binary CUDA_VISIBLE_DEVICES=0 python -u main.py
```

All four base ablations:

```bash
CCPSO_MODE=all CUDA_VISIBLE_DEVICES=0 python -u main.py
```

## 5. Interpretation of Current Results

### A: direct + binary

This mode performs poorly. The likely reason is that the binary reward is too
coarse to teach DDPG how to directly control a second-order convergence gain.

### B: direct + continuous

The continuous reward improves the feedback signal, but direct control remains
unstable.

Observed behavior:

- The actor may learn a rough high-to-low trend on F1.
- The learned `Conv_a` has high variance.
- On F11, the actor may saturate at a high `Conv_a`.
- Very low exploration noise can still lead to boundary-jumping actions.

Conclusion:

```text
Continuous reward helps, but direct Conv_a control is not stable enough by itself.
```

### C: progress_prior + binary

This mode can perform very well on F11, even with binary reward. This does not
mean binary reward is ideal for CCPSO. It means the progress prior provides a
strong exploration-to-exploitation structure:

```text
early stage: larger Conv_a
late stage:  smaller Conv_a
```

The actor only learns a residual instead of the whole control law.

However, C can still stagnate on F1. This supports the Q-collapse diagnosis:

```text
If pbest/gbest/Q collapse early, a decreasing Conv_a can accelerate convergence
around the wrong center.
```

### D: progress_prior + continuous

This is currently the strongest original combination. It combines:

- stable progress prior
- denser CCPSO-specific reward

It is more convincing when interpreted together with A/B/C:

```text
A vs C: progress prior matters under binary reward.
B vs D: progress prior matters under continuous reward.
C vs D: continuous reward matters when progress prior is fixed.
```

## 6. Progress Prior

In `progress_prior`, the final `Conv_a` is:

```text
Conv_a = base_progress_prior + actor_residual + stagnation_boost
```

The base prior decreases from:

```text
conv_a_max = 1.5
```

to:

```text
conv_a_min = 0.2
```

The actor only controls a residual:

```text
conv_a_delta = raw_action * conv_a_delta_scale
```

This mechanism is important because `Conv_a` is a convergence gain in a
second-order dynamic system. Letting DDPG directly control the full value every
step tends to produce unstable or saturated actions.

## 7. Q Collapse / Early Stagnation

Q collapse happens when `pbest`, `gbest`, and therefore `Q` all become
concentrated in a wrong region.

Symptoms:

- low swarm diversity
- low Q diversity
- low pbest diversity
- small distance between Q and gbest
- low relative gbest improvement over a recent FE window
- continued convergence around a wrong region

This cannot be solved reliably by only changing `Conv_a` because `Conv_a` does
not move Q. It only controls motion around Q.

The same early-maturity risk also exists in basic PSO. Standard PSO is also
driven by `pbest` and `gbest`:

```text
v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)
```

If `pbest` and `gbest` become concentrated in a wrong region, the swarm can keep
making tiny improvements while still being practically trapped. Therefore, a
single poor PSO baseline curve on F1 should not be interpreted as Q-reset
breaking PSO. Q-reset is implemented only in `ConvPsoSwarm`; it does not affect
`PsoSwarm`.

## 8. General Q-reset Mechanism

Q-reset is introduced as a new anti-collapse mechanism. It is no longer tied only
to C. It can be applied to A, B, C, or D.

The mechanism detects:

```text
sufficient progress
+ low relative gbest improvement over the stagnation window
+ low pbest diversity
+ cooldown satisfied
```

Then it schedules a conservative particle-level soft restart for the worst
small fraction of particles. The current default parameters are:

```text
anti_q_collapse = True
collapse_min_progress = 0.55
collapse_stagnation_fe_ratio = 0.18
collapse_gbest_improvement_threshold = 1e-3
collapse_pbest_diversity_threshold = 0.003
collapse_reset_ratio = 0.05
collapse_reset_cooldown_fe_ratio = 0.15
collapse_restart_radius_ratio = 0.08
collapse_global_restart_probability = 0.25
```

The default reset action is:

```text
schedule the worst 5% non-best particles
apply the restart at the beginning of the next CCPSO step
mostly restart near the current gbest, with a small global-random fraction
sync x, x_old, v, and pbest
keep the current best pbest/gbest untouched
set restarted pbest fitness to infinity until the normal next evaluation
```

This changes the pbest distribution so future Q centers can spread again.
The stagnation condition is not based on whether `gbest` changed at all. It uses
the relative improvement over the recent window, so tiny numerical improvements
do not prevent Q-reset from detecting practical stagnation.

Important boundary:

```text
Q-reset is a new mechanism.
It should be evaluated separately from the original A/B/C/D ablation.
```

## 9. Q-reset Modes

Base modes remain unchanged. To apply Q-reset explicitly:

```bash
CCPSO_MODE=a_q_reset CUDA_VISIBLE_DEVICES=0 python -u main.py
CCPSO_MODE=b_q_reset CUDA_VISIBLE_DEVICES=0 python -u main.py
CCPSO_MODE=c_q_reset CUDA_VISIBLE_DEVICES=0 python -u main.py
CCPSO_MODE=d_q_reset CUDA_VISIBLE_DEVICES=0 python -u main.py
```

Equivalent aliases:

```bash
CCPSO_MODE=direct_binary_q_reset CUDA_VISIBLE_DEVICES=0 python -u main.py
CCPSO_MODE=direct_continuous_q_reset CUDA_VISIBLE_DEVICES=0 python -u main.py
CCPSO_MODE=progress_binary_q_reset CUDA_VISIBLE_DEVICES=0 python -u main.py
CCPSO_MODE=progress_continuous_q_reset CUDA_VISIBLE_DEVICES=0 python -u main.py
```

All four base modes with Q-reset:

```bash
CCPSO_MODE=all_q_reset CUDA_VISIBLE_DEVICES=0 python -u main.py
```

Or apply Q-reset to any selected mode through an environment variable:

```bash
CCPSO_MODE=direct_continuous CCPSO_Q_RESET=1 CUDA_VISIBLE_DEVICES=0 python -u main.py
CCPSO_MODE=progress_binary CCPSO_Q_RESET=1 CUDA_VISIBLE_DEVICES=0 python -u main.py
```

## 10. Noise Sensitivity for B

B-mode noise sensitivity is controlled by:

```bash
CCPSO_MODE=direct_continuous CCPSO_SIGMA=0.05 CUDA_VISIBLE_DEVICES=0 python -u main.py
CCPSO_MODE=direct_continuous CCPSO_SIGMA=0.02 CUDA_VISIBLE_DEVICES=0 python -u main.py
```

Sigma affects DDPG training exploration. It is not evaluation noise. Lower sigma
does not necessarily mean smoother learned `Conv_a`. Low sigma can reduce
exploration and still allow actor saturation.

Current interpretation:

```text
Sigma affects B, but sigma alone cannot solve direct Conv_a control instability.
```

## 11. Plotting

Use:

```bash
python -u plot_final_battle.py <top_task_md5> final_battle_plots/<folder_name>
```

Examples:

```bash
python -u plot_final_battle.py <top_task_md5> final_battle_plots/C_q_reset
python -u plot_final_battle.py <top_task_md5> final_battle_plots/D_q_reset
```

The plotting code uses CEC2013 optimality gap:

```text
fitness_error = fitness - f_opt
```

For CEC2013 F1:

```text
f_opt = -1400
```

Do not take log of raw F1 fitness.

## 12. Diagnosis

Use `evaluate/q_collapse_diagnosis.py` for lightweight F1 diagnosis. It does not
train a model.

Example without Q-reset:

```bash
CUDA_VISIBLE_DEVICES=0 python -u evaluate/q_collapse_diagnosis.py \
  --runs 5 \
  --dim 30 \
  --max-fe 10000 \
  --n-part 100 \
  --group 1 \
  --conv-a-schedule progress_prior \
  --model "<F1_actor_model>" \
  --output-dir data/diagnosis/C_F1
```

Example with Q-reset:

```bash
CUDA_VISIBLE_DEVICES=0 python -u evaluate/q_collapse_diagnosis.py \
  --runs 5 \
  --dim 30 \
  --max-fe 10000 \
  --n-part 100 \
  --group 1 \
  --conv-a-schedule progress_prior \
  --anti-q-collapse \
  --model "<F1_actor_model>" \
  --output-dir data/diagnosis/C_q_reset_F1
```

Important output fields:

- `q_diversity`
- `pbest_diversity`
- `mean_distance_q_gbest`
- `mean_distance_pbest_gbest`
- `mean_distance_x_q`
- `conv_a`
- `gbest_window_relative_improvement`
- `pbest_reset_count`

## 13. Recent Diagnosis Notes

The first conservative Q-reset experiment recovered the previous F11 behavior,
but F1 still stagnated. The F1 diagnosis in:

```text
data/diagnosis/C_q_reset_soft_F1
```

showed an important failure mode:

```text
pbest_reset_count = 0 for all 5 diagnosis runs
```

So the issue was not that soft restart was triggered and ineffective. The issue
was that the old trigger did not fire. It used `last_best_update_fe`, so any tiny
`gbest` improvement reset the stagnation counter.

Example diagnosis pattern:

```text
run 1 gap:
5500 FE  -> 983.68
7000 FE  -> 958.56
10000 FE -> 958.32
```

This is practical stagnation, even though `gbest` still changes by a tiny amount.
The trigger was therefore changed to a window-relative condition:

```text
past 18% FE window relative gbest improvement < 1e-3
+ low pbest diversity
```

Offline replay on the old diagnosis CSV indicated that the new rule would have
triggered around:

```text
run 1: ~8700 FE
run 2: ~8600 FE
run 4: ~8500 FE
```

This means the next experiment should check two things:

```text
1. Does pbest_reset_count become nonzero on F1?
2. After reset, do q_diversity and pbest_diversity recover enough to move Q?
```

If `pbest_reset_count` remains zero, the trigger is still too strict. If it is
nonzero but F1 remains stagnant, the restart location or reset strength is still
insufficient.

## 14. Current Empirical Takeaways

The current interpretation of the experiment path is:

```text
A: direct + binary
   Poor. Binary reward and direct Conv_a control conflict strongly.

B: direct + continuous
   Continuous reward improves feedback, but direct Conv_a control remains noisy
   and unstable. Sigma changes the behavior, but sigma alone is not a solution.

C: progress_prior + binary
   F11 can be strong even under binary reward. This supports progress prior as a
   key structural mechanism. F1 can still stagnate because progress prior cannot
   move Q once pbest/gbest have collapsed.

D: progress_prior + continuous
   Stronger original combination. It supports the joint story: progress prior
   stabilizes the action interface, and continuous reward improves the feedback
   signal.

C + Q-reset
   A new mechanism beyond the original A/B/C/D ablation. It should be evaluated
   separately. The first crude reset was too disruptive; the conservative soft
   restart recovered F11 but initially did not trigger on F1 until the stagnation
   condition was changed to a window-relative improvement rule.
```

Important caution:

```text
Do not over-interpret a single PSO baseline curve.
PSO can also show random early maturity on F1 because it depends on pbest/gbest.
Use repeated runs, fixed seeds when needed, and the same top-task comparison
protocol before making claims.
```

## 15. Suggested Paper Narrative

A clean narrative is:

```text
1. RLPSO is effective but mainly controls traditional PSO coefficients.
2. CCPSO provides a Q-centered convergence control object.
3. Direct RL control of Conv_a is unstable because Conv_a is a delayed second-order convergence gain.
4. A progress prior stabilizes the control interface by giving an exploration-to-exploitation structure.
5. Continuous reward improves the feedback signal for CCPSO but is not sufficient without a stable action interface.
6. Q collapse remains a limitation because Conv_a cannot move Q.
7. Q-reset is introduced as an additional anti-collapse mechanism that changes pbest distribution and therefore future Q centers.
```

Do not present Q-reset as part of the original A/B/C/D ablation. It is a new
mechanism and should have its own ablation.

## 16. README and SKILL Roles

Use these two files differently:

```text
README.md
  Human-facing research notes and paper-thinking record.
  Keep experiment logic, empirical interpretation, caveats, and narrative here.

skills/introduction/SKILL.md
  Model-facing compact operational memory.
  In a new Codex conversation, explicitly ask the model to read it first.
```

Recommended prompt for a new conversation:

```text
I am working in D:\code\RL_Tensorflow.
Please read skills/introduction/SKILL.md first, then continue the RL-CCPSO analysis.
```

The SKILL file is not a universal memory across all tools or all machines. It is
a local project context file. It works best when explicitly requested in the
same Codex environment.

## 17. Current Research Discipline

Keep these boundaries clear:

- Do not change learning rate or gamma for the current ablation story.
- Do not mix new mechanisms into old A/B/C/D conclusions.
- Treat failed B or F1 stagnation as evidence, not as something to hide.
- Use Q-reset as a new mechanism motivated by the observed Q-collapse diagnosis.
- When reporting results, distinguish:
  - reward effect
  - progress-prior effect
  - noise sensitivity
  - Q-reset anti-collapse effect

The goal is not to tune one curve until it looks good. The goal is to build a
defensible explanation of why each mechanism is needed.
