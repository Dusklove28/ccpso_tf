EXPERIMENT_FUNCTIONS = [1, 11]
EXPERIMENT_RUNTIMES = 10
EXPERIMENT_SEPARATE_TRAINS = [True]
EXPERIMENT_GROUPS = [1]
EXPERIMENT_DIMS = [30]
# Fast research iteration budget. Each episode uses about 100 optimizer steps
# for max_fe=10000 and n_part=100, so 100 episodes is roughly 10000 steps.
EXPERIMENT_TRAIN_MAX_EPISODE = 100
EXPERIMENT_TRAIN_MAX_STEPS = 10000
EXPERIMENT_TRAIN_TIMES = 1
EXPERIMENT_MAX_FE = int(1e4)
EXPERIMENT_N_PART = 100

EXPERIMENT_LR_CRITIC = 1e-4
EXPERIMENT_LR_ACTOR = 1e-6
EXPERIMENT_GAMMA = 0.85

EXPERIMENT_CCPSO_LR_ACTOR = EXPERIMENT_LR_ACTOR
EXPERIMENT_CCPSO_LR_CRITIC = EXPERIMENT_LR_CRITIC
EXPERIMENT_CCPSO_GAMMA = EXPERIMENT_GAMMA

EXPERIMENT_ENV_CONFIG = {
    'reward_mode': 'binary',
}

EXPERIMENT_CCPSO_ENV_CONFIG = {
    'reward_mode': 'ccpso_continuous',
    'reward_gbest_weight': 1.0,
    'reward_mean_weight': 0.25,
    'reward_instability_weight': 0.2,
    'reward_clip': 2.0,
}

EXPERIMENT_CCPSO_CONFIG = {
    'conv_a_clip_min': 0.00,
    'conv_a_clip_max': 2.0,
    'anti_collapse_fe_ratio': 0.2,
    'anti_collapse_q_div_threshold': 0.03,
    'anti_collapse_q_gbest_threshold': 0.05,
}


def get_primary_experiment_signature():
    if len(EXPERIMENT_DIMS) != 1:
        raise ValueError("Standalone evaluation expects exactly one configured dimension.")
    if len(EXPERIMENT_GROUPS) != 1:
        raise ValueError("Standalone evaluation expects exactly one configured group.")
    if len(EXPERIMENT_SEPARATE_TRAINS) != 1:
        raise ValueError("Standalone evaluation expects exactly one separate_train setting.")

    return {
        'dim': EXPERIMENT_DIMS[0],
        'group': EXPERIMENT_GROUPS[0],
        'separate_train': EXPERIMENT_SEPARATE_TRAINS[0],
    }
