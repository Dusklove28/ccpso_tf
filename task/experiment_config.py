EXPERIMENT_FUNCTIONS = [1, 11]
EXPERIMENT_RUNTIMES = 10
EXPERIMENT_SEPARATE_TRAINS = [True]
EXPERIMENT_GROUPS = [1]
EXPERIMENT_DIMS = [30]
EXPERIMENT_TRAIN_MAX_EPISODE = 100
EXPERIMENT_TRAIN_MAX_STEPS = EXPERIMENT_TRAIN_MAX_EPISODE * 100
EXPERIMENT_TRAIN_TIMES = 1
EXPERIMENT_MAX_FE = int(1e4)
EXPERIMENT_N_PART = 100

EXPERIMENT_LR_CRITIC = 1e-4
EXPERIMENT_LR_ACTOR = 1e-6
EXPERIMENT_GAMMA = 0.85

# Keep the existing CCPSO learning-rate/gamma convention unchanged.
EXPERIMENT_CCPSO_LR_ACTOR = EXPERIMENT_LR_CRITIC
EXPERIMENT_CCPSO_LR_CRITIC = EXPERIMENT_LR_ACTOR
EXPERIMENT_CCPSO_GAMMA = EXPERIMENT_GAMMA

EXPERIMENT_ENV_CONFIG = {
    'reward_mode': 'binary',
}

EXPERIMENT_CCPSO_ENV_CONFIG = {
    'reward_mode': 'ccpso_continuous',
    'reward_gbest_weight': 8.0,
    'reward_mean_weight': 2.0,
    'reward_diversity_weight': 0.5,
    'reward_instability_weight': 0.3,
    'reward_clip': 2.0,
}

EXPERIMENT_CCPSO_CONFIG = {
    'conv_a_delta_scale': 0.2,
    'conv_a_clip_min': 0.00,
    'conv_a_clip_max': 2.0,
    'stagnation_boost_max': 0.25,
    'stagnation_boost_fe_ratio': 0.2,
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
