EXPERIMENT_FUNCTIONS = [1,5,11]
EXPERIMENT_RUNTIMES = 10
EXPERIMENT_SEPARATE_TRAINS = [True]
EXPERIMENT_GROUPS = [1]
EXPERIMENT_DIMS = [30]
EXPERIMENT_TRAIN_MAX_EPISODE = 400
EXPERIMENT_TRAIN_MAX_STEPS = EXPERIMENT_TRAIN_MAX_EPISODE * 100
EXPERIMENT_TRAIN_TIMES = 1
EXPERIMENT_MAX_FE = int(1e4)
EXPERIMENT_N_PART = 100
EXPERIMENT_LR_CRITIC = 1e-4
EXPERIMENT_LR_ACTOR = 1e-6
EXPERIMENT_GAMMA = 0.85

EXPERIMENT_CCPSO_LR_ACTOR = 3e-6
EXPERIMENT_CCPSO_LR_CRITIC = 3e-4
EXPERIMENT_CCPSO_GAMMA = 0.95

EXPERIMENT_CCPSO_CONFIG = {
    'ccpso_update_mode': 'second_order',
    'conv_a_schedule': 'progress_prior',
    'conv_a_max': 1.5,
    'conv_a_min': 0.2,
    'conv_a_delta_scale': 0.2,
    'conv_a_clip_min': 0.05,
    'conv_a_clip_max': 1.8,
    'stagnation_boost_max': 0.25,
    'stagnation_boost_fe_ratio': 0.2,
    'first_order_sigma_floor': 0.001,
}

EXPERIMENT_ENV_CONFIG = {
    'reward_mode': 'binary',
}

EXPERIMENT_CCPSO_ENV_CONFIG = {
    'reward_mode': 'continuous',
    'reward_gbest_weight': 8.0,
    'reward_mean_weight': 2.0,
    'reward_diversity_weight': 0.5,
    'reward_instability_weight': 0.3,
    'reward_clip': 2.0,
}

EXPERIMENT_CCPSO_DIRECT_CONFIG = dict(EXPERIMENT_CCPSO_CONFIG, conv_a_schedule='direct')
EXPERIMENT_CCPSO_PROGRESS_PRIOR_CONFIG = dict(EXPERIMENT_CCPSO_CONFIG, conv_a_schedule='progress_prior')

EXPERIMENT_CCPSO_ABLATION_CONFIGS = [
    {
        'name': 'CCPSO_original_reward',
        'env_class': 'NormalEnv',
        'optimizer_config': EXPERIMENT_CCPSO_DIRECT_CONFIG,
        'env_config': EXPERIMENT_ENV_CONFIG,
        'lr_actor': EXPERIMENT_LR_ACTOR,
        'lr_critic': EXPERIMENT_LR_CRITIC,
        'gamma': EXPERIMENT_GAMMA,
    },
    {
        'name': 'CCPSO_progress_prior',
        'env_class': 'NormalEnv',
        'optimizer_config': EXPERIMENT_CCPSO_PROGRESS_PRIOR_CONFIG,
        'env_config': EXPERIMENT_ENV_CONFIG,
        'lr_actor': EXPERIMENT_LR_ACTOR,
        'lr_critic': EXPERIMENT_LR_CRITIC,
        'gamma': EXPERIMENT_GAMMA,
    },
    {
        'name': 'CCPSO_continuous_reward',
        'env_class': 'ConvEnv',
        'optimizer_config': EXPERIMENT_CCPSO_DIRECT_CONFIG,
        'env_config': EXPERIMENT_CCPSO_ENV_CONFIG,
        'lr_actor': EXPERIMENT_CCPSO_LR_ACTOR,
        'lr_critic': EXPERIMENT_CCPSO_LR_CRITIC,
        'gamma': EXPERIMENT_CCPSO_GAMMA,
    },
    {
        'name': 'CCPSO_DualC_full',
        'env_class': 'ConvEnv',
        'optimizer_config': EXPERIMENT_CCPSO_PROGRESS_PRIOR_CONFIG,
        'env_config': EXPERIMENT_CCPSO_ENV_CONFIG,
        'lr_actor': EXPERIMENT_CCPSO_LR_ACTOR,
        'lr_critic': EXPERIMENT_CCPSO_LR_CRITIC,
        'gamma': EXPERIMENT_CCPSO_GAMMA,
    },
]


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
