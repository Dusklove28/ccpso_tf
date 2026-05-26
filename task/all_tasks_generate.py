import copy

from env.ConvEnv import ConvEnv
from env.NormalEnv import NormalEnv
from matAgent.ccpso import ConvPsoSwarm
from matAgent.pso import PsoSwarm
from task.experiment_config import (
    EXPERIMENT_CCPSO_ABLATION_CONFIGS,
    EXPERIMENT_CCPSO_CONFIG,
    EXPERIMENT_CCPSO_ENV_CONFIG,
    EXPERIMENT_CCPSO_GAMMA,
    EXPERIMENT_CCPSO_LR_ACTOR,
    EXPERIMENT_CCPSO_LR_CRITIC,
    EXPERIMENT_DIMS,
    EXPERIMENT_ENV_CONFIG,
    EXPERIMENT_FUNCTIONS,
    EXPERIMENT_GROUPS,
    EXPERIMENT_GAMMA,
    EXPERIMENT_LR_ACTOR,
    EXPERIMENT_LR_CRITIC,
    EXPERIMENT_MAX_FE,
    EXPERIMENT_N_PART,
    EXPERIMENT_RUNTIMES,
    EXPERIMENT_SEPARATE_TRAINS,
    EXPERIMENT_TRAIN_MAX_EPISODE,
    EXPERIMENT_TRAIN_MAX_STEPS,
    EXPERIMENT_TRAIN_TIMES,
)


def _env_class_from_name(name):
    env_classes = {
        'NormalEnv': NormalEnv,
        'ConvEnv': ConvEnv,
    }
    if name not in env_classes:
        raise ValueError(f"unknown env_class in experiment config: {name}")
    return env_classes[name]


def ccpso_ablation_optimizer_pairs():
    optimizer_pairs = []
    for config in EXPERIMENT_CCPSO_ABLATION_CONFIGS:
        optimizer_pairs.append({
            'name': config['name'],
            'phase_name': config['name'],
            'train_optimizer': ConvPsoSwarm,
            'evaluate_optimizer': ConvPsoSwarm,
            'env_class': _env_class_from_name(config['env_class']),
            'optimizer_config': copy.deepcopy(config['optimizer_config']),
            'env_config': copy.deepcopy(config['env_config']),
            'lr_actor': config['lr_actor'],
            'lr_critic': config['lr_critic'],
            'gamma': config['gamma'],
        })
    return optimizer_pairs


def all_tasks_generate():
    rl_optimizer_pairs = [
        {
            'name': 'RLPSO_original_env',
            'train_optimizer': PsoSwarm,
            'evaluate_optimizer': PsoSwarm,
            'env_class': NormalEnv,
            'optimizer_config': {},
            'env_config': EXPERIMENT_ENV_CONFIG,
            'lr_actor': EXPERIMENT_LR_ACTOR,
            'lr_critic': EXPERIMENT_LR_CRITIC,
            'gamma': EXPERIMENT_GAMMA,
        },
        {
            'name': 'CCPSO_DualC_new_env',
            'train_optimizer': ConvPsoSwarm,
            'evaluate_optimizer': ConvPsoSwarm,
            'env_class': ConvEnv,
            'optimizer_config': EXPERIMENT_CCPSO_CONFIG,
            'env_config': EXPERIMENT_CCPSO_ENV_CONFIG,
            'lr_actor': EXPERIMENT_CCPSO_LR_ACTOR,
            'lr_critic': EXPERIMENT_CCPSO_LR_CRITIC,
            'gamma': EXPERIMENT_CCPSO_GAMMA,
        },
    ]

    task = {
        'type': 'top',
        'baseline_optimizers': [PsoSwarm],
        'rl_optimizer_pairs': rl_optimizer_pairs,
        'evaluate_function': EXPERIMENT_FUNCTIONS,
        'runtimes': EXPERIMENT_RUNTIMES,
        'separate_trains': EXPERIMENT_SEPARATE_TRAINS,
        'groups': EXPERIMENT_GROUPS,
        'train_max_episode': EXPERIMENT_TRAIN_MAX_EPISODE,
        'train_max_steps': EXPERIMENT_TRAIN_MAX_STEPS,
        'dims': EXPERIMENT_DIMS,
        'train_times': EXPERIMENT_TRAIN_TIMES,
        'max_fe': EXPERIMENT_MAX_FE,
        'n_part': EXPERIMENT_N_PART,
        'lr_critic': EXPERIMENT_LR_CRITIC,
        'lr_actor': EXPERIMENT_LR_ACTOR,
        'gamma': EXPERIMENT_GAMMA,
    }

    return [task]


if __name__ == '__main__':
    print(all_tasks_generate())
