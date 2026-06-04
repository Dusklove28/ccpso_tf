import copy

from env.NormalEnv import NormalEnv
from matAgent.ccpso import ConvPsoSwarm
from matAgent.pso import PsoSwarm
from task.experiment_config import (
    EXPERIMENT_CCPSO_CONFIG,
    EXPERIMENT_CCPSO_ENV_CONFIG,
    EXPERIMENT_CCPSO_GAMMA,
    EXPERIMENT_CCPSO_LR_ACTOR,
    EXPERIMENT_CCPSO_LR_CRITIC,
    EXPERIMENT_DIMS,
    EXPERIMENT_ENV_CONFIG,
    EXPERIMENT_FUNCTIONS,
    EXPERIMENT_GAMMA,
    EXPERIMENT_GROUPS,
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


def _rlpso_pair():
    return {
        'name': 'RLPSO',
        'train_optimizer': PsoSwarm,
        'evaluate_optimizer': PsoSwarm,
        'env_class': NormalEnv,
        'optimizer_config': {},
        'env_config': copy.deepcopy(EXPERIMENT_ENV_CONFIG),
        'lr_actor': EXPERIMENT_LR_ACTOR,
        'lr_critic': EXPERIMENT_LR_CRITIC,
        'gamma': EXPERIMENT_GAMMA,
    }


def _rlccpso_pair():
    return {
        'name': 'RLCCPSO',
        'phase_name': 'RLCCPSO',
        'train_optimizer': ConvPsoSwarm,
        'evaluate_optimizer': ConvPsoSwarm,
        'env_class': NormalEnv,
        'optimizer_config': copy.deepcopy(EXPERIMENT_CCPSO_CONFIG),
        'env_config': copy.deepcopy(EXPERIMENT_CCPSO_ENV_CONFIG),
        'lr_actor': EXPERIMENT_CCPSO_LR_ACTOR,
        'lr_critic': EXPERIMENT_CCPSO_LR_CRITIC,
        'gamma': EXPERIMENT_CCPSO_GAMMA,
    }


def all_tasks_generate():
    task = {
        'type': 'top',
        'baseline_optimizers': [PsoSwarm],
        'rl_optimizer_pairs': [
            _rlpso_pair(),
            _rlccpso_pair(),
        ],
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
