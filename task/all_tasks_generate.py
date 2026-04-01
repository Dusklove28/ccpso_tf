from matAgent.ccpso import ConvPsoSwarm
from matAgent.pso import PsoSwarm
from task.experiment_config import (
    EXPERIMENT_DIMS,
    EXPERIMENT_FUNCTIONS,
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


def all_tasks_generate():
    rl_optimizer_pairs = [
        {
            'train_optimizer': PsoSwarm,
            'evaluate_optimizer': PsoSwarm,
        },
        {
            'train_optimizer': ConvPsoSwarm,
            'evaluate_optimizer': ConvPsoSwarm,
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
    }

    return [task]


if __name__ == '__main__':
    print(all_tasks_generate())
