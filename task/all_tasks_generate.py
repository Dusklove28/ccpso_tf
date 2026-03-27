from matAgent.ccpso_50d import FiftyDimCCPsoSwarm
from matAgent.clpso import ClpsoSwarm
from matAgent.pso import PsoSwarm
from matAgent.rlepso import RlepsoSwarm
from matAgent.rl_ccpso_eval import RlCCPsoSwarm
from matAgent.testpso import TestpsoSwarm


def all_tasks_generate():
    baseline_optimizers = [ ]
    rl_optimizer_pairs = [
        {
            'train_optimizer': TestpsoSwarm,
            'evaluate_optimizer': RlepsoSwarm,
            'train_profile': 'original_rlepso',
            'train_al_type': 'testpso',
        },
        {
            'train_optimizer': FiftyDimCCPsoSwarm,
            'evaluate_optimizer': RlCCPsoSwarm,
            'train_profile': 'original_rlepso',
            'train_al_type': 'ccpso_50d',
        },
    ]

    task = {
        'type': 'top',
        'baseline_optimizers': baseline_optimizers,
        'rl_optimizer_pairs': rl_optimizer_pairs,
        'evaluate_function': list(range(1, 29)),
        # 'evaluate_function': [1, 15, 23],
        'runtimes': 5,
        # 'separate_trains': [False],
        'separate_trains': [True],
        'groups': [5],
        'train_max_episode': 200,
        'train_max_steps': 8000,
        'dims': [30],
        'train_times': 1,
        'max_fe': int(1e4),
        'n_part': 100,
        'lr_critic': 1e-7,
        'lr_actor': 1e-9,
    }

    return [task]


if __name__ == '__main__':
    print(all_tasks_generate())
