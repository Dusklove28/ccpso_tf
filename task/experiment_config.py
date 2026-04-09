EXPERIMENT_FUNCTIONS = [1,11]
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
