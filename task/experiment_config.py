from plot_final_battle import model_path

EXPERIMENT_FUNCTIONS = [1,11]
EXPERIMENT_RUNTIMES = 10
EXPERIMENT_SEPARATE_TRAINS = [True]
EXPERIMENT_GROUPS = [1]
EXPERIMENT_DIMS = [30]
EXPERIMENT_TRAIN_MAX_EPISODE = 200
EXPERIMENT_TRAIN_MAX_STEPS = EXPERIMENT_TRAIN_MAX_EPISODE * 100
EXPERIMENT_TRAIN_TIMES = 1
EXPERIMENT_MAX_FE = int(1e4)
EXPERIMENT_N_PART = 100
EXPERIMENT_LR_CRITIC = 1e-4
EXPERIMENT_LR_ACTOR = 1e-6
EXPERIMENT_Model_Path = model_path

class update():
    def __init__(self, model_path=None):
        self.model_path = model_path
        # 关键：修改全局变量 EXPERIMENT_Model_Path
        if model_path:
            global EXPERIMENT_Model_Path
            EXPERIMENT_Model_Path = model_path


def get_primary_experiment_signature():
    if len(EXPERIMENT_DIMS) != 1:
        raise ValueError("Standalone evaluation expects exactly one configured dimension.")
    if len(EXPERIMENT_GROUPS) != 1:
        raise ValueError("Standalone evaluation expects exactly one configured group.")
    if len(EXPERIMENT_SEPARATE_TRAINS) != 1:
        raise ValueError("Standalone evaluation expects exactly one separate_train setting.")
    if len(EXPERIMENT_Model_Path) != 1:
        raise ValueError("Model Path not exist")

    return {
        'dim': EXPERIMENT_DIMS[0],
        'group': EXPERIMENT_GROUPS[0],
        'separate_train': EXPERIMENT_SEPARATE_TRAINS[0],
    }
