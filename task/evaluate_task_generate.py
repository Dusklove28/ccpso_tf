from utils.db.db import get_optimizer_train_result
from task.experiment_config import (
    EXPERIMENT_FUNCTIONS,
    EXPERIMENT_MAX_FE,
    EXPERIMENT_N_PART,
    EXPERIMENT_RUNTIMES,
    get_primary_experiment_signature,
)

try:
    from matAgent.pso import PsoSwarm
    from matAgent.ccpso import ConvPsoSwarm
except ImportError as e:
    print(f"算法模块导入失败: {e}。请检查 matAgent 目录。")
    raise

funs = EXPERIMENT_FUNCTIONS
no_model_fun_model = {fun: [None] for fun in funs}


def _get_required_train_result(optimizer_cls, display_name):
    signature = get_primary_experiment_signature()
    train_result = get_optimizer_train_result(
        optimizer_cls.optimizer_name,
        signature['dim'],
        signature['group'],
        signature['separate_train'],
        EXPERIMENT_MAX_FE,
        EXPERIMENT_N_PART,
    )
    if not train_result:
        raise RuntimeError(
            f"缺少 {display_name} 的训练模型，请先运行主实验训练流程后再执行独立评估。"
        )
    return train_result


def generate_evaluate_tasks():
    signature = get_primary_experiment_signature()
    optimizer_model_list = [
        {
            'optimizer': PsoSwarm,
            'fun_model': no_model_fun_model,
        },
        {
            'optimizer': PsoSwarm,
            'fun_model': _get_required_train_result(PsoSwarm, 'RL+基础PSO'),
        },
        {
            'optimizer': ConvPsoSwarm,
            'fun_model': _get_required_train_result(ConvPsoSwarm, 'RL+基础PSO+收敛性策略'),
        },
    ]

    return [{
        'type': 'new_result_evaluate',
        'optimizer_model_list': optimizer_model_list,
        'evaluate_function': funs,
        'group': signature['group'],
        'max_fe': EXPERIMENT_MAX_FE,
        'n_part': EXPERIMENT_N_PART,
        'dim': signature['dim'],
        'runtimes': EXPERIMENT_RUNTIMES,
    }]


if __name__ == '__main__':
    print("\n--- 独立评测任务生成校验 ---")
    tasks = generate_evaluate_tasks()

    print("\n[待评测算法序列]:")
    for idx, opt_dict in enumerate(tasks[0]['optimizer_model_list']):
        opt_name = opt_dict['optimizer'].__name__
        model_status = "装备RL模型" if opt_dict['fun_model'] != no_model_fun_model else "纯传统启发式"
        print(f"  {idx + 1}. 核心引擎: {opt_name.ljust(15)} | 驱动模式: {model_status}")
    print("---------------------------------------\n")
