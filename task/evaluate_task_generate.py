import os
from utils.db.db import get_optimizer_train_result

# --- 1. 精简导入：只保留真正的核心环境 ---
try:
    from matAgent.pso import PsoSwarm
    from matAgent.ccpso import ConvPsoSwarm
except ImportError as e:
    print(f"算法模块导入失败: {e}。请检查 matAgent 目录。")
    raise e

funs = list(range(1, 29, 1))
no_model_fun_model = {fun: [None] for fun in funs}

# --- 严格的 30D 正常参数对齐 ---
dim = 30
runtimes = 10
max_fe = 10000
group = 1
separate_train = True
n_part = 100

def generate_evaluate_tasks():
    optimizer_model_list = []

    # ==========================================
    # 实验序列 1：纯 PSO 基准 (Native Baseline)
    # ==========================================
    optimizer_model_list.append({
        'optimizer': PsoSwarm,
        'fun_model': no_model_fun_model,
    })

    # ==========================================
    # 实验序列 2：RL + 基础 PSO (对照组)
    # ==========================================
    fun_model_rl_pso = get_optimizer_train_result(
        PsoSwarm.optimizer_name, dim, group, separate_train, max_fe, n_part
    )
    if not fun_model_rl_pso:
        print(f"未能在 DB 找到 【RL+基础PSO ({PsoSwarm.optimizer_name})】 的有效模型！")
    else:
        print(f"【RL+基础PSO】 模型就绪。")

    optimizer_model_list.append({
        'optimizer': PsoSwarm,
        'fun_model': fun_model_rl_pso if fun_model_rl_pso else no_model_fun_model,
    })

    # ==========================================
    # 实验序列 3：RL + 收敛策略 PSO (核心实验组)
    # ==========================================
    fun_model_rl_conv = get_optimizer_train_result(
        ConvPsoSwarm.optimizer_name, dim, group, separate_train, max_fe, n_part
    )
    if not fun_model_rl_conv:
        print(f"未能在 DB 找到 【RL+收敛策略 ({ConvPsoSwarm.optimizer_name})】 的有效模型！")
    else:
        print(f"【RL+收敛策略】 模型就绪。")

    # 注意：这里已经彻底替换为 ConvPsoSwarm
    optimizer_model_list.append({
        'optimizer': ConvPsoSwarm,
        'fun_model': fun_model_rl_conv if fun_model_rl_conv else no_model_fun_model,
    })

    # --- 封装评估任务 ---
    new_result_evaluate_task_dic = {
        'type': 'new_result_evaluate',
        'optimizer_model_list': optimizer_model_list,
        'evaluate_function': funs,
        'group': group,
        'max_fe': max_fe,
        'n_part': n_part,
        'dim': dim,
        'runtimes': runtimes,
    }

    return [new_result_evaluate_task_dic]

if __name__ == '__main__':
    # 启动前的健康度检查预演
    print("\n--- Rlpso(tf版) 30D 评测任务生成校验 ---")
    tasks = generate_evaluate_tasks()

    print("\n[待评测算法序列]:")
    for idx, opt_dict in enumerate(tasks[0]['optimizer_model_list']):
        opt_name = opt_dict['optimizer'].__name__
        model_status = "装备RL模型" if opt_dict['fun_model'] != no_model_fun_model else "纯传统启发式"
        print(f"  {idx + 1}. 核心引擎: {opt_name.ljust(15)} | 驱动模式: {model_status}")
    print("---------------------------------------\n")