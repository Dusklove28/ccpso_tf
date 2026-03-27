# from matAgent.awpso import AwpsoSwarm
# from matAgent.clpso import ClpsoSwarm
# from matAgent.epso import EpsoSwarm
# from matAgent.fdrpso import FdrpsoSwarm
# from matAgent.hpso_tvac import HpsotvacSwarm
# from matAgent.lips import LipsSwarm
# from matAgent.olpso import OlpsoSwarm
# from matAgent.pppso import PppsoSwarm
# from matAgent.pso import PsoSwarm
# from matAgent.shpso import ShpsoSwarm
# from matAgent.hrlepso_base import HrlepsoBaseSwarm
# from matAgent.swarm.gwo import GwoSwarm
#
# from matAgent.adaptionPso.f1pso import FT1PsoSwarm
# from matAgent.adaptionPso.f2pso import FT2PsoSwarm
# from matAgent.adaptionPso.success_history_pso import SuccessHistoryPsoSwarm
# from matAgent.adaptionPso.qlpso import QlpsoSwarm
#
# from utils.db.db import get_optimizer_train_result
#
# new_result_evaluate_task_test_dic = {
#     'type': 'new_result_evaluate',
#     'optimizer_model_list': [
#         {
#             'optimizer': 'optimizer_class',
#             'fun_model': {
#                 1: [
#                     'model_path1',
#                 ]
#             }
#         }
#     ],
#     'evaluate_function': list(range(1, 29, 1)),
#     'group': 1,
#     'max_fe': 1e4,
#     'n_part': 100,
#     'dim': 20,
#     'runtimes': 10,
# }
#
# funs = list(range(1, 29, 1))
# no_model_fun_model = {}
# for fun in funs:
#     no_model_fun_model[fun] = [None]
# dim = 30
# runtimes = 10
# max_fe = 1e4
# group = 5
# separate_train = False
# n_part = 100
#
#
# def generate_evaluate_tasks():
#     # no_train_optimizers = [FT1PsoSwarm, FT2PsoSwarm, SuccessHistoryPsoSwarm, QlpsoSwarm]
#     train_optimizers = [PsoSwarm, ]
#     no_train_optimizers = [PsoSwarm, ]
#
#     # no_train_optimizers = [ClpsoSwarm, FdrpsoSwarm, LipsSwarm, PsoSwarm, ShpsoSwarm, AwpsoSwarm, PppsoSwarm, EpsoSwarm]
#     # train_optimizers = [ClpsoSwarm, FdrpsoSwarm, HpsotvacSwarm, LipsSwarm, PsoSwarm, ShpsoSwarm, ]
#     optimizer_model_list = []
#
#     for no_train_optimizer in no_train_optimizers:
#         optimizer_model_list.append({
#             'optimizer': no_train_optimizer,
#             'fun_model': no_model_fun_model,
#         })
#     for train_optimizer in train_optimizers:
#         fun_model = get_optimizer_train_result(train_optimizer.optimizer_name, dim, group, separate_train, max_fe,
#                                                n_part)
#         fun_model = no_model_fun_model if fun_model is None else fun_model
#         optimizer_model_list.append({
#             'optimizer': train_optimizer,
#             'fun_model': fun_model,
#         })
#
#     new_result_evaluate_task_dic = {
#         'type': 'new_result_evaluate',
#         'optimizer_model_list': optimizer_model_list,
#         'evaluate_function': list(range(1, 29, 1)),
#         'group': group,
#         'max_fe': max_fe,
#         'n_part': n_part,
#         'dim': dim,
#         'runtimes': runtimes,
#     }
#
#     return [new_result_evaluate_task_dic]
#
#
# if __name__ == '__main__':
#     a = generate_evaluate_tasks()
#     print(a)
#     pass

from matAgent.testpso import TestpsoSwarm
from matAgent.ccpso_50d import FiftyDimCCPsoSwarm
from matAgent.rlepso import RlepsoSwarm
from matAgent.rl_ccpso_eval import RlCCPsoSwarm
from utils.db.db import get_optimizer_train_result


def generate_evaluate_tasks():
    # 1. 【欺骗数据库】：用当初训练配置时的参数去找回已经存好的模型权重
    # 虽然你底层硬编码了 50 维，但配置单上写的是 30，所以模型存在 30 的记录下
    train_dim_in_db = 30
    group = 5
    separate_train = True
    max_fe = 10000
    n_part = 100

    # 2. 【修正测试环境】：将真正的评估维度强制锁定为 50 维！
    real_eval_dim = 50
    runtimes = 5

    optimizer_model_list = []

    # --- 提取原版 RLEPSO 模型 ---
    fun_model_rlepso = get_optimizer_train_result(
        TestpsoSwarm.optimizer_name,
        train_dim_in_db, group, separate_train, max_fe, n_part
    )
    if fun_model_rlepso:
        optimizer_model_list.append({
            'optimizer': RlepsoSwarm,
            'fun_model': fun_model_rlepso,
        })
        print("✅ 成功找回原作者 RLEPSO 的历史训练模型！")

    # --- 提取你的 RL_CCPSO50D 模型 ---
    fun_model_ccpso = get_optimizer_train_result(
        FiftyDimCCPsoSwarm.optimizer_name,
        train_dim_in_db, group, separate_train, max_fe, n_part
    )
    if fun_model_ccpso:
        optimizer_model_list.append({
            'optimizer': RlCCPsoSwarm,
            'fun_model': fun_model_ccpso,
        })
        print("✅ 成功找回你的 RL_CCPSO 的历史训练模型！")

    # 构建纯粹的【评估任务】下发给系统
    new_result_evaluate_task_dic = {
        'type': 'new_result_evaluate',
        'optimizer_model_list': optimizer_model_list,
        'evaluate_function': list(range(1, 29, 1)),
        'group': group,
        'max_fe': max_fe,
        'n_part': n_part,
        'dim': real_eval_dim,  # <--- 核心修复点：告诉测试器用 50 维进行真实考核！
        'runtimes': runtimes,
    }

    return [new_result_evaluate_task_dic]


if __name__ == '__main__':
    print(generate_evaluate_tasks())