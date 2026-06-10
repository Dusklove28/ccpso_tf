import copy
import os
import time
import traceback

import numpy as np

from env.NormalEnv import NormalEnv
from log import logger
from settings import TASK_PATH
from task.task_run_utils.common import get_task_result, get_tasks_result, result_process
from task.task_run_utils.result_evaluate_task import (
    new_result_evaluate_task_run,
    result_evaluate_task_run,
)
from task.task_run_utils.top_task_run import top_task_run
from task.utils.all_task_final_result_process.all_task_final_result_process import (
    all_task_final_result_process,
)
from task.utils.evluate_optimizer import evluate_optimizer
from train.ddpg import get_ddpg_object
from utils.db.db import save_optimizer
from utils.task_hash import get_task_hash

DEFAULT_LR_CRITIC = 1e-4
DEFAULT_LR_ACTOR = 1e-6
DEFAULT_GAMMA = 0.85
DEFAULT_NOISE = 'norm'
DEFAULT_SIGMA = 0.15

TASK_TYPE_LABELS = {
    'all': '旧版总任务',
    'train': '训练调度',
    'single_train': '训练',
    'evaluate_models': '模型筛选',
    'evaluate_multi_times': '测试调度',
    'single_evaluate': '测试',
    'result_evaluate': '旧版结果汇总',
    'new_result_evaluate': '最终对比',
    'top': '总任务',
}


def _format_function(fun_num):
    return f"F{fun_num}函数"


def _format_functions(fun_nums):
    if fun_nums is None:
        return "函数未指定"
    if not isinstance(fun_nums, (list, tuple, set)):
        fun_nums = [fun_nums]
    return "、".join(_format_function(fun_num) for fun_num in fun_nums)


def _phase_label(task, fallback=None):
    phase_name = task.get('phase_name')
    if phase_name and phase_name != '最终对比':
        return str(phase_name)

    optimizer = (
        task.get('optimizer')
        or task.get('evaluate_optimizer')
        or fallback
    )
    if optimizer is None:
        return "未命名算法"

    optimizer_name = getattr(optimizer, 'optimizer_name', getattr(optimizer, '__name__', str(optimizer)))
    if optimizer_name == 'PSO':
        return 'PSO'
    if str(optimizer_name).startswith('Conv_PSO'):
        return 'RLCCPSO'
    return optimizer_name


def _train_optimizer_label(task):
    optimizer = task.get('optimizer')
    if optimizer is None:
        return _phase_label(task)

    optimizer_name = getattr(optimizer, 'optimizer_name', getattr(optimizer, '__name__', str(optimizer)))
    if optimizer_name == 'PSO':
        return 'PSO'
    if str(optimizer_name).startswith('Conv_PSO'):
        return 'CCPSO'
    return optimizer_name


def _task_label(task):
    task_type = task.get('type')
    phase = _phase_label(task)

    if task_type == 'single_train':
        phase = _train_optimizer_label(task)
        return f"训练：{phase} + {_format_functions(task.get('fun_nums'))}"
    if task_type == 'train':
        phase = _train_optimizer_label(task)
        return f"训练调度：{phase} + {_format_functions(task.get('fun_nums'))}"
    if task_type in ('single_evaluate', 'evaluate_multi_times'):
        return f"测试：{phase} + {_format_function(task.get('evaluate_function'))}"
    if task_type == 'evaluate_models':
        return f"模型筛选：{phase} + {_format_functions(task.get('evaluate_functions'))}"
    if task_type in ('new_result_evaluate', 'result_evaluate'):
        return f"最终对比：{_format_functions(task.get('evaluate_function'))}"
    if task_type == 'top':
        return f"总任务：{_format_functions(task.get('evaluate_function'))}"

    return TASK_TYPE_LABELS.get(task_type, str(task_type))


def _save_train_result_to_db(task, train_result):
    optimizer = task['optimizer']
    save_optimizer([{
        'optimizer': optimizer.optimizer_name,
        'dim': task['dim'],
        'group': task['group'],
        'separate_train': task['separate_train'],
        'max_fe': task['max_fe'],
        'n_part': task['n_part'],
        'train_result': train_result,
    }])


def _trace_fe(row):
    if isinstance(row, dict):
        return int(row.get('fe', 0))
    return int(row[0])


def _trace_value(row, key='conv_a'):
    if isinstance(row, dict):
        if key not in row:
            return None
        return float(row[key])
    if key == 'conv_a' and len(row) > 1:
        return float(row[1])
    return None


def _summarize_trace_metric(trace_runs, key='conv_a'):
    if not trace_runs:
        return None
    fe_value_map = {}
    for run_trace in trace_runs:
        for row in run_trace:
            value = _trace_value(row, key)
            if value is None:
                continue
            fe = _trace_fe(row)
            if fe not in fe_value_map:
                fe_value_map[fe] = []
            fe_value_map[fe].append(value)

    if not fe_value_map:
        return None

    fe_points = sorted(fe_value_map.keys())
    mean_vals = []
    std_vals = []
    var_vals = []
    count_vals = []

    for fe in fe_points:
        values = np.asarray(fe_value_map[fe], dtype=float)
        mean_vals.append(float(np.mean(values)))
        std_vals.append(float(np.std(values)))
        var_vals.append(float(np.var(values)))
        count_vals.append(int(values.size))

    return {
        'fe': fe_points,
        'mean': mean_vals,
        'std': std_vals,
        'var': var_vals,
        'count': count_vals,
    }


def _summarize_control_runs(trace_runs):
    metric_keys = [
        'conv_a',
        'raw_action',
        'conv_a_norm',
        'progress',
        'swarm_diversity',
        'pbest_diversity',
        'q_diversity',
        'q_gbest_distance',
        'x_q_distance',
        'collapse_risk',
        'recent_gbest_improvement',
        'recent_mean_improvement',
        'boundary_ratio',
        'velocity_clip_ratio',
        'instability_penalty',
    ]
    stats = {}
    for key in metric_keys:
        metric_stats = _summarize_trace_metric(trace_runs, key)
        if metric_stats is not None:
            stats[key] = metric_stats
    return stats or None


def _summarize_conv_runs(conv_runs):
    return _summarize_trace_metric(conv_runs, 'conv_a')


def task_run(task, mq=None):
    task_md5 = get_task_hash(task)
    logger.info(f"开始任务 | {_task_label(task)} | 类型={TASK_TYPE_LABELS.get(task.get('type'), task.get('type'))} | 任务ID={task_md5}")

    result = get_task_result(task) if task['type'] not in ['top', 'new_result_evaluate'] else None
    try:
        if result:
            logger.info(f"命中缓存 | {_task_label(task)} | 任务ID={task_md5}")
            if task['type'] == 'train' and result.get('result') is not None:
                _save_train_result_to_db(task, result['result'])
            return result_process(task, result, write=False, mq=mq)
        if task['type'] == 'all':
            return all_task_run(task, mq)
        if task['type'] == 'train':
            return train_task_run(task, mq)
        if task['type'] == 'single_train':
            return single_train_task_run(task, mq)
        if task['type'] == 'evaluate_models':
            return evaluate_models_task_run(task, mq)
        if task['type'] == 'evaluate_multi_times':
            return evaluate_multi_times_task_run(task, mq)
        if task['type'] == 'single_evaluate':
            return single_evaluate_task_run(task, mq)
        if task['type'] == 'result_evaluate':
            return result_evaluate_task_run(task, mq)
        if task['type'] == 'new_result_evaluate':
            return new_result_evaluate_task_run(task, mq)
        if task['type'] == 'top':
            return top_task_run(task, mq)
        raise ValueError(f"unknown task type: {task['type']}")
    except Exception as exc:
        with open('error.txt', 'a') as file:
            traceback.print_exc(file=file)
        logger.info(f"任务出错开始 | {_task_label(task)} | 任务ID={task_md5}")
        traceback.print_exc()
        logger.info(f"任务出错结束 | {_task_label(task)} | 任务ID={task_md5}")
        time.sleep(20)
        raise exc


def all_task_run(task, mq=None):
    assert task['type'] == 'all'

    optimizer = task['evaluate_optimizer']
    train_task = {
        'type': 'train',
        'optimizer': optimizer,
        'group': task['group'],
        'train_max_steps': task['train_max_steps'],
        'train_max_episode': task['train_max_episode'],
        'fun_nums': task['evaluate_function'],
        'train_num': task['train_times'],
        'separate_train': task['separate_train'],
        'runtimes': task['runtimes'],
        'dim': task['dim'],
        'max_fe': task['max_fe'],
        'n_part': task['n_part'],
        'env_class': task.get('env_class') or NormalEnv,
        'lr_critic': task.get('lr_critic', DEFAULT_LR_CRITIC),
        'lr_actor': task.get('lr_actor', DEFAULT_LR_ACTOR),
        'gamma': task.get('gamma', DEFAULT_GAMMA),
        'noise': task.get('noise', DEFAULT_NOISE),
        'sigma': task.get('sigma', DEFAULT_SIGMA),
        'actor_units': task.get('actor_units'),
        'critic_units': task.get('critic_units'),
        'optimizer_config': copy.deepcopy(task.get('optimizer_config', {})),
        'env_config': copy.deepcopy(task.get('env_config', {})),
    }

    results = get_tasks_result([train_task])
    if results is None:
        task_result = {
            'result': None,
            'md5': get_task_hash(task),
            'needs': [train_task],
        }
        return result_process(task, task_result, write=False, mq=mq)

    train_result = results[0]['result']
    evaluate_task = {
        'type': 'result_evaluate',
        'optimizer': optimizer,
        'group': task['group'],
        'base_evaluate_optimizer': task['base_evaluate_optimizers'],
        'separate_train': task['separate_train'],
        'runtimes': task['runtimes'],
        'dim': task['dim'],
        'max_fe': task['max_fe'],
        'model': train_result,
        'n_part': task['n_part'],
        'optimizer_config': copy.deepcopy(task.get('optimizer_config', {})),
    }

    results = get_tasks_result([evaluate_task])
    if results is None:
        task_result = {
            'result': None,
            'md5': get_task_hash(task),
            'needs': [evaluate_task],
        }
        return result_process(task, task_result, write=False, mq=mq)

    task_result = copy.deepcopy(task)
    task_result['result'] = [
        all_task_final_result_process(result, task['evaluate_optimizer'])
        for result in results
    ]
    task_result['md5'] = get_task_hash(task)
    task_result['train_result'] = train_result
    return result_process(task, task_result, mq)


train_task_test_dic = {
    'optimizer': None,
    'group': 5,
    'train_max_steps': 0,
    'train_max_episode': 0,
    'fun_nums': [1],
    'train_num': 3,
    'separate_train': True,
    'runtimes': 10,
    'dim': 20,
    'max_fe': 1e4,
}


def train_task_run(task, mq=None):
    assert task['type'] == 'train'

    optimizer = task['optimizer']
    tasks = []

    if task['separate_train']:
        for fun_num in task['fun_nums']:
            single_train_task = {
                'type': 'single_train',
                'phase_name': task.get('phase_name'),
                'optimizer': optimizer,
                'group': task['group'],
                'train_max_steps': task['train_max_steps'],
                'train_max_episode': task['train_max_episode'],
                'fun_nums': [fun_num],
                'train_num': task['train_num'],
                'runtimes': task['runtimes'],
                'dim': task['dim'],
                'max_fe': task['max_fe'],
                'n_part': task['n_part'],
                'env_class': task.get('env_class') or NormalEnv,
                'lr_critic': task.get('lr_critic', DEFAULT_LR_CRITIC),
                'lr_actor': task.get('lr_actor', DEFAULT_LR_ACTOR),
                'gamma': task.get('gamma', DEFAULT_GAMMA),
                'noise': task.get('noise', DEFAULT_NOISE),
                'sigma': task.get('sigma', DEFAULT_SIGMA),
                'actor_units': task.get('actor_units'),
                'critic_units': task.get('critic_units'),
                'optimizer_config': copy.deepcopy(task.get('optimizer_config', {})),
                'env_config': copy.deepcopy(task.get('env_config', {})),
            }
            tasks.append(single_train_task)
    else:
        single_train_task = {
            'type': 'single_train',
            'phase_name': task.get('phase_name'),
            'optimizer': optimizer,
            'group': task['group'],
            'train_max_steps': task['train_max_steps'],
            'train_max_episode': task['train_max_episode'],
            'fun_nums': task['fun_nums'],
            'train_num': task['train_num'],
            'runtimes': task['runtimes'],
            'dim': task['dim'],
            'max_fe': task['max_fe'],
            'n_part': task['n_part'],
            'env_class': task.get('env_class') or NormalEnv,
            'lr_critic': task.get('lr_critic', DEFAULT_LR_CRITIC),
            'lr_actor': task.get('lr_actor', DEFAULT_LR_ACTOR),
            'gamma': task.get('gamma', DEFAULT_GAMMA),
            'noise': task.get('noise', DEFAULT_NOISE),
            'sigma': task.get('sigma', DEFAULT_SIGMA),
            'actor_units': task.get('actor_units'),
            'critic_units': task.get('critic_units'),
            'optimizer_config': copy.deepcopy(task.get('optimizer_config', {})),
            'env_config': copy.deepcopy(task.get('env_config', {})),
        }
        tasks.append(single_train_task)

    results = get_tasks_result(tasks)
    if results is None:
        task_result = {
            'result': None,
            'md5': get_task_hash(task),
            'needs': tasks,
        }
        return result_process(task, task_result, write=False, mq=mq)

    real_result = {}
    for result in results:
        for fun_num in result['fun_nums']:
            real_result[fun_num] = result['result']

    _save_train_result_to_db(task, real_result)

    task_result = copy.deepcopy(task)
    task_result['result'] = real_result
    task_result['md5'] = get_task_hash(task)
    return result_process(task, task_result, mq)


single_train_task_test_dic = {
    'optimizer': None,
    'group': 5,
    'train_max_steps': 0,
    'train_max_episode': 0,
    'fun_nums': [1],
    'train_num': 3,
    'runtimes': 10,
    'dim': 20,
    'max_fe': 1e4,
}


def _build_train_env_and_limits(task):
    optimizer = task['optimizer']
    env_class = task.get('env_class') or NormalEnv
    env_config = dict(task.get('env_config') or {})
    gym_env = env_class(
        obs_shape=(optimizer.obs_space,),
        action_shape=(optimizer.action_space * task['group'],),
        target_optimizer=optimizer,
        fun_nums=task['fun_nums'],
        max_fe=task['max_fe'],
        n_part=task['n_part'],
        n_dim=task['dim'],
        group=task['group'],
        optimizer_config=copy.deepcopy(task.get('optimizer_config', {})),
        **env_config,
    )
    gym_env.phase_name = task.get('phase_name')

    save_freq = max(1, int(task['train_max_episode'] / 20))
    train_limits = {
        'max_episodes': task['train_max_episode'],
        'max_epochs': task['train_max_steps'],
        'max_steps': task['train_max_steps'],
        'save_freq': save_freq,
        'memory_cap': 10000000,
    }
    return gym_env, train_limits


def single_train_task_run(task, mq=None):
    assert task['type'] == 'single_train'

    optimizer = task['optimizer']
    fun_nums = task['fun_nums']
    group = task['group']
    runtimes = task['runtimes']
    max_fe = task['max_fe']
    dim = task['dim']
    lr_critic = task.get('lr_critic', DEFAULT_LR_CRITIC)
    lr_actor = task.get('lr_actor', DEFAULT_LR_ACTOR)
    gamma = task.get('gamma', DEFAULT_GAMMA)
    noise = task.get('noise', DEFAULT_NOISE)
    sigma = task.get('sigma', DEFAULT_SIGMA)
    actor_units = task.get('actor_units')
    critic_units = task.get('critic_units')

    gym_env, train_limits = _build_train_env_and_limits(task)

    assert gym_env.action_space.high == -gym_env.action_space.low
    is_discrete = False
    task_md5 = get_task_hash(task)
    task_dir = TASK_PATH.joinpath(f'{task_md5}/')
    logger.info(
        f"训练配置 | {_task_label(task)} | 任务ID={task_md5} | "
        f"噪声={noise} | sigma={sigma} | "
        f"actor学习率={lr_actor} | critic学习率={lr_critic} | gamma={gamma}"
    )

    for train_index in range(task['train_num']):
        if os.path.exists(task_dir.joinpath(f"ddpg_actor_final_round{train_index}.h5")):
            logger.info(
                f"跳过训练 | {_task_label(task)} | round={train_index} | "
                f"原因=最终Actor已存在 | 任务ID={task_md5}"
            )
            continue

        ddpg = get_ddpg_object(
            gym_env,
            discrete=is_discrete,
            memory_cap=train_limits['memory_cap'],
            lr_critic=lr_critic,
            lr_actor=lr_actor,
            gamma=gamma,
            noise=noise,
            sigma=sigma,
            actor_units=actor_units,
            critic_units=critic_units,
        )
        ddpg.train(
            max_episodes=train_limits['max_episodes'],
            max_epochs=train_limits['max_epochs'],
            max_steps=train_limits['max_steps'],
            task_path=task_dir,
            train_num=train_index,
            save_freq=train_limits['save_freq'],
        )

    new_task = {
        'type': 'evaluate_models',
        'phase_name': task.get('phase_name'),
        'evaluate_optimizers': [],
        'evaluate_functions': fun_nums,
        'dims': [dim],
        'groups': [group],
        'runtimes': runtimes if runtimes < 5 else 5,
        'max_fe': max_fe,
        'n_part': task['n_part'],
        'optimizer_config': copy.deepcopy(task.get('optimizer_config', {})),
    }
    model_candidates = sorted(task_dir.glob('ddpg_actor*.h5'))
    for model in model_candidates:
        new_task['evaluate_optimizers'].append({
            'optimizer': optimizer,
            'model': model,
            'optimizer_config': copy.deepcopy(task.get('optimizer_config', {})),
        })

    logger.info(
        f"模型筛选 | {_task_label(task)} | 候选模型={len(model_candidates)} | "
        f"测试调度数={len(model_candidates)} | 预计单次测试数={len(model_candidates) * new_task['runtimes']}"
    )

    results = get_tasks_result([new_task])
    if results is None:
        task_result = {
            'result': None,
            'md5': get_task_hash(task),
            'needs': [new_task],
        }
        return result_process(task, task_result, write=False, mq=mq)

    real_results = results[0]['result'][:3]
    new_models = [real_result['model'] for real_result in real_results]

    task_result = copy.deepcopy(task)
    task_result['result'] = new_models
    task_result['md5'] = get_task_hash(task)
    return result_process(task, task_result, mq)


evaluate_task_test_dic = {
    'evaluate_optimizers': [
        {
            'optimizer': None,
            'model': 'model_path',
        },
    ],
    'evaluate_functions': [1],
    'dims': [20],
    'groups': [5],
    'run_times': 1,
    'max_fe': 1e4,
}


def evaluate_models_task_run(task, mq=None):
    assert task['type'] == 'evaluate_models'

    tasks = []
    for evaluate_optimizer in task['evaluate_optimizers']:
        for evaluate_function in task['evaluate_functions']:
            for dim in task['dims']:
                for group in task['groups']:
                    tasks.append({
                        'type': 'evaluate_multi_times',
                        'phase_name': task.get('phase_name'),
                        'evaluate_optimizer': evaluate_optimizer['optimizer'],
                        'model': evaluate_optimizer['model'],
                        'evaluate_function': evaluate_function,
                        'dim': dim,
                        'group': group,
                        'max_fe': task['max_fe'],
                        'runtimes': task['runtimes'],
                        'n_part': task['n_part'],
                        'optimizer_config': copy.deepcopy(
                            evaluate_optimizer.get('optimizer_config', task.get('optimizer_config', {}))
                        ),
                    })

    logger.info(
        f"模型筛选展开 | {_task_label(task)} | 生成测试调度任务={len(tasks)}"
    )

    results = get_tasks_result(tasks)
    if results is None:
        task_result = {
            'result': None,
            'md5': get_task_hash(task),
            'needs': tasks,
        }
        return result_process(task, task_result, write=False, mq=mq)

    results.sort(key=lambda result: result['result'][-1][2])

    task_result = copy.deepcopy(task)
    task_result['result'] = results
    task_result['md5'] = get_task_hash(task)
    return result_process(task, task_result, mq)


def evaluate_multi_times_task_run(task, mq=None):
    assert task['type'] == 'evaluate_multi_times'

    tasks = []
    for run_index in range(task['runtimes']):
        copy_task = copy.deepcopy(task)
        copy_task['type'] = 'single_evaluate'
        del copy_task['runtimes']
        copy_task['run_index'] = run_index
        tasks.append(copy_task)

    logger.info(
        f"测试调度 | {_task_label(task)} | 重复次数={len(tasks)} | 模型={task.get('model')}"
    )

    results = get_tasks_result(tasks)
    if results is None:
        task_result = {
            'result': None,
            'md5': get_task_hash(task),
            'needs': tasks,
        }
        return result_process(task, task_result, write=False, mq=mq)

    average_ress = np.average(np.array([result['result'] for result in results]), axis=0)
    conv_runs = [result.get('conv_trace', []) for result in results if result.get('conv_trace')]
    conv_stats = _summarize_conv_runs(conv_runs)
    control_stats = _summarize_control_runs(conv_runs)

    task_result = copy.deepcopy(task)
    task_result['result'] = average_ress
    task_result['md5'] = get_task_hash(task)
    task_result['conv_runs'] = conv_runs
    if conv_stats is not None:
        task_result['conv_stats'] = conv_stats
    if control_stats is not None:
        task_result['control_stats'] = control_stats
    return result_process(task, task_result, mq)


single_evaluate_task_test_dic = {
    'evaluate_optimizer': None,
    'model': 'model_path',
    'evaluate_function': 1,
    'dim': 20,
    'group': 5,
    'run_index': 1,
    'max_fe': 1e4,
}


def single_evaluate_task_run(task, mq=None):
    assert task['type'] == 'single_evaluate'

    task_result = get_task_result(task)
    if not task_result:
        result, conv_trace = evluate_optimizer(task, return_trace=True)
        task_result = copy.deepcopy(task)
        task_result['result'] = result
        if conv_trace is not None:
            task_result['conv_trace'] = conv_trace

        task_result['md5'] = get_task_hash(task)

    return result_process(task, task_result, mq)
