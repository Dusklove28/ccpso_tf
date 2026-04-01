import copy

import numpy as np

from task.task_run_utils.common import get_tasks_result, result_process
from utils.task_hash import get_task_hash
from log import logger


def _get_train_phase_name(optimizer_class):
    optimizer_name = getattr(optimizer_class, 'optimizer_name', optimizer_class.__name__)
    if optimizer_name == 'PSO':
        return 'Stage1-RL+BasicPSO'
    if optimizer_name == 'Conv_PSO':
        return 'Stage2-RL+BasicPSO+Convergence'
    return f"Train-{optimizer_name}"


def _build_train_tasks(task):
    tasks = []

    max_fe = task.get('max_fe', 1e4)
    n_part = task.get('n_part', 100)
    lr_critic = task.get('lr_critic', 1e-7)
    lr_actor = task.get('lr_actor', 1e-9)

    for optimizer_pair in task['rl_optimizer_pairs']:
        train_optimizer = optimizer_pair['train_optimizer']
        evaluate_optimizer = optimizer_pair['evaluate_optimizer']
        phase_name = _get_train_phase_name(train_optimizer)
        for separate_train in task['separate_trains']:
            for group in task['groups']:
                for dim in task['dims']:
                    train_task = {
                        'type': 'train',
                        'phase_name': phase_name,
                        'optimizer': train_optimizer,
                        'evaluate_optimizer': evaluate_optimizer,
                        'group': group,
                        'train_max_steps': task['train_max_steps'],
                        'train_max_episode': task['train_max_episode'],
                        'fun_nums': task['evaluate_function'],
                        'train_num': task['train_times'],
                        'separate_train': separate_train,
                        'runtimes': task['runtimes'],
                        'dim': dim,
                        'max_fe': max_fe,
                        'n_part': n_part,
                        'lr_critic': lr_critic,
                        'lr_actor': lr_actor,
                    }
                    tasks.append(train_task)
    return tasks


def _build_compare_tasks(task, train_tasks, train_results):
    baseline_optimizers = task.get('baseline_optimizers')
    if not baseline_optimizers:
        raise ValueError("top task requires at least one baseline optimizer.")

    compare_task_map = {}
    baseline_fun_model = {
        f_num: [None]
        for f_num in task['evaluate_function']
    }

    for train_task, train_result in zip(train_tasks, train_results):
        key = (
            train_task['separate_train'],
            train_task['group'],
            train_task['dim'],
        )
        if key not in compare_task_map:
            optimizer_model_list = []
            for optimizer in baseline_optimizers:
                optimizer_model_list.append({
                    'optimizer': optimizer,
                    'fun_model': copy.deepcopy(baseline_fun_model),
                })
            compare_task_map[key] = optimizer_model_list

        compare_task_map[key].append({
            'optimizer': train_task['evaluate_optimizer'],
            'fun_model': train_result['result'],
        })

    compare_tasks = []
    for (separate_train, group, dim), optimizer_model_list in compare_task_map.items():
        compare_tasks.append({
            'type': 'new_result_evaluate',
            'phase_name': 'Stage3-FinalCompare',
            'optimizer_model_list': optimizer_model_list,
            'evaluate_function': task['evaluate_function'],
            'group': group,
            'max_fe': task.get('max_fe', 1e4),
            'n_part': task.get('n_part', 100),
            'dim': dim,
            'runtimes': task['runtimes'],
            'separate_train': separate_train,
        })

    return compare_tasks


def _calculate_average_ranks(compare_result):
    optimizer_rank_cache = {}

    for _, optimizer_results in compare_result['result'].items():
        ordered_results = sorted(
            optimizer_results.items(),
            key=lambda item: item[1]['result'][-1][2]
        )
        for rank, (optimizer_name, _) in enumerate(ordered_results, start=1):
            if optimizer_name not in optimizer_rank_cache:
                optimizer_rank_cache[optimizer_name] = []
            optimizer_rank_cache[optimizer_name].append(rank)

    return {
        optimizer_name: float(np.mean(ranks))
        for optimizer_name, ranks in optimizer_rank_cache.items()
    }


def top_task_run(task, mq=None):
    train_tasks = _build_train_tasks(task)

    train_results = get_tasks_result(train_tasks)
    if train_results is None:
        task_result = {
            'result': None,
            'md5': get_task_hash(task),
            'needs': train_tasks,
        }
        return result_process(task, task_result, write=False, mq=mq)

    compare_tasks = _build_compare_tasks(task, train_tasks, train_results)
    compare_results = get_tasks_result(compare_tasks)
    if compare_results is None:
        task_result = {
            'result': None,
            'md5': get_task_hash(task),
            'needs': compare_tasks,
        }
        return result_process(task, task_result, write=False, mq=mq)

    summary_results = []
    for compare_task, compare_result in zip(compare_tasks, compare_results):
        average_ranks = _calculate_average_ranks(compare_result)
        summary_results.append({
            'type': (
                f"compare-separate_train{compare_task['separate_train']}"
                f"-group{compare_task['group']}-dim{compare_task['dim']}"
            ),
            'average_ranks': average_ranks,
            'functions': sorted(compare_result['result'].keys()),
            'result': compare_result['result'],
        })

    task_result = copy.deepcopy(task)
    task_result['result'] = summary_results
    task_result['compare_result'] = compare_results
    task_result['md5'] = get_task_hash(task)

    logger.info(f"最终比较任务路径:{task_result['md5']}")
    return result_process(task, task_result, mq)
