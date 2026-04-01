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


def task_run(task, mq=None):
    task_md5 = get_task_hash(task)
    logger.info(f"run task {task_md5}-{task.get('type')}-{task}")

    result = get_task_result(task) if task['type'] not in ['top', 'new_result_evaluate'] else None
    try:
        if result:
            logger.info(f'{task_md5} cache hit')
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
        logger.info(f"error-start-{task_md5}-{task.get('type')}-{task}")
        traceback.print_exc()
        logger.info(f"error-end-{task_md5}-{task.get('type')}-{task}")
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
        'lr_critic': task.get('lr_critic', 1e-7),
        'lr_actor': task.get('lr_actor', 1e-9),
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
                'lr_critic': task.get('lr_critic', 1e-7),
                'lr_actor': task.get('lr_actor', 1e-9),
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
            'lr_critic': task.get('lr_critic', 1e-7),
            'lr_actor': task.get('lr_actor', 1e-9),
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
    gym_env = NormalEnv(
        obs_shape=(optimizer.obs_space,),
        action_shape=(optimizer.action_space * task['group'],),
        target_optimizer=optimizer,
        fun_nums=task['fun_nums'],
        max_fe=task['max_fe'],
        n_part=task['n_part'],
        n_dim=task['dim'],
        group=task['group'],
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
    lr_critic = task.get('lr_critic', 1e-7)
    lr_actor = task.get('lr_actor', 1e-9)

    gym_env, train_limits = _build_train_env_and_limits(task)

    assert gym_env.action_space.high == -gym_env.action_space.low
    is_discrete = False
    task_md5 = get_task_hash(task)
    task_dir = TASK_PATH.joinpath(f'{task_md5}/')

    for train_index in range(task['train_num']):
        if os.path.exists(task_dir.joinpath(f"ddpg_actor_final_round{train_index}.h5")):
            continue

        ddpg = get_ddpg_object(
            gym_env,
            discrete=is_discrete,
            memory_cap=train_limits['memory_cap'],
            lr_critic=lr_critic,
            lr_actor=lr_actor,
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
    }
    model_candidates = sorted(task_dir.glob('ddpg_actor*.h5'))
    for model in model_candidates:
        new_task['evaluate_optimizers'].append({
            'optimizer': optimizer,
            'model': model,
        })

    logger.info(
        f"[{task.get('phase_name', optimizer.optimizer_name)}] "
        f"model_selection candidates={len(model_candidates)} "
        f"evaluate_multi_times={len(model_candidates)} "
        f"expected_single_evaluate={len(model_candidates) * new_task['runtimes']}"
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
                    })

    logger.info(
        f"[{task.get('phase_name', 'UnlabeledPhase')}] "
        f"evaluate_models discovered={len(tasks)} evaluate_multi_times tasks"
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
        f"[{task.get('phase_name', 'UnlabeledPhase')}] "
        f"evaluate_multi_times runtimes={len(tasks)} model={task.get('model')}"
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

    task_result = copy.deepcopy(task)
    task_result['result'] = average_ress
    task_result['md5'] = get_task_hash(task)
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
        result = evluate_optimizer(task)
        task_result = copy.deepcopy(task)
        task_result['result'] = result
        task_result['md5'] = get_task_hash(task)

    return result_process(task, task_result, mq)
