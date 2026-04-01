import os
import time
from multiprocessing import freeze_support

import multiprocessing as mp
import numpy as np
import psutil

from display.top_task_result_display import top_task_result_display
from log import logger
from task.all_tasks_generate import all_tasks_generate
from task.tasks_run import task_run
from utils.task_hash import get_task_hash

# Compatibility patch for older codepaths that still expect these aliases.
np.int = int
np.float = float
np.bool = bool
np.object = object

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

task_progress = {}
task_progress_seen = set()
phase_progress = {}
phase_progress_seen = set()


def _ensure_progress_bucket(progress_map, key):
    if key is None:
        return
    if key not in progress_map:
        progress_map[key] = {
            'all': 0,
            'finish': 0,
        }


def _update_progress(progress_map, seen_set, key, task_md5, start=0, finish=0):
    if key is None:
        return

    _ensure_progress_bucket(progress_map, key)

    if start and task_md5 not in seen_set:
        seen_set.add(task_md5)
        progress_map[key]['all'] += 1

    if finish:
        progress_map[key]['finish'] += 1


def _get_phase_name(task):
    return task.get('phase_name')


def task_statistic(task, start=0, finish=0):
    global task_progress, task_progress_seen, phase_progress, phase_progress_seen

    task_type = task.get('type')
    if task_type is None:
        return

    task_md5 = get_task_hash(task)
    _update_progress(task_progress, task_progress_seen, task_type, task_md5, start=start, finish=finish)
    _update_progress(
        phase_progress,
        phase_progress_seen,
        _get_phase_name(task),
        task_md5,
        start=start,
        finish=finish,
    )


def _progress_status(progress):
    if progress['all'] == 0:
        return 'pending'
    if progress['finish'] >= progress['all']:
        return 'done'
    return 'running'


def _progress_fragment(name, progress):
    remaining = max(progress['all'] - progress['finish'], 0)
    status = _progress_status(progress)
    return (
        f"[{name}] status={status} "
        f"done/discovered={progress['finish']}/{progress['all']} "
        f"remaining={remaining}"
    )


def print_task_progress():
    global task_progress, phase_progress

    if phase_progress:
        stage_summary = "--- Stage Progress --- | " + " | ".join(
            _progress_fragment(stage_name, progress)
            for stage_name, progress in sorted(phase_progress.items())
        )
        logger.info(stage_summary)

    if task_progress:
        task_summary = "--- Task Progress --- | " + " | ".join(
            _progress_fragment(task_type, progress)
            for task_type, progress in sorted(task_progress.items())
        )
        logger.info(task_summary)

    with open('progress.txt', 'w', encoding='utf-8') as file:
        if phase_progress:
            file.write("[stages]\n")
            for stage_name, progress in sorted(phase_progress.items()):
                remaining = max(progress['all'] - progress['finish'], 0)
                file.write(
                    f"{stage_name} status={_progress_status(progress)} "
                    f"done/discovered={progress['finish']}/{progress['all']} "
                    f"remaining={remaining}\n"
                )
        if task_progress:
            file.write("[tasks]\n")
            for task_type, progress in sorted(task_progress.items()):
                remaining = max(progress['all'] - progress['finish'], 0)
                file.write(
                    f"{task_type} status={_progress_status(progress)} "
                    f"done/discovered={progress['finish']}/{progress['all']} "
                    f"remaining={remaining}\n"
                )


def _process_task_result(result, running_tasks, wait_result_tasks, need_run_tasks, task_detail):
    result_task_md5 = result['md5']
    child_tasks = result.get('needs')

    if child_tasks:
        child_task_md5s = []
        for child_task in child_tasks:
            child_task_md5s.append(get_task_hash(child_task))
            task_statistic(child_task, start=1)
            need_run_tasks.append(child_task)

        wait_result_tasks[result_task_md5] = child_task_md5s
        logger.info(f"task waiting: {result_task_md5} -> child_tasks={len(child_task_md5s)}")
        return

    if result_task_md5 in running_tasks:
        running_tasks.remove(result_task_md5)

    logger.info(f"task completed: {result_task_md5} type={result.get('type')}")
    task_statistic(result, finish=1)

    del_keys = []
    for parent_task_md5, needs in wait_result_tasks.items():
        if result_task_md5 in needs:
            needs.remove(result_task_md5)
        if len(needs) == 0:
            logger.info(f"dependencies satisfied; requeue parent: {parent_task_md5}")
            del_keys.append(parent_task_md5)
            need_run_tasks.append(task_detail[parent_task_md5])

    for del_key in del_keys:
        del wait_result_tasks[del_key]

    if result['type'] == 'top':
        top_task_result_display(result)


def main(processes=1):
    global task_progress, task_progress_seen, phase_progress, phase_progress_seen

    if processes > mp.cpu_count():
        processes = mp.cpu_count()

    task_progress = {}
    task_progress_seen = set()
    phase_progress = {}
    phase_progress_seen = set()

    logger.info(f"main run at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    logger.info(f"processes:{processes}")

    need_run_tasks = all_tasks_generate()
    for task in need_run_tasks:
        task_statistic(task, start=1)

    running_tasks = []
    wait_result_tasks = {}
    task_detail = {}
    async_results = {}
    pool = None
    run_epoch = 0

    try:
        if processes > 1:
            pool = mp.Pool(processes=processes)

        while len(need_run_tasks) + len(running_tasks) > 0:
            run_epoch += 1
            if run_epoch % 10 == 0:
                print_task_progress()
                mem = psutil.virtual_memory()
                if mem.available < 5 * 1024 * 1024 * 1024:
                    logger.info(f"free memory:{mem.available / 1024 / 1024 / 1024}G")
                    if pool is not None:
                        pool.terminate()
                        pool.join()
                        pool = None
                    return 'restart'

            result = None

            if processes > 1:
                for task_md5, async_result in list(async_results.items()):
                    if not async_result.ready():
                        continue
                    try:
                        result = async_result.get()
                    except Exception:
                        logger.exception(f"multiprocess task failed: {task_md5}")
                        if pool is not None:
                            pool.terminate()
                            pool.join()
                            pool = None
                        raise
                    finally:
                        del async_results[task_md5]
                    break

            if result is None and need_run_tasks:
                task = need_run_tasks.pop()
                running_task_md5 = get_task_hash(task)
                task_detail[running_task_md5] = task

                if running_task_md5 not in running_tasks:
                    running_tasks.append(running_task_md5)

                if processes > 1:
                    if running_task_md5 not in async_results:
                        async_results[running_task_md5] = pool.apply_async(task_run, args=(task,))
                        logger.debug(f"add multiprocess task {running_task_md5}")
                else:
                    result = task_run(task)

            if result:
                _process_task_result(
                    result,
                    running_tasks=running_tasks,
                    wait_result_tasks=wait_result_tasks,
                    need_run_tasks=need_run_tasks,
                    task_detail=task_detail,
                )
            elif not need_run_tasks:
                time.sleep(1)
    finally:
        if pool is not None:
            pool.close()
            pool.join()


if __name__ == '__main__':
    freeze_support()

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    res = 'restart'
    processes_count = 2

    while res == 'restart':
        res = main(processes_count)
        logger.info(f'main run finish res:{res}')
        time.sleep(60)
