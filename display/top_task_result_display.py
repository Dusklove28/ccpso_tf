from log import logger


def top_task_result_display(result):
    logger.info('top_task_result_display')

    evaluate_result = result.get('result', [])
    if not evaluate_result:
        logger.info('top result is empty')
        return

    for res in evaluate_result:
        compare_type = res.get('type', 'compare')
        average_ranks = res.get('average_ranks', {})
        ordered_ranks = sorted(average_ranks.items(), key=lambda item: item[1])
        logger.info(f'{compare_type} average_ranks: {ordered_ranks}')
        print(f'{compare_type} average_ranks: {ordered_ranks}')
