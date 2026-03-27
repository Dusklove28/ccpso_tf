import logging
import sys

def get_logger(name="RL_PSO"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 避免在多次调用时重复添加 Handler
    if not logger.handlers:
        # 定义日志格式：[时间] [日志级别] 消息
        formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 1. 控制台输出
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 2. 写入文件（可选，这在放服务器上跑时非常有用）
        file_handler = logging.FileHandler('experiment.log', encoding='utf-8', mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# 暴露出全局可用的 logger 实例
logger = get_logger()