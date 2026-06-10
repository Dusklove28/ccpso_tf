import logging
import sys


class ConsoleHandler(logging.StreamHandler):
    """控制台日志处理器，避免日志和 Keras 进度条挤在同一行。"""

    def emit(self, record):
        try:
            if hasattr(self.stream, "isatty") and self.stream.isatty():
                self.stream.write("\n")
        except Exception:
            pass
        super().emit(record)


def get_logger(name="RL_PSO"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(levelname)s] [%(processName)s]\n%(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        console_handler = ConsoleHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler('experiment.log', encoding='utf-8', mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = get_logger()
