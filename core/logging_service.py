"""UTF-8 轮转日志，限制单文件体积。"""
import logging
from logging.handlers import RotatingFileHandler

def setup_logging(directory, level="INFO"):
    """创建项目专用日志，不改变其他程序的根日志配置。"""
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("flight_delay")
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    handler = RotatingFileHandler(directory / "application.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger
