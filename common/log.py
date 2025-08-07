import os

from loguru import logger

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
log_dir = os.path.join(BASE_DIR, 'logs')
os.makedirs(log_dir, exist_ok=True)

logger.add(
    os.path.join(log_dir, 'feifei_{time:YYYY-MM-DD}.log'),
    rotation="00:00",
    retention="30 days",
    enqueue=True,
    encoding="utf-8",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | "
           "{level: <8} | "
           "thread-{thread} | "
           "{name}.{function} ｜ "
           "{message}",
    backtrace=True,
    diagnose=True,
)

logger.add(
    os.path.join(log_dir, "error_{time:YYYY-MM-DD}.log"),
    level="ERROR",
    rotation="00:00",
    retention="30 days",
    encoding="utf-8",
)
