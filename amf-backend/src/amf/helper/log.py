"""Logging functionality."""  # TODO replace with loguru

import datetime
import sys
from typing import Any

from loguru import logger as logger

ERR_NO_DATA = 10
ERR_INVALID_DATA = 11
ERR_NO_PRETRAINED_MODEL = 20
ERR_INVALID_MODEL_SHAPE = 21
ERR_INVALID_ANNOTATION_LEVEL = 22
ERR_MISSING_ARCHIVE = 30
ERR_MISSING_ANNOTATIONS = 32
ERR_INVALID_MODEL = 40
ERR_NO_DATABASE_CONNECTION = 42

logger.remove(0)


def formatter(record: Any) -> str:
    if record["level"].no == 10:
        return "{time:MMMM D, YYYY - HH:mm:ss} | {message}\n"
    return "{time:MMMM D, YYYY - HH:mm:ss} | {level} | {message}\n"


logger.add(f"{datetime.date.today()}.log", format=formatter, level=0)
logger.add(sys.stderr, format=formatter, level=0)
