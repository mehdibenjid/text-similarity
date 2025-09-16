# src/rti_sim/logging.py
from __future__ import annotations
import logging
import os
import sys
from typing import Optional

_LOGGER_NAME = "rti-sim"
_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}

_singleton_logger: Optional[logging.Logger] = None


def _env_level() -> int:
    env = os.getenv("RTI_LOG_LEVEL", "INFO").upper()
    return _LEVELS.get(env, logging.INFO)


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get the app logger (singleton) or a namespaced child.
    Idempotent: no duplicate handlers, no propagation to root.
    """
    global _singleton_logger

    if _singleton_logger is None:
        logger = logging.getLogger(_LOGGER_NAME)
        logger.setLevel(_env_level())
        logger.propagate = False  # don't double-print via root

        if not logger.handlers:  # idempotent guard
            handler = logging.StreamHandler(stream=sys.stdout)
            handler.setLevel(logger.level)
            handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
            logger.addHandler(handler)

        _singleton_logger = logger

    if name:
        # return a child logger that inherits level/handlers but doesn't re-add them
        return _singleton_logger.getChild(name)

    return _singleton_logger


def set_log_level(level: str | int) -> None:
    """
    Dynamically change log level for the app logger and its handlers.
    """
    logger = get_logger()
    if isinstance(level, str):
        lvl = _LEVELS.get(level.upper(), logging.INFO)
    else:
        lvl = int(level)

    logger.setLevel(lvl)
    for h in logger.handlers:
        h.setLevel(lvl)
