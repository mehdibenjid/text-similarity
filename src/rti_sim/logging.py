from __future__ import annotations
import logging
from rich.logging import RichHandler

def get_logger(name: str = "rti-sim") -> logging.Logger:
    """
    Minimal Rich logger. Single handler, INFO level.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = RichHandler(rich_tracebacks=True, show_time=True, show_path=False)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger