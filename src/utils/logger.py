"""
Logging setup for Analytics Copilot.

Configures file + console logging based on config.yaml settings.
"""

import logging
import os
from pathlib import Path

_initialized = False
_PROJECT_ROOT = Path(__file__).parent.parent.parent


def setup_logging(log_level: str = "INFO", log_file: str = "logs/pipeline.log"):
    """
    Initialize logging with file and console handlers.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file relative to project root
    """
    global _initialized
    if _initialized:
        return

    log_path = _PROJECT_ROOT / log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=fmt,
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler(),
        ],
    )

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """Get a named logger, initializing logging if needed."""
    if not _initialized:
        setup_logging()
    return logging.getLogger(name)
