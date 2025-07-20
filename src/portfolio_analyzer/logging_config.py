"""Logging configuration for the portfolio analyzer application."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(level=logging.INFO, log_dir="logs"):
    """Set up logging configuration.

    Args:
        level (int): Logging level, default is logging.INFO.
        log_dir (str): Directory to store log files, default is "logs".

    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / "portfolio_analyzer.log"

    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)-8s - %(name)-25s - %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(console_formatter)
    stream_handler.setLevel(level)
    root_logger.addHandler(stream_handler)

    file_handler = RotatingFileHandler(log_file_path, maxBytes=5 * 1024 * 1024, backupCount=3)
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)

    logging.info(
        "Logging configured. Console level: %s, File level: DEBUG", logging.getLevelName(level)
    )
