"""Provides functions to create loggers."""

import logging
import os
import sys
from typing import Text, Union


def get_console_handler() -> logging.StreamHandler:
    """Get console handler.
    Returns:
        logging.StreamHandler which logs into stdout
    """

    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
    console_handler.setFormatter(formatter)

    return console_handler

def get_file_handler(log_file: str = "app.log") -> logging.FileHandler:
    """Get file handler.
    Args:
        log_file: The path to the log file.
    Returns:
        logging.FileHandler which logs into a file.
    """

    file_handler = logging.FileHandler(log_file,mode= 'w')
    formatter = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
    file_handler.setFormatter(formatter)
    return file_handler

def get_logger(name: Text = __name__, log_level: Union[Text, int] = logging.DEBUG, log_file: str = "chest-x-ray.log", log_to_file: bool = True) -> logging.Logger:
    """Get logger.
    Args:
        name {Text}: logger name
        log_level {Text or int}: logging level; can be string name or integer value
        log_file {str}: path to log file.
        log_to_file {bool}: whether to log to file or not.
    Returns:
        logging.Logger instance
    """

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Prevent duplicate outputs in Jypyter Notebook
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(get_console_handler())

    if log_to_file:
        logger.addHandler(get_file_handler(log_file))

    logger.propagate = False

    return logger