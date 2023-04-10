import logging
from pathlib import Path

_log_format = f"%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s).%(funcName)s(%(lineno)d) - %(message)s"


def get_file_handler(logging_level: str = "INFO",
                     log_filename: str = "./log_report.log"):
    """
    Write log to file.
    """
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.getLevelName(logging_level))
    file_handler.setFormatter(logging.Formatter(_log_format))
    return file_handler


def get_stream_handler(logging_level: str = "INFO"):
    """
    Write log to console.
    """
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.getLevelName(logging_level))
    stream_handler.setFormatter(logging.Formatter(_log_format))
    return stream_handler


def get_logger(name: str,
               logging_level: str = "INFO",
               log_to_file: bool = True,
               log_to_console: bool = True,
               log_filename: str = "./log_report.log"):
    """
    Write log to file and to console.
    (default choice)
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.getLevelName(logging_level))
    if log_to_file:
        logger.addHandler(get_file_handler(logging_level, log_filename))
    if log_to_console:
        logger.addHandler(get_stream_handler(logging_level))
    return logger


def clear_log_files(log_files_path: str = "."):
    """
    Search and clean all .log files if provided path.
    :param log_files_path: path used for previous logging
    :type log_files_path: str
    """
    log_files_path = Path(log_files_path)
    if not log_files_path.exists():
        raise FileExistsError(f"Logging path do not exists: {log_files_path}")
    # Clear all files
    for filename in log_files_path.glob("*.log"):
        filename.unlink()

