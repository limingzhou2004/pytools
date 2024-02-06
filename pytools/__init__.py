__author__ = "limingzhou"


import logging
import os
from os.path import expanduser

__version__ = '0.0.3'

def get_logger(name, folder=None, level=logging.INFO):
    logger = logging.getLogger(name)
    if folder is None:
        folder = expanduser("~")
    full_file_name = os.path.join(folder, "log", name)
    handler = MakeFileHandler(f"{full_file_name}.log")
    logging.basicConfig()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)
    logger.setLevel(level)
    return logger


def get_file_folder(cur_file: str) -> str:
    """
    Return the current file path

    Args:
        cur_file: current file name from __file__

    Returns: current path

    """

    return os.path.dirname(cur_file)


def mkdir_path(path):
    os.makedirs(path, exist_ok=True)


class MakeFileHandler(logging.FileHandler):
    def __init__(self, filename, mode="a", encoding=None, delay=0):
        mkdir_path(os.path.dirname(filename))
        logging.FileHandler.__init__(self, filename, mode, encoding, delay)
