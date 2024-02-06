import logging
from multiprocessing import Pool
import os


import xarray as xr
import numpy as np
import pandas as pd

def get_absolute_path(cur_path: str, file_name) -> str:
    """
    Return the absolute full path and file name.

    Args:
        cur_path: the path-file name
        file_name: relative file name

    Returns: full path-name for the file_name

    """
    return os.path.join(os.path.dirname(cur_path), file_name)


def get_files_from_a_folder(fd:str):
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [join(fd, f) for f in listdir(fd) if isfile(join(fd, f))]
    return onlyfiles


def get_now_str():
    from datetime import datetime

    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%Y-%m-%dT%H-%M-%S")
    return dt_string


def get_logger(level=logging.INFO, file_name=f"{get_now_str()}.log"):
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)

    # our first handler is a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler_format = '%(asctime)s | %(levelname)s: %(message)s'
    console_handler.setFormatter(logging.Formatter(console_handler_format))
    logger.addHandler(console_handler)

    # the second handler is a file handler
    file_handler = logging.FileHandler(file_name)
    file_handler.setLevel(level)
    file_handler_format = '%(asctime)s | %(levelname)s | %(lineno)d: %(message)s'
    file_handler.setFormatter(logging.Formatter(file_handler_format))
    logger.addHandler(file_handler)

    return logger


def parallelize_dataframe(df, func, n_cores=7):
    # usage train = parallelize_dataframe(train_df, add_features)
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df
