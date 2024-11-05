import logging
import os
import os.path as osp


import numpy as np
import pandas as pd
import dask.bag as bag


def get_file_path(fn,this_file_path):   
    """
    Get the absolute path of a file path. If not found, use the this_file_path given by __FILE__

    Args:
        fn (function): full path of a file, or relative path to __FILE__
        this_file_path (_type_): __FILE__ from a given py file

    Returns: absolute path of the file

    """
    if osp.exists(fn):
        return fn
    else:
        return osp.join(osp.dirname(this_file_path), fn)


def get_absolute_path(cur_path: str, file_name) -> str:
    """
    Return the absolute full path and file name.

    Args:
        cur_path: the path-file name
        file_name: relative file name

    Returns: full path-name for the file_name

    """
    return osp.join(os.path.dirname(cur_path), file_name)


def get_files_from_a_folder(fd:str, min_size=1000):
    from os import listdir
    from os.path import isfile, join, getsize
    onlyfiles = [join(fd, f) for f in listdir(fd) if isfile(join(fd, f)) if getsize(join(fd,f )) > min_size and not f.startswith('.')]
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


def parallelize_dataframe(df, func, n_cores=7, partition_size=1):
    # usage train = parallelize_dataframe(train_df, add_features)
    from dask.diagnostics import ProgressBar
    df_split = np.array_split(df, n_cores)
    file_bag = bag.from_sequence(df_split, partition_size=partition_size)
    with ProgressBar():
        res = file_bag.map(func).compute()

    return res
