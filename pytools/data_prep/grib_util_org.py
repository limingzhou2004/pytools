"""
This module summarize the hrrr data folders. Make the list of all file names, datetime, fst hour, type hrrr|nc, and statics, count, missing, count by type
"""


from functools import partial
import glob
from itertools import chain
import os
from math import ceil, floor
import shutil
import sys
import time
from typing import List, Tuple, Union, Dict

import duckdb
import pandas as pd
import pendulum as pu

from pytools.retry.api import retry

folder_table_name = 'hrrr_folder_info'
folder_info_file = 'data/hrrr_obs_folder.txt'


def build_folder_info_table(batch_no=0, insert=False):
    df = pd.read_csv(folder_info_file)
    duckdb.sql(f"CREATE TABLE my_table AS SELECT * FROM my_df")
    return


def main():
    return


if __name__ == '__main__':
    main()