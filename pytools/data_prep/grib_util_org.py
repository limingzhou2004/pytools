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

import pandas as pd
import pendulum as pu
import polars as pl

from pytools.data_prep.grib_utils import decide_grib_type, get_all_files_iter
from pytools.data_prep.weather_data_prep import get_datetime_from_grib_file_name


folder_table_name = 'hrrr_folder_info'
cur_path = os.path.dirname(os.path.realpath(__file__)), 
folder_info_file = os.path.join(cur_path, '../data/hrrr_obs_folder.txt')

col_folder = 'folder'
col_type = 'type'
col_batch = 'batch'
col_filename = 'filename'
col_timestamp = 'timestamp'
col_complete_timestamp = 'cplt_timestamp'


def load_files(batch_no=0):

    df = pd.read_csv(folder_info_file, dtype={'folder':str, 'batch':int})
    df = df[df[col_batch] ==0]
    ds = []

    def get_timestamp(fn:str):
        # automatically decide wheter it is hrrr-obs, hrrr-fst, or nc-obs
        grib_type = decide_grib_type(fn)
        if grib_type.startswith('hrrr'):
            return get_datetime_from_grib_file_name(hour_offset=0, filename=fn,nptime=True)
        else:
            
            return




    for f in df['folder']:
        df0 = pd.DataFrame(data={col_filename:get_all_files_iter(f)})
        df0[col_folder] = f
        df0[col_type] = df0[col_folder].apply(lambda fn: os.path.splitext(fn)[1])
        # generate timestamp from filename
        df0[col_timestamp] 
        df0

        # generate the complete timestamp and join

        ds.append(df0)

    fo = pd.concat(ds)
    fo.to_pickle(os.join(cur_path, f'../data/grib2_folder_{batch_no}.pkl'))

    


def main():
    return


if __name__ == '__main__':
    load_files(batch_no=0)
    main()