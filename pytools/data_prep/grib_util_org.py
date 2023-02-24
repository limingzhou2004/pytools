"""
This module summarize the hrrr data folders. Make the list of all file names, datetime, fst hour, type hrrr|nc, and statics, count, missing, count by type
"""


from functools import partial
import glob
from itertools import chain
import os
from os.path import exists

from math import ceil, floor
import shutil
import sys
import time
from typing import List, Tuple, Union, Dict

import pandas as pd
import pendulum as pu
import polars as pl
from tqdm import tqdm

from pytools.data_prep.grib_utils import decide_grib_type, get_all_files_iter, produce_full_timestamp
from pytools.data_prep.weather_data_prep import get_datetime_from_grib_file_name, get_datetime_from_utah_file_name


folder_table_name = 'hrrr_folder_info'
cur_path = os.path.dirname(os.path.realpath(__file__))
folder_info_file = os.path.join(cur_path, '../data/hrrr_obs_folder.txt')

col_folder = 'folder'
col_type = 'type'
col_batch = 'batch'
col_filename = 'filename'
col_timestamp = 'timestamp'
col_complete_timestamp = 'cplt_timestamp'


def make_stats(fn='hrrr_stats_summary.csv'):
    #load all pickle files for batch no from 0
    dfs = []
    for i in range(100):
        pkl_path = os.path.join(os.path.dirname(__file__), f'../data/grib2_folder_{i}.pkl')
        if exists(pkl_path):
            dfs.append(pd.read_pickle(pkl_path))
        else:
            break
    df = pd.concat(dfs)
    df['year'] = df[col_complete_timestamp].apply(lambda t: t.year)
    df['month'] = df[col_complete_timestamp].apply(lambda t: t.month)
    dfg = df[df[col_timestamp].isna()==False].groupby(by=['year','month'])[col_complete_timestamp].count()
    dfg2 = df[df[col_timestamp].isna()].groupby(['year', 'month'])[col_complete_timestamp].count()
    dfg = pd.merge(dfg, dfg2, on=['year', 'month'], how='left')
    dfg.columns = ['count', 'missing']
    dfg.to_csv(fn)
    


def load_files(batch_no=0):
    """
    create the pickle file for each hour for missing hour labeling.

    Args:
        batch_no (int, optional): The batch no to match the hrrr_obs_folder.txt file. Defaults to 0.

    Returns:
        None
    """

    df = pd.read_csv(folder_info_file, dtype={'folder':str, 'batch_no':int})
    df = df[df[col_batch] ==batch_no]
    ds = []

    def get_timestamp(fn:str, grib_type:str):
        # automatically decide wheter it is hrrr-obs, hrrr-fst, or nc-obs
        # grib_type = decide_grib_type(fn)
        if grib_type.startswith('hrrr'):
            return get_datetime_from_grib_file_name(hour_offset=0, filename=fn,nptime=True)
        else:
            return get_datetime_from_utah_file_name(filename=fn)
    
    tqdm.pandas()
    for f in df['folder']:
        df0 = pd.DataFrame(data={col_filename:[v.name for v in get_all_files_iter(f, exclude_small_files=True)]})
        df0[col_folder] = f
        df0[col_type] = df0[col_filename].apply(decide_grib_type)
        # generate timestamp from filename
        print(f'processing {f}...')
        df0[col_timestamp] = df0.progress_apply(lambda x: get_timestamp(x[col_filename], x[col_type]), axis=1)
        ds.append(df0)

    fo = pd.concat(ds)
    # generate the complete timestamp and join
    full_timestamp = produce_full_timestamp(fo[col_timestamp].values)
    full_timestamp = pd.DataFrame(data=full_timestamp, columns=[col_complete_timestamp])
    fo = pd.merge(left=fo, right=full_timestamp, how='right', left_on=col_timestamp, right_on=col_complete_timestamp) 
    fo = fo.sort_values(by=col_complete_timestamp)

    fo.to_pickle(os.path.join(cur_path, f'../data/grib2_folder_{batch_no}.pkl'))

    
def main():
    return


if __name__ == '__main__':
# usage, python -m  pytools.data_prep.grib_util_org 0
    if len(sys.argv) > 1:
        load_files(batch_no=int(sys.argv[1]))
    else:
        make_stats()
