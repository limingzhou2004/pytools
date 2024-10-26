"""
This module summarize the hrrr data folders. Make the list of all file names, datetime, fst hour, type hrrr|nc, and statics, count, missing, count by type
"""


from datetime import timedelta
from functools import partial
import glob
from itertools import chain
import os
from os.path import exists

from math import ceil, floor, isnan
import shutil
import sys
import time
from typing import List, Tuple, Union, Dict

import pandas as pd
import numpy as np
import pendulum as pu
#import polars as pl
from tqdm import tqdm
from pytools.config import Config
from pytools.data_prep.get_datetime_from_grib_file_name import get_datetime_from_utah_file_name

from pytools.data_prep.grib_utils import decide_grib_type, get_all_files_iter, produce_full_timestamp
from pytools.data_prep.get_datetime_from_grib_file_name import get_datetime_from_grib_file_name
from pytools.data_prep.herbie_wrapper import download_obs_data_as_files, get_timestamp_from_herbie_folder_filename
from pytools.retry.api import retry


folder_table_name = 'hrrr_folder_info'
cur_path = os.path.dirname(os.path.realpath(__file__))
folder_info_file = os.path.join(cur_path, '../data/hrrr_obs_folder.txt')

col_folder = 'folder'
col_type = 'type'
col_batch = 'batch'
col_filename = 'filename'
folder_name = 'folder'
col_timestamp = 'timestamp'
col_complete_timestamp = 'cplt_timestamp'


def make_stats(fn=None, i=1):
    if not fn:
        fn = os.path.join(os.path.dirname(__file__),'../data/hrrr_stats_summary.csv')
    #load all pickle files for batch no from 0
    dfs = []
    #for i in range(100):
    pkl_path = os.path.join(os.path.dirname(__file__), f'../data/grib2_folder_{i}.pkl')
    if exists(pkl_path):
        dfs.append(pd.read_pickle(pkl_path))
    else:
        print(f"{fn} file not found!")
        return
    df = pd.concat(dfs)
    df['year'] = df[col_complete_timestamp].apply(lambda t: t.year)
    df['month'] = df[col_complete_timestamp].apply(lambda t: t.month)
    dfg = df[df[col_timestamp].isna() == False].groupby(by=['year','month'])[col_complete_timestamp].count()
    dfg2 = df[df[col_timestamp].isna()].groupby(['year', 'month'])[col_complete_timestamp].count()
    dfg = pd.merge(dfg, dfg2, on=['year', 'month'], how='left')
    dfg.columns = ['count', 'missing']
    dfg['missing'] = dfg['missing'].apply(lambda x: x if isnan(x) else int(x))

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

    def get_timestamp(fn:str, grib_type:str, folder_name:str=''):
        # automatically decide wheter it is hrrr-obs, hrrr-fst, or nc-obs
        # grib_type = decide_grib_type(fn)
        if grib_type.startswith('hrrr'):
            return get_datetime_from_grib_file_name(hour_offset=0, filename=fn,nptime=True)
        elif grib_type.startswith('herbie'):
            return get_timestamp_from_herbie_folder_filename(folder=folder_name, fn=fn)
        else:
            return get_datetime_from_utah_file_name(filename=fn)
    
    tqdm.pandas()
    for f in df['folder']:
        fs = list(get_all_files_iter(f, recursive=True, exclude_small_files=True))
        df0 = pd.DataFrame(data={col_filename:[v.name for v in fs ]})
        df0[col_folder] = pd.DataFrame(data={col_filename:[v.path for v in fs ]})
        df0[col_type] = df0[col_filename].apply(decide_grib_type)
        # generate timestamp from filename
        print(f'processing {f}...')
        df0[col_timestamp] = df0.progress_apply(lambda x: get_timestamp(x[col_filename], x[col_type], x[folder_name]), axis=1)
        ds.append(df0)

    fo = pd.concat(ds)
    # generate the complete timestamp and join
    full_timestamp = produce_full_timestamp(fo[col_timestamp].values)
    full_timestamp = pd.DataFrame(data=full_timestamp, columns=[col_complete_timestamp])
    fo = pd.merge(left=fo, right=full_timestamp, how='right', left_on=col_timestamp, right_on=col_complete_timestamp) 
    fo = fo.sort_values(by=col_complete_timestamp)

    fo.to_pickle(os.path.join(cur_path, f'../data/grib2_folder_{batch_no}.pkl'))


@retry(tries=5, delay=20)
def download_herbie_file_extract(cur_date:np.datetime64, tgt_folder:str):
    time.sleep(3)
    config_file='pytools/config/albany_test.toml'
    c = Config(config_file)
    paras_file = c.automate_path(c.weather_pdt.hrrr_paras_file)
    download_obs_data_as_files(t0=str(cur_date), t1=str(cur_date + np.timedelta64(1,'h')), paras_file=paras_file,save_dir=tgt_folder)


def fillmissing_from_pickle(batch_no, tgt_folder:str):
    """
    The batch no indicates the latest pickle file, to include previous batches

    Args:
        batch_no (int): start from 0. 
        tgt_folder (str): target folder
    """
    forecast_hour = 0
    fn = os.path.join(os.path.dirname(__file__), f'../data/grib2_folder_{batch_no}.pkl')
    df = pd.read_pickle(fn)
    df = df[df['timestamp'].isna()]
    print('start processing...\n')
    for t in tqdm(df['cplt_timestamp']):
        if t.to_datetime64()<np.datetime64('2020-01-01'):
            continue
        download_herbie_file_extract(cur_date=t, tgt_folder=tgt_folder)

    
def main():
    return


if __name__ == '__main__':
# usage, python -m  pytools.data_prep.grib_util_org 0
    if len(sys.argv) > 1:
        if sys.argv[1]=='summary':
            make_stats(i=int(sys.argv[2]))

        elif sys.argv[1] == '-fill':
            tgt_folder = sys.argv[2]
            batch_no = int(sys.argv[3])
            fillmissing_from_pickle(batch_no=batch_no, tgt_folder=tgt_folder)

        else:
            load_files(batch_no=int(sys.argv[1]))
    
