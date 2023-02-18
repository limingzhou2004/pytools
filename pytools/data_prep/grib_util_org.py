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

from pytools.data_prep.grib_utils import get_all_files


folder_table_name = 'hrrr_folder_info'
cur_path = os.path.dirname(os.path.realpath(__file__)), 
folder_info_file = os.path.join(cur_path, '../data/hrrr_obs_folder.txt')


def load_files(batch_no=0):
    df = pd.read_csv(folder_info_file, dtype={'folder':str, 'batch':int})
    df = df[df['batch'] ==0]
    ds = []
    for f in df['folder']:
        df0 = pd.DataFrame(data={'filename':get_all_files(f)})
        df0['folder'] = f
        ds.append(df0)
        
    os.path.splitext(

    fo = pd.concat(ds)
    fo.to_pickle(f'{os.join(cur_path}../data/grib2_folder_{batch}.pkl')

    


def main():
    return


if __name__ == '__main__':
    load_files(0)
    main()