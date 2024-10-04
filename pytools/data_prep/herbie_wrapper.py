
from collections import OrderedDict
from typing import List
from herbie import Herbie, FastHerbie, HerbieLatest  #, HerbieWait
import numpy as np
import pandas as pd
import xarray as xr

from pytools.data_prep.grib_utils import extract_data_from_grib2, get_herbie_str_from_cfgrib_file, get_paras_from_cfgrib_file



def download_obs_data_as_files(t0:str, t1:str, paras_file:str, save_dir:str, threads:int=8):
    fst_hr = 0
    paras_str = get_herbie_str_from_cfgrib_file(paras_file=paras_file)
    t_range = pd.date_range(start=t0, end=t1, freq='h')

    if threads>1:
        FH = FastHerbie(t_range, model="hrrr", fxx=range(fst_hr, fst_hr+1), save_dir=save_dir, max_threads=threads)
        FH.download(paras_str)
    else:
        for t in t_range:
            h = Herbie(
                str(t),
                model='hrrr',
                product='sfc',
                fxx=fst_hr, save_dir=save_dir)
            h.download(paras_str, verbose=False, overwrite=True)
               

def download_latest_data(paras_file:str, max_hrs, envelopes:List )->List[OrderedDict]:
    """
    returns the tuple (timestamp array, list of weather arrays for each envlope)

    Args:
        paras_file (str): _description_
        max_hrs (_type_): _description_
        envelopes (List): _description_

    Returns:
        List[OrderedDict]: timestamp array, weather array
    """
    paras_str = get_herbie_str_from_cfgrib_file(paras_file=paras_file)
    group_paras, _ = get_paras_from_cfgrib_file(paras_file=paras_file)

    if isinstance(max_hrs, int):
        max_hrs = range(1, max_hrs+1)
    arr_list = []
    for t in max_hrs:
        H = HerbieLatest(model="hrrr", product='sfc', fxx=t)
        arr = H.xarray(search=paras_str,verbose=False,)
        arr_list.append({H.date + pd.to_timedelta(t, unit='h'): arr})

    ret_timestamp = None
    ret_array = []
    for ev in envelopes:
        time_stamps = []
        dat_list = []
        for hrs, arr in zip(max_hrs,arr_list):
            try: 
                dat = extract_data_from_grib2(fn_arr=arr,paras=group_paras,envelope=ev, return_latlon=False)
                if ret_timestamp is None:
                    t = H.date + pd.to_timedelta(hrs, unit='h')
                    time_stamps.append(t)
                dat_list.append(dat)               
            except TimeoutError as ex:
                print(ex.strerror)

        if ret_timestamp is None:
            ret_timestamp = np.ndarray(time_stamps)
        ret_array.append(np.stack(dat_list,axis=0))

    return ret_timestamp, ret_array


def download_hist_fst_data(dt:str, fst_hr):
    """
    Get historical forecast, or get the most recent obs weather

    Args:
        dt (str): _description_
        fst_hr (_type_): _description_
    """

    return


def extract_data_from_file(fn):

    return

