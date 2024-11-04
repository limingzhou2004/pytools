import pickle
import sys
import os.path as osp
from pathlib import Path
from collections import OrderedDict

from typing import List
from herbie import Herbie, FastHerbie, HerbieLatest  #, HerbieWait
import numpy as np
import pandas as pd
from tqdm import tqdm

from pytools import get_logger
from pytools.config import Config
from pytools.data_prep.grib_utils import extract_data_from_grib2, get_herbie_str_from_cfgrib_file, get_paras_from_cfgrib_file


logger = get_logger('herbie_log')

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
               

def download_latest_data(paras_file:str, max_hrs, envelopes:List)->List[OrderedDict]:
    """
    returns the tuple (timestamp array, list of weather arrays for each envlope)

    Args:
        paras_file (str): _description_
        max_hrs (_type_): _description_
        envelopes (List): _description_
        save_dir (string): directory to save the data

    Returns:
        List[OrderedDict]: timestamp array, the envelope list of weather array
    """
    paras_str = get_herbie_str_from_cfgrib_file(paras_file=paras_file)
    group_paras, _ = get_paras_from_cfgrib_file(paras_file=paras_file)

    if isinstance(max_hrs, int):
        max_hrs = range(1, max_hrs+1)
    arr_list = []
    for t in max_hrs:
        try:
            H = HerbieLatest(model="hrrr", product='sfc', fxx=t)
            #arr is a list of three, 10 m, 2 m and surface
            arr = H.xarray(search=paras_str,verbose=False,)
            arr_list.append(arr)
        except TimeoutError as ex:
            print(ex.strerror)
            break

    ret_timestamp = None
    ret_array_list = []
    for ev in envelopes:
        time_stamps = []
        dat_list = []
        for hrs, arr in zip(max_hrs[:len(arr_list)],arr_list):

            dat, _ = extract_data_from_grib2(fn_arr=arr,paras=group_paras,envelope=ev, return_latlon=False)
            if ret_timestamp is None:
                t = H.date + pd.to_timedelta(hrs, unit='h')
                time_stamps.append(t)
            dat_list.append(dat)               

        if ret_timestamp is None:
            ret_timestamp = time_stamps
        ret_array_list.append(np.stack(dat_list,axis=0))

    return ret_timestamp, ret_array_list


def download_hist_fst_data(t_start, t_end, fst_hr:int,  paras_file:str, envelopes:List, freq='D', save_dir='~/tmp_data'):
    """
    Get historical forecast, or get the most recent obs weather. It will be a large size, 
    so we only extract a subset based on the envelopes, and save them locally.

    Args:
        dt (str): current timestamp
        fst_hr (_type_): time horizon, hours
        envelopes: 
        freq str: frequency, d for day
        save_dir: directory to save the weather data.

    Returns: 
        spot timestamp list, list of envelopes with elements of [fst timestamp list, weather array list]
    """
    paras_str = get_herbie_str_from_cfgrib_file(paras_file=paras_file)
    group_paras, _ = get_paras_from_cfgrib_file(paras_file=paras_file)
    spot_time_list = []
    envelope_arr_list = [[[], []] for _ in range(len(envelopes))]    

    if not isinstance(envelopes[0], List):
        envelopes = [envelopes]

    for cur_t in tqdm(pd.date_range(start=t_start, end=t_end, freq=freq)):
        logger.info(cur_t)
        for h in range(fst_hr):
            logger.info(f'forecast hour...{h}')
            try:    
                H = Herbie(cur_t, model="hrrr", fxx=h, save_dir=save_dir)
                arr = H.xarray(search=paras_str, remove_grib=True)
                for i in range(len(envelopes)):
                    ev = envelopes[i]
                    dat, _ = extract_data_from_grib2(fn_arr=arr,paras=group_paras,envelope=ev, return_latlon=False)
                    envelope_arr_list[i][0].append(cur_t + pd.Timedelta(h, unit='h'))
                    envelope_arr_list[i][1].append(dat)
            except Exception as ex:
                logger.error(str(ex))
        spot_time_list.append(cur_t)          

    return spot_time_list, envelope_arr_list


def get_timestamp_from_herbie_folder_filename(folder, fn):
    #  ~/tmp/hist/hrrr/20200101/subset_75ef4997__hrrr.t00z.wrfsfcf00.grib2
    date_str = folder.split('/')[-2][-8:]
    time_str = fn.split('.')[1][1:3]

    return pd.Timestamp(date_str[0:4]+'-'+date_str[4:6]+'-'+date_str[6:8]+' '+time_str).to_numpy()
   
def extract_data_from_file(fn):

    return


def main(args):

    config_file='pytools/config/albany_test.toml'

    fn = None
    save_dir = None

    for i, p in enumerate(args):
        if p == '-config_file':
            config_file = args[i+1]
        if p == '-save-dir':
            save_dir = args[i+1]
        if p == '-t0':
            t0 = args[i+1]
        if p == '-t1':
            t1 = args[i+1]
        if p == '-fst_hr':
            fst_hr = int(args[i+1])    
        if p == '-fn':
            fn = args[i+1]
        
    if save_dir is None:
        raise ValueError('-save-dir cannot be empty!')

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    c = Config(config_file)
    envs = c.weather_pdt.envelope
    paras_file = c.automate_path(c.weather_pdt.hrrr_paras_file)

    if args[1] == '-obs':
        download_obs_data_as_files(t0=t0, t1=t1, paras_file=paras_file,save_dir=save_dir)
    elif args[1] == '-fst':
        if fn is None:
            raise ValueError('-fn cannot be empty!')
        res = download_hist_fst_data(t_start=t0, t_end=t1,fst_hr=fst_hr, paras_file=paras_file,envelopes=envs )
        path_full = osp.join(save_dir, fn)
        with open(path_full, 'wb') as h:
            pickle.dump(res, h, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise ValueError('has to be -obs|-fst')


if __name__ == '__main__':

    main(sys.argv.split(','))