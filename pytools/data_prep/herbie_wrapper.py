
from collections import OrderedDict
from typing import List
from herbie import Herbie, FastHerbie, HerbieLatest  #, HerbieWait
import pandas as pd
import xarray as xr

from pytools.data_prep.grib_utils import get_herbie_str_from_cfgrib_file, get_paras_from_cfgrib_file



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
               

def process_xarray(keys:List[str], arr:xr.DataArray)->OrderedDict:
    for k in keys:
        for a in arr:
            if k in a.variables:
                
    return


def download_latest_data_file(paras_file:str, max_hrs)->List[xr.DataArray]:
    paras_str = get_herbie_str_from_cfgrib_file(paras_file=paras_file)
    _, keys = get_paras_from_cfgrib_file(paras_file=paras_file)
    if isinstance(max_hrs, int):
        max_hrs = range(1, max_hrs+1)
    ret = OrderedDict()
    for i in max_hrs:
        try: 
            H = HerbieLatest(model="hrrr", product='sfc', fxx=i)
            dt = H.xarray(search=paras_str,verbose=False,)

            for d in dt:
                list(dt[0].keys())

            ret[H.date + pd.to_timedelta(i, unit='h')] = dt
        except TimeoutError as ex:
            print(ex.strerror)

    return ret


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

