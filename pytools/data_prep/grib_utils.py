
from functools import partial
import glob
from itertools import chain
import os
from math import ceil, floor
import shutil
import sys
import time
from typing import List, Tuple, Union, Dict

from pytools.retry.api import retry
import pandas as pd
import pendulum as pu
import geopandas
import requests
import numpy as np
from tqdm import tqdm
from shapely.geometry import Point
import xarray as xr

from pytools.data_prep.weather_data_prep import get_datetime_from_grib_file_name


hrrr_url_str_template = 'http://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl?file=hrrr.t{HH}z.wrfsfcf{FHH}.grib2&lev_10_m_above_ground=on&lev_2_m_above_ground=on&lev_surface=on&var_APCP=on&var_ASNOW=on&var_DPT=on&var_DSWRF=on&var_GUST=on&var_HPBL=on&var_PRES=on&var_RH=on&var_SNOD=on&var_SNOWC=on&var_SPFH=on&var_TCDC=on&var_TMP=on&var_UGRD=on&var_VBDSF=on&var_VDDSF=on&var_VGRD=on&var_VIS=on&var_WIND=on&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fhrrr.YYYYMMDD%2Fconus'



def print_grib2_info(fn:str):
    ds = xr.open_dataset(fn, engine="pynio")
    ds['gridlat_0'].data 
    ds['gridlat_0'].Dx #Dy
    ds['gridlon_0']
    print(ds)
    channels = ds.variables.mapping.keys()
    print(channels)
    for c in channels:
        print(c+'\n')
        print(ds[c].attrs)
        print('------\n')


def _find_nearest_index(val:float, arr:np.ndarray) -> Tuple[int,int]:
    arr = np.abs(arr-val)
    ind = np.argmin(arr)

    return np.unravel_index(ind, np.array(arr).shape)


def find_ind_fromlatlon(lon:float, lat:float, arr_lon:np.ndarray, arr_lat:np.ndarray) -> Tuple[int,int]:
    x_ind = _find_nearest_index(lon, arr_lon)[1]
    y_ind = _find_nearest_index(lat, arr_lat)[0]

    return x_ind, y_ind


def extract_data_from_grib2(fn:str, lon:float, lat:float, radius:Union[int,Tuple[int, int, int, int]], paras:List[str])->np.ndarray:
    """
    Extract a subset, based on a rectangle area. We assume all paras share the same grid. Both lat/lon are increasing in the grid. The hrrr data has a grid of 1799 by 1059

    Args:
        fn (str): file name of the grib2 file.
        lon (float): longitude, as x
        lat (float): latitutde, as y
        radius (Union[int,Tuple[int, int, int, int]]): distance in kms from the center
        paras (List[str]): the weather parameters

    Returns:
        np.ndarray: 3D tensor extracted np array, west->east:south->north:parameter
    """
    if fn.endswith('.grib2'):
        ds = xr.load_dataset(fn, engine="pynio")
    elif fn.endswith('.nc'):
        ds = xr.load_dataset(fn, engine="scipy")
    else:
        raise Exception('only pynio and scipy are supported for engine')

    delta_x = ds['gridlat_0'].Dx
    delta_y = ds['gridlat_0'].Dy
    if isinstance(radius, int):
        east_dist = radius; west_dist = radius; north_dist = radius; south_dist=radius
    else:
        east_dist, west_dist, south_dist, north_dist = radius
 
    arr_lat = ds['gridlat_0'].data 
    arr_lon = ds['gridlon_0'].data 
    center_x_ind, center_y_ind = find_ind_fromlatlon(lat=lat, lon=lon, arr_lon=arr_lon, arr_lat=arr_lat)
    east_ind = ceil(center_x_ind+east_dist/delta_x)
    west_ind = floor(center_x_ind-west_dist/delta_x)
    south_ind = floor(center_y_ind-south_dist/delta_y)
    north_ind = ceil(center_y_ind+north_dist/delta_y) 
    arr_list = []
    for p in paras:
        x = ds[p].data[west_ind:east_ind+1, south_ind:north_ind+1]
        arr_list.append(x)
    return np.stack(arr_list, axis=2)


def extract_a_file(fn:str, para_file:str, lon:float, lat:float, radius:Union[int, Tuple[int, int, int, int]]) -> np.ndarray:
    """
    Extract a grib2 file

    Args:
        fn (str): full file name
        para_file (str): pamaremeter file
        lon (float): center longitute
        lat (float): center latitude
        radius (Union[int, Tuple[int, int, int, int]]): distance from the center

    Returns:
        np.ndarray: dimension of x, y, channel
    """
    a = {}
    with open(para_file) as f:
        for line in f:
            kv = line.strip().split(',')
            k = kv[0]; v = kv[1]
            if int(v) == 1:
                a[k] = int(v)
    data = extract_data_from_grib2(fn=fn, lon=lon, lat=lat, radius=radius, paras=a)
    return data


def read_utah_file_and_save_a_subset(fn:str, para_file:str, tgt_folder:str, rename_var:Dict={'APCP_P8_L1_GLC0_acc1h':'APCP_P8_L1_GLC0_acc'}):
    a = []
    with open(para_file) as f:
        for line in f:
            kv = line.strip().split(',')
            k = kv[0]; v = kv[1]
            # use 1h precipitation for Utah data
            if k == 'APCP_P8_L1_GLC0_acc':
                k = k + '_1h'
            if int(v) == 1:
                a.append(int(v))
    ds = xr.load_dataset(fn, engine='pynio')
    ds2 = ds[a]
    ds2 = ds2.rename_vars(rename_var)
    ds2.to_netcdf(path=os.path.join(tgt_folder, fn.replace('grib2', 'nc')), engine='scipy')


def get_all_files(folders: Union[str, Tuple[str]], size_kb_fileter:int=2000) -> List[str]:
    pfn = 'hrrrfiles.csv'
    if os.path.exists(pfn):
        x = pd.read_csv(pfn)

        return [f for f in x['name'] ]
    if isinstance(folders, str):
        filenames = glob.glob(folders+ "/*.grib2")
    else:
        filenames = []
        for f in folders:
            filenames.extend(glob.glob(f+ "/*.grib2"))
    
    df = pd.DataFrame(filenames, columns = ['name'])
    df.to_csv(pfn)

    return filenames
    #return [f for f in filenames if os.path.getsize(f)/1024 >= size_kb_fileter]


def get_all_files_iter(folders: Union[str, Tuple[str]], size_kb_fileter:int=2000) -> List[str]:
    file_iter = iter([])
    if isinstance(folders, str):
        file_iter = chain(file_iter, os.scandir(folders))
    else:        
        for f in folders:
            file_iter = chain(file_iter, os.scandir(f))

    for f in file_iter:
        yield f
        #if os.path.getsize(f)/1024 >= size_kb_fileter:
          #  yield f
           


def find_missing_grib2(folders:Union[str, List[str]], tgt_folder:str='.', t0:str=None, t1:str=None)->List[str]:
    """
    Find missing hours of hrrr grib2 files

    Args:
        folders (Union[str, List[str]]): source data folders
        tgt_folder (str, optional): target folder to write to. The default is .
        t0 (str, optional): starting time, 'yyyy-mm-dd hh'. Defaults to None, to use the min time of all files.
        t1 (str, optional): ending time, 'yyyy-mm-dd hh'. Defaults to None, to use the max time of all files.

    Returns:
        List[str]: a list of file names to download from Utah U's website.
    """
    # get the list all obs grib files
    full_filenames = get_all_files(folders)
    filenames = [os.path.basename(f.strip()) for f in list(full_filenames)]
    cur_dates = list(map(partial(get_datetime_from_grib_file_name,hour_offset=0,nptime=True, get_fst_hour=False), filenames))
   
    # find the min and max
    cur_dates = np.array(cur_dates)

    def process_time(t, minmax='min'):  
        if t:
            return pd.DatetimeIndex([t])[0]
        else:
            if minmax == 'min':
                return cur_dates.min()
            else:
                return cur_dates.max()

    t0 = process_time(t0)
    t1 = process_time(t1)

    # find the missing hours
    full_timestamp = np.arange(t0, t1, np.timedelta64(1, "h"))
    set_diff = np.setdiff1d(full_timestamp, cur_dates)
    # processing each missing hour
    counter = 0
    for t in tqdm(set_diff):
        counter += 1
        if counter%100 == 0:
            print('wait 60 sec...')
            time.sleep(60)
        print(f'\nprocessing {t}...')
        download_utah_file_extract(cur_date=t, fst_hour=0, tgt_folder=tgt_folder)


@retry(tries=5, delay=20)
def download_utah_file_extract(cur_date:np.datetime64, fst_hour:int, tgt_folder:str):
    # convert to utah file name convention
    time.sleep(3)
    cur_d = pd.DatetimeIndex([cur_date])
    yyyy = str(cur_d.year[0]).zfill(4)
    mm = str(cur_d.month[0]).zfill(2)
    dd = str(cur_d.day[0]).zfill(2)
    hh = str(cur_d.hour[0]).zfill(2)
    
    fhh = str(fst_hour).zfill(2)
    base_url = f'https://pando-rgw01.chpc.utah.edu/hrrr/sfc/{yyyy}{mm}{dd}/hrrr.t{hh}z.wrfsfcf{fhh}.grib2'

    r = requests.get(base_url, stream=True)
    print(base_url)
    fn = f'{yyyy}{mm}{dd}.hrrr.t{hh}z.wrfsfcf{fhh}.grib2'
    if r.status_code == 200:
        print('succssful...')
        with open(os.path.join(tgt_folder, fn), 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f) 
    else:
        print('failure to retriee...')


def get_stats(folders:Union[str, List[str]], utah_folders:Union[str, List[str]], t0:str, t1:str)->pd.DataFrame:
    """
    Stats of the grib files, number, start, end, missing. 
    Summary files by year: summary_year.csv, source, missing, starting datetime, ending datetime
    Spec file by hour: spec_hour.csv, date-hour, source/hrrr|utah, missing

    Args:
        folders (Union[str, List[str]]): dirctories
        utah_folders (Union[str, List[str]]): utah downloaded files, with a different naming convention to extract datetime info
        t0 (str): starting date yyyy-mm-dd hh:mm
        t1 (str): ending date yyyy-mm-dd hh:mm

    Returns:
        pd.DataFrame: column of number, start time, end time, missing number
    """
    # TODO

    return


def fillmissing(sourcefolder:str, targetfolder:str, t0:str=None, t1:str=None, hourfst:int=0):
    find_missing_grib2(folders=sourcefolder.split(','), tgt_folder=targetfolder, t0=t0, t1=t1)
    print('start...')
    

def extract_datetime_from_utah_files(fn:str) -> np.datetime64:
    # TODO
    # convert string to datetime with regex
    
    return

#@retry(tries=1, delay=20, backoff=3)
def download_hrrr(cur_date:pu.datetime, fst_hour:int, tgt_folder:str):
    """
    Download hrrr data

    Args:
        cur_date (pu.datetime): The hour the forecast is made from
        fst_hour (int): 0 for analysis, forecast so fare is 1-48
        tgt_folder (str): target folder to save the grib2 file

    Raises:
        RuntimeError: If the file is not generated, or network error
    """
    # from the now time {execution_date}, derive the latest obs time
    cur_d = cur_date # pd.DatetimeIndex([cur_date])
    yyyy = str(cur_d.year).zfill(4)
    mm = str(cur_d.month).zfill(2)
    dd = str(cur_d.day).zfill(2)
    hh = str(cur_d.hour).zfill(2)
    
    fhh = str(fst_hour).zfill(2)
    cur_url = hrrr_url_str_template.replace('YYYYMMDD', yyyy+mm+dd).replace('{HH}', hh). replace('{FHH}', fhh)
    r = requests.get(cur_url, stream=True)
    fn = f'hrrrsub_{hh}_{yyyy}_{mm}_{dd}_{hh}F{str(fst_hour)}.grib2' if fst_hour > 0 else f'hrrrsub_{yyyy}_{mm}_{dd}_{hh}F{str(fst_hour)}.grib2' 
    if r.status_code == 200:
        with open(os.path.join(tgt_folder, fn), 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f) 
    else:
        print(f'{cur_url}--failure to retrieve the data...')
        raise RuntimeError(f'{cur_url} data not found!')


def download_hrrr_by_hour(exe_date:pu.datetime, fst_hour:int, tgt_folder):

    # from the now time, derive the fst time; if the 1st success, the remaining 48 should be available.
    if exe_date.minute < 10:
        exe_date = exe_date.add(hours=-2)
    else:
        exe_date = exe_date.add(hours=-1)
    # publishing time is 1 hour and 10 min after the cur_date, based on the observation
    print('start downloading...')
    download_hrrr(cur_date=exe_date,fst_hour=fst_hour, tgt_folder=tgt_folder)
    


    """
import re 
from datetime import datetime 
  
# Input string 
string = '2020-07-17T14:30:00'
  
# Using regex expression to match the pattern 
regex = r'(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})'
  
# Finding all the matches 
matches = re.findall(regex, string) 
  
# Converting the string into datetime object 
date_time_obj = datetime(*[int(i) for i in matches[0]]) 
  
# Printing the datetime object 
print(date_time_obj)    """

if __name__ == "__main__":
    fillmissing(*sys.argv[1:])