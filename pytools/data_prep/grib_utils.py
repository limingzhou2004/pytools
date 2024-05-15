
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

from pytools.data_prep.get_datetime_from_grib_file_name import get_datetime_from_grib_file_name


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
    x_ind = _find_nearest_index(lon, arr_lon)[0]
    y_ind = _find_nearest_index(lat, arr_lat)[1]

    return x_ind, y_ind

def extract_data_from_grib2(fn:str, lon:float, lat:float, radius:Union[int,Tuple[int, int, int, int]], paras:List[str], 
return_latlon:bool=False, 
is_utah=False)->np.ndarray:
    """
    Extract a subset, based on a rectangle area. We assume all paras share the same grid. 
    Both lat/lon are increasing in the grid. The hrrr data has a grid of 1799 by 1059
    The order of the paras is decided by the paras file. 
    Args:
        fn (str): file name of the grib2 file.
        lon (float): longitude, as x
        lat (float): latitutde, as y
        radius (Union[int,Tuple[int, int, int, int]]): distance in kms from the center
        paras (List[str]): the weather parameters
        return_latlon (bool): wheter to return lat lon as the second item

    Returns:
        np.ndarray: 3D tensor extracted np array, west->east:south->north:parameter
    """
    if fn.endswith('.grib2'):
        print(f'process...{fn}')
        ds = xr.load_dataset(fn, engine="pynio")
    elif fn.endswith('.nc'):
        ds = xr.load_dataset(fn, engine="scipy")
    else:
        raise Exception('only pynio and scipy are supported for engine')

    delta_x = ds['gridlon_0'].Dx
    delta_y = ds['gridlat_0'].Dy
    if isinstance(radius, int):
        east_dist = radius
        west_dist = radius 
        north_dist = radius 
        south_dist = radius
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
    # for u, v wind, the 3 dim, dim 0 for 10 and 80 m
    ground_2m_dim=0
  
    for p in paras:
        x = ds[p].data[west_ind:(east_ind+1), south_ind:(north_ind+1)]
        if len(x.shape)>2:
            x = ds[p].data[ground_2m_dim, west_ind:(east_ind+1), south_ind:(north_ind+1)]
            
        arr_list.append(x)

    if return_latlon:
        return (np.stack(arr_list, axis=2), 
                ds['gridlon_0'][west_ind:east_ind+1, south_ind:north_ind+1], 
                ds['gridlat_0'][west_ind:east_ind+1, south_ind:north_ind+1])
    else:
        return np.stack(arr_list, axis=2)

def get_paras_from_pynio_file(para_file:str, is_utah=False) -> Dict:
    a = {}
    # It is possible to use file size to decide whether it is a utah file, >100 mb

    with open(para_file) as f:
        for line in f:
            kv = line.strip().split(',')
            k = kv[0].strip(); v = kv[1].strip()
            # use 1h precipitation for Utah data
            if is_utah:
                if k == 'APCP_P8_L1_GLC0_acc':
                    k = k + '_1h'
            # 1 to use;        
            if int(v) == 1:
                a[k] = int(v)

    return a


def extract_a_file(fn:str, para_file:str, lon:float, lat:float, radius:Union[int, Tuple[int, int, int, int]], min_utah_size_mb=100) -> np.ndarray:
    """
    Extract a grib2 file

    Args:
        fn (str): full file name
        para_file (str): pamaremeter file
        lon (float): center longitute
        lat (float): center latitude
        radius (Union[int, Tuple[int, int, int, int]]): distance from the center
        min_utah_file_size: a utah file is larger than this size in mb

    Returns:
        np.ndarray: dimension of x, y, channel
    """

    # get the size of the file
    file_size_mb=os.path.getsize(fn)/1e6
    is_utah=True if file_size_mb>min_utah_size_mb else False
    para_list = get_paras_from_pynio_file(fn, is_utah=is_utah)
    data = extract_data_from_grib2(fn=fn, lon=lon, lat=lat, radius=radius, paras=para_list, is_utah=is_utah)
    return data


def read_utah_file_and_save_a_subset(fn:str, para_file:str, tgt_folder:str,
 rename_var:Dict={'APCP_P8_L1_GLC0_acc_1h':'APCP_P8_L1_GLC0_acc'}
 ):
    # a = []
    # with open(para_file) as f:
    #     for line in f:
    #         kv = line.strip().split(',')
    #         k = kv[0]; v = kv[1]
    #         # use 1h precipitation for Utah data
    #         if k == 'APCP_P8_L1_GLC0_acc':
    #             k = k + '_1h'
    #         if int(v) == 1:
    #             a.append(int(v))
    a = get_paras_from_pynio_file(is_utah=True)
    ds = xr.load_dataset(fn, engine='pynio')
    ds2 = ds[a]
    ds2 = ds2.rename_vars(rename_var)
    ds2.to_netcdf(path=os.path.join(tgt_folder, fn.replace('grib2', 'nc')), engine='scipy')


def get_all_files(folders: Union[str, Tuple[str]], exclude_small_files=False, size_kb_fileter=1024) -> List[str]:
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
    
    # It's better to use Linux command to remove files less than a certain size.
    if exclude_small_files: 
        filenames = [f for f in filenames if os.path.getsize(f)/1024 >= size_kb_fileter]
    df = pd.DataFrame(filenames, columns=['name'])
    df.to_csv(pfn)

    return filenames


def get_all_files_iter(folders: Union[str, Tuple[str]], exclude_small_files=False, size_kb_fileter=1024) -> List[str]:
    file_iter = iter([])
    if isinstance(folders, str):
        file_iter = chain(file_iter, os.scandir(folders))
    else:        
        for f in folders:
            file_iter = chain(file_iter, os.scandir(f))

    for f in file_iter:
        if exclude_small_files:
            if os.path.getsize(f)/1024 >=size_kb_fileter:
                yield f
        else:
            yield f
          

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
    t1 = process_time(t1, 'max')

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
        download_utah_file_extract(cur_date=t, fst_hour=forecast_hour, tgt_folder=tgt_folder)


def produce_full_timestamp(cur_dates):
    t0 = cur_dates.min()
    t1 = cur_dates.max()
    # find the missing hours
    return np.arange(t0, t1+np.timedelta64(1, "h"), np.timedelta64(1, "h"),)

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


def fillmissing(sourcefolder:str, targetfolder:str, t0:str=None, t1:str=None, hourfst:int=0):
    find_missing_grib2(folders=sourcefolder.split(','), tgt_folder=targetfolder, t0=t0, t1=t1)
    print('start...')
    

def decide_grib_type(fn:str): 
    """
    #     hrrr_obs = 'hrrrsub_2020_01_01_00F0.grib2'
    #     hrrr_fst = 'hrrrsub_12_2020_01_01_18F1.grib2'
    #     utah_grib = '20200105.hrrr.t14z.wrfsfcf00.grib2'
   
    Args:
        fn (str): grib filename

    Returns:
        str: hrrr_obs|hrrr_fst|utah_grib|utah_nc
    """
    import re
    p = re.compile(r'hrrrsub_\d\d\d\d_\d\d_\d\d_\d\dF\w*')
    if p.match(fn): 
        return 'hrrr_obs'
    p = re.compile(r'hrrrsub_\d\d_\d\d\d\d_\d\d_\d\d_\d\d\w*')
    if p.match(fn): 
        return 'hrrr_fst'
    p = re.compile(r'\d\d\d\d\d\d\d\d.hrrr.\w*.\w*')
    if p.match(fn): 
        return 'utah_grib'
    raise ValueError(f'{fn} unrecognized!')
    

# def extract_datetime_from_utah_files(fn:str) -> np.datetime64:
#     # TODO
#     # convert string to datetime with regex
    
#     return


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

@retry(tries=5, delay=20)
def download_hrrr_by_hour(exe_date:pu.datetime, fst_hour:int, tgt_folder):

    # from the now time, derive the fst time; if the 1st success, the remaining 48 should be available.
    # if exe_date.minute < 10:
    #     exe_date = exe_date.add(hours=-1)
    # else:
    #     exe_date = exe_date.add(hours=-0)
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
    # fillmissing(*sys.argv[1:])
    # python -m pytools.data_prep.grib_utils 
    if len(sys.argv)<3:
        tgt_folder = "/Users/limingzhou/zhoul/work/energy/utah_2"
    else:
        tgt_folder = sys.argv[2]
    
    if len(sys.argv) <2:
        batch_no=0
    else:
        batch_no=int(sys.argv[1])

    fillmissing_from_pickle(batch_no=batch_no, tgt_folder=tgt_folder) 