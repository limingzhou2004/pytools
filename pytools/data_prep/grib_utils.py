
from collections import OrderedDict
from functools import partial
import glob
from itertools import chain
import os
from math import ceil, floor
import shutil
import time
from typing import List, Tuple, Union, Dict


import cartopy.crs as ccrs
import pandas as pd
import pendulum as pu
import requests
import numpy as np
from tqdm import tqdm
import xarray as xr

from pytools.retry.api import retry
from pytools.data_prep.get_datetime_from_grib_file_name import get_datetime_from_grib_file_name


hrrr_url_str_template = 'http://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl?file=hrrr.t{HH}z.wrfsfcf{FHH}.grib2&lev_10_m_above_ground=on&lev_2_m_above_ground=on&lev_surface=on&var_APCP=on&var_ASNOW=on&var_DPT=on&var_DSWRF=on&var_GUST=on&var_HPBL=on&var_PRES=on&var_RH=on&var_SNOD=on&var_SNOWC=on&var_SPFH=on&var_TCDC=on&var_TMP=on&var_UGRD=on&var_VBDSF=on&var_VDDSF=on&var_VGRD=on&var_VIS=on&var_WIND=on&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fhrrr.YYYYMMDD%2Fconus'


# HRRR settings
HRRR_DX = 3
HRRR_DY = 3
HRRR_lat_name = 'latitude'
HRRR_lon_name = 'longitude'


def get_LCC_proj():
    return ccrs.LambertConformal(central_longitude=262.5, 
                                   central_latitude=38.5, 
                                   standard_parallels=(38.5, 38.5),
                                    globe=ccrs.Globe(semimajor_axis=6371229,
                                                     semiminor_axis=6371229))

@DeprecationWarning
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

    return ind

    #return np.unravel_index(ind, np.array(arr).shape)


def find_ind_fromlatlon(lon:float, lat:float, arr_lon:np.ndarray, arr_lat:np.ndarray) -> Tuple[int,int]:
    # array dim, lat, lon
    x_ind = _find_nearest_index(lon, arr_lon)
    y_ind = _find_nearest_index(lat, arr_lat)

    return x_ind, y_ind


def _extract_a_group(fn:str, group:str, paras: List[str], extract_latlon:bool=False):
    # group of 2 m, 10 m, and surface
    backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}}

    if group=='2 m':
        backend_kwargs['filter_by_keys']['level'] = 2
    if group=='surface':
        backend_kwargs['filter_by_keys'] = {'stepType': 'instant',                                    'typeOfLevel': 'surface'}
    dat = xr.open_dataset(fn, engine='cfgrib', backend_kwargs=backend_kwargs)
    dr = xr.Dataset()
    lat = HRRR_lat_name
    lon = HRRR_lon_name
    if extract_latlon:
        return dat[lon].data - 360 if dat[lon].data.max()> 180 else dat[lon].data, dat[lat].data
    for p in paras:
        dr[p] = dat[p]
    return dr


def _extract_xrray(arr_ds:List[xr.Dataset], paras:List[str], extract_latlon=False):
    dr = xr.Dataset()
    lat = HRRR_lat_name
    lon = HRRR_lon_name
    if extract_latlon:
        dat =arr_ds[0]
        return dat[lon].data - 360 if dat[lon].data.max()> 180 else dat[lon].data, dat[lat].data
    for p in paras:
        for d in arr_ds:
            if p in d:
                dr[p] = d[p]
    return dr


def _extract_fn_arr(fn_arr, group, paras, extract_latlon=False):
    if isinstance(fn_arr, str):
        return _extract_a_group(fn=fn_arr, group=group, paras=paras, extract_latlon=extract_latlon)
    elif isinstance(fn_arr, List):    
        return _extract_xrray(arr_ds=fn_arr, paras=paras, extract_latlon=extract_latlon)
    else:
        raise ValueError('the fn_arr must be a string of a file name, or a list of xr.Dataset')


def _get_evelope_ind(lon:float,lat:float, radius, arr_lon, arr_lat):
    proj_latlon = ccrs.PlateCarree()
    projection = get_LCC_proj()
    coords = projection.transform_points(src_crs=proj_latlon, x=arr_lon, y=arr_lat)
    arr_lat_to_lcc = coords[:,0,1]
    arr_lon_to_lcc = coords[0,:,0] 
    coord = projection.transform_point(x=lon, y=lat, src_crs=proj_latlon)
    lat_to_lcc = coord[1]
    lon_to_lcc = coord[0]
    delta_x = HRRR_DX
    delta_y = HRRR_DY

    if isinstance(radius, int):
        east_dist = radius
        west_dist = radius 
        north_dist = radius 
        south_dist = radius
    else:
        east_dist, west_dist, south_dist, north_dist = radius
    
    center_x_ind, center_y_ind = find_ind_fromlatlon(lat=lat_to_lcc, lon=lon_to_lcc, arr_lon=arr_lon_to_lcc, arr_lat=arr_lat_to_lcc)
    east_ind = ceil(center_x_ind+east_dist/delta_x)
    west_ind = floor(center_x_ind-west_dist/delta_x)
    south_ind = floor(center_y_ind-south_dist/delta_y)
    north_ind = ceil(center_y_ind+north_dist/delta_y) 
    return west_ind, east_ind, south_ind, north_ind


def extract_data_from_grib2(fn_arr:str, lon:float=None, lat:float=None, radius:Union[int,Tuple[int, int, int, int]]=None, 
    paras:OrderedDict=None, return_latlon:bool=False, envelope:List=None)->Tuple[np.ndarray,]:
    """
    Extract a subset, based on a rectangle area. We assume all paras share the same grid. 
    Both lat/lon are increasing in the grid. The hrrr data has a grid of 1799 by 1059
    The order of the paras is decided by the paras file. 
    Args:
        fn (str): file name of the grib2 file. Or a list of xr.Dataset
        lon (float): longitude, as x
        lat (float): latitutde, as y
        radius (Union[int,Tuple[int, int, int, int]]): distance in kms from the center
        paras (OrderedDict): the weather parameters by layers
        return_latlon (bool): wheter to return lat lon as the second item
        envelope (List): index [left-west, right-east, lower-south, upper-north] 

    Returns:
        np.ndarray: 3D tensor extracted np array, envelopes, west->east:south->north:parameter 
    """

    if paras is None:
        raise ValueError('paras cannot be none!')
    ds_data = {}
    # for each paras group
    for k in paras:
        ds_data[k] = _extract_fn_arr(fn_arr, k, paras[k])

    group='2 m'
    arr_lon, arr_lat = _extract_fn_arr(fn_arr, k, paras[group], extract_latlon=True)
    if envelope is None:
        if lon is None or lat is None or radius is None:
            raise ValueError('lon, lat, radius cannot be all Nones if envelope is None!')
        envelope = _get_evelope_ind(lon=lon, lat=lat, radius=radius, arr_lon=arr_lon, arr_lat=arr_lat) 

    west_ind, east_ind, south_ind, north_ind = envelope[0], envelope[1], envelope[2], envelope[3] 
    arr_list = []
    # for u, v wind, the 3 dim, dim 0 for 10 and 80 m
    ground_2m_dim = 0
  
    for k in paras:
        for p in paras[k]:
            ds = ds_data[k]
            x = ds[p].data[south_ind:(north_ind+1), west_ind:(east_ind+1)]
            if len(x.shape)>2:
                x = ds[p].data[ground_2m_dim, south_ind:(north_ind+1), west_ind:(east_ind+1)]
                
            arr_list.append(x)

    if return_latlon:
        return (np.stack(arr_list, axis=2), envelope,
                ds[HRRR_lon_name][south_ind:north_ind+1, west_ind:east_ind+1], 
                ds[HRRR_lat_name][south_ind:north_ind+1, west_ind:east_ind+1])
    else:
        return np.stack(arr_list, axis=2), envelope
    

def get_herbie_str_from_cfgrib_file(paras_file:str):
    qstr = ':'
    with open(paras_file) as f:
        f.readline() #skip the header row
        for line in f:

            kv = line.strip().split(',')
            if kv[0].strip()=='0':
                continue
            code= kv[6].strip()
            layer=kv[4].strip()
            if layer.endswith('m'):
                layer = layer.replace('m',' m').replace('  ', ' ')
            qstr = qstr + code + ':' + layer + '|'        
    # important to remove the last |; otherwise, all parameters will be included!
    return qstr[:-1]


def get_paras_from_cfgrib_file(paras_file:str)->Tuple[Dict, List[str]]:
    with open(paras_file) as f:
        f.readline() #skip the header row
        #p_dict = OrderedDict([('2m', list()), ('10m', list()), ('surface', list())])
        p_dict = OrderedDict()
        keys = list()
        for line in f:
            kv = line.strip().split(',')
            flag = kv[0].strip()
            k = kv[4].strip()
            if k not in p_dict:
                p_dict[k] = list()
            v = kv[1].strip()
            keys.append(v)
            if flag=='1':
                p_dict[k].append(v) 

    return p_dict, keys


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


def scantree(path):
    """Recursively yield DirEntry objects for given directory."""
    for entry in os.scandir(path):
        if entry.is_dir(follow_symlinks=False):
            yield from scantree(entry.path)  # see below for Python 2.x
        else:
            yield entry

def get_all_files_iter(folders: Union[str, Tuple[str]], recursive=False, exclude_small_files=False, size_kb_fileter=1024):
    file_iter = iter([])
    if isinstance(folders, str):
        if recursive:
            file_iter = chain(file_iter, scantree(folders))
        else:
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


# def fillmissing(sourcefolder:str, targetfolder:str, t0:str=None, t1:str=None, hourfst:int=0):
#     find_missing_grib2(folders=sourcefolder.split(','), tgt_folder=targetfolder, t0=t0, t1=t1)
#     print('start...')
    

def decide_grib_type(fn:str): 
    """

    #     herbie_obs = 'subset_75ef4997__hrrr.t00z.wrfsfcf00.grib2'
    #     hrrr_obs = 'hrrrsub_2020_01_01_00F0.grib2'
    #     hrrr_fst = 'hrrrsub_12_2020_01_01_18F1.grib2'
    #     utah_grib = '20200105.hrrr.t14z.wrfsfcf00.grib2'
   
    Args:
        fn (str): grib filename

    Returns:
        str: hrrr_obs|hrrr_fst|utah_grib|utah_nc
    """
    import re
    if fn.startswith('subset_'):
        return 'herbie_obs'
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
    


 
# if __name__ == "__main__":
#     # fillmissing(*sys.argv[1:])
#     # python -m pytools.data_prep.grib_utils 
#     if len(sys.argv)<3:
#         tgt_folder = "/Users/limingzhou/zhoul/work/energy/utah_2"
#     else:
#         tgt_folder = sys.argv[2]
    
#     if len(sys.argv) <2:
#         batch_no=0
#     else:
#         batch_no=int(sys.argv[1])

    # fillmissing_from_pickle(batch_no=batch_no, tgt_folder=tgt_folder) 