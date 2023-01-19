import os 

import numpy as np
import xarray as xr
import rioxarray

from pytools.data_prep.grib_utils import download_utah_file_extract, find_missing_grib2, print_grib2_info, find_ind_fromlatlon, extract_a_file

def test_read_grib2():
    hrrr_obs_path = '/Users/limingzhou/zhoul/work/energy/grib2/hrrrdata'
    fn = hrrr_obs_path + '/hrrrsub_2020_01_01_00F0.grib2'
    #ds = xr.open_dataset(fn, engine="pynio")
    #arr=ds['gridlat_0'].data 
    #ds['gridlat_0'].Dx #Dy
    #dd=ds['APCP_P8_L1_GLC0_acc'].data
    #ds['gridlon_0']
    #channels = ds.variables.mapping.keys()
    fn = '/Users/limingzhou/zhoul/work/energy/grib2/utah/hrrr.t00z.wrfsfcf03.grib2'
    #print_grib2_info(fn)
    para_file = '/Users/limingzhou/zhoul/work/energy/pytools/pytools/data_prep/hrrr_paras_pynio.txt'
    #extract_a_file(fn=fn, para_file=para_file,lat=40, lon=-137, radius=10)
    assert 1==1
    

def test_read_write_grib2():
    fn = '/Users/limingzhou/zhoul/work/energy/grib2/utah/hrrr.t00z.wrfsfcf03.grib2'
    ds = xr.load_dataset(fn, engine='pynio') #netcdf4
    para_file = '/Users/limingzhou/zhoul/work/energy/pytools/pytools/data_prep/hrrr_paras_pynio.txt'
    a = []
    with open(para_file) as f:
        for line in f:
            kv = line.strip().split(',')
            k = kv[0]; v = kv[1]
            # use 1h precipitation for Utah data
            if k == 'APCP_P8_L1_GLC0_acc':
                k = k + '1h'
            if int(v) == 1:
                a.append(k)
    ds2 = ds[a]
    ds2 = ds2.rename_vars({'APCP_P8_L1_GLC0_acc1h':'APCP_P8_L1_GLC0_acc'})
    ds2.to_netcdf(path='newdata.nc', engine='scipy')
    assert 1==1

def test_read_nc():
    ds = xr.load_dataset('newdata.nc',engine='scipy')
    assert 1==1




def test_find_ind_fromlatlon():
    arrx = np.array([[2,3,4], [5,6,7], [9, 10, 12]])
    arry = np.array([[30, 31, 32], [34, 35, 36], [31, 31, 32]])
    indx, indy = find_ind_fromlatlon(lon=9.3, lat=33.4, arr_lon=arrx, arr_lat=arry)
    assert(indx==0)
    assert(indy==1)


def test_download_utah_file_extract():
    download_utah_file_extract(cur_date=np.datetime64('2022-10-23 10:00'), fst_hour=0, tgt_folder='.')
    assert 1==1


def test_find_missing_grib2():
    src = ['/Users/limingzhou/zhoul/work/energy/grib2/hrrrdata', ]
    tgt = '.'
    find_missing_grib2(folders=src, tgt_folder=tgt, t0='2019-12-31 22:00', t1='2020-01-05 15:00')
    assert 1==1