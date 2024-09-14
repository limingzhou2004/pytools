import os
import pathlib
import sys 

import numpy as np
import pendulum as pu
import pytest
import xarray as xr
# import rioxarray


from pytools.data_prep.grib_utils import download_hrrr, download_utah_file_extract, extract_data_from_grib2, find_missing_grib2, get_herbie_str_from_cfgrib_file, get_paras_from_cfgrib_file, print_grib2_info, find_ind_fromlatlon, extract_a_file, decide_grib_type


hrrr_obs_path = '/Users/limingzhou/zhoul/work/energy/grib2/hrrrdata'
fn = hrrr_obs_path + '/hrrrsub_2020_01_01_00F0.grib2'


cfgrib_paras_file = 'pytools/data_prep/hrrr_paras_cfgrib.txt'

paras = """
0	19	0	0	1	0	255	0	Visibility
0	2	22	0	1	0	255	0	Wind speed (gust)
0	3	0	0	1	0	255	0	Pressure
0	0	0	0	1	0	255	0	Temperature
0	0	0	0	103	2	255	0	Temperature
0	1	0	0	103	2	255	0	Specific humidity
0	0	6	0	103	2	255	0	Dewpoint temperature
0	1	1	0	103	2	255	0	Relative humidity
0	2	2	0	103	10	255	0	u-component of wind
0	2	3	0	103	10	255	0	v-component of wind
0	4	7	0	1	0	255	0	Downward short-wave radiation flux
0	3	18	0	1	0	255	0	Planetary boundary layer height
0	1	8	0	1	0	255	0	Total precipitation
""".split('\n')


def test_extract_data_from_grib2():
    paras = get_paras_from_cfgrib_file(cfgrib_paras_file)
    ret, envelope = extract_data_from_grib2(fn=fn, lat=43, lon=-73, radius=30, paras=paras,return_latlon=False)
    #south-north, west-east, paras
    assert ret.shape==(21,21,16)


def test_():
    ret = get_herbie_str_from_cfgrib_file(cfgrib_paras_file)

    assert ret.startswith(':TMP')
    assert ret.endswith('10 m')


def test_get_paras_from_cfgrib_file():
    ret= get_paras_from_cfgrib_file(cfgrib_paras_file)

    assert len(ret['2m'])==4
    assert len(ret['10m'])==3
    assert len(ret['surface'])==9

@pytest.mark.skip(reason='large binary files needed for the test')
def test_read_grib2():
    # hrrr_obs_path = '/Users/limingzhou/zhoul/work/energy/grib2/hrrrdata'
    # fn = hrrr_obs_path + '/hrrrsub_2020_01_01_00F0.grib2'
    ds = xr.open_dataset(fn, engine="cfgrib")
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
    

#@pytest.mark.skip(reason='large binary files needed for the test')
def test_read_write_grib2():
    fn = '/Users/limingzhou/zhoul/work/energy/grib2/utah/20200105.hrrr.t14z.wrfsfcf00.grib2'
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


@pytest.mark.skip(reason='large binary files needed for the test')
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


@pytest.mark.skip(reason='download large files')
def test_download_hrrr():
  
    cur_date = pu.now(tz='UTC')
    if cur_date.minute < 10:
        cur_date = cur_date.add(hours=-2)
    else:
        cur_date = cur_date.add(hours=-1)
    # publishing time is 1 hour and 10 min after the cur_date, based on the observation
    download_hrrr(cur_date=cur_date,fst_hour=0, tgt_folder='.')
    assert 1==1

@pytest.mark.skip(reason='no grib2 files in code repo')
def test_decide_grib_type():
    hrrr_obs = 'hrrrsub_2020_01_01_00F0.grib2'
    hrrr_fst = 'hrrrsub_12_2020_01_01_18F1.grib2'
    utah_grib = '20200105.hrrr.t14z.wrfsfcf00.grib2'

    assert decide_grib_type(hrrr_obs) == 'hrrr_obs'
    assert decide_grib_type(hrrr_fst) == 'hrrr_fst'
    assert decide_grib_type(utah_grib) == 'utah_grib'
    assert decide_grib_type('Nosense') is None

