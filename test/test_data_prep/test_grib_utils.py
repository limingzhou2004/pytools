import os 

import xarray as xr


def test_read_grib2():
    hrrr_obs_path = '/Users/limingzhou/zhoul/work/energy/grib2/hrrrdata'
    fn = hrrr_obs_path + '/hrrrsub_2020_01_01_00F0.grib2'
    ds = xr.open_dataset(fn, engine="pynio")
    ds=3
