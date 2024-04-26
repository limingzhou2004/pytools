import numpy as np

from pytools.data_prep.get_datetime_from_grib_file_name import get_datetime_from_grib_file_name_utah


def test_get_datetime_from_grib_file_name_utah():
    utah_fn = '20200105.hrrr.t15z.wrfsfcf00.grib2'

    dt = get_datetime_from_grib_file_name_utah(filename=utah_fn,hour_offset=0,nptime=True,get_fst_hour=False)
    
    assert dt==np.datetime64('2020-01-05T15:00')
    
    dt = get_datetime_from_grib_file_name_utah(filename=utah_fn,hour_offset=5,nptime=True,get_fst_hour=True)
    assert dt==0