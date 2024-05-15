import numpy as np
import pendulum as pu

import datetime as dt
import os
import re
from typing import Union


def get_datetime_from_grib_file_name(
    filename, hour_offset, nptime=True, get_fst_hour=False
) -> Union[int, np.datetime64, dt.datetime]:
    """
    Derive the datetime from a single grib file name: hours ahead|np.datetime64|dt.datetime
    Args:
        filename:
        hour_offset: the hours to add to convert UTC time that grib files use to local datetime, e.g. -5 for EST
        nptime: whether to return np.datetime64 or python datetime
        get_fst_hour: whether to return the forecast hours ahead or the datetime itself
    Returns: either hours ahead as an integer, or target datetime (np.datetime64, or python native datetime)
    """

    filename = os.path.basename(filename)  # no directory used
    if len(filename) < 10:
        return
    p = re.compile(r".*(\d\d+).*(20\d\d)_(\d+)_(\d+)_(\d+)F(\d+).*")
    # nam_12_2016_02_03_XX starts with 12 as fst time.
    # The hour XX will be the hour data produced rather than the hour forecasted.
    m = p.match(filename)
    if m is None:
        p = re.compile(r".*(\d+).*(20\d\d)_(\d+)_(\d+)_(\d+)F(\d+).*")
        m = p.match(filename)
    if m is None:
        p = re.compile(r".*(20\d\d)_(\d+)_(\d+)_(\d+)F(\d+).*")
        m = p.match(filename)
    if m.lastindex == 5:
        nums = [int(m.group(i)) for i in range(1, 6)]
        nums = [-1] + nums
        t = dt.datetime(nums[1], nums[2], nums[3], nums[4]) + dt.timedelta(hours=nums[5]) + dt.timedelta(hours=hour_offset)
        
    else:
        nums = [int(m.group(i)) for i in range(1, 7)]
        t = dt.datetime(nums[1], nums[2], nums[3], nums[0])+ dt.timedelta(hours=nums[5]) + dt.timedelta(hours=hour_offset)
    
    if t is None:
        raise ValueError("Unrecognized file name format for hrrr!")
    if get_fst_hour:
        return int(nums[5])
    else:
        if nptime:
            return np.datetime64(t)
        else:
            return t
        

def get_datetime_from_grib_file_name_utah(
    filename, hour_offset, nptime=True, get_fst_hour=False
) -> Union[int, np.datetime64, dt.datetime]:
    filename = os.path.basename(filename)  # no directory used
    if len(filename) < 10:
        return
    p = re.compile(r'(20\d\d)(\d\d)(\d\d).hrrr.t(\d\d)z.wrfsfcf(\d\d).grib2')
    m = p.match(filename)
    nums = [int(m.group(i)) for i in range(1, 6)]
    nums = [-1] + nums
    if get_fst_hour:
        return int(nums[5])
    else:
        t =  dt.datetime(nums[1], nums[2], nums[3]) + dt.timedelta(hours=nums[4]) + dt.timedelta(hours=hour_offset)
        if nptime:
            return np.datetime64(t)
        else:
            return t
    

def get_datetime_from_utah_file_name(filename:str, get_fst_hour=False, numpy=True):
    t = pu.parse(filename[0:8])
    chour = int(filename[15:17])
    fhour = int(filename[26:28])
    t = t.add(hours=chour)
    if numpy:
        t = np.datetime64(t)

    if get_fst_hour:
        return t, fhour
    else:
        return t