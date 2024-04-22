
import datetime as dt
import os
from typing import Callable, List
import asyncio

import numpy as np
import pandas as pd

from pytools.data_prep.weather_data_prep import GribType, HistFst, WeatherDataPrep
from pytools.data_prep.grib_util_org import col_folder, col_batch, col_complete_timestamp,col_filename,col_timestamp,col_type
from pytools.utilities import parallelize_dataframe

default_pickle_file = os.path.join(os.path.dirname(__file__), '../data/grib2_folder_0.pkl')

def create_weather_data(pickle_file:str=default_pickle_file, fout:str='0', parallel=False, t0:str=None, t1:str=None, lon:float=0, lat:float=0, radius=0, paras=[]) -> np.ndarray:
    """
    Read the grib files and extract the npy array. The pickle file schema, file_name, folder, type, timestamp

    Args:
        pickle_file (str): pickle file name
        fout (str): '0' for historical data
        parallel (bool): True for multi process
        t0 (str): starting time
        t1 (str): ending time
        

    Returns:
        np.ndarray: np array, sample * x * y * channels
    """

    df = pd.read_pickle(pickle_file)


    parallelize_dataframe()

    return df





class WeatherDataPrepPynio(WeatherDataPrep):
    """
    _summary_

    Args:
        WeatherDataPrep (_type_): _description_
    """

    def __init__(
        para_file: str,
        weather_folder: List[str],
        dest_npy_folder: str,
        hist_fst_flag: HistFst,
        timestamp_fmt,
        t0: dt.datetime,
        t1: dt.datetime,
        prefix,
        para_num: int = 12,
        grib_type=GribType.hrrr,
        utc_hour_offset: int = None,
    ):
        super().__init__()

    def make_npy_data(
        self,
        center: str,
        rect: str,
        weather_df: pd.DataFrame = None,
        folder_out: str = None,
        parallel: bool = True,
        grib_name_filter: Callable = None,
    ):
        """
        Create npy data from a given pickle file of dataframe.

        Args:
            center: str format, (lon,lat)
            rect: str format, ()
            weather_df: dataframe from a pickle file
            folder_out: for non default npy output folder
            last_time: only get files after the last_time, yyyy-mm-dd hh:mm
            parallel: True or False
            grib_name_filter: further screening the grib files, a must for nam historic and predictions

        Returns: npy file, sample * x * y * para

        """
        prefix = self.prefix
        if weather_folder is None:
            weather_folder = self.weather_folder
        if folder_out is not None:
            pj.set_folder_out(folder_out)

        def make_npy(a_folder):
            # file name length has to be greater than 20
            valid_fn = os.listdir(a_folder)
            if grib_name_filter is not None:
                if "grib_filter_func" in str(grib_name_filter.func):
                    valid_fn = grib_name_filter(valid_fn)
                else:
                    valid_fn = list(filter(grib_name_filter, valid_fn))
            if last_time is None:
                exclude = []
            else:
                exclude = [
                    i
                    for i in valid_fn
                    if (
                        self.extract_datetime_from_grib_filename(i, nptime=False)
                        < last_time
                    )
                ]
            if self.check_grib_name_filter is not None:
                additional_exclude = [
                    self.check_grib_name_filter(
                        self.extract_datetime_from_grib_filename(i, get_fst_hour=True)
                    )
                    for i in valid_fn
                ]
                exclude.extend(additional_exclude)
            pj.set_folder_in(a_folder)
            pj.process_folders(
                out_prefix=prefix,
                out_suffix=".npy",
                exclude=exclude,
                include_files=None if grib_name_filter is None else valid_fn,
                parallel=parallel,
            )
            return len(valid_fn)

        if isinstance(weather_folder, list):
            row_count = 0
            for f in weather_folder:
                row_count += make_npy(f)
            return row_count
        else:
            return make_npy(weather_folder)


        return

    @classmethod
    def build_hrrr():

        return     