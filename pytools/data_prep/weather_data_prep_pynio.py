
import datetime as dt

from typing import Callable, List
from pytools.data_prep.weather_data_prep import GribType, HistFst, WeatherDataPrep


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
        weather_df: str = None,
        folder_out: str = None,
        parallel: bool = True,
        grib_name_filter: Callable = None,
    ):
        """
        Create npy data from a given pickle file of dataframe.

        Args:
            center:
            rect:
            weather_df: dataframe from a pickle file
            folder_out: for non default npy output folder
            last_time: only get files after the last_time, yyyy-mm-dd hh:mm
            parallel: True or False
            grib_name_filter: further screening the grib files, a must for nam historic and predictions

        Returns: npy file, sample * x * y * para

        """



        return