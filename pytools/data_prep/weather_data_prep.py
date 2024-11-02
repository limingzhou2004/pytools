import shutil
from enum import Enum
import datetime as dt
import os
from typing import Dict, List, Optional, Callable
import uuid

import numpy as np
import pandas as pd
import dask.bag as bag

from pytools.data_prep import weather_data as wd
from pytools.data_prep.get_datetime_from_grib_file_name import get_datetime_from_grib_file_name, get_datetime_from_grib_file_name_utah
from pytools.data_prep.grib_utils import extract_data_from_grib2, get_paras_from_cfgrib_file
from pytools.utilities import get_file_path, parallelize_dataframe

def grib_filter_func(
    file_names: list,
    func_timestamp: callable,
    func_fst_hours: callable,
    min_filename_length=20,
    predict=False,
    max_fst_hours=None,
    t_after=None,
    time_box=None,
) -> list:
    """
    Return filtered files for the latest forecast

    Args:
        file_names:  input file names
        func_timestamp: function to get the timestamp from the file name
        func_fst_hours: function to derive the forecast hours ahead, 0 for actual weather
        min_filename_length: minimum filename length to consider as a valid grib2 file
        predict: weather prediction of not
        max_fst_hours: if predict == False, use the fst hours <= max_fst_hours as actual weather; if predict == True,
        max_fst_hours can be set as a range, so only weather forecast made between certain hours are considered,
        t_after: To get files after a specific datetime
        time_box: [t0, t1];

    Returns: there are multiple files for the same target hour, so we find the most recent file

    """

    res = [
        (a, func_timestamp(a), func_fst_hours(os.path.basename(a)))
        for a in file_names
        if len(os.path.basename(a)) >= min_filename_length
    ]
    if not max_fst_hours:
        max_fst_hours = 1000
    if not res:
        raise ValueError("No grib weather files found!")
    df = pd.DataFrame(res, columns=["fn", "timestamp", "hours"])
    if time_box:
        if isinstance(time_box[0], str):
            time_box = [pd.to_datetime(t) for t in time_box]
        df = df[df["timestamp"] >= time_box[0]]
        df = df[df["timestamp"] <= time_box[1]]
    if not predict:
        df = df[df["hours"] <= max_fst_hours]
        if t_after is not None:
            df = df[df["timestamp"] >= np.datetime64(t_after)]
    else:
        if isinstance(max_fst_hours, int):
            max_fst_hours = [0, max_fst_hours]
        if len(max_fst_hours) == 2:
            df = df[max_fst_hours[0] <= df["hours"]]
            df = df[df["hours"] <= max_fst_hours[1]]
        else:
            raise ValueError(
                "max_fst_hours must have two elements, starting time and ending time!"
            )
    grouped = df.groupby("timestamp")
    xf = df.loc[grouped["hours"].idxmin()]
    return xf["fn"].tolist()


def grib_filter_func_elemental(
    file_name: str,
    func_timestamp: callable,
    func_fst_hours: callable,
    min_filename_length=20,
    predict=False,
    max_fst_hours=None,
    t_after=None,
) -> bool:
    """
    Filter non grib files

    Args:
         file_name:  input file names
         func_timestamp: function to get the timestamp from the file name
         func_fst_hours: function to derive the forecast hours ahead, 0 for actual weather
         min_filename_length: minimum filename length to consider as a valid grib2 file
         predict: weather prediction of not
         max_fst_hours: if predict == False, use the fst hours <= max_fst_hours as actual weather; if predict == True,
         max_fst_hours can be set as a range, so only weather forecast made between certain hours are considered,
         t_after: To get files after a specific datetime

     Returns: bool

    """
    if len(os.path.basename(file_name)) <= min_filename_length:
        return False
    return False


class NameException(Exception):
    pass


def has_numbers(input_string) -> bool:
    """
    Whether a string contains a number

    Args:
        input_string: a string

    Returns:

    """
    return any(char.isdigit() for char in input_string)


class GribType(Enum):
    hrrr = 0
    nam = 1


class HistFst(Enum):
    hist = 0
    fst = 1


class NoTrainingDataError(Exception):
    pass


class WeatherDataPrep:

    earliest_time = dt.datetime.strptime("2016-12-01", "%Y-%m-%d")
    hrrr_fmt = "%Y_%m_%d_%H"

    def __init__(
        self,
        para_file: str,
        weather:Dict,
        dest_npy_folder: str,
        hist_fst_flag: HistFst,
        timestamp_fmt,
        t0: dt.datetime,
        t1: dt.datetime,
        prefix,
        #para_num: int = 12,
        utc_hour_offset: int = None,
    ):
        """
        Weather Data preparation

        Args:
            para_file: parameter file
            weather_folder: folder for source grib files, either str or list[str]
            dest_npy_folder: npy folder
            hist_fst_flag: HistFst 0 for hist, 1 for fst (predict)
            timestamp_fmt: to parse datetime from grib2 file names
            t0: start datetime
            t1: end datetime
            prefix: prfix str
            para_num: number of weather channel
            utc_hour_offset: number of hours to add to convert to UTC,

        """

        self.para_file = para_file
        self.weather = weather
        self.weather_obj = None
        self.weather_obj_fn = ""
        self.uuid = uuid.uuid4()
        self.dest_npy_folder = dest_npy_folder
        self.dest_predict_npy_folder = None
        self.hist_fst_flag = hist_fst_flag
        self.timestamp_fmt = timestamp_fmt
        self.t0 = t0
        self.t1 = t1
        self.prefix = prefix
        if has_numbers(prefix):
            raise NameException("no digits in prefix")
        self.suffix = ".npy"
        self.weather_train_data: Optional[wd.WeatherData] = None
        self.data_shape = None  # nan * m * n * para
        self.weather_predict_data: Optional[wd.WeatherData] = None
        self.utc_to_local_hours = utc_hour_offset
        self.check_grib_name_filter = None
        self.min_filename_length = 20
        self.hrrr_paras:Dict = get_paras_from_cfgrib_file(para_file)[0]
        #self.hrrr_paras:Dict = get_paras_from_pynio_file(para_file,False)
        #self.utah_paras:Dict = get_paras_from_pynio_file(para_file,True)
        # lon 2D
        self.x_grid = None
        # lat 2D
        self.y_grid = None

    def extract_datetime_from_grib_filename(
        self, filename: str, hour_offset: int = None, nptime=True, get_fst_hour=False, 
    ):
        """
        Get the datetime from an hrrr grib file name

        Args:
            filename:  file name
            hour_offset: hours to add to convert grib datetime(UTC) to local
            nptime:
            get_fst_hour:

        Returns: python datetime or numpy timestamp64

        """
        if hour_offset is None:
            hour_offset = self.utc_to_local_hours
        return get_datetime_from_grib_file_name(
            filename=filename,
            hour_offset=hour_offset,
            nptime=nptime,
            get_fst_hour=get_fst_hour,
        )

    def extract_datetime_from_grib_filename_utah():

        return

    def set_npy_predict_folder(self, folder, clean=True):
        """
        set the npy folder for predictions

        Args:
            folder: path for the npy data
            clean: clear the files if true

        Returns:

        """
        self.dest_predict_npy_folder = folder
        if clean:
            shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)

    def timestamp_to_npy_file(self, t):
        t = pd.to_datetime(t)
        t_str = t.strftime(self.timestamp_fmt)
        return os.path.join(self.dest_npy_folder, self.prefix + t_str + self.suffix)

    def make_npy_data_from_inventory(
            self, 
            center:List[float],
            rect:List[float],
            inventory_file:str=None,
            parallel:bool=False,
            folder_col_name:str='folder',
            filename_col_name:str='filename',
            type_col_name:str='type',
            t0: np.datetime64=np.datetime64('2018-01-01'),
            t1: np.datetime64=np.datetime64('2018-01-03'),
            n_cores=7,
            )->wd.WeatherData:
        df = pd.read_pickle(get_file_path(fn=inventory_file, this_file_path=__file__))
        df = df[(df['timestamp']>=t0) & (df['timestamp']<=t1)]

        envelope = []

        def single_row_process(row):
            nonlocal envelope

            filename = getattr(row,filename_col_name)
            fn = os.path.join(getattr(row,folder_col_name), filename)

            if getattr(row,type_col_name).startswith('hrrr'):
                is_utah=False
                p=self.hrrr_paras
            else:
                is_utah=True
                p=self.utah_paras

            timestamp = get_datetime_from_grib_file_name_utah(filename,hour_offset=0, nptime=True, get_fst_hour=False)  if is_utah \
                else get_datetime_from_grib_file_name(
                filename=filename, 
                hour_offset=0,
                nptime=True, 
                get_fst_hour=False)
            
            return_latlon = False
            if self.x_grid is None:
                return_latlon = True
            res = extract_data_from_grib2(
                fn_arr=fn, lon=center[0],  
                lat=center[1], radius=rect, paras=p, 
                return_latlon=return_latlon, envelope=envelope)
            envelope = res[1] 
            if self.x_grid is None:
                self.x_grid = res[2].data
                self.y_grid = res[3].data
            return timestamp, res[0]
        
        def df_block_process(df_sub):
            data_dict={}
            for row in df_sub.itertuples():
                k, v = single_row_process(row)
                data_dict[k] = v

            return data_dict  #np.stack(arr, axis=0)
        
        if parallel:
            dict_list = parallelize_dataframe(df, df_block_process, n_cores=n_cores )
            dict_ta = {}
            for d in dict_list:
                dict_ta.update(d)

        else:
            dict_ta = df_block_process(df)

        w_data = wd.WeatherData(dict_data=dict_ta, prediction=False, paras=self.hrrr_paras, grid_x=self.x_grid, grid_y=self.y_grid)
        self.weather_train_data = w_data

        return w_data
            
    def make_npy_data(
        self,
        center: List[float],
        rect: List[float],
        weather_folder: str = None,
        folder_out: str = None,
        last_time: dt.datetime = None,
        parallel: bool = True,
        grib_name_filter: Callable = None,
    ):
        """
        Create npy data from a given folder into another given folder; the folder is defined in the constructor

        Args:
            center:
            rect:
            weather_folder: for none default grib2 input folder
            folder_out: for non default npy output folder
            last_time: only get files after the last_time, yyyy-mm-dd hh:mm
            parallel: True or False
            grib_name_filter: further screening the grib files, a must for nam historic and predictions

        Returns: None

        """
        # extract subset of grib2 files into npy files.
        pj = self.get_pj(center=center, rect=rect)
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

    def get_weather_train(self, folders=None) -> wd.WeatherData:
        """
        Derive a single multi npy array from multiple folders of npy data

        Args:
            folders:

        Returns: WeatherData object

        """
        # if folders is None:
        #     folders = self.dest_npy_folder
        # self.weather_train_data = self.load_all_npy(
        #     folders=folders, para_num=self.para_num
        # )
        #self.weather_train_data.standardize()
        return self.weather_train_data

    def get_weather_predict(
        self, folders: str = None, predict: bool = True
    ) -> wd.WeatherData:
        """
        Load npy forecast data. Scale using the training weather data scaler.

        Args:
            folders: folder for reading npy files.
            predict: True for prediction, False for training

        Returns: the scaled wd.WeatherData object

        """
        if folders is None:
            folders = self.dest_predict_npy_folder
        if not isinstance(folders, list):
            folders = [folders]
        predict_data = self.load_all_npy(
            folders=folders, predict=predict, para_num=self.para_num
        )
        if predict_data.data.shape[0] == 0:
            raise ValueError("No Npy files found!")
        predict_data.scaler = self.weather_train_data.scaler
        predict_data.transform()
        return predict_data

    def load_all_npy(
        self,
        folders: List[str],
        predict: bool = False,
        para_num: int = None,
        npy_ending: str = ".npy",
    ) -> wd.WeatherData:
        """
        Load npy data; for predictions,

        Args:
            folders: list of folders to process
            predict: for prediction weather
            para_num: channel number
            npy_ending: check file name ending with .npy

        Returns: WeatherData.data has the sample X height X Width X channel data

        """
        if not para_num:
            para_num = self.para_num
        file_names = []
        if not isinstance(folders, list):
            folders = [folders]
        if isinstance(predict, str):
            predict = pd.to_datetime(dt).astype(dt.datetime)
        for f in folders:
            if isinstance(predict, dt.datetime):
                file_names.extend(
                    [
                        os.path.join(f, fn)
                        for fn in os.listdir(f)
                        if self.extract_datetime_from_grib_filename(fn) > predict
                    ]
                )
            else:
                file_names.extend([os.path.join(f, fn) for fn in os.listdir(f)])
        if npy_ending:
            file_names = list(filter(lambda x: x.endswith(".npy"), file_names))
        dat = (
            bag.from_sequence(file_names)
            .map(
                lambda x: (
                    self.extract_datetime_from_grib_filename(x),
                    self.load_a_npy(x, para_num=para_num),
                )
            )
            .compute()
        )
        val = {k: v for (k, v) in dat}
        if predict:
            if self.weather_train_data is None:
                raise NoTrainingDataError(
                    "No training data processed, needed to define the scaler"
                )
            return wd.build_predict_data(self.weather_train_data, val)
        else:
            return wd.WeatherData(dict_data=val)

    def load_a_npy(self, file_name: str, para_num: int):
        """
        load a npy file

        Args:
            file_name: file name
            para_num: parameter number

        Returns: npy array data if found

        """

        if os.path.isfile(file_name):
            data = np.load(file_name)
            dim = self.impute_shape(data=data, para_num=para_num)
            data = np.swapaxes(data.reshape(dim), 0, 2)
            if self.data_shape is None:
                self.data_shape = dim
            return data

    def set_utc_to_local_hours(self, hours):
        self.utc_to_local_hours = hours

    def impute_shape(self, data, para_num):
        dim = data.shape
        return para_num, int(dim[0] / para_num), dim[1]

    @classmethod
    def build_hrrr(
        cls,
        weather,
        dest_npy_folder,
        utc_hour_offset: int,
        weather_para_file=None,
    ):

        return cls(
            para_file=weather_para_file,
            weather=weather,
            dest_npy_folder=dest_npy_folder,
            hist_fst_flag=HistFst.hist,
            timestamp_fmt=cls.hrrr_fmt,
            t0=cls.earliest_time,
            t1=dt.datetime.now(),
            prefix="hrrr_hist_",
            utc_hour_offset=utc_hour_offset,
        )
