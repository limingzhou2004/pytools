import os
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import enum
import dill as pickle
from typing import Tuple, Union
from uuid import uuid4

from pytools.data_prep.weather_scaling import WeatherScaler
from pytools.data_prep import weather_data_prep as wp
from pytools.data_prep.weather_data_prep import WeatherDataPrep
from pytools.data_prep import weather_data as wd
from pytools.data_prep.weather_data_prep import GribType
from pytools.data_prep import load_data_prep as ldp
from pytools.config import Config


# pd.set_option('mode.chained_assignment', 'raise')
pd.options.mode.chained_assignment = None
"""
Manages all data prep for modeling purposes
site name and description
working folder for a site
parameters
starting and ending time

check range
fill na by imputations
create lagged load

weather data
"""


class DataType(enum.Enum):
    LoadData = 0
    CalendarData = 1
    WeatherData = 2


class DataPrepManager:
    """
    category, to identify like nyiso_hit_load and other iso * type of data
    t0 start time,
    t1 end time
    process scaling of load data, and weather data
    """

    def __init__(
        self,
        category: str,
        site_name: str,
        site_alias: str,
        site_description: str,
        site_folder: str,
        t0: str,
        t1: str,
        load_data: ldp.LoadData,
        load_limit: Tuple[float, float],
        max_load_lag_start: int = 1,
        load_lag_order: int = 168,
        utc_to_local_hours: int = -5,
        weather_type: GribType = GribType.hrrr,
        load_name: str = "load",
        timestamp_name: str = "timestamp",
        load_scaler: float = None,
        uuid=None,
        para_num=12,
    ):
        # category used for table name
        """

        Args:
            category: ny_load, pjm_load, ne_load, ny_price etc. Used as table name in the database
            site_name: capital, the name used to identify the zone
            site_alias: the name for the folder
            site_description: description like zone name, location,
            site_folder: the folder for all data and models for a given site
            t0: str for starting datetime, yyyy-mm-dd hh:mm
            t1: str for ending datetime, yyyy-mm-dd hh:mm
            load_data: historical load data
            load_limit: the (min, max) load that makes sens, e.g. you don't expect -10000 GW load or 1000000 GW
            max_load_lag_start: the first lag of load
            load_lag_order: the overall lag order, 0 for one lag, lag_order + lag_start is the max lag
            utc_to_local_hours: utc datetime + utc_to_local_hours = local datetime
            weather_type: hrrr or nem
            load_name: load column name, default 'load'
            timestamp_name: timestamp column name, default 'timestamp'
            load_scaler: a standard or minmax scalar
            uuid: uuid passed from a builder
        """
        self.category = category
        self.site_name = site_name
        self.site_alias = site_alias
        self.site_description = site_description
        # self.site_folder = os.path.join(site_folder, category, site_name)
        self.site_folder = site_folder
        if not os.path.exists(self.site_folder):
            os.makedirs(self.site_folder)
        self._load_data: ldp.LoadData = load_data
        self.t0 = t0
        self.t1 = t1
        self.load_name = load_name
        self.timestamp_name = timestamp_name
        self.load_limit = load_limit
        if load_scaler is None:
            self.load_scalar = []
        else:
            self.load_scalar = load_scaler
        self.weather_scaler = []
        raw_load_data = self.range_check_clean(
            load_data.train_data, load_name, min(load_limit), max(load_limit)
        )
        # standardize load data
        self.data_standard_load, self.load_scalar = self.standardize(
            raw_load_data, field=load_name
        )
        del raw_load_data[load_name]
        self.data_calendar = raw_load_data
        self.max_load_lag_start = max_load_lag_start
        self.load_lag_order = load_lag_order
        self.utc_to_local_hours = utc_to_local_hours
        self.weather_type = weather_type
        self.data_standard_load_lag = self.add_lag(
            self.data_standard_load, start=max_load_lag_start, order=load_lag_order
        )
        self.weather: wp.WeatherDataPrep = None
        self.center = None
        self.rect = None
        self.grib_name_filter_hist = None
        self.grib_name_filter_predict = None
        if uuid is None:
            self.uuid = uuid4()
        else:
            self.uuid = uuid
        self._weather_para_file = ""
        self._weather_predict_folder = ""
        self.para_num = para_num

    def process_load_data(
        self, load_data: ldp.LoadData, max_lag_start=None
    ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """
        Process the LoadData to generate lag load, calendar, and target load

        Args:
            load_data: LoadData.train_data
            max_lag_start: the max lag start, effectively, hours ahead to forecast; always starting from 1

        Returns: lag load, calendar, and target load

        """
        raw_load_data = self.range_check_clean(
            load_data.train_data,
            self.load_name,
            min(self.load_limit),
            max(self.load_limit),
        )
        data_standard_load, _ = self.standardize(raw_load_data, field=self.load_name)
        del raw_load_data[self.load_name]
        calendar_data = raw_load_data
        lag_start = self.max_load_lag_start if not max_lag_start else max_lag_start
        lag_data = self.add_lag(
            self.data_standard_load, start=lag_start, order=self.load_lag_order
        )
        return lag_data, calendar_data, data_standard_load

    def add_lag(self, df: pd.DataFrame, start, order):
        """
        Process df to append lag columns.

        Args:
            df: target dataframe
            start: the order from which the lag starts
            order: how many lags

        Returns: appended dataframe

        """
        mean = df.mean()
        col_name = list(df)[0]
        df_lag = (
            df.shift(1).fillna(value=mean).rename(columns={col_name: col_name + "_1"})
        )
        for i in range(2, start + order):
            df_lag["load_" + str(i)] = df.shift(i).fillna(value=mean)
        return df_lag

    def build_weather(
        self,
        weather_folder,
        jar_address: str,
        center: str,
        rect: str,
        weather_para_file: str = None,
    ):
        """
        Set up a Weatherdata_prep object, and set up hist weather.

        Args:
            weather_folder: grib file folder, as a dict of hrrr_hist, hrrr_predict, nam_hist, nam_predict
            jar_address: jar lib address
            center: (lat, lon)
            rect: (height, width) in km
            weather_para_file: weather parameter file

        Returns: None

        """
        if self.weather_type == GribType.hrrr:
            weather_folder = weather_folder["hrrr_hist"]
        elif self.weather_type == GribType.nam:
            weather_folder = weather_folder["nam_hist"]
        else:
            raise ValueError("un-recognized grib type, must be hrrr or nam")
        if weather_para_file is None:
            weather_para_file = self._weather_para_file
        w = self.build_hist_weather(
            weather_folder=weather_folder,
            weather_para_file=weather_para_file,
            jar_address=jar_address,
            grib_type=self.weather_type,
            para_num=self.para_num,
        )
        w.set_utc_to_local_hours(self.utc_to_local_hours)
        self.weather = w
        self.center = center
        self.rect = rect

    def make_npy_train(self, filter_func=None, parallel=True):
        """
        Make npy files for training

        Args:
            filter_func:
            parallel:

        Returns: None

        """
        self.weather.make_npy_data(
            center=self.center,
            rect=self.rect,
            grib_name_filter=filter_func,
            parallel=parallel,
        )

    def make_npy_predict(
        self,
        in_folder=None,
        out_folder=None,
        time_after=dt.datetime.now(),
        filter_func=None,
        parallel=True,
    ):
        if out_folder is None:
            out_folder = self.get_npy_folder(self.weather.grib_type, hist=False)
        if in_folder is None:
            in_folder = self._weather_predict_folder
        self.weather.set_npy_predict_folder(out_folder)
        return self.weather.make_npy_data(
            center=self.center,
            rect=self.rect,
            weather_folder=in_folder,
            folder_out=out_folder,
            last_time=time_after,
            grib_name_filter=filter_func,
            parallel=parallel,
        )

    def get_load_scalar(self):
        """
        Get the scaling factors for load data

        Returns: scalars, minmax, standard, etc...

        """
        return self.load_scalar

    def get_npy_folder(self, hrrr=GribType.hrrr, hist: bool = True):
        """
        Get the folder for a site

        Args:
            hrrr: GribType.hrrr or nam
            hist: hist or predict

        Returns: None

        """
        w_type = "hrrr" if hrrr == GribType.hrrr else "nam"
        if hist:
            return os.path.join(self.site_folder, "weather", "npy_train_", w_type)
        else:
            return os.path.join(self.site_folder, "weather", "npy_predict_", w_type)

    def get_predict_weather(self, predict_weather_folder=None):
        """
        Load all npy data in the folder

        Args:
            predict_weather_folder:

        Returns:

        """

        if self.weather is None:
            raise ValueError("no hist weather loaded")
        if predict_weather_folder is None:
            self.weather.set_npy_predict_folder(
                self.get_npy_folder(hrrr=self.weather_type, hist=False), clean=False
            )
        else:
            self.weather.set_npy_predict_folder(predict_weather_folder, clean=False)
        return self.weather.get_weather_predict()

    def build_hist_weather(
        self,
        weather_folder: str,
        jar_address: str,
        grib_type=GribType.hrrr,
        weather_para_file=None,
        para_num=None,
    ) -> wp.WeatherDataPrep:
        """
        Build a WeatherDataPrep object

        Args:
            weather_folder: grib2 folder
            jar_address:
            grib_type:
            weather_para_file:
            para_num:

        Returns:

        """
        if grib_type == GribType.hrrr:
            self.weather = wp.WeatherDataPrep.build_hrrr(
                weather_folder=weather_folder,
                jar_address=jar_address,
                dest_npy_folder=self.get_npy_folder(),
                weather_para_file=weather_para_file,
                utc_hour_offset=self.utc_to_local_hours,
                para_num=para_num,
            )
        elif grib_type == GribType.nam:
            self.weather = wp.WeatherDataPrep.build_nam(
                weather_folder=weather_folder,
                jar_address=jar_address,
                dest_npy_folder=self.get_npy_folder(hrrr=self.weather_type),
                weather_para_file=weather_para_file,
                utc_hour_offset=self.utc_to_local_hours,
                para_num=para_num,
            )
        else:
            raise ValueError("non valid grib type, must be hrrr or nam")
        return self.weather

    def get_train_weather(self) -> np.array:
        """

        Load all train weather npy data in the folder

        Returns: m X n X channel nd array

        """

        if self.weather is None:
            raise ValueError("no hist weather loaded")
        return self.weather.get_weather_train()

    def get_prediction_load(
        self,
        t0: Union[str, dt.datetime],
        t1: Union[str, dt.datetime],
        tc: Union[str, dt.datetime],
    ) -> pd.DataFrame:
        """
        Get predict load/calendar data

        Args:
            t0: str or datetime for starting time
            t1: str or datetime for ending time
            tc: str or datetime for the current time

        Returns: dataframe, calendar, load with prediction load as nan

        """

        def datetime2str(t):
            if isinstance(t, dt.datetime):
                return str(t)
            else:
                return t

        t0 = datetime2str(t0)
        t1 = datetime2str(t1)
        tc = datetime2str(tc)

        return self._load_data.query_predict_data(t0=t0, t1=t1, tc=tc)

    @property
    def load_data(self):
        return self._load_data

    def range_check_clean(
        self, df: pd.DataFrame, field: str, min_val: float, max_val: float
    ) -> pd.DataFrame:  # load data have to be checked
        df.loc[(df[field] < min_val) | (df[field] > max_val)] = np.nan
        return df.fillna(method="backfill")

    def reconcile(
        self, load_df: pd.DataFrame, date_column: str, w_data: wd.WeatherData
    ) -> (pd.DataFrame, np.array):
        """
        Reconcile load data with weather data based on timestamps; consider the timezone difference

        Args:
            load_df: load dataframe
            date_column: name of date column in load_df
            w_data: WeatherData to reconcile

        Returns:

        """
        joint_time, join_weather_array = w_data.query_timestamp_sorted_data(
            load_df[date_column]
        )
        load_df = load_df.set_index(date_column)
        return load_df.loc[joint_time], join_weather_array

    def setup_grib_para_file(self, fn: str):
        self._weather_para_file = fn

    def setup_grib_predict_folder(self, folder: str):
        self._weather_predict_folder = folder

    def standardize(self, df: pd.DataFrame, field, method="minmax"):
        if not isinstance(field, list):
            field = [field]
        if method == "minmax":
            if not self.load_scalar:
                scaler: MinMaxScaler = MinMaxScaler()
                scaler.fit(df[field])
                self.load_scalar = scaler
            else:
                scaler = self.load_scalar
            return (
                pd.DataFrame(scaler.transform(df[field]), columns=list(df[field])),
                scaler,
            )

    def standardize_predictions(self, prediction_data: ldp.LoadData) -> ldp.LoadData:
        """

        Args:
            prediction_data: prediction_data:

        Returns: scaled prediction

        """
        if isinstance(prediction_data, ldp.LoadData):
            d = prediction_data.get_data()[self.load_name]
        elif isinstance(prediction_data, pd.DataFrame):
            d = prediction_data[self.load_name]
        else:
            raise ValueError("Data type either be LoadData or DataFrame")

        ind_none = d.apply(lambda x: (x is None) | np.isnan(x))
        ind = np.bitwise_not(ind_none)
        d.loc[ind_none] = np.nan
        if ind.max() > False:
            d.loc[ind] = self.load_scalar.transform(
                d[ind].values.reshape((-1, 1))
            ).reshape((-1))
        return prediction_data

    def standardize_weather(
        self, weather_array: np.ndarray, overwrite: bool = False
    ) -> np.ndarray:
        if not self.weather_scaler or overwrite:
            self.weather_scaler = WeatherScaler(data=weather_array)
        return self.weather_scaler.scale(data=weather_array)


def save(config: Config, dmp: DataPrepManager, prefix="", suffix=""):
    fn = config.get_model_file_name(
        class_name="_data_manager_", prefix=prefix, suffix=suffix
    )
    with open(fn, "wb") as dill_file:
        pickle.dump(dmp, dill_file)
    return fn


def load(
    config: Config, fn: str = None, prefix="", suffix=""
) -> Union[bool, DataPrepManager]:
    if fn is None:
        fn = config.get_model_file_name(
            class_name="_data_manager_", prefix=prefix, suffix=suffix
        )
    if not os.path.exists(fn):
        return False
    with open(fn, "rb") as fr:
        return pickle.load(fr)
