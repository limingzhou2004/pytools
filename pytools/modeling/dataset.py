import pickle
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import Tensor
from torch.utils import data

from pytools.config import Config
from pytools.config import DataType
from pytools.modeling.weather_net import WeatherPara


class WeatherDataSet(data.Dataset):
    def __init__(
        self,
        weather: np.ndarray,
        lag_load: np.ndarray,
        calendar_data: np.ndarray,
        target: np.ndarray,
    ):
        """
        Move the channel to the second dimension in np wea arrays.

        Args:
            weather: np arrays, sample, x, y, channel
            lag_load:  sample, load
            calendar_data: sample, cal
            target: sample, load
        """
        # weather dim batch, height, width, channel --> batch, channel, height, width
        original_channel_num = 3
        new_channel_num = 1
        self._weather = np.moveaxis(
            weather, original_channel_num, new_channel_num
        ).astype(np.float32)
        self._lag_load = np.expand_dims(lag_load, 1).astype(np.float32)
        self._calendar_data = calendar_data.astype(np.float32)
        self._target = target.astype(np.float32)

    def __len__(self):
        """
        Denotes the total number of samples

        Returns: sample number

        """
        return self._target.shape[0]

    def __getitem__(self, index) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Generates one sample of data

        Args:
            index: a string

        Returns: weather, calendar, y-label

        """
        return (
            self._weather[index, :, :, :],
            self._lag_load[index, :],
            self._calendar_data[index, :],
            self._target[index, :],
        )


def read_past_fst_weather(config:Config, year=-1, month=-1):
    fnw = config.get_load_data_full_fn(DataType.Past_fst_weatherData, 'pkl', year=year)
    with open(fnw, 'rb') as file:
        dat = pickle.load(file) 

    #[spot_time, envlopes[fst_timestamp, array[x,y,channel]]]

    
    with open(fnw, 'wb') as file:
        #pickle.dump()
        pass
    return

def check_fix_missings(df_load:np.ndarray, w_timestamp:np.ndarray, w_arr:np.ndarray)->Tuple[np.ndarray, np.ndarray]:
    # sync data, fill missings
    w_timestamp = pd.DatetimeIndex(list(w_timestamp)).tz_localize('UTC')
    t_str = 'timestamp'

    t0 = max(df_load[0][0], w_timestamp[0])
    t1 = min(df_load[0][-1], w_timestamp[-1])
    t = pd.DataFrame(pd.date_range(start=t0, end=t1, freq='h'), columns=[t_str])
    df_tl = t.join(df_load, on=t_str, how='left')
    df_tl.interpolate(inplace=True)
    df_tw = t.join(pd.DataFrame(w_timestamp, columns=[t_str]), on=t_str, how='left' ) 

    return

def read_weather_data_from_config(config:Config, year=-1):
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    fn_load = config.get_load_data_full_fn(DataType.LoadData, 'npz', year=-1)
    fn_wea = config.get_load_data_full_fn(DataType.Hist_weatherData, 'npz', year=year)
    with np.load(fn_load) as dat:
        load_data = dat[DataType.LoadData.name]
    with np.load(fn_wea) as dat:
        paras = dat['paras']
        w_timestamp = dat['timestamp']
        w_data = dat[DataType.Hist_weatherData.name]

    return load_data, paras, w_timestamp, w_data


class WeatherDataSetBuilder:
    """
    The lag_load is queried by lag 1 for training data, essentially shifted by one from target.
    For predictions, only the load column is used.
    """

    def __init__(
        self,
        weather: np.ndarray,
        lag_load: np.ndarray = [],
        calendar_data: np.ndarray = None,
        y_labels: np.ndarray = None,
        cat_fraction=None,
    ):
        """
        Initialization. The original data has lag 1 to lag 168 by default.

        Args:
            weather: np.ndarray for sample * x * y * channel
            lag_load: lagged load; if none, use y_label shifted by one
            calendar_data: 2d npy data array
            y_labels:
            cat_fraction: dict {'train':0.8, 'test':0.1, 'validation':0.1}
            #hours_ahead: forecast starting, 1 hour, 7 hours, 25 hours, 49 hours etc
        """
        self._weather = weather.squeeze()
        self._hours_ahead = None
        y_labels = y_labels.squeeze()
        if len(y_labels.shape) < 2:
            y_labels = y_labels.reshape((-1, 1))
        self._y_labels = y_labels
        self._calendar_data = calendar_data.squeeze()
        if len(lag_load) == 0:
            lag_load = np.roll(y_labels, 1, axis=0)
        self._lag_load = lag_load.squeeze()
        if isinstance(cat_fraction, List):
            cat_fraction = {
                "train": cat_fraction[0],
                "test": cat_fraction[1],
                "validation": cat_fraction[2],
            }
        self._cat_fraction = (
            {"train": 0.75, "test": 0.1, "validation": 0.15}
            if not cat_fraction
            else cat_fraction
        )
        assert np.isclose(1, sum(self._cat_fraction.values())), \
            "sum of train, test, validation fractions should be 1!"
        self._cat_index = {"train": None, "test": None, "validation": None}
        self._cat = "train"
        assert (
            weather.shape[0] == calendar_data.shape[0]
        ), "npy file and calendar sizes not match"
        assert len(calendar_data) == len(
            y_labels
        ), "calendar data and y_labels sizes not match"
        self._all_sample_index = range(self._y_labels.shape[0])

    def extract_data(self, cat: str, fst_hours: int) -> WeatherDataSet:
        """
        Set up the mode to get data, train|test|validation, fst hours ahead

        Args:
            cat: train, test, or validation; if train==1, return the full dataset for training, and None, None for val, test
            fst_hours: hours ahead to forecast, be between 0.01 and 169
        """
        assert isinstance(fst_hours, int)
        max_fst_hours = 169
        min_fst_hours = 0.1
        assert max_fst_hours > fst_hours >= min_fst_hours
        self._hours_ahead = fst_hours
        assert cat in self._cat_fraction
        self._cat = cat
        if self._cat_fraction["train"] == 1:
            return WeatherDataSet(
                weather=self._weather,
                lag_load=np.roll(self._lag_load, self._hours_ahead - 1, axis=0),
                calendar_data=self._calendar_data,
                target=self._y_labels,
            )

        tmp, self._cat_index["validation"] = train_test_split(
            self._all_sample_index, test_size=self._cat_fraction.get("validation")
        )
        self._cat_index["train"], self._cat_index["test"] = train_test_split(
            tmp,
            train_size=self._cat_fraction.get("train")
            / (1 - self._cat_fraction.get("validation")),
        )
        return WeatherDataSet(
            weather=self.weather,
            lag_load=self.lag_loads,
            calendar_data=self.calendar,
            target=self.target,
        )

    @property
    def weather(self):
        return self._weather[self._cat_index[self._cat], :, :, :]

    @property
    def target(self):
        return self._y_labels[self._cat_index[self._cat], :]

    @property
    def lag_loads(self):
        # lag_load already lagged by one
        tmp_loads = np.roll(self._lag_load, self._hours_ahead - 1, axis=0)
        return tmp_loads[self._cat_index[self._cat], :]

    @property
    def calendar(self):
        return self._calendar_data[self._cat_index[self._cat], :]

    def get_weather_para(self):
        w = list(self._weather.shape[1:])
        w.append(self._lag_load.shape[1])
        w.append(self._calendar_data.shape[1])
        return WeatherPara(*w)
