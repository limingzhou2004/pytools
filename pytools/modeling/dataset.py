import math
from pathlib import Path
import pickle
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
from torch import Tensor
from torch.utils import data

from pytools.config import Config
from pytools.config import DataType
from pytools.modeling.scaler import Scaler, load
from pytools.modeling.weather_net import WeatherPara



class WeatherDataSet(data.Dataset):
    def __init__(
        self,
        flag:str,
        tabular_data: np.ndarray,
        wea_arr: np.ndarray,
        timestamp: np.ndarray,
        config: Config,
        sce_ind: int,
        to_scale:bool = True,
    ):
        # the scaler and model file has flag and year information

        self._config = config
        self._flag = flag
        target_ind = config.model_pdt.target_ind
        seq_length = config.model_pdt.seq_length
        fst_horizon = config.model_pdt.forecast_horizon
        self._target = tabular_data[:, target_ind]
        self._ext = np.delete(tabular_data, target_ind, axis=1)
        self._wea_arr = wea_arr
        self._sce_ind = sce_ind

        self._seq_length = seq_length
        self._pred_length = fst_horizon[-1]
        self._fst_horizeon = fst_horizon
        self._wea_ar_embedding_dim = config.model_pdt.wea_ar_embedding_dim
        self._wea_embedding_dim = config.model_pdt.wea_embedding_dim
        self._ext_embedding_dim = config.model_pdt.ext_embedding_dim

        self._fn_scaler = config.get_model_file_name(class_name='scaler')

        if to_scale:
            if Path(self._fn_scaler).exists():
                scaler = load(self._fn_scaler)
            else:
                scaler = Scaler(self._target, self._wea_arr, scaler_type=config.model_pdt.scaler_type)
                scaler.save(self._fn_scaler)

            self._target = scaler.scale_target(self._target)
            self._wea_arr = scaler.scale_arr([self._wea_arr])[0]
      
        # load the data, and selec the subset, based on flag, ind
        # filter by t0 and t1

        if flag.startswith('cv'):
            tt = config.model_pdt.cv_settings[sce_ind]
            t0 = tt[0]
            t1 = tt[1]
        elif flag.startswith('final_train'):
            tt = config.model_pdt.final_train_hist[sce_ind]
            t0 = tt[0]
            t1 = tt[1]
        elif flag.startswith('forward_forecast'):
            tt = config.model_pdt.final_train_hist[sce_ind]
            t0 = tt[2]
            t1 = tt[3]
        else:
            raise ValueError(f'Unkown flag of{flag}.  It has to be cv|final_train|forward_forecast')

        # weather dim batch, height, width, channel --> batch, channel, height, width
        t_flag = (pd.DataFrame(timestamp) >= pd.Timestamp(t0)) & (pd.DataFrame(timestamp) <= pd.Timestamp(t1))
        self._target = self._target[t_flag]
        self._ext = self._ext[t_flag]
        self._wea_arr = self._wea_arr[t_flag, ...]       

        fs = [self._config.model_pdt.final_train_frac, self._config.model_pdt.final_train_frac_yr1]\
              if self._flag.startswith('final_train') else \
        [self._config.model_pdt.frac_yr1, self._config.model_pdt.frac_split ]
        first_yr = fs[0]
        frac = fs[1]

        full_length = self._target.shape[0] - self._seq_length - self._pred_length +1 
        train_iter, test_iter, val_iter = self._config.get_sample_segmentation_borders(full_length=full_length, 
                                                     fst_scenario=self._sce_ind,
                                                     first_yr_frac=first_yr,
                                                     fractions=frac)
        
        if 'test' in self._flag:
            self._sample_iter = train_iter
        elif 'val' in self._flag:
            self._sample_iter = test_iter
        else: 
            self._sample_iter = val_iter

        self._sample_list = list(self._sample_iter)

    def __len__(self):
        """
        Denotes the total number of samples

        Returns: sample number

        """
 
        return len(self._sample_iter)


    def __getitem__(self, index) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Generates one sample of data

        Args:
            index: an int

        Returns: weather, tabular(calendar), target-AR, target

        """
        index = self._sample_list[index]

        target_ind0 = index + self._seq_length 
        target_ind1 = target_ind0 + self._pred_length
        wea_ind0 = target_ind0 - self._wea_embedding_dim
        wea_ind1 = target_ind1 
        ext_ind0 = target_ind0 - self._ext_embedding_dim
        ext_ind1 = target_ind1 
        ar_ind0 = index
        ar_ind1 = index + self._seq_length
        return (
            self._wea_arr[wea_ind0:wea_ind1, ...],
            self._ext[ext_ind0:ext_ind1, :],
            self._target[ar_ind0:ar_ind1, :],
            self._target[target_ind0:target_ind1,:]
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

def check_fix_missings(load_arr:np.ndarray, w_timestamp:np.ndarray, w_arr:np.ndarray)->Tuple[np.ndarray, np.ndarray]:
    # sync data, fill missings
    w_timestamp = pd.DatetimeIndex(list(w_timestamp)).tz_localize('UTC')
    t_str = 'timestamp'
    t0 = max(load_arr[0][0].tz_convert('UTC'), w_timestamp[0])
    t1 = min(load_arr[-1][0].tz_convert('UTC'), w_timestamp[-1])
    load_arr = load_arr[np.logical_and(load_arr[:,0] >= t0, load_arr[:,0] <=t1)]
    w_arr =  w_arr[np.logical_and(w_timestamp.to_numpy()>=t0,  w_timestamp.to_numpy()<=t1)]
    t = pd.DataFrame(pd.date_range(start=t0, end=t1, freq='h'), columns=[t_str])
    df_load = pd.DataFrame(load_arr).set_index(0)
    df_tl = t.set_index(t_str).join(df_load, how='left')
    #df_tl.interpolate(inplace=True)
    df_tl.fillna(method='ffill', inplace=True)
    df_w = pd.DataFrame(w_timestamp, columns=[t_str]).set_index(t_str)
    df_w['value'] = 1
    df_tw = t.set_index(t_str).join(df_w, how='left' ) 

    # the first hour will not be empty, and we will do a ffill
    for i in range(0, len(df_tw)):
        if math.isnan(df_tw.iloc[i]['value']):
            w_arr = np.insert(w_arr, [i], w_arr[i], axis=0)

    return df_tl.values.astype(float), w_arr.astype(float), t

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


# class WeatherDataSetBuilder:
#     """
#     The lag_load is queried by lag 1 for training data, essentially shifted by one from target.
#     For predictions, only the load column is used.
#     """

#     def __init__(
#         self,
#         weather: np.ndarray,
#         lag_load: np.ndarray = [],
#         calendar_data: np.ndarray = None,
#         y_labels: np.ndarray = None,
#         cat_fraction=None,
#     ):
#         """
#         Initialization. The original data has lag 1 to lag 168 by default.

#         Args:
#             weather: np.ndarray for sample * x * y * channel
#             lag_load: lagged load; if none, use y_label shifted by one
#             calendar_data: 2d npy data array
#             y_labels:
#             cat_fraction: dict {'train':0.8, 'test':0.1, 'validation':0.1}
#             #hours_ahead: forecast starting, 1 hour, 7 hours, 25 hours, 49 hours etc
#         """
#         self._weather = weather.squeeze()
#         self._hours_ahead = None
#         y_labels = y_labels.squeeze()
#         if len(y_labels.shape) < 2:
#             y_labels = y_labels.reshape((-1, 1))
#         self._y_labels = y_labels
#         self._calendar_data = calendar_data.squeeze()
#         if len(lag_load) == 0:
#             lag_load = np.roll(y_labels, 1, axis=0)
#         self._lag_load = lag_load.squeeze()
#         if isinstance(cat_fraction, List):
#             cat_fraction = {
#                 "train": cat_fraction[0],
#                 "test": cat_fraction[1],
#                 "validation": cat_fraction[2],
#             }
#         self._cat_fraction = (
#             {"train": 0.75, "test": 0.1, "validation": 0.15}
#             if not cat_fraction
#             else cat_fraction
#         )
#         assert np.isclose(1, sum(self._cat_fraction.values())), \
#             "sum of train, test, validation fractions should be 1!"
#         self._cat_index = {"train": None, "test": None, "validation": None}
#         self._cat = "train"
#         assert (
#             weather.shape[0] == calendar_data.shape[0]
#         ), "npy file and calendar sizes not match"
#         assert len(calendar_data) == len(
#             y_labels
#         ), "calendar data and y_labels sizes not match"
#         self._all_sample_index = range(self._y_labels.shape[0])

#     def extract_data(self, cat: str, fst_hours: int) -> WeatherDataSet:
#         """
#         Set up the mode to get data, train|test|validation, fst hours ahead

#         Args:
#             cat: train, test, or validation; if train==1, return the full dataset for training, and None, None for val, test
#             fst_hours: hours ahead to forecast, be between 0.01 and 169
#         """
#         assert isinstance(fst_hours, int)
#         max_fst_hours = 169
#         min_fst_hours = 0.1
#         assert max_fst_hours > fst_hours >= min_fst_hours
#         self._hours_ahead = fst_hours
#         assert cat in self._cat_fraction
#         self._cat = cat
#         if self._cat_fraction["train"] == 1:
#             return WeatherDataSet(
#                 weather=self._weather,
#                 lag_load=np.roll(self._lag_load, self._hours_ahead - 1, axis=0),
#                 calendar_data=self._calendar_data,
#                 target=self._y_labels,
#             )

#         tmp, self._cat_index["validation"] = train_test_split(
#             self._all_sample_index, test_size=self._cat_fraction.get("validation")
#         )
#         self._cat_index["train"], self._cat_index["test"] = train_test_split(
#             tmp,
#             train_size=self._cat_fraction.get("train")
#             / (1 - self._cat_fraction.get("validation")),
#         )
#         return WeatherDataSet(
#             weather=self.weather,
#             lag_load=self.lag_loads,
#             calendar_data=self.calendar,
#             target=self.target,
#         )

#     @property
#     def weather(self):
#         return self._weather[self._cat_index[self._cat], :, :, :]

#     @property
#     def target(self):
#         return self._y_labels[self._cat_index[self._cat], :]

#     @property
#     def lag_loads(self):
#         # lag_load already lagged by one
#         tmp_loads = np.roll(self._lag_load, self._hours_ahead - 1, axis=0)
#         return tmp_loads[self._cat_index[self._cat], :]

#     @property
#     def calendar(self):
#         return self._calendar_data[self._cat_index[self._cat], :]

#     def get_weather_para(self):
#         w = list(self._weather.shape[1:])
#         w.append(self._lag_load.shape[1])
#         w.append(self._calendar_data.shape[1])
#         return WeatherPara(*w)
