import math
from pathlib import Path
import pickle
from typing import Tuple, List

import numpy as np
import pandas as pd
from scipy import interpolate

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
from torch import Tensor
from torch.utils import data

from pytools.config import Config
from pytools.config import DataType
from pytools.modeling.scaler import Scaler, load
#from pytools.modeling.weather_net import WeatherPara

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)


def _get_dt_range(t0, t1, t):
    t1 = pd.Timestamp(t1, tz='UTC') 
    t0 = pd.Timestamp(t0,tz='UTC') 
    dt=t1-t0
    dtn= dt/ np.timedelta64(1, 'h')
    t_new = t[(t0<=t) & (t<=t1)]
    return range(int(dtn))
    #return range(len(t_new))

def create_datasets(config:Config, flag, tabular_data, wea_arr, timestamp, sce_ind):
    target_ind = config.model_pdt.target_ind

    _target = tabular_data[:, target_ind]
    _ext = np.delete(tabular_data, target_ind, axis=1)
    _wea_arr = wea_arr   

    _fn_scaler = config.get_model_file_name(class_name='scaler')
 
    if Path(_fn_scaler).exists():
        scaler = load(_fn_scaler)
    else:
        scaler = Scaler(_target, _wea_arr, scaler_type=config.model_pdt.scaler_type)
        scaler.save(_fn_scaler)

    target_mean = _target.mean()
    target_std = _target.std()
    _target = scaler.scale_target(_target)
    _wea_arr = scaler.scale_arr([_wea_arr])[0]   
    tt = config.model_pdt.cv_settings[sce_ind]

    # weather dim batch, height, width, channel --> batch, channel, height, width
    t_flag = (timestamp >= pd.Timestamp(tt[0],tz='UTC')) & (timestamp<= pd.Timestamp(tt[-1],tz='UTC'))
    t_flag = t_flag.to_numpy().reshape((-1))
    _target = _target[t_flag]
    _ext = _ext[t_flag]
    _wea_arr = _wea_arr[t_flag, ...]   

    train_range = _get_dt_range(tt[0], tt[1], timestamp[t_flag])
    fst_ind = 0
    pre_length = config.model_pdt.forecast_horizon[fst_ind][-1]
    full_length = _target.shape[0] - config.model_pdt.seq_length
    full_length -= pre_length
    train_size = len(train_range)
    val_range = None
    test_range = None

    if flag.startswith('cv'):
        test_range = range(train_range[-1], full_length)
        if config.model_pdt.train_frac >= 1:
            val_range = None
        else:
            # train_range, val_range = train_test_split(train_range, train_size=int(config.model_pdt.train_frac*train_size))
            train_len = len(train_range)
            new_train_len = int(train_len * config.model_pdt.train_frac)
            train_range = range(new_train_len)
            val_range = range(new_train_len,train_len)




    # elif flag.startswith('final_'):
    #     return train_range
    # return train, val, test datasets

    train_ds =  WeatherDataSet(train_range, config=config, scaler=scaler,target_mean=target_mean, 
                               target_std=target_std, target=_target, wea_arr=_wea_arr, ext_arr=_ext) 
    val_ds =  WeatherDataSet(val_range, config=config, scaler=scaler,target_mean=target_mean, 
                             target_std=target_std, target=_target, wea_arr=_wea_arr, ext_arr=_ext) if val_range else None      
    test_ds = WeatherDataSet(test_range, config=config, scaler=scaler,target_mean=target_mean,
                             target_std=target_std, target=_target, wea_arr=_wea_arr, ext_arr=_ext) if test_range else None  
    return train_ds, val_ds, test_ds


class WeatherDataSet(data.Dataset):

    def __init__(self, sample_iter, scaler, target_mean, target_std, target, wea_arr, ext_arr,config:Config):
        self.scaler = scaler
        self.target_mean = target_mean
        self.target_std = target_std 
        self._target = target 
        self._wea_arr = wea_arr 
        self._ext = ext_arr
        self._sample_iter = sample_iter
        seq_length = config.model_pdt.seq_length
        fst_horizon = config.model_pdt.forecast_horizon[0]
        self._seq_length = seq_length
        self._pred_length = fst_horizon[-1]
        self._fst_horizeon = fst_horizon
        self._wea_ar_embedding_dim = config.model_pdt.wea_ar_embedding_dim
        self._wea_embedding_dim = config.model_pdt.wea_ar_embedding_dim
        self._ext_embedding_dim = config.model_pdt.ext_ar_embedding_dim
        self._sample_list = list(sample_iter)

    def __init00__(
        self,
        flag:str,
        tabular_data: np.ndarray,
        wea_arr: np.ndarray,
        timestamp: np.ndarray,
        config: Config,
        sce_ind: int, # cv scenario
        to_scale:bool = True,
    ):
        # the scaler and model file has flag and year information

        self._config = config
        self._flag = flag
        target_ind = config.model_pdt.target_ind
        seq_length = config.model_pdt.seq_length
        fst_horizon = config.model_pdt.forecast_horizon[0]



        if to_scale:
            if Path(self._fn_scaler).exists():
                scaler = load(self._fn_scaler)
            else:
                scaler = Scaler(self._target, self._wea_arr, scaler_type=config.model_pdt.scaler_type)
                scaler.save(self._fn_scaler)

            self.target_mean = self._target.mean()
            self.target_std = self._target.std()
            self._target = scaler.scale_target(self._target)
            self._wea_arr = scaler.scale_arr([self._wea_arr])[0]
            self.scaler = scaler
      
        # load the data, and selec the subset, based on flag, ind
        # filter by t0 and t1

        if flag.startswith('cv'):
            tt = config.model_pdt.cv_settings[sce_ind]
            t0 = tt[0]
            t1 = tt[1]
            self._flag = self._flag.split('_')[-1]
        elif flag.startswith('final_train'):
            tt = config.model_pdt.final_train_hist[sce_ind]
            t0 = tt[0]
            t1 = tt[1]
            self._flag = self._flag.split('_')[-1]
        elif flag.startswith('forward_forecast'):
            tt = config.model_pdt.final_train_hist[sce_ind]
            t0 = tt[2]
            t1 = tt[3]
        else:
            raise ValueError(f'Unkown flag of{flag}.  It has to be cv|final_train|forward_forecast')

        # weather dim batch, height, width, channel --> batch, channel, height, width
        t_flag = (timestamp >= pd.Timestamp(t0,tz='UTC')) & (timestamp<= pd.Timestamp(t1,tz='UTC'))
        t_flag = t_flag.to_numpy().reshape((-1))
        self._target = self._target[t_flag]
        self._ext = self._ext[t_flag]
        self._wea_arr = self._wea_arr[t_flag, ...]       

        fs = [self._config.model_pdt.final_train_frac_yr1, self._config.model_pdt.final_train_frac]\
              if self._flag.startswith('final_train') else \
        [self._config.model_pdt.frac_yr1, self._config.model_pdt.frac_split ]
        first_yr = fs[0]
        frac = fs[1]
        full_length = self._target.shape[0] - self._seq_length - self._pred_length +1 

        train_iter, test_iter, val_iter = self._config.get_sample_segmentation_borders(full_length=full_length, 
        fst_scenario=self._sce_ind,
        first_yr_frac=first_yr,
        fractions=frac)
        
        if 'train' in self._flag:
            self._sample_iter = train_iter
        elif 'test' in self._flag:
            self._sample_iter = test_iter
        elif 'val' in self._flag: 
            self._sample_iter = val_iter
        else:
            raise ValueError('flag must be ened as _train|_test|_val')

        self._sample_list = list(self._sample_iter)

    def __len__(self):
        """
        Denotes the total number of samples

        Returns: sample number

        """
 
        return len(self._sample_list)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Generates one sample of data [seq_wea, seq_ext, seq_target, wea_arr, ext_arr, target]

        Args:
            index: an int

        Returns: seq weather, seq ext vars, seq target, weather array, 
        tabular(calendar), target

        """
        index = self._sample_list[index]
        target_ind0 = index + self._seq_length 
        
        target_ind1 = target_ind0 + self._pred_length
        wea_ind0 = target_ind0 - self._wea_embedding_dim + self._fst_horizeon[0] - 1
        wea_ind1 = target_ind1 
        ext_ind0 = target_ind0 - self._ext_embedding_dim + self._fst_horizeon[0] - 1
        ext_ind1 = target_ind1 
        ar_ind0 = index
        ar_ind1 = index + self._seq_length
        target_ind0 += self._fst_horizeon[0] - 1
        # print(f'ar_ind0={ar_ind0},ar_ind1={ar_ind1}, target_ind0={target_ind0},target_ind1={target_ind1}')
        return (
            torch.tensor(self._wea_arr[ar_ind0:ar_ind1, ...]),
            torch.tensor(self._ext[ar_ind0:ar_ind1, :]),
            torch.tensor(self._target[ar_ind0:ar_ind1,...]),
            torch.tensor(self._wea_arr[wea_ind0:wea_ind1, ...]),
            torch.tensor(self._ext[ext_ind0:ext_ind1, :]),
            torch.tensor(self._target[target_ind0:target_ind1,...])
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

def check_fix_missings(load_arr:np.ndarray, w_timestamp:np.ndarray, w_arr:np.ndarray,month=-1)->Tuple[np.ndarray, np.ndarray]:
    # sync data, fill missings
    #w_timestamp = pd.DatetimeIndex(list(w_timestamp)).tz_localize('UTC')
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

    ret_df = df_tl.values.astype(float)
    ret_w = w_arr.astype(float)
    if month==-1:
        return ret_df, ret_w, t
    else:        
        mths=[month, month+1 if month+1<=12 else 1, month-1 if month-1>0 else 12]
        ind_months = t[t_str].apply(lambda x: x.month in mths)
        return ret_df[ind_months,...], ret_w[ind_months], t[ind_months]


def read_weather_data_from_config(config:Config, year=-1):
    # load data are not separated by year
    fn_load = config.get_load_data_full_fn(DataType.LoadData, 'npz', year=-1)
    fn_wea =  config.get_load_data_full_fn(DataType.Hist_weatherData, 'npz', year=year)
    
    with np.load(fn_load) as dat:
        load_data = dat[DataType.LoadData.name]
    with np.load(fn_wea) as dat:
        paras = dat['paras']
        w_timestamp = dat['timestamp']
        w_data = dat[DataType.Hist_weatherData.name]

    return load_data, paras, w_timestamp, w_data, 


def read_past_weather_data_from_config(config:Config, year=-1):
    fn_load = config.get_load_data_full_fn(DataType.LoadData, 'npz', year=-1)
    fn_wea =  config.get_load_data_full_fn(DataType.Past_fst_weatherData, 'pkl', year=year)
    with np.load(fn_load) as dat:
        load_data = dat[DataType.LoadData.name]
    with open(fn_wea, 'rb') as fr:
        wea_dat = pickle.load(fr)

    # load_data, :, 10, timestamp, load, ...
    # wea_data= [spot timestamp], [fst timestamp-[1][0][0:] , arr-[1][1][0:]]

    return load_data, wea_dat


def create_rolling_fst_data(load_data:np.ndarray, cur_t:pd.Timestamp, 
                            w_timestamp, wea_data:np.ndarray, 
                            rolling_fst_horizon:int =48, 
                            config:Config=None, fst_ind=0,
                            default_seq_length = 168):
    # returns seq_wea_arr, seq_ext_arr, seq_arr, wea_arr, ext_arr, target
    # need to use local timezone
    # tz = cur_t.timetz
    #default_fst_horizon = 1

    if config:
       # fst_horizon = config.model_pdt.forecast_horizon[fst_ind][1], 
        seq_length = config.model_pdt.seq_length
    else:
        #fst_horizon = default_fst_horizon
        seq_length = default_seq_length
    
    # necessary hist time range for load, fill missing 
    t0 = cur_t - pd.Timedelta(seq_length-1,'h')
    t1 = cur_t + pd.Timedelta(rolling_fst_horizon, 'h')
    dft = pd.date_range(t0, t1, freq='h')
    df = pd.DataFrame()
    df.index=dft
    df=df.join(load_data, how='left')
    df=df.ffill()

    wet_arr = np.zeros((rolling_fst_horizon, *wea_data[0].shape)) + np.nan

    #w_timestamp_local = pd.DatetimeIndex(list(w_timestamp)).tz_localize('UTC')
    w_timestamp_local = list(w_timestamp) #list(w_timestamp_local)
    # necessary weather range for weather, fill missing
    valid_inds = []
    for h in range(rolling_fst_horizon):
        tp = cur_t + pd.Timedelta(h+1, 'h') 
        if tp in w_timestamp_local:
            ind = w_timestamp_local.index(tp)
            wet_arr[h, ...] = wea_data[ind]
            valid_inds.append(h)
    nan_inds = set(range(rolling_fst_horizon)) - set(valid_inds)
    wea_data = np.stack(wea_data,axis=0)
    tmp = interpolate.interp1d(np.array(valid_inds), wea_data,axis=0,fill_value='extrapolate')
    for i in nan_inds:
        wet_arr[i,...] = tmp(np.array(i))

    return df, wet_arr


def get_hourly_fst_data(target_arr, ext_arr, wea_arr, hr, seq_length):

    #seq_wea_arr, seq_ext_arr, seq_arr, wea_arr, ext_arr, target
    seq_w_shape = list(wea_arr.shape)
    seq_w_shape[0] = seq_length
    seq_wea_arr = np.zeros(seq_w_shape) #wea_arr[hr-1:seq_length+hr-1,...]
    seq_ext_arr = ext_arr[hr-1:seq_length+hr-1,...]
    ext_arr = ext_arr[seq_length+hr-1:seq_length+hr,  ...]
    seq_target = target_arr[hr-1:seq_length+hr-1,...]
    wea_arr = wea_arr[hr-1:hr, ...]

    res=[ seq_wea_arr,seq_ext_arr, seq_target, wea_arr, ext_arr, target_arr[seq_length+hr-1,...]]
    
    return [torch.from_numpy(x.astype(np.float32))[None,...] for x in res ]

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
