import pytest
import torch
#from torch.utils import data
import numpy as np
import pandas as pd

from pytools.modeling.dataset import WeatherDataSet, check_fix_missings, create_rolling_fst_data, get_hourly_fst_data, read_weather_data_from_config, read_past_fst_weather


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# torch.backends.cudnn.benchmark = True

# Parameters
params = {"batch_size": 10, "shuffle": True}  # , "num_workers": 1}
max_epochs = 100

# Datasets
labels = np.arange(0, 100, 1)  # Labels


# Generators
sample_size = 100
wea = torch.randn(sample_size, 34, 34, 13, dtype=torch.float).numpy()
embed_load = torch.randn(sample_size, 168, dtype=torch.float).numpy()
calendar = torch.randn(sample_size, 7, dtype=torch.float).numpy()
target = np.roll(embed_load[:, 0], shift=-1, axis=0)


def test_train_test_val(config):

    assert 1==1


def test_build_from_config(config):
    load_data, w_paras, w_timestamp, w_data = read_weather_data_from_config(config, year=-1)

    assert load_data.shape[1]>1
    assert w_data.shape == (49,21,21,16)
    assert w_timestamp.shape[0] == 49
    w_timestamp[6] = np.nan
    load_arr, wea_arr, t = check_fix_missings(load_arr=load_data, w_timestamp=w_timestamp, w_arr=w_data)
    assert np.allclose(wea_arr[3], wea_arr[2])
    wds, val, test = WeatherDataSet(flag='cv_train',tabular_data=load_arr, wea_arr=wea_arr, timestamp=t, config=config, sce_ind=0)
    # wds1, val, test = WeatherDataSet(flag='final_train',tabular_data=load_arr, wea_arr=wea_arr, timestamp=t, config=config, sce_ind=0)
    

def test_read_past_fst_weather(config):
    dat =read_past_fst_weather(config, year=2029)



    assert 1==1


def test_create_rolling_fst_data(config,):
    cur_t_str = '2020-04-01'
    cur_t = pd.Timestamp(cur_t_str,tz='US/Eastern')
    t_load = pd.date_range('2020-03-01', '2020-05-01',tz='US/Eastern',freq='h')
    data = np.random.rand(len(t_load), 9)
    df_load = pd.DataFrame(data)
    df_load.index =t_load
    t_wea_list = []
    wea_arr_list = []
    rolling_forecast_horizeon=23
    for i in range(rolling_forecast_horizeon):
        if i==3:
            continue
        else:
            t_wea_list.append(pd.Timestamp(cur_t_str,tz='UTC')+pd.Timedelta(i,'h'))
            wea_arr_list.append(np.random.rand(21,21,14))
    seq_length=168
    # seq_wea_arr, seq_ext_arr, seq_arr, wea_arr, ext_arr, target 
    df_tab, wet_arr = create_rolling_fst_data(
        load_data=df_load, cur_t=cur_t, wea_data=wea_arr_list,rolling_fst_horizon=rolling_forecast_horizeon+2, w_timestamp=t_wea_list, default_seq_length=seq_length)

    assert df_tab.shape[0]==193
    assert wet_arr.shape[0]==193

    res = get_hourly_fst_data(target_arr=df_tab.loc[:,0].values, ext_arr=df_tab.loc[:,1:].values, wea_arr=wet_arr, hr=1, seq_length=seq_length)
    res = get_hourly_fst_data(target_arr=df_tab.loc[:,0].values, ext_arr=df_tab.loc[:,1:].values, wea_arr=wet_arr, hr=2, seq_length=seq_length)
    
    assert len(res) == 6

# @pytest.mark.parametrize(
#     "para, wea_shape, lag_load_shape",
#     [
#         ("train 1", (75, 34, 34, 13), ()),
#     ],
# )
# def test_dataset(para, wea_shape, lag_load_shape):
#     p = para.split()
#     cat = p[0]
#     fst_hour = int(p[1])
#     w = WeatherDataSetBuilder(
#         weather=wea, lag_load=embed_load, calendar_data=calendar, y_labels=target
#     )
#     w.extract_data(cat=cat, fst_hours=fst_hour)
#     assert w.weather.shape == wea_shape
#     assert np.allclose(w._lag_load[1:, 0], w._y_labels[:-1, 0])

