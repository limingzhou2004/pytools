import numpy as np
import torch


from pytools.config import Config
from pytools.modeling.dataset import WeatherDataSet, check_fix_missings, read_weather_data_from_config
from pytools.modeling.ts_weather_net import TSWeatherNet, WeaCov
from pytools.modeling.weather_net import WeatherLayer


def test_construct_weathernet(config:Config):
    load_data, w_paras, w_timestamp, w_data = read_weather_data_from_config(config, year=-1)

    assert load_data.shape[1]>1
    assert w_data.shape == (49,21,21,16)
    assert w_timestamp.shape[0] == 49
   
    load_arr, wea_arr, t = check_fix_missings(load_arr=load_data, w_timestamp=w_timestamp, w_arr=w_data)

    # wds = WeatherDataSet(flag='cv',tabular_data=load_arr, wea_arr=wea_arr, timestamp=t, config=config, sce_ind=0, fst_horizon_ind=0)
    wds1 = WeatherDataSet(flag='final_train',tabular_data=load_arr, wea_arr=wea_arr, timestamp=t, config=config, sce_ind=0, fst_horizon_ind=1)
    for name, p in wds1.named_parameters():
        print(name, p.shape)
    # wea_arr[1:5, 0, ...].squeeze()

def test_ts_weather_net(config:Config):
    # [batch, x, y, wea_para]
    input_shapes = [[20, 8, 8, 10], [20, 5, 5, 10],[20, 2, 1, 10]]
    for input_shape in input_shapes:
        w=WeaCov(input_shape=input_shape, layer_paras=config.model_pdt.cov_layer)
        w_arr = torch.rand(input_shape)
        y = w.forward(wea_arr=w_arr)
        assert y.shape[0] == input_shape[0]
        assert list(y.shape) == w.output_shape


def test_ts_TSWeather(config:Config):
    input_shape = [20, 12, 8, 8, 10]

    w = TSWeatherNet(
        wea_arr_shape=input_shape, 
        wea_layer_paras=config.model_pdt.cov_layer, 
        ts_layer_paras=config.model_pdt.lstm_layer, 
        pred_length=,
        )
    for name, p in w.named_parameters():
        if 'multi_linear' in name:
            print(name, p.shape)




