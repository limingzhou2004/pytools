import numpy as np
import torch


from pytools.config import Config
from pytools.modeling.dataset import WeatherDataSet, check_fix_missings, read_weather_data_from_config
from pytools.modeling.ts_weather_net import WeaCov
from pytools.modeling.weather_net import WeatherLayer


def test_construct_weathernet(config:Config):
    load_data, w_paras, w_timestamp, w_data = read_weather_data_from_config(config, year=-1)

    assert load_data.shape[1]>1
    assert w_data.shape == (49,21,21,16)
    assert w_timestamp.shape[0] == 49
   
    load_arr, wea_arr, t = check_fix_missings(load_arr=load_data, w_timestamp=w_timestamp, w_arr=w_data)

    # wds = WeatherDataSet(flag='cv',tabular_data=load_arr, wea_arr=wea_arr, timestamp=t, config=config, sce_ind=0, fst_horizon_ind=0)
    wds1 = WeatherDataSet(flag='final_train',tabular_data=load_arr, wea_arr=wea_arr, timestamp=t, config=config, sce_ind=0, fst_horizon_ind=1)

    w=WeaCov(input_shape=[] ,layer_paras=config.model_pdt.cov_layer)
    w.forward(wea_arr=wea_arr[1:5, 0, ...].squeeze())

    x = torch.rand(3,2,2)

    assert 1==1