import numpy as np
import torch
import torch.utils.data as data

from pytools.config import Config
from pytools.modeling.dataset import WeatherDataSet, check_fix_missings, read_weather_data_from_config
from pytools.modeling.ts_weather_net import MixedOutput, TSWeatherNet, WeaCov
from pytools.modeling.weather_net import WeatherLayer


def test_construct_weathernet(config:Config):
    load_data, w_paras, w_timestamp, w_data = read_weather_data_from_config(config, year=-1)

    assert load_data.shape[1]>1
    assert w_data.shape == (49,21,21,16)
    assert w_timestamp.shape[0] == 49
   
    load_arr, wea_arr, t = check_fix_missings(load_arr=load_data, w_timestamp=w_timestamp, w_arr=w_data)

    # wds = WeatherDataSet(flag='cv_train',tabular_data=load_arr, wea_arr=wea_arr, timestamp=t, config=config, sce_ind=0, )
    wds1 = WeatherDataSet(flag='final_train',tabular_data=load_arr, wea_arr=wea_arr, timestamp=t, config=config, sce_ind=0,)

    sample = wds1.__getitem__(1)
    assert len(sample)  == 6

    def to_np(x):
        # add batch dimesion
        return torch.from_numpy(x.astype(np.float32))[None,...]
    sample = map(to_np, sample)
    [seq_wea_arr, seq_ext_arr, seq_arr, wea_arr, ext_arr, target] = sample

    wea_input_shape = list(seq_wea_arr.shape)
    
    #wea_input_shape = [20, 12, 8, 8, 16] # B, seq, x, y, channel

    m = TSWeatherNet(wea_arr_shape=wea_input_shape, config=config)
    y = m.forward(seq_wea_arr=seq_wea_arr, seq_ext_arr=seq_ext_arr, seq_target=seq_arr, wea_arr=wea_arr, ext_arr=ext_arr)
    assert y.shape == target.shape




    # for i, sample in enumerate(wds1):
    #     sample
    #     print(sample)

    # for name, p in wds1.named_parameters():
    #     print(name, p.shape)
    # wea_arr[1:5, 0, ...].squeeze()


def test_mixed_output(config:Config):
    pred_len=23

    m = MixedOutput(seq_arr_dim=8, filternet_hidden_size=5, ext_dim=4, wea_arr_dim=8, pred_len=pred_len, model_paras=config.model_pdt.mixed_net)
    seq_arr = torch.rand(20, 8, 5)
    ext_arr = torch.rand(20, pred_len, 4)
    wea_arr = torch.rand(20, pred_len, 8)
    y = m.forward(seq_arr=seq_arr, ext_arr=ext_arr, wea_arr=wea_arr)
    assert list(y.shape) == [20, 23]

def test_ts_weather_net(config:Config):
    # [batch, x, y, wea_para]
    input_shapes = [[20, 8, 8, 10], [20, 5, 5, 10],[20, 2, 1, 10]]
    for input_shape in input_shapes:
        w=WeaCov(input_shape=input_shape, layer_paras=config.model_pdt.cov_net)
        w_arr = torch.rand(input_shape)
        y = w.forward(wea_arr=w_arr)
        assert y.shape[0] == input_shape[0]
        assert list(y.shape)[1] == w.output_shape


def test_ts_TSWeather(config:Config):
    input_shape = [20, 12, 8, 8, 10]
    fst_hz = config.model_pdt.forecast_horizon

    w = TSWeatherNet(
        wea_arr_shape=input_shape, 
        wea_layer_paras=config.model_pdt.cov_net, 
        lstm_layer_paras=config.model_pdt.lstm_net, 
        pred_length=fst_hz[0][1] -fst_hz[0][0],
        )
    for name, p in w.named_parameters():
        if 'multi_linear' in name:
            print(name, p.shape)


def test_weather_net_train_a_minibatch(config):
    load_data, w_paras, w_timestamp, w_data = read_weather_data_from_config(config, year=-1)
    load_arr, wea_arr, t = check_fix_missings(load_arr=load_data, w_timestamp=w_timestamp, w_arr=w_data)
    wds = WeatherDataSet(flag='final_train',tabular_data=load_arr, wea_arr=wea_arr, timestamp=t, config=config, sce_ind=0,)

    
    train_loader = data.DataLoader(train_set, batch_size=256, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    val_loader = data.DataLoader(val_set, batch_size=256, shuffle=False, drop_last=False, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=256, shuffle=False, drop_last=False, num_workers=4)


    wea_input_shape = wea_arr.shape
    m = TSWeatherNet(wea_arr_shape=wea_input_shape, config=config)
    train_loader = data.DataLoader(dataset=wds,)




