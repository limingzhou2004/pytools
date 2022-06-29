import pytest
import torch
from torch.optim import adam

# from torch.utils import data
# import numpy as np

from pytools.modeling.weather_net import (
    WeatherLayer,
    EnumOptimizer,
    WeatherPara,
    WeatherNet,
)
from pytools.modeling.weather_net import ModelSettings, default_layer_sizes


@pytest.fixture()
def hrrr_weather_para():
    return WeatherPara(34, 34, 11, 168, 10)


@pytest.fixture()
def model_setting():
    return ModelSettings(
        device="cpu",
        batch_size=100,
        epoch_num=10,
        optimizer=EnumOptimizer.adam,
        learning_rate=0.001,
        seed=0,
        dropout=0.01,
        log_interval=1,
    )


@pytest.fixture()
def generator_data(hrrr_weather_para):
    def generator_data_fun(
        batch_size, hrrr: WeatherPara = hrrr_weather_para, max_count=10
    ):

        wea = torch.randn(
            batch_size, hrrr.channel, hrrr.x_dim, hrrr.y_dim, dtype=torch.float
        )
        embed_load = torch.randn(batch_size, 1, hrrr.embed_load_dim, dtype=torch.float)
        calendar = torch.randn(batch_size, hrrr.calendar_dim, dtype=torch.float)
        target = torch.randn(batch_size, dtype=torch.float)
        for _ in range(max_count):
            yield wea, embed_load, calendar, target

    return generator_data_fun


def test_weather_layer(generator_data, hrrr_weather_para, model_setting):
    nn = WeatherLayer(
        weather_para=hrrr_weather_para,
        layer_paras=default_layer_sizes,
        model_settings=model_setting,
    )

    for wea, embed_load, calendar, target in generator_data(
        model_setting.batch_size, max_count=1
    ):
        ypre = nn.forward(
            tensor_wea=wea, tensor_load_embed=embed_load, tensor_calendar=calendar
        )
        assert len(ypre.shape) == 2
        assert ypre.shape[0] == model_setting.batch_size


def test_weather_net_train_a_minibatch(
    generator_data, hrrr_weather_para, model_setting
):
    nn = WeatherNet(
        model_file_path="resources",
        model_file_name="test",
        hrs_ahead=1,
        weather_para=hrrr_weather_para,
        layer_paras=default_layer_sizes,
        model_settings=model_setting,
    )
    for wea, embed_load, calendar, target in generator_data(
        model_setting.batch_size, max_count=5
    ):
        loss = nn.training_step((wea, embed_load, calendar, target), None)
        # print(nn.model.weather_conv2_layer[0].weight.grad[0,0])
        print(loss)
        assert loss["loss"] >= 0
        loss_pre = nn.test_step((wea, embed_load, calendar, target), None)
        assert loss_pre["test_loss"] >= 0
        y_pre = nn(wea, embed_load, calendar)
        assert y_pre.shape == (100, 1)
