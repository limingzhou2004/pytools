import pytest
import torch
#from torch.utils import data
import numpy as np

from pytools.modeling.dataset import WeatherDataSetBuilder, check_fix_missings, read_weather_data_from_config


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


def test_build_from_config(config):
    load_data, w_paras, w_timestamp, w_data = read_weather_data_from_config(config, year=-1)

    assert load_data.shape[1]>1
    assert w_data.shape == (49,21,21,16)

    check_fix_missings(df_load=load_data, w_timestamp=w_timestamp, w_arr=w_data)


@pytest.mark.parametrize(
    "para, wea_shape, lag_load_shape",
    [
        ("train 1", (75, 34, 34, 13), ()),
    ],
)
def test_dataset(para, wea_shape, lag_load_shape):
    p = para.split()
    cat = p[0]
    fst_hour = int(p[1])
    w = WeatherDataSetBuilder(
        weather=wea, lag_load=embed_load, calendar_data=calendar, y_labels=target
    )
    w.extract_data(cat=cat, fst_hours=fst_hour)
    assert w.weather.shape == wea_shape
    assert np.allclose(w._lag_load[1:, 0], w._y_labels[:-1, 0])
