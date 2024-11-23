import torch.nn as nn
import torch.nn.functional as F

from pytools.modeling.weather_net import WeatherNet


class WeaCov(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, wea_arr):

        return
    

class TSWeatherNet(WeatherNet):

    def __init__(self, *args, **kwargs):
        super().super().__init__(*args, **kwargs)(self,)

        return