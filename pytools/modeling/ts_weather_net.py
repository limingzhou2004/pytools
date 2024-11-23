import torch.nn as nn
import torch.nn.functional as F

from pytools.modeling.weather_net import WeatherNet


class WeaCov(nn.Module):

    def __init__(self, input_shape, layer_paras):
        # input_shape, (batch, x, y, channel)
        super().__init__()
        self.weather_conv1_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=weather_para.channel,
                out_channels=layer_paras["w_out1_channel"],
                kernel_size=layer_paras["w_kernel1"],
                stride=layer_paras["w_stride1"],
                padding=layer_paras["w_padding_1"],
            ),
            nn.LayerNorm(num_features=layer_paras["w_out1_channel"]),
            nn.ReLU(),
            # nn.Dropout2d(p=model_settings.dropout),
        )
        self.weather_conv2_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=layer_paras["w_out1_channel"],
                out_channels=layer_paras["w_out2_channel"],
                kernel_size=layer_paras["w_kernel2"],
                stride=layer_paras["w_stride2"],
                padding=layer_paras["w_padding_2"],
            ),
            nn.LayerNorm(num_features=layer_paras["w_out2_channel"]),
            nn.ReLU(),
            # nn.Dropout2d(p=model_settings.dropout),
        )

    def forward(self, wea_arr):

        return
    

class TSWeatherNet(WeatherNet):

    def __init__(self, *args, **kwargs):
        super().super().__init__(*args, **kwargs)

        return