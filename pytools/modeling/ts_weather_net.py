import torch
import torch.nn as nn
import torch.nn.functional as F

from pytools.modeling.weather_net import WeatherNet


class DirectFC(nn.Module):
    def __init__(self, input_feature, output_feature):
        super().__init__()
        self.dfc = nn.Sequential(
            nn.Linear(in_features=input_feature, out_features=output_feature),
            nn.LayerNorm(normalized_shape=output_feature),
            nn.ReLU() 
        ) 

    def forward(self, wea_arr):
         wea_arr = torch.reshape(wea_arr, (-1,))
         return self.dfc(wea_arr)
    

class WeaCov(nn.Module):

    def __init__(self, input_shape, layer_paras, min_cv1_size=3,
        min_cv2_size=5):
        # input_shape, (x, y, channel)
        super().__init__()
        m_list = []

        if input_shape[1] > min_cv1_size:
            weather_conv1_layer = nn.Conv2d(
                in_channels=input_shape[-1],
                out_channels=layer_paras['cov1']['output_channel'],
                kernel_size=layer_paras['cov1']['kernel'],
                stride=layer_paras['cov1']['stride'],
                padding=layer_paras['cov1']['padding'],
            )
            output_shape1= weather_conv1_layer(torch.rand(input_shape).permute([0,3,1,2])).shape
            m = nn.Sequential(weather_conv1_layer,
            nn.LayerNorm(normalized_shape=output_shape1[1:]), #layer_paras['cov1']['output_channel']),
            nn.ReLU(),
            )
            m_list.append(m)
        else:
            m_list.append(DirectFC(layer_paras['cov1']['output_channel']))

        # if less than 5 X 5, no need for the 2nd Cov2d

        if input_shape[1] >= min_cv2_size:
            weather_conv2_layer = nn.Conv2d(
                in_channels=layer_paras['cov1']['output_channel'],
                out_channels=layer_paras['cov2']['output_channel'],
                kernel_size=layer_paras['cov2']['kernel'],
                stride=layer_paras['cov2']['stride'],
                padding=layer_paras['cov2']['padding'],
            )
            output_shape2 = weather_conv2_layer(torch.rand(output_shape1)).shape
            m = nn.Sequential(weather_conv2_layer,
            nn.LayerNorm(normalized_shape=output_shape2[1:]),
            nn.ReLU())
            m_list.append(m)
        elif input_shape >= min_cv2_size:
            m_list.append(DirectFC(layer_paras['cov2']['output_channel']))

        self.module_list = nn.ModuleList(m_list)

    def forward(self, wea_arr):
        wea_arr = wea_arr.permute([0, 3, 1, 2])
        return self.module_list(wea_arr)        
    

class TSWeatherNet(WeatherNet):

    def __init__(self, ):
        super().super().__init__()

        return