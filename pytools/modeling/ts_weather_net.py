from functools import reduce
from operator import mul

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytools.modeling.weather_net import WeatherNet


class DirectFC(nn.Module):
    def __init__(self, input_shape, output_feature):
        super().__init__()
        input_feature = reduce(mul, input_shape[1:])
        self.dfc = nn.Sequential(
            nn.Linear(in_features=input_feature, out_features=output_feature),
            nn.LayerNorm(normalized_shape=output_feature),
            nn.ReLU(), 
        ) 

    def forward(self, wea_arr):
         wea_arr = torch.flatten(wea_arr, start_dim=1)
         return self.dfc(wea_arr)
    

class WeaCov(nn.Module):

    def __init__(self, input_shape, layer_paras, min_cv1_size=3,
        min_cv2_size=5):
        # input_shape, (paras/channel, x, y)
        super().__init__()
        m_list = []

        if input_shape[1] >= min_cv1_size:
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
            m_list.append(DirectFC(input_shape, layer_paras['cov1']['output_channel']))
            output_shape1 = m_list[-1].forward(torch.rand(input_shape).permute([0,3,1,2])).shape
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
        else:
            m_list.append(DirectFC(output_shape1, layer_paras['cov2']['output_channel']))
            output_shape2 = m_list[-1].forward(torch.rand(output_shape1)).shape

        self.output_shape = [output_shape2[0], reduce(mul, output_shape2[1:])]
        self.module_list = nn.ModuleList(m_list)

    def forward(self, wea_arr):
        wea_arr = wea_arr.permute([0, 3, 1, 2])
        for _, layer in enumerate(self.module_list):
            wea_arr = layer(wea_arr)

        return torch.flatten(wea_arr,1)     
    

class TSWeatherNet(WeatherNet):

    def __init__(self, wea_arr_shape, wea_layer_paras, seq_dim=1):
        # wea_arr_shape, N, Seq, x, y, channel/para
        super().super().__init__()
        self._wea_arr_shape = wea_arr_shape.copy()
        self._seq_dim = seq_dim
        del wea_arr_shape[seq_dim]
        self.wea_net = WeaCov(input_shape=wea_arr_shape, layer_paras=wea_layer_paras)


        return
    
    def forward(self, seq_wea_arr, ext_wea_arr):
        seq_wea_arr = seq_wea_arr.detach().clone()
        channel_num = self.wea_net.output_shape[1]
        wea_len = ext_wea_arr.shape[self._seq_dim]
        seq_length = seq_wea_arr.shape[self._seq_dim]
        seq_pred = torch.zeros([seq_length,channel_num])
        wea_pred = torch.zeros(ext_wea_arr.shape[self._seq_dim], channel_num)
        for i in range(seq_length):
            seq_pred[:,i,:] = self.wea_net.forward(seq_wea_arr[:,i,...])
        for i in range(wea_len):
            wea_pred[:,i,:] = self.wea_net.forward(ext_wea_arr[:,i,...])


        wea_embed = nn.Linear()

        return seq_pred, wea_pred