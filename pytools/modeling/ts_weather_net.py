from functools import reduce
from operator import mul
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
# from einops import rearrange, repeat, pack, unpack
# from xlstm import (
#     xLSTMBlockStack,
#     xLSTMBlockStackConfig,
#     sLSTMBlockConfig,
#     sLSTMLayerConfig,
#     mLSTMBlockConfig,
# )

from pytools.modeling.StandardNorm import Normalize as RevIN

from pytools.config import Config
from pytools.modeling.TexFilter import SeqModel


class DirectFC(nn.Module):
    def __init__(self, input_shape, output_feature):
        super().__init__()
        input_feature = reduce(mul, input_shape[1:])
        self.dfc = nn.Sequential(
            nn.Linear(in_features=input_feature, out_features=output_feature),
            nn.LayerNorm(normalized_shape=output_feature),
            nn.LeakyReLU(), 
        ) 

    def forward(self, wea_arr):
         wea_arr = torch.flatten(wea_arr, start_dim=1)
         return self.dfc(wea_arr)
    

class MixedOutput(nn.Module):
    def __init__(self, seq_arr_dim, filternet_hidden_size, ext_dim, wea_arr_dim, pred_len, model_paras, ):
        super().__init__()
        target_dim = 1
        self._pred_len = pred_len
        self.wea_cov1d = nn.Conv1d(in_channels=wea_arr_dim, **model_paras['cov1d'])
        self.ext_cov1d = nn.Conv1d(in_channels=ext_dim, **model_paras['ext_cov1d'])

        in_dim = seq_arr_dim * filternet_hidden_size + model_paras['cov1d']['out_channels'] + model_paras['cov1d']['out_channels']

        self.mixed_model = nn.ModuleList(
            nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=in_dim//2),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_dim//2, out_features=target_dim),
            ) for _ in range(pred_len)
            )

    def forward(self, seq_arr, ext_arr, wea_arr):
        # B, pred_len, channel
        B = wea_arr.shape[0]
        seq_cross = seq_arr.shape[1] * seq_arr.shape[2]
        y = torch.zeros(B, self._pred_len)
        wea_arr = self.wea_cov1d.forward(torch.permute(wea_arr,[0, 2, 1]))
        ext_arr = self.ext_cov1d.forward(torch.permute(ext_arr, [0, 2, 1]))
        delta =  wea_arr.shape[-1] - self._pred_len
        if delta >=0:
            wea_arr = wea_arr[..., delta:]
        else:
            raise ValueError(f'weather length is less than pred_len {self._pred_len} by {-delta}!')
        delta = ext_arr.shape[-1] - self._pred_len
        if delta >=0 :
            ext_arr = ext_arr[..., delta:]
        else:
            raise ValueError(f'ext length is less than pred_len {self._pred_len} by {-delta}!')
        for i, layer in enumerate(self.mixed_model):
            y[:, i] = layer(torch.cat([torch.reshape(seq_arr, (B, seq_cross)), ext_arr[:, :, i], wea_arr[:, :, i]], dim=1)).squeeze()

        return y


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
            nn.LeakyReLU(),
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
            nn.LeakyReLU())
            m_list.append(m)
            output_shape_from_cov = [output_shape2[0], reduce(mul, output_shape2[1:])]
            m_list.append(DirectFC(output_shape_from_cov, layer_paras['last']['channel']))

        else:
            m_list.append(DirectFC(output_shape1, layer_paras['last']['channel']))
            output_shape2 = m_list[-1].forward(torch.rand(output_shape1)).shape

        self.output_shape = layer_paras['last']['channel']
        self.module_list = nn.ModuleList(m_list)

    def forward(self, wea_arr):
        wea_arr = wea_arr.permute([0, 3, 1, 2])
        for _, layer in enumerate(self.module_list):
            wea_arr = layer(wea_arr)

        return torch.flatten(wea_arr,1)     
    

class TSWeatherNet(pl.LightningModule):

    def __init__(self, wea_arr_shape, 
                 #wea_layer_paras, lstm_layer_paras, pred_length, seq_dim=1,
                 config:Config,
                  ):
        # wea_arr_shape, N, Seq, x, y, channel/para
        super().__init__()
        self.model_settings = config.model_pdt.model_settings
        fn = config.get_model_file_name(class_name='model', extension='.ckpt')
        self.checkpoint_callback = ModelCheckpoint(
            dirpath=osp.dirname(fn),
            filename=osp.basename(fn),
            verbose=True,
            monitor="val_loss",
            mode="min",
        )
        self._mdl_logger: TensorBoardLogger = None
        self._wea_arr_shape = wea_arr_shape.copy()

        seq_dim = config.model_pdt.seq_dim
        wea_layer_paras = config.model_pdt.cov_net
        filter_net_paras = config.model_pdt.filter_net
        fst_ind=0        
        pred_length = config.model_pdt.forecast_horizon[fst_ind][1] - config.model_pdt.forecast_horizon[fst_ind][0] + 1
        self._seq_dim = seq_dim
        del wea_arr_shape[seq_dim]
        self.wea_net = WeaCov(input_shape=wea_arr_shape, layer_paras=wea_layer_paras)
        # lstm hidden state + ext channel * length  + ext weather channel * length
        # mcf = config.model_pdt
        # multi_linear_input_dim = lstm_layer_paras['hidden_dim'] *2 + mcf.ext_embedding_dim * mcf.ext_layer['output'] + \
        #     mcf.wea_embedding_dim * mcf.cov_layer['cov2']['output_channel']
        # pred length
        self._pred_length = pred_length

        #filter_net
        in_channel = 1 if isinstance(config.model_pdt.target_ind, int) else len(config.model_pdt.target_ind)
        self.revin_layer = RevIN(in_channel, affine=True, subtract_last=False)
        self.filter_net = SeqModel(seq_len=seq_dim, filter_net_paras=filter_net_paras)

        self.ext_net = nn.Linear(in_features=config.model_pdt.ext_net['input_channel'], out_features=config.model_pdt.ext_net['output_channel'])
        
        # prediction weather 1D cov
        self.mixed_output = MixedOutput(
            seq_arr_dim=config.model_pdt.filter_net['embed_size'],
            filternet_hidden_size=config.model_pdt.filter_net['hidden_size'],
            ext_dim=config.model_pdt.ext_net['output_channel'],
            wea_arr_dim=config.model_pdt.cov_net['cov2']['output_channel'], 
            pred_len=self._pred_length, 
            model_paras=config.model_pdt.mixed_net)

        # self.multi_linear = nn.Linear(multi_linear_input_dim, pred_length)

    def configure_optimizers(self, label='multi_linear'):
        # REQUIRED
        m_linear = [p for name, p in self.named_parameters() if label in name]
        others = [p for name, p in self.named_parameters() if label not in name]

        return torch.optim.Adam([{'paras':m_linear}, {'paras':others, 'weight_decay':0}], 
                                weight_decay=self.model_settings['weight_decay'], lr=self.model_settings['lr'])
    
    def forward(self, seq_wea_arr, seq_ext_arr, seq_target, wea_arr, ext_arr):
        seq_wea_arr = seq_wea_arr.detach().clone()
        seq_ext_arr = seq_ext_arr.detach().clone()

        # seq pass to time series


        channel_num = self.wea_net.output_shape[1]
        wea_len = wea_arr.shape[self._seq_dim]
        seq_length = seq_wea_arr.shape[self._seq_dim]

        seq_pred = torch.zeros([seq_length,channel_num])
        wea_pred = torch.zeros(wea_arr.shape[self._seq_dim], channel_num)
        for i in range(seq_length):
            seq_pred[:,i,:] = self.wea_net.forward(seq_wea_arr[:,i,...])

        for i in range(wea_len):
            wea_pred[:,i,:] = self.wea_net.forward(wea_arr[:,i,...])


        return seq_pred, wea_pred