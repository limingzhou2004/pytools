from copy import deepcopy
from functools import reduce
from operator import mul
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


import lightning as pl
#from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
# from einops import rearrange, repeat, pack, unpack
# from xlstm import (
#     xLSTMBlockStack,
#     xLSTMBlockStackConfig,
#     sLSTMBlockConfig,
#     sLSTMLayerConfig,
#     mLSTMBlockConfig,
# )

from pytools.modeling.RevIN import  RevIN

from pytools.config import Config
#from pytools.modeling.TexFilter import SeqModel
#from pytools.modeling.utilities import extract_a_field


class DirectFC(nn.Module):
    def __init__(self, input_shape, output_feature,dropout=0.001):
        super().__init__()
        input_feature = reduce(mul, input_shape[1:])
        self.dfc = nn.Sequential(
            nn.Linear(in_features=input_feature, out_features=output_feature),
            nn.LayerNorm(normalized_shape=output_feature),
            nn.Dropout(dropout),
            nn.ReLU(), 
        ) 

    def forward(self, wea_arr):
         wea_arr = torch.flatten(wea_arr, start_dim=1)
         return self.dfc(wea_arr)
    

class MixedOutput(nn.Module):
    def __init__(self, seq_arr_dim, seq_latent_dim, filternet_hidden_size, ext_dim, wea_arr_dim, pred_len, model_paras):
        super().__init__()
        target_dim = 1
        self._pred_len = pred_len
        in_dim = seq_latent_dim * filternet_hidden_size + wea_arr_dim + ext_dim
        self.mixed_model = nn.ModuleList(
            nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=model_paras['channel0']),
            nn.Dropout(model_paras['dropout']),
            nn.ReLU(),
            nn.Linear(in_features=model_paras['channel0'], out_features=model_paras['channel1']),
            nn.ReLU(),
            nn.Linear(in_features=model_paras['channel1'],out_features=target_dim)
            ) for _ in range(pred_len)
            )

    def forward(self, seq_arr, ext_arr, wea_arr):
        # B, pred_len, channel
        device = seq_arr.device
        B = wea_arr.shape[0]
        seq_cross = seq_arr.shape[1] * seq_arr.shape[2]
        y = torch.zeros(B, self._pred_len, device=device)
        wea_arr=torch.permute(wea_arr,[0, 2, 1])
        ext_arr=torch.permute(ext_arr, [0, 2, 1])

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
            y[:, i] = layer(torch.cat([torch.reshape(seq_arr, (B, seq_cross)), ext_arr[:, :, i], torch.reshape(wea_arr,(B,-1))], dim=1)).squeeze()

        return y


class WeaCov(nn.Module):

    def __init__(self, input_shape, layer_paras, min_cv1_size=3,
        min_cv2_size=5):
        # input_shape, (paras/channel, x, y)
        super().__init__()
        m_list = []
        self.dropout = layer_paras['dropout']

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
            nn.Dropout(layer_paras['dropout']),
            nn.LeakyReLU(),
            )
            m_list.append(m)          
        else:
            m_list.append(DirectFC(input_shape, layer_paras['cov1']['output_channel'],dropout=self.dropout))
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
            self.output_dim = output_shape_from_cov[1]
    
        else:
            m_list.append(DirectFC(output_shape1, layer_paras['last']['channel'],dropout=self.dropout))
            output_shape2 = m_list[-1].forward(torch.rand(output_shape1)).shape
            self.output_dim = output_shape2[1] *output_shape2[2] 

        self.output_shape = self.output_dim 
        self.module_list = nn.ModuleList(m_list)

    def forward(self, wea_arr):
        wea_arr = wea_arr.permute([0, 3, 1, 2])
        for _, layer in enumerate(self.module_list):
            wea_arr = layer(wea_arr)

        return torch.flatten(wea_arr,1)     
    

class TSWeatherNet(pl.LightningModule):

    def __init__(self, wea_arr_shape, config:Config):
        # wea_arr_shape, N, Seq, x, y, channel/para
        self.save_hyperparameters()
        super().__init__()
        self.hyper_options = config.model_pdt.hyper_options
        fn = config.get_model_file_name(class_name='model', extension='.ckpt')
        self.lr = config.model_pdt.hyper_options['lr']
        self.batch_size = config.model_pdt.hyper_options['batch_size']
        self.checkpoint_callback = ModelCheckpoint(
            dirpath=osp.dirname(fn),
            filename=osp.basename(fn),
            verbose=True,
            monitor='validation_epoch_average',
            mode='min',
        )
        self._wea_arr_shape = deepcopy(wea_arr_shape)
        seq_dim = config.model_pdt.sample_data_seq_dim
        wea_layer_paras = config.model_pdt.cov_net
        fst_ind=0        
        pred_length = config.model_pdt.forecast_horizon[fst_ind][1] - config.model_pdt.forecast_horizon[fst_ind][0] + 1
        self._seq_dim = seq_dim
        self._seq_length = config.model_pdt.seq_length
        del self._wea_arr_shape[seq_dim]
        self.wea_net = WeaCov(input_shape=self._wea_arr_shape, layer_paras=wea_layer_paras)
        self._pred_length = pred_length

        self.validation_step_outputs = []
        in_channel = 1 if isinstance(config.model_pdt.target_ind, int) else len(config.model_pdt.target_ind)
        self.wea_channels = self.wea_net.output_shape 
        self.beta=nn.Parameter(0.01*torch.randn(self._wea_arr_shape[-1]))   
        para_copy = deepcopy(config.model_pdt.ts_net)
        if self.hyper_options['zero_inflation']:
            x = torch.zeros(1,in_channel+self.wea_channels, self._seq_length)
            para_copy['in_channels'] += self.wea_channels
        else:
            x = torch.zeros(1,in_channel, self._seq_length)
        self.filter_net = nn.Conv1d(**para_copy) 
        x=self.filter_net(x)

        self.ext_channels = config.model_pdt.ext_net['output_channel']
        
        # prediction weather 1D cov
        self.mixed_output = MixedOutput(
            seq_arr_dim=config.model_pdt.seq_length,
            seq_latent_dim=x.shape[2], 
            filternet_hidden_size=x.shape[1],
            ext_dim=config.model_pdt.ext_net['input_channel'],
            wea_arr_dim=self.wea_channels, 
            pred_len=self._pred_length, 
            model_paras=config.model_pdt.mixed_net)

    def configure_optimizers(self, label='filter_net'):
        # REQUIRED
        #print(self.named_parameters)
        # ts_net = [p for name, p in self.named_parameters() if label in name]
        # others = [p for name, p in self.named_parameters() if label not in name]
        # return torch.optim.Adam([{'params':others}, {'params':ts_net, 'lr':0.0001*self.hyper_options['lr']}],
        #                         #weight_decay=self.hyper_options['weight_decay'], 
        #                         lr=self.hyper_options['lr'])
        return torch.optim.Adam(self.parameters(), lr=(self.lr))
    
    def forward(self, seq_wea_arr, seq_ext_arr, seq_target, wea_arr, ext_arr):
        device = seq_wea_arr.device

        #print(f'seq target shape : {seq_target.shape}')
        B= seq_wea_arr.shape[0]
        seq_wea_arr = seq_wea_arr.detach().clone()
        seq_ext_arr = seq_ext_arr.detach().clone()
        # seq pass to time series
        wea_channel_num = self.wea_net.output_shape
        seq_length = self._seq_length
        w_dim = wea_arr.shape[1]
        e_dim = ext_arr.shape[1]
        
        seq_wea_y = torch.zeros(B, seq_length, wea_channel_num, device=device)
        wea_y = torch.zeros(B, w_dim, wea_channel_num, device=device)
        #seq_ext_y = torch.zeros(B, seq_length, self.ext_channels, device=device)
        #ext_y = torch.zeros([B, e_dim, self.ext_channels], device=device)    

        #w_shape = tuple(seq_wea_arr.shape) 

        #for i in range(seq_length):
         #   seq_wea_y[:,i,:] = self.wea_net(seq_wea_arr[:,i,...])
        #seq_wea_y = self.wea_net(seq_wea_arr.reshape((-1,*w_shape[2:]))).reshape((*w_shape[0:2],-1))
            #seq_ext_y[:,i,:] = self.ext_net(seq_ext_arr[:,i,...])
        for i in range(1,w_dim):            
            wea_y[:,i,:] = self.wea_net(wea_arr[:,i,...] + torch.mul(wea_arr[:,i-1,...],self.beta) )
        # for i in range(e_dim):
        #     ext_y[:,i,:] = self.ext_net(ext_arr[:,i,...])
        #seq_y = torch.cat([seq_target[...,None], seq_ext_y, seq_wea_y],dim=2).permute([0, 2, 1])
        if self.hyper_options['zero_inflation']:
            seq_y = torch.cat([seq_target[...,None], seq_wea_y],dim=2).permute([0, 2, 1])
        else:
            seq_y = torch.cat([seq_target[...,None]],dim=2).permute([0, 2, 1])
        seq_y = self.filter_net(seq_y)

        y = self.mixed_output(seq_y, ext_arr, wea_y)
        #y = self.revin_layer(y, 'denorm')
        return y    

    def training_step(self, batch, batch_nb):
        # REQUIRED
        seq_wea_arr, seq_ext_arr, seq_arr, wea_arr, ext_arr, target = batch
        y_hat = self(seq_wea_arr, seq_ext_arr, seq_arr, wea_arr, ext_arr)
        loss = F.mse_loss(y_hat, target)
        self.log('training RMSE loss',torch.sqrt(loss), on_epoch=True)
        return loss

    def _busi_loss_metrics(self, pred, target):
        abs_bias = (torch.mean(pred)-torch.mean(target)).item()
        pred = self._scaler.unscale_target(pred.cpu().reshape((-1, 1))) 
        target = self._scaler.unscale_target(target.cpu().reshape((-1, 1))) 
        abs_loss = F.l1_loss(torch.tensor(pred),torch.tensor(target)).item()
        relative_loss = abs_loss / self._target_mean
        abs_bias_perc = abs_bias/self._target_mean
        return {
            'abs_loss(MAE)': abs_loss,
            'relative_loss(RMAE)': relative_loss,
            'y_mean': self._target_mean,
            'y_std': self._target_std,
            'abs_bias':abs_bias,
            'abs_bias_perc':abs_bias_perc,
        }

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        seq_wea_arr, seq_ext_arr, seq_arr, wea_arr, ext_arr, target = batch
        y_hat = self(seq_wea_arr, seq_ext_arr, seq_arr, wea_arr, ext_arr)
        loss = F.mse_loss(y_hat, target)        
        self.log('val RMSE loss', torch.sqrt(loss), on_epoch=True)
        return loss

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        seq_wea_arr, seq_ext_arr, seq_arr, wea_arr, ext_arr, target = batch
        y_hat = self(seq_wea_arr, seq_ext_arr, seq_arr, wea_arr, ext_arr)
        loss = F.mse_loss(y_hat, target)
        self.log('test RMSE loss', torch.sqrt(loss), on_epoch=True)
        self.log_dict(self._busi_loss_metrics(y_hat, target))
        return loss
 
    def setup_mean(self, target_mean, target_std, scaler):
        self._target_mean = target_mean
        self._scaler = scaler
        self._target_std = target_std


class TsWeaDataModule(pl.LightningDataModule):

    def __init__(self, ds_train, ds_val, ds_test, batch_size, num_worker, ):
        super().__init__()
        self._batch_size = batch_size
        self.allow_zero_length_dataloader_with_multiple_devices = True
        self._num_worker = num_worker
        self._ds_train = ds_train
        self._ds_val = ds_val
        self._ds_test = ds_test 

    def train_dataloader(self):
        return DataLoader(self._ds_train, batch_size=self._batch_size,num_workers=self._num_worker, persistent_workers=True, shuffle=True, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self._ds_test, batch_size=self._batch_size,num_workers=self._num_worker, persistent_workers=True,shuffle=False)
    
    def val_dataloader(self):
        return DataLoader(self._ds_val, batch_size=self._batch_size,num_workers=self._num_worker, persistent_workers=True,shuffle=False)
    


