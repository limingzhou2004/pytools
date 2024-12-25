from copy import deepcopy
from functools import reduce
from operator import mul
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.utils.data.dataloader import DataLoader

import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
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
from pytools.modeling.TexFilter import SeqModel
from pytools.modeling.utilities import extract_a_field


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
    def __init__(self, seq_arr_dim, seq_latent_dim, filternet_hidden_size, ext_dim, wea_arr_dim, pred_len, model_paras, ):
        super().__init__()
        target_dim = 1
        self._pred_len = pred_len
        self.wea_cov1d = nn.Conv1d(in_channels=wea_arr_dim, **model_paras['cov1d'])
        self.ext_cov1d = nn.Conv1d(in_channels=ext_dim, **model_paras['ext_cov1d'])

        in_dim = seq_latent_dim * filternet_hidden_size + model_paras['cov1d']['out_channels'] + model_paras['ext_cov1d']['out_channels']

        self.ts_latent_model = nn.Linear(in_features=seq_arr_dim,out_features=seq_latent_dim)

        self.mixed_model = nn.ModuleList(
            nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=target_dim), #in_dim//2),
          #  nn.LeakyReLU(),
          #  nn.Linear(in_features=in_dim//2, out_features=target_dim),
            ) for _ in range(pred_len)
            )

    def forward(self, seq_arr, ext_arr, wea_arr):
        # B, pred_len, channel
        device = seq_arr.device
        B = wea_arr.shape[0]
        seq_arr = self.ts_latent_model(seq_arr)
        seq_cross = seq_arr.shape[1] * seq_arr.shape[2]

        y = torch.zeros(B, self._pred_len, device=device)
        wea_arr = self.wea_cov1d(torch.permute(wea_arr,[0, 2, 1]))
        ext_arr = self.ext_cov1d(torch.permute(ext_arr, [0, 2, 1]))
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

    def __init__(self, wea_arr_shape, config:Config):
        # wea_arr_shape, N, Seq, x, y, channel/para
        self.save_hyperparameters()
        super().__init__()
        self.model_options = config.model_pdt.model_options
        fn = config.get_model_file_name(class_name='model', extension='.ckpt')
        self.lr = config.model_pdt.model_options['lr']
        self.batch_size = config.model_pdt.model_options['batch_size']
        self.checkpoint_callback = ModelCheckpoint(
            dirpath=osp.dirname(fn),
            filename=osp.basename(fn),
            verbose=True,
            monitor='validation_epoch_average',
            mode='min',
        )
        #self._mdl_logger: TensorBoardLogger = None
        self._wea_arr_shape = deepcopy(wea_arr_shape)
        seq_dim = config.model_pdt.sample_data_seq_dim
        wea_layer_paras = config.model_pdt.cov_net
        filter_net_paras = config.model_pdt.filter_net
        fst_ind=0        
        pred_length = config.model_pdt.forecast_horizon[fst_ind][1] - config.model_pdt.forecast_horizon[fst_ind][0] + 1
        self._seq_dim = seq_dim
        self._seq_length = config.model_pdt.seq_length
        del self._wea_arr_shape[seq_dim]
        self.wea_net = WeaCov(input_shape=self._wea_arr_shape, layer_paras=wea_layer_paras)
        self._pred_length = pred_length

        self.validation_step_outputs = []

        #filter_net
        in_channel = 1 if isinstance(config.model_pdt.target_ind, int) else len(config.model_pdt.target_ind)
        self.revin_layer = RevIN(in_channel, affine=True, subtract_last=False)
        self.filter_net = SeqModel(config.filternet_input, filter_net_paras=filter_net_paras)
        self.ext_channels = config.model_pdt.ext_net['output_channel']
        self.ext_net = nn.Linear(in_features=config.model_pdt.ext_net['input_channel'], out_features=self.ext_channels)   
        self.wea_channels = config.model_pdt.cov_net['last']['channel']     
        
        # prediction weather 1D cov
        mix_input_dim = in_channel + self.ext_channels + self.wea_channels
        self.mixed_output = MixedOutput(
            seq_arr_dim=config.model_pdt.seq_length,
            seq_latent_dim=config.model_pdt.mixed_net['ts_latent_dim'],
            filternet_hidden_size=mix_input_dim,
            ext_dim=config.model_pdt.ext_net['output_channel'],
            wea_arr_dim=self.wea_channels, 
            pred_len=self._pred_length, 
            model_paras=config.model_pdt.mixed_net)

        # self.multi_linear = nn.Linear(multi_linear_input_dim, pred_length)

    def configure_optimizers(self, label='multi_linear'):
        # REQUIRED
        # m_linear = [p for name, p in self.named_parameters() if label in name]
        # others = [p for name, p in self.named_parameters() if label not in name]

        #return torch.optim.Adam([{'paras':m_linear}, {'paras':others, 'weight_decay':0}], 
                                #weight_decay=self.model_options['weight_decay'], lr=self.#model_options['lr'])
        return torch.optim.Adam(self.parameters(), lr=(self.lr))
    
    def forward(self, seq_wea_arr, seq_ext_arr, seq_target, wea_arr, ext_arr):
        device = seq_wea_arr.device
        self.revin_layer(seq_target,'norm')
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
        seq_ext_y = torch.zeros(B, seq_length, self.ext_channels, device=device)
        ext_y = torch.zeros([B, e_dim, self.ext_channels], device=device)        

        for i in range(seq_length):
            seq_wea_y[:,i,:] = self.wea_net(seq_wea_arr[:,i,...])
            seq_ext_y[:,i,:] = self.ext_net(seq_ext_arr[:,i,...])
        for i in range(w_dim):
            wea_y[:,i,:] = self.wea_net(wea_arr[:,i,...])
        for i in range(e_dim):
            ext_y[:,i,:] = self.ext_net(ext_arr[:,i,...])
        seq_y = torch.cat([seq_target[...,None], seq_ext_y, seq_wea_y],dim=2).permute([0, 2, 1])
        seq_y = self.filter_net(seq_y)

        y = self.mixed_output(seq_y, ext_y, wea_y)
        y = self.revin_layer(y, 'denorm')
        return y    

    def training_step(self, batch, batch_nb):
        # REQUIRED
        seq_wea_arr, seq_ext_arr, seq_arr, wea_arr, ext_arr, target = batch
        y_hat = self(seq_wea_arr, seq_ext_arr, seq_arr, wea_arr, ext_arr)
        loss = F.mse_loss(y_hat, target)
        self.log('training RMSE loss',torch.sqrt(loss), on_epoch=True)
        return loss #torch.sqrt(loss)

    # def on_training_epoch_end(self, outputs):
    #     #  the function is called after every epoch is completed
    #     avg_loss = torch.stack([x["RMSE loss"] for x in outputs]).mean()
    #     # self._mdl_logger.experiment.add_scalar(
    #     #     "loss/train", avg_loss, self.current_epoch
    #     # )
    #     # for k, v in self._busi_loss_metrics(avg_loss).items():
    #     #     self._mdl_logger.experiment.add_scalar(k, v, self.current_epoch)
    #     #     self.log(k, v)
    #     self.log("train_rmse_loss", float(avg_loss.squeeze()))

    def _busi_loss_metrics(self, scaled_loss):
        abs_loss = self._scaler.unscale_target(scaled_loss.cpu().reshape((-1, 1)))
        relative_loss = abs_loss / self._target_mean
        return {
            "abs_loss(MAE)": float(abs_loss.squeeze()),
            "relative_loss(RMAE)": float(relative_loss.squeeze()),
            "y_mean": self._target_mean,
        }

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        seq_wea_arr, seq_ext_arr, seq_arr, wea_arr, ext_arr, target = batch
        y_hat = self(seq_wea_arr, seq_ext_arr, seq_arr, wea_arr, ext_arr)
        loss = F.mse_loss(y_hat, target)
        self.log('val RSME loss', torch.sqrt(loss))#, on_epoch=True)
        return loss

    # def on_validation_epoch_end(self):
    #     # OPTIONAL
    #     # avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
    #     # self._mdl_logger.experiment.add_scalar("loss/val", avg_loss, self.current_epoch)
    #     # for k, v in self._busi_loss_metrics(avg_loss).items():
    #     #     self._mdl_logger.experiment.add_scalar(k, v, self.current_epoch)
    #     #     self.log(k, v)
    #     # self.log("val_loss", float(avg_loss.squeeze()))
    #     epoch_average = torch.stack(self.validation_step_outputs).mean()
    #     self.log('validation_epoch_average', epoch_average)
    #     self.validation_step_outputs.clear()  # free memory


    def test_step(self, batch, batch_nb):
        # OPTIONAL
        seq_wea_arr, seq_ext_arr, seq_arr, wea_arr, ext_arr, target = batch
        y_hat = self(seq_wea_arr, seq_ext_arr, seq_arr, wea_arr, ext_arr)
        loss = F.mse_loss(y_hat, target)
        self.log('test RSME loss', loss, on_epoch=True)
        self.log_dict(self._busi_loss_metrics(loss))
        return loss

    # def on_test_epoch_end(self):
    #     # OPTIONAL
    #     #avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
    #     self._mdl_logger.experiment.add_scalar(
    #         "loss/test", avg_loss, self.current_epoch
    #     )
    #     for k, v in self._busi_loss_metrics(avg_loss).items():
    #         self._mdl_logger.experiment.add_scalar(k, v, self.current_epoch)
    #         self.log(k, v)
    #     self.log("test_loss", float(avg_loss.squeeze()))
    #     self._mdl_logger.experiment.add_hparams(
    #         dict(self.model_options._asdict()), {"test_loss": avg_loss}
    #     )

    def setup_mean(self, target_mean, scaler):
        self._target_mean = target_mean
        self._scaler = scaler


class TsWeaDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.allow_zero_length_dataloader_with_multiple_devices = True


def cv_train_ts_weather_net(sce_id, config:Config):

    return

def final_train_ts_weather_net(config:Config):

    return



