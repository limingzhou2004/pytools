from collections import namedtuple
from enum import Enum
from typing import Dict

import torch.nn as nn
import torch.nn.functional as F
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torch import optim
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data.dataloader import DataLoader

from pytools.config import Config
from pytools.data_prep.data_prep_manager import DataPrepManager
from pytools.data_prep.load_data_prep import LoadData
from pytools.modeling.utilities import (
    get_cnn1d_dim,
    get_cnn2d_dim,
    get_cnn_padding,
    extract_a_field,
)

default_layer_sizes = {
    "w_out1_channel": 51,
    "w_stride1": 3,
    "w_kernel1": (5, 5),
    "w_out2_channel": 8,
    "w_stride2": 3,
    "w_kernel2": (5, 5),
    "l_channel": 1,
    "l_out1_channel": 6,
    "l_kernel1": 5,
    "l_stride1": 3,
    "l_out2_channel": 3,
    "l_kernel2": 3,
    "l_stride2": 2,
    "c_dense1_in": 6,
    "c_dense1_out": 12,
    "c_dense2_out": 6,
    "final_dense_out_1": 40,
    "final_dense_out_2": 20,
    "final_dense_out_3": 1,
}


class EnumOptimizer(Enum):
    sgd = 1
    adam = 2
    rsprop = 3


WeatherPara = namedtuple(
    "WeatherPara", "x_dim y_dim channel embed_load_dim calendar_dim"
)

ModelSettings = namedtuple(
    "ModelSetting",
    [
        "device",
        "batch_size",
        "epoch_num",
        "epoch_step",
        "min_delta",
        "patience",
        "optimizer",
        "dropout",
        "learning_rate",
        "seed",
        "log_interval",
    ],
    defaults=("cpu", 100, 100, 1, 0, 3, "adams", 0.01, 0.01, 0, 0.01),
)


class WeatherLayer(nn.Module):
    def __init__(
        self,
        weather_para: WeatherPara,
        layer_paras: Dict,
        model_settings: ModelSettings,
    ):
        """
        Initialize the network

        Args:
            weather_para: WeatherPara for x_dim, y_dim...
            layer_paras: dict for sizes, keys including "w_channel, ". "kernel" and "stride" could be an int or a tuple
            model_settings:
        """
        super(WeatherLayer, self).__init__()
        self.weather_para = weather_para
        layer_paras = self._add_padding_parameters(layer_paras)
        self.layer_paras = layer_paras
        self.model_settings = model_settings
        self.weather_conv1_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=weather_para.channel,
                out_channels=layer_paras["w_out1_channel"],
                kernel_size=layer_paras["w_kernel1"],
                stride=layer_paras["w_stride1"],
                padding=layer_paras["w_padding_1"],
            ),
            nn.BatchNorm2d(num_features=layer_paras["w_out1_channel"]),
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
            nn.BatchNorm2d(num_features=layer_paras["w_out2_channel"]),
            nn.ReLU(),
            # nn.Dropout2d(p=model_settings.dropout),
        )
        self.load_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=layer_paras["l_channel"],
                out_channels=layer_paras["l_out1_channel"],
                kernel_size=layer_paras["l_kernel1"],
                stride=layer_paras["l_stride1"],
                padding=layer_paras["l_padding_1"],
            ),
            nn.ReLU(),
        )
        self.load2_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=layer_paras["l_out1_channel"],
                out_channels=layer_paras["l_out2_channel"],
                kernel_size=layer_paras["l_kernel2"],
                stride=layer_paras["l_stride2"],
                padding=layer_paras["l_padding_2"],
            ),
            nn.ReLU(),
            # nn.Dropout(p=model_settings.dropout),
        )
        self.calendar_layer = nn.Sequential(
            nn.Linear(
                in_features=layer_paras["c_dense1_in"],
                out_features=layer_paras["c_dense1_out"],
            ),
            nn.Linear(
                in_features=layer_paras["c_dense1_out"],
                out_features=layer_paras["c_dense2_out"],
            ),
        )
        final_h, final_w = self._get_cnn_output_dim_2d()
        self.final_dense_layer_input_channel = (
            layer_paras["c_dense2_out"]
            + layer_paras["l_out2_channel"] * self._get_cnn_output_dim_1d()
            + layer_paras["w_out2_channel"] * final_h * final_w
        )

        self.final_dense_layer = nn.Sequential(
            nn.Linear(
                in_features=self.final_dense_layer_input_channel,
                out_features=layer_paras["final_dense_out_1"],
            ),
            nn.Linear(
                in_features=layer_paras["final_dense_out_1"],
                out_features=layer_paras["final_dense_out_2"],
            ),
            nn.Linear(
                in_features=layer_paras["final_dense_out_2"],
                out_features=layer_paras["final_dense_out_3"],
            ),
        )

    def _add_padding_parameters(self, layer_para):
        """
        Add padding integers into the dict

        Args:
            layer_para:

        Returns: processed layer_para

        """
        layer_para["l_padding_1"] = get_cnn_padding(layer_para["l_kernel1"])
        layer_para["l_padding_2"] = get_cnn_padding(layer_para["l_kernel2"])
        layer_para["w_padding_1"] = get_cnn_padding(layer_para["w_kernel1"])
        layer_para["w_padding_2"] = get_cnn_padding(layer_para["w_kernel2"])
        return layer_para

    def _get_cnn_output_dim_1d(self):
        out_dim1 = get_cnn1d_dim(
            self.weather_para.embed_load_dim,
            self.layer_paras["l_kernel1"],
            self.layer_paras["l_stride1"],
            self.layer_paras["l_padding_1"],
        )
        return get_cnn1d_dim(
            out_dim1,
            self.layer_paras["l_kernel2"],
            self.layer_paras["l_stride2"],
            self.layer_paras["l_padding_2"],
        )

    def _get_cnn_output_dim_2d(self):
        out_dim1 = get_cnn2d_dim(
            self.weather_para.x_dim,
            self.weather_para.y_dim,
            self.layer_paras["w_kernel1"],
            self.layer_paras["w_stride1"],
            self.layer_paras["w_padding_1"],
        )
        return get_cnn2d_dim(
            *out_dim1,
            self.layer_paras["w_kernel2"],
            self.layer_paras["w_stride2"],
            self.layer_paras["w_padding_2"],
        )

    def forward(self, tensor_wea, tensor_load_embed, tensor_calendar):
        weather_x = self.weather_conv1_layer(tensor_wea)
        weather_x = self.weather_conv2_layer(weather_x)
        sample_size = weather_x.shape[0]
        weather_x = weather_x.view((sample_size, -1))
        load_x = self.load_layer(tensor_load_embed)
        load_x = self.load2_layer(load_x).view(sample_size, -1)
        calendar_x = self.calendar_layer(tensor_calendar)
        dense_x = torch.cat((weather_x, load_x, calendar_x), 1)
        dense_x = self.final_dense_layer(dense_x)
        return dense_x


def loss_fn(y, y_pre, cat="mae"):
    """

    Args:
        y: actual y
        y_pre: predicted y
        cat: mae | rse

    Returns:

    """
    if cat == "mae":
        return torch.mean(torch.abs(y - y_pre))
    elif cat == "rse":
        return torch.mean(torch.square(y - y_pre))


class WeatherNet(pl.LightningModule):
    def __init__(
        self,
        model_file_path,
        model_file_name,
        hrs_ahead,
        layer_paras: Dict,
        model_settings: ModelSettings,
        weather_para: WeatherPara,
        hist_load: DataPrepManager,
    ):
        super(WeatherNet, self).__init__()
        self.model = WeatherLayer(weather_para, layer_paras, model_settings)
        self.weather_para = weather_para
        self.layer_paras = layer_paras
        self.model_settings = model_settings
        self.checkpoint_callback = ModelCheckpoint(
            filepath=f"${model_file_path}/{model_file_name}_hrs_{hrs_ahead}.ckpt",  # '/path/to/store/weights.ckpt',
            verbose=True,
            monitor="val_loss",
            mode="min",
        )
        self._hrs_ahead = hrs_ahead
        self._mdl_logger: TensorBoardLogger = None
        self._load_mean, self._load_scaler, self._y_name = (
            hist_load.load_data.y_mean,
            hist_load.get_load_scalar(),
            hist_load.load_data.y_label,
        )

    def add_a_logger(self, cfg: Config):
        self._mdl_logger = TensorBoardLogger(
            name=f"model_{self._hrs_ahead}",
            save_dir=cfg.get_model_file_name(class_name="_log"),
        )

    def forward(self, tensor_wea, tensor_load_embed, tensor_calendar):
        return self.model(tensor_wea, tensor_load_embed, tensor_calendar)

    def training_step(self, batch, batch_nb):
        # REQUIRED
        wea, lag, c, y = batch
        y_hat = self(wea, lag, c)
        loss = F.mse_loss(y_hat, y)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        #  the function is called after every epoch is completed
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self._mdl_logger.experiment.add_scalar(
            "loss/train", avg_loss, self.current_epoch
        )
        for k, v in self._busi_loss_metrics(avg_loss).items():
            self._mdl_logger.experiment.add_scalar(k, v, self.current_epoch)
            self.log(k, v)
        self.log("train_loss", float(avg_loss.squeeze()))

    def _busi_loss_metrics(self, scaled_loss):
        abs_loss = self._load_scaler.inverse_transform(scaled_loss.reshape((-1, 1)))
        relative_loss = abs_loss / self._load_mean
        return {
            "abs_loss": float(abs_loss.squeeze()),
            "relative_loss": float(relative_loss.squeeze()),
            "y_mean": self._load_mean,
        }

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        wea, lag, c, y = batch
        y_hat = self(wea, lag, c)
        loss = F.mse_loss(y_hat, y)
        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self._mdl_logger.experiment.add_scalar("loss/val", avg_loss, self.current_epoch)
        for k, v in self._busi_loss_metrics(avg_loss).items():
            self._mdl_logger.experiment.add_scalar(k, v, self.current_epoch)
            self.log(k, v)
        self.log("val_loss", float(avg_loss.squeeze()))

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        wea, lag, c, y = batch
        y_hat = self(wea, lag, c)
        loss = F.mse_loss(y_hat, y)
        return {"loss": loss, **self._busi_loss_metrics(loss)}

    def test_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self._mdl_logger.experiment.add_scalar(
            "loss/test", avg_loss, self.current_epoch
        )
        for k, v in self._busi_loss_metrics(avg_loss).items():
            self._mdl_logger.experiment.add_scalar(k, v, self.current_epoch)
            self.log(k, v)
        self.log("test_loss", float(avg_loss.squeeze()))
        self._mdl_logger.experiment.add_hparams(
            dict(self.model_settings._asdict()), {"test_loss": avg_loss}
        )

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=self.model_settings.learning_rate)

    def train_dataloader(self):
        # REQUIRED
        batch_size = extract_a_field(self.model_settings, "batch_size", 32)
        return DataLoader(0, batch_size)

    def val_dataloader(self):
        # OPTIONAL
        return None

    def test_dataloader(self):
        # OPTIONAL
        return DataLoader()

    def add_model_specific_args(self):
        return
