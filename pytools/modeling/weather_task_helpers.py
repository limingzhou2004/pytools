import multiprocessing
import os
import datetime as dt

import pandas as pd
import numpy as np
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

from pytools.config import Config
from pytools.data_prep import data_prep_manager as dpm
from pytools.data_prep.data_prep_manager import DataPrepManager
from pytools.modeling.dataset import WeatherDataSetBuilder
from pytools.modeling.utilities import load_npz_as_dict, extract_model_settings
from pytools.modeling.weather_net import WeatherNet, ModelSettings, default_layer_sizes


def get_npz_train_weather_file_name(cfg: Config, suffix) -> str:
    save_folder = os.path.join(
        cfg.site_parent_folder, f"sync_data_train_{suffix}.npz"
    )
    return save_folder


def get_training_data(cfg: Config, suffix, cat_fraction):
    """
    Get training weather data

    Args:
        cfg: the Config object
        cat_fraction: fraction for train|validate|test

    Returns: WeatherData object

    """
    npz_file = get_npz_train_weather_file_name(cfg, suffix)
    if os.path.exists(npz_file):
        data = load_npz_as_dict(npz_file)
        return WeatherDataSetBuilder(**data, cat_fraction=cat_fraction)
    else:
        raise ValueError(
            "Please run task 3 to generate the npz file for model training data sets"
        )


def prepare_train_data(data, ahead_hrs, batch_size, num_workers=None, full_data=False):
    """
    Load the train, validation, test triplet

    Args:
        data: loaded from the npz file
        ahead_hrs: hours ahead
        batch_size: batch size
        num_workers: number of cpus to use
        full_data: get the full dataset.
    Returns: DataLoaders for train, val, test

    """
    if not num_workers:
        num_workers = multiprocessing.cpu_count()
    train_data = DataLoader(
        data.extract_data(cat="train", fst_hours=ahead_hrs),
        batch_size=batch_size,
        num_workers=num_workers,
    )
    if full_data:
        return train_data, None, None
    validation_data = DataLoader(
        data.extract_data(cat="validation", fst_hours=ahead_hrs),
        batch_size=batch_size,
        num_workers=num_workers,
    )
    test_data = DataLoader(
        data.extract_data(cat="test", fst_hours=ahead_hrs),
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return train_data, validation_data, test_data


def prepare_train_model(
    cfg: Config, weather_para, ahead_hours, train_options, hist_load
):
    mdl_setting = ModelSettings(
        **extract_model_settings(train_options, ModelSettings._fields)
    )
    model = WeatherNet(
        model_file_path=cfg.site_parent_folder,
        model_file_name=cfg.base_folder,
        layer_paras=default_layer_sizes,
        model_settings=mdl_setting,
        weather_para=weather_para,
        hrs_ahead=ahead_hours,
        hist_load=hist_load,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=mdl_setting.min_delta,
        patience=mdl_setting.patience,
        verbose=False,
        mode="min",
    )
    trainer = pl.Trainer(
        auto_lr_find=True,
        callbacks=[early_stop_callback],
        check_val_every_n_epoch=mdl_setting.epoch_step,
        max_epochs=mdl_setting.epoch_num,
    )
    return model, trainer


def get_model_checkpoint_path(cfg: Config, ahead_hours):
    return cfg.get_model_file_name(
        class_name="_model", suffix=f"_fst_{ahead_hours}.ckpt"
    )


def decide_start_cur_end_time(cfg: Config, d: DataPrepManager, t_cr=None, max_hours=1):
    """
    Decide the t0, current time, t1
    # t_now = t_now.replace(tzinfo=pytz.utc)  # NOTE: it works only with a fixed utc offset

    Args:
        cfg: config
        d: dataManager
        t_cr: the first hour to predict
        max_hours: hours ahead to forecast

    Returns: lag starting time, current time, forecast end time

    """
    if not t_cr:
        t_cr = d.load_data.query_max_load_time()
        # round to hour by zeroing out min, sec, ms
        t_cr = pd.to_datetime(t_cr.values[0][0]).replace(
            microsecond=0, second=0, minute=0
        )
        t_cr = t_cr + np.timedelta64(1, "h")
    if isinstance(t_cr, str):
        t_cr = pd.to_datetime(t_cr)
    t0 = t_cr - dt.timedelta(hours=cfg.load["lag_hours"])
    t1 = t_cr + dt.timedelta(hours=max_hours)
    return t0, t_cr, t1


def prepare_predict_data(data, batch_size):
    """
    Prepare the input data for a prediction
    Args:
        data:
        batch_size:

    Returns:

    """
    raise NotImplementedError
