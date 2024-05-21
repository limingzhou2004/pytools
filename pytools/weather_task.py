import sys
import datetime as dt
from functools import partial
import os.path as osp
from typing import Tuple

import pytz
import torch
from dateutil import parser
import mlflow
import numpy as np
import pandas as pd

# from torch.utils.data.dataloader import DataLoader

# from pytools.modeling.dataset import WeatherDataSet
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader

from pytools.arg_class import ArgClass
from pytools.modeling.rolling_forecast import RollingForecast
from pytools.modeling.utilities import extract_model_settings
from pytools.modeling.weather_net import WeatherNet, default_layer_sizes, ModelSettings
from pytools.modeling.mlflow_helper import save_model, load_model
from pytools.modeling.weather_task_helpers import (
    get_npz_train_weather_file_name,
    get_training_data,
    prepare_train_data,
    prepare_train_model,
    get_model_checkpoint_path,
    decide_start_cur_end_time,
)
from pytools import get_logger
from pytools.data_prep.data_prep_manager import DataPrepManager
from pytools.data_prep import data_prep_manager as dpm
from pytools.data_prep.data_prep_manager_builder import (
    DataPrepManagerBuilder as Dpmb,
)
from pytools.data_prep import load_data_prep as ldp
from pytools.data_prep.get_datetime_from_grib_file_name import get_datetime_from_grib_file_name
from pytools.data_prep import weather_data_prep as wp
from pytools.config import Config

# use hour 0-5 forecast as history to train models, also called analysis

#nam_hist_max_fst_hour = 5
#nam_hist_max_fst_hour = 5
hrrr_hist_max_fst_hour = 0
logger = get_logger("weather_tasks")
weather_data_file_name = 'weather_data.npz'

def hist_load(
    config_file: str,
    t0: str = None,
    t1: str = None,
    create: bool = False,
    prefix='',
    suffix='v0'
) -> DataPrepManager:
    """
    Get historical load data to train a model.

    config_file: the configuration file from a toml file
    grib_type: grib file type, hrrr or nam
    t0: historical load data starting datetime. Override the t0 in the config.
    t1: historical load data ending datetime. Override the t1 in the config.
    create: create a new one without loading previous

    Returns: a data manager

    """
    config = Config(config_file)
    logger.info(f"... Check hist_load at {config.site_parent_folder}\n")
    dm = dpm.load(config, prefix=prefix, suffix=suffix)
    if not dm or create:
        logger.info("No existing manager detected or replace existing manager...\n")
        dm = Dpmb(
            config_file=config_file, t0=t0, t1=t1
        ).build_dm_from_config_weather(config=config)
        dm.build_weather(
            weather=config.weather,
            center=config.site["center"],
            rect=config.site["rect"],
        )
        # save the data manager with a weather object
        dpm.save(config=config, dmp=dm, prefix=prefix, suffix=suffix)
    else:
        logger.info("Use the existing manager...\n")
    return dm


def hist_weather_prepare_from_report(config_file:str, n_cores=1,prefix='',suffix='v0'):
    d = hist_load(config_file=config_file, create=False)
    logger.info(f"Creating historical npy data from {d.t0} to {d.t1}...\n")

    config = Config(config_file)
    #hour_offset = config.load["utc_to_local_hours"]
    d.build_weather(
        weather=config.weather,
        center=config.site["center"],
        rect=config.site["rect"],)

    parallel=False
    if n_cores > 1:
        parallel = True
    w_obj = d.weather.make_npy_data_from_inventory(
        center=config.site['center'],
        rect=config.site['rect'],
        inventory_file=config.weather_pdt.hist_weather_pickle,
        parallel=parallel,
        folder_col_name=config.weather_pdt.folder_col_name,
        filename_col_name=config.weather_pdt.filename_col_name,
        type_col_name=config.weather_pdt.type_col_name,
        t0=d.t0,
        t1=d.t1,
        n_cores=n_cores,
        )
    #w_obj.save_scaled_npz(osp.join(config.site_parent_folder, weather_data_file_name))
    #w_obj.save_unscaled_npz(osp.join(config.site_parent_folder, 'unscaled_'+weather_data_file_name))
    dpm.save(config=config, dmp=d, suffix=suffix)

    return d


def train_data_assemble(
    config_file: str, fst_horizon=1,suffix='v0'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Assemble training data. Save to a npz file.

    Args:
        config_file:
        grib_type:

    Returns: DataPrepManager with load and hist weather organized for training

    """
    d: DataPrepManager = hist_load(config_file=config_file, create=False)
   # w_data = np.load(osp.join(config.site_parent_folder, weather_data_file_name))
    h_weather = d.weather.get_weather_train()#.standardized_data #w_data['data']

    lag_data, calendar_data, data_standard_load = d.process_load_data(
        d.load_data, max_lag_start=fst_horizon,
    )
    cols_lag = list(lag_data)
    cols_calendar = list(calendar_data)
    cols_load = list(data_standard_load)
    join_load, join_wdata = d.reconcile(
        pd.concat([lag_data, calendar_data, data_standard_load], axis=1),
        d.load_data.date_col,
        h_weather,
    )
    cols_calendar.remove(d.load_data.date_col)
    # Let's set up the weather data scaler, and save the file with all weather data
    #join_wdata = d.standardize_weather(weather_array=join_wdata)
    cfg = Config(config_file)
    dpm.save(config=cfg, dmp=d, suffix=suffix)
    save_fn = get_npz_train_weather_file_name(cfg=cfg, suffix=suffix)
    np.savez_compressed(
        save_fn,
        weather=join_wdata,
        load_lag=join_load[cols_lag].values,
        calendar=join_load[cols_calendar].values,
        target=join_load[cols_load].values,
    )
    return (
        join_wdata,
        join_load[cols_lag],
        join_load[cols_calendar],
        join_load[cols_load],
    )


def train_model(
    config_file: str,
    ahead_hours: int,
    train_options,
    tracking_uri,
    model_uri,
    experiment_name,
    tags="",
    grib_type: wp.GribType = wp.GribType.hrrr,
):
    cfg = Config(config_file)
    cat_fraction = (
        train_options.cat_fraction if hasattr(train_options, "cat_fraction") else None
    )
    data = get_training_data(cfg, grib_type, cat_fraction)
    weather_para = data.get_weather_para()
    d = hist_load(config_file=config_file, create=False)
    train_data, validation_data, test_data = prepare_train_data(
        data,
        ahead_hrs=ahead_hours,
        batch_size=train_options.batch_size,
        num_workers=None,
        full_data=cat_fraction[0] == 1,
    )
    model, trainer = prepare_train_model(
        cfg=cfg,
        weather_para=weather_para,
        ahead_hours=ahead_hours,
        train_options=train_options,
        hist_load=d,
    )

    model.add_a_logger(cfg=cfg)
    ckpt_path = get_model_checkpoint_path(cfg, ahead_hours)
    if cat_fraction[0] == 1:
        sce_name = cfg.site["name"] + f"-fst-{ahead_hours}" + "-full"
        trainer.fit(model, train_data)
    else:
        trainer.fit(model, train_data, validation_data)
        sce_name = cfg.site["name"] + f"-fst-{ahead_hours}"
        trainer.test(test_dataloaders=test_data, ckpt_path=ckpt_path)

    trainer.save_checkpoint(ckpt_path)
    # save metric and model in airflow tracking
    save_model(
        metric=trainer.logged_metrics,
        artifact_file=config_file,
        model=model,
        sub_folder=sce_name,
        tracking_uri=tracking_uri,
        model_uri=model_uri,
        experiment_name=experiment_name,
        tags={s.split("=")[0]: s.split("=")[1] for s in tags.split(",")}
        if tags
        else {},
        parameters=None,
        run_name=sce_name,
    )
    return trainer


def run_predict(d: dpm, tp0: str, tp1: str) -> ldp.LoadData:
    """
    Get prediction load data

    Args:
        d: a data manager
        tp0: prediction starting time
        tp1: prediction end time

    Returns: LoadData

    """
    return d.get_prediction_load(t0=tp0, t1=tp1)


def predict_weather_prepare(
    config_file: str,
    max_hours_ahead=36,
    grib_type=wp.GribType.hrrr,
    current_time=None,
    rebuild_npy=False,
):
    """
    Make prediction weather npy files.

    Args:
        config_file: file name
        max_hours_ahead: max hours ahead. If a string, parse for a list
        grib_type:
        current_time: if not provided, will use the latest load time.
        rebuild_npy: if True, create new npy files. If False, reuse existing npy files.
      #  time_box: e.g. 2020-1-1T13:00,2020-1-12T16:00, UTC time. If one time is given, the starting time is t_cr
    Returns: None
    """
    config = Config(config_file)
    hour_offset = config.load["utc_to_local_hours"]
    d = hist_load(config_file=config_file, grib_type=grib_type, create=False)
    t0, t_cr, t1 = decide_start_cur_end_time(
        config, d, t_cr=current_time, max_hours=max_hours_ahead
    )
    filter_func_predict = partial(
        dpm.wp.grib_filter_func,
        func_timestamp=partial(
            get_datetime_from_grib_file_name,
            hour_offset=hour_offset,
            get_fst_hour=False,
        ),
        func_fst_hours=partial(
            get_datetime_from_grib_file_name, hour_offset=hour_offset, get_fst_hour=True
        ),
        predict=True,
        max_fst_hours=max_hours_ahead,
        time_box=[t_cr, t1],
    )
    # use all available prediction weather
    grib_folder = config.weather_folder["hrrr_predict"]
    if rebuild_npy:
        return d.make_npy_predict(
            time_after=t_cr,
            filter_func=filter_func_predict,
            in_folder=grib_folder,
        )


def get_predict_data(config_file, grib_type, current_time, max_hours_ahead):
    config = Config(config_file)
    d = hist_load(config_file=config_file, grib_type=grib_type, create=False)
    t0, t_cr, t1 = decide_start_cur_end_time(
        config, d, t_cr=current_time, max_hours=max_hours_ahead
    )
    df_pre = d.get_prediction_load(t0=t0, t1=t1, tc=t_cr)
    ind = df_pre[d.load_name].notna()
    df_pre.loc[ind, d.load_name] = d.load_scalar.transform(
        df_pre.loc[ind, [d.load_name]]
    )

    # read npy prediction weather
    p_weather = d.get_predict_weather()
    # join_load_pre, join_wdata_pre, load_scaler
    return (
        df_pre,
        d.load_data.date_col,
        d.load_name,
        p_weather,
        config.load["lag_hours"],
        d.load_scalar,
    )


def main(args):
    """
    The logic is:
    * Define a DataManager from a config; if exists, load it; otherwise generate it for train data.
    * Make hist weather; store the scaler--the weather data will be loaded into memory be part
    * Train model: load all npy data(include scaling), save model, generate report
    * Make predict weather
    * Load load prediction data, and sync with
    * Make prediction

    Returns: None

    """

    pa = ArgClass(args)
    args = pa.construct_args()
    task_dict = {
        "task_1": task_1,
        "task_2": task_2,
        "task_3": task_3,
        "task_4": task_4,
        "task_5": task_5,
        "task_6": task_6,
        "task_7": task_7,
    }
    fun = task_dict[args.pop("option")]
    return fun(**args)


def task_1(**args):
    dm = hist_load(**args)
    return dm


def task_2(**args):
    dm = hist_weather_prepare_from_report(**args)
    return dm


def task_3(**args):
    return train_data_assemble(**args)


def task_4(**args):
    return train_model(**args)


def task_5(**args):
    return predict_weather_prepare(**args)


def task_6(
    config_file: str,
    max_hours_ahead=36,
    grib_type=wp.GribType.hrrr,
    current_time=None,
    report_t0=None,
    report_t1=None,
):
    (
        df_load,
        load_timestamp_name,
        load_col_name,
        wea,
        load_embed_dim,
        load_scaler,
    ) = get_predict_data(
        config_file=config_file,
        grib_type=grib_type,
        current_time=current_time,
        max_hours_ahead=max_hours_ahead,
    )
    roll_f = RollingForecast(
        df_load=df_load,
        wea=wea,
        load_embed_dim=load_embed_dim,
        timestamp_name="timestamp",
    )
    cfg = Config(config_file)
    models = cfg.model["models"]
    for m in models:
        roll_f.add_model(hour_ahead=m["ahead"], model_path=m["uri"])
    y, start_time, hrs = roll_f.predict(hrs=max_hours_ahead)
    y[load_col_name] = load_scaler.inverse_transform(
        y[load_col_name].values.reshape((-1, 1))
    )
    if not report_t0:
        report_t0 = start_time
    if not report_t1:
        report_t1 = y["timestamp"].max()
    y.set_index("timestamp", inplace=True)
    dfr = y[report_t0:report_t1][load_col_name]
    cfg.report_predictions(
        df=dfr,
        start_time=report_t0,
        hours_ahead=(report_t1 - start_time) // np.timedelta64(1, "h"),
    )
    return dfr


def task_7(**args):
    raise NotImplementedError


if __name__ == "__main__":
    main(sys.argv[1:])
