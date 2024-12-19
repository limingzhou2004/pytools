import pickle
import sys
import datetime as dt
from functools import partial
import os.path as osp
from typing import Tuple

import torch
from dateutil import parser
import mlflow
import numpy as np
import pandas as pd

# from torch.utils.data.dataloader import DataLoader

# from pytools.modeling.dataset import WeatherDataSet
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.tuner import Tuner
import lightning as pl
from torch.utils.data import DataLoader

from pytools.arg_class import ArgClass
from pytools.data_prep.herbie_wrapper import download_hist_fst_data
from pytools.modeling.dataset import WeatherDataSet, check_fix_missings, read_weather_data_from_config
from pytools.modeling.rolling_forecast import RollingForecast
from pytools.modeling.ts_weather_net import TSWeatherNet, TsWeaDataModule
from pytools.modeling.utilities import extract_model_settings
from pytools.modeling.weather_net import WeatherNet, default_layer_sizes, ModelSettings
from pytools.modeling.mlflow_helper import save_model, load_model
from pytools.modeling.weather_task_helpers import (
  #  get_npz_train_weather_file_name,
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
from pytools.config import Config, DataType

# use hour 0-5 forecast as history to train models, also called analysis

#nam_hist_max_fst_hour = 5
#nam_hist_max_fst_hour = 5
hrrr_hist_max_fst_hour = 0
logger = get_logger("weather_tasks")
#weather_data_file_name = 'weather_data.npz'

train_loader_settings = {'batch_size':256, 'shuffle':True, 'drop_last':True, 'pin_memory':True, 'num_workers':4}
test_loader_settings = {'batch_size':256, 'shuffle':False, 'drop_last':False, 'num_workers':4}


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
    dm = None
    if not create:
        dm = dpm.load(config, prefix=prefix, suffix=suffix)
    if not dm or create:
        logger.info("No existing manager detected or replace existing manager...\n")
        dm = Dpmb(
            config_file=config_file, t0=t0, t1=t1
        ).build_dm_from_config_weather(config=config)
        # save the data manager with a weather object
        dpm.save(config=config, dmp=dm, prefix=prefix, suffix=suffix)

        # write the load data to npy 
        fn = config.get_load_data_full_fn(data_type=DataType.LoadData, extension='npz')
        cols, load = dm.export_data(DataType.LoadData)
        logger.info(f'column names: {cols}')
        np.savez_compressed(fn, **{DataType.LoadData.name:load, 'columns':cols})       
        
    else:
        logger.info("Use the existing manager...\n")
    return dm


def hist_weather_prepare_from_report(config_file:str, n_cores=1, suffix='v0', create=False, fst_hour=48,year=-1):
    d = hist_load(config_file=config_file, create=create)
    logger.info(f"Creating historical npy data from {d.t0} to {d.t1}\n")
    if year>0:
        logger.info(f'process year {year} only...')

    config = Config(config_file)
    d.build_weather(
        weather=config.weather,
        center=config.site["center"],
        rect=config.site["rect"],)

    parallel=False
    if n_cores > 1:
        parallel = True
    w_obj = d.weather.make_npy_data_from_inventory(
        config=config,
        parallel=parallel,
        t0=d.t0,
        t1=d.t1,
        n_cores=n_cores,
        year=year
        )
    #w_obj.save_scaled_npz(osp.join(config.site_parent_folder, weather_data_file_name))
    #w_obj.save_unscaled_npz(osp.join(config.site_parent_folder, 'unscaled_'+weather_data_file_name))
    fn = config.get_load_data_full_fn(data_type=DataType.Hist_weatherData, extension='npz', year=year)
    # if year>0:
    #     fn = f'{fn}_{year}'
    # use paras[()] to access the OrderedDict in the 0-dim paras array.
    paras, w_timestamp, wdata = d.export_data(DataType.Hist_weatherData)

    np.savez_compressed(fn, **{'paras':paras, 'timestamp':w_timestamp, DataType.Hist_weatherData.name:wdata})
    #dpm.save(config=config, dmp=d, suffix=suffix)

    return d


def past_fst_weather_prepare(config_file:str, fst_hour=48, year=-1, month=-1):
    logger.info('Create past weather forecast, with forecast horizon of {fst_hour}...\n')
    if year>0:
        logger.info(f'Process year {year} only...')
    c = Config(config_file)
    paras_file = c.automate_path(c.weather_pdt.hrrr_paras_file)
    spot_time, weather_arr = download_hist_fst_data(t_start=c.site_pdt.back_fst_window[0], t_end=c.site_pdt.back_fst_window[1], fst_hr=fst_hour, 
    paras_file=paras_file,envelopes=c.weather_pdt.envelope, year=year)
    fn = c.get_load_data_full_fn(DataType.Past_fst_weatherData, extension='pkl', year=year, month=month)
    with open(fn, 'wb') as f:
        pickle.dump([spot_time, weather_arr],f)
    

def get_trainer(config:Config):
    setting = config.model_pdt.model_settings

    early_stop_callback = EarlyStopping(
        monitor="validation_epoch_average",
        min_delta=setting['min_delta'],
        patience=setting['patience'],
        verbose=False,
        mode="min",
    )
    trainer = pl.Trainer(
        callbacks=early_stop_callback,
        check_val_every_n_epoch=setting['epoch_step'],
        max_epochs=setting['epoch_num'],
    )
    return trainer


# def train_data_assemble(
#     config_file: str, suffix='v0', 
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Assemble training data. Fill missing load and/or weather data.

#     Args:
#         config_file:
#         fst_horizon: forecast horizon, 1, 6, 24
#         suffix: id, will add fst_horizon

#     Returns: DataPrepManager with load and hist weather organized for training

#     """
#     #d: DataPrepManager = hist_load(config_file=config_file, create=False)
#     #h_weather = d.weather.get_weather_train()  
#     c: Config = Config(config_file)

    
#     lag_data, calendar_data, data_standard_load = d.process_load_data(
#         d.load_data, lag_hours=c.load_pdt.lag_hours, fst_horizon=c.load_pdt.fst_hours
#     )
#     cols_lag = list(lag_data.keys())
#     cols_calendar = list(calendar_data)
#     cols_load = list(data_standard_load)
#     join_load, join_wdata = d.reconcile(
#         pd.concat([lag_data, calendar_data, data_standard_load], axis=1),
#         d.load_data.date_col,
#         h_weather,
#     )
#     cols_calendar.remove(d.load_data.date_col)
#     # Let's set up the weather data scaler, and save the file with all weather data
#     #join_wdata = d.standardize_weather(weather_array=join_wdata)
#     cfg = Config(config_file)
#     dpm.save(config=cfg, dmp=d, suffix=suffix)
#     save_fn = get_npz_train_weather_file_name(cfg=cfg, suffix=suffix)
#     np.savez_compressed(
#         save_fn,
#         weather=join_wdata,
#         load_lag=join_load[cols_lag].values,
#         calendar=join_load[cols_calendar].values,
#         target=join_load[cols_load].values,
#     )
#     return (
#         join_wdata,
#         join_load[cols_lag],
#         join_load[cols_calendar],
#         join_load[cols_load],
#     )


def train_model(
    config_file: str,
   # fst_hours: int,
    train_options,
    tracking_uri,
    model_uri,
    experiment_name,
    tags="",
):
    cfg = Config(config_file)
    cat_fraction = (
        train_options.cat_fraction if hasattr(train_options, "cat_fraction") else None
    )
    data = get_training_data(cfg, cat_fraction, fst_hour=fst_hours)
    weather_para = data.get_weather_para()
    d = hist_load(config_file=config_file, create=False)
    train_data, validation_data, test_data = prepare_train_data(
        data,
        ahead_hours=fst_hours,
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

    task_dict = {
        "task_1": task_1,
        "task_2": task_2,
        "task_3": task_3,
        "task_4": task_4,
        "task_5": task_5,
        "task_6": task_6,
        "task_7": task_7,
    }
    pa = ArgClass(args, list(task_dict.values()))
    fun, args = pa.construct_args_dict()

    return fun(**args)


def task_1(**args):
    dm = hist_load(**args)
    return dm


def task_2(**args):
    flag = args['flag']
    del args['flag']
    if 'h' in flag:
        del args['month']
        hist_weather_prepare_from_report(**args)
    if 'f' in flag:
        past_fst_weather_prepare(config_file=args['config_file'], fst_hour=args['fst_hour'], year=args['year'])


def task_3(**args):
    flag = args['flag']
    config = Config(args['config_file'])
    load_data, w_paras, w_timestamp, w_data = read_weather_data_from_config(config, year=-1)
    logger.info(f'Use these weather parameters... {w_paras}')
    load_arr, wea_arr, t = check_fix_missings(load_arr=load_data, w_timestamp=w_timestamp, w_arr=w_data)
    wea_arr = wea_arr.astype(np.float32)
    load_arr = load_arr.astype(np.float32)

    ind = args['ind']
    if flag.startswith('cv'):
        prefix = 'cv'
        ds_test =  WeatherDataSet(flag=f'{prefix}_test',tabular_data=load_arr, wea_arr=wea_arr, timestamp=t, config=config, sce_ind=ind)
        loader_test = DataLoader(ds_test, **test_loader_settings)
        ds_val =  WeatherDataSet(flag=f'{prefix}_val',tabular_data=load_arr, wea_arr=wea_arr, timestamp=t, config=config, sce_ind=ind)
        loader_val = DataLoader(ds_val, **test_loader_settings)
    elif flag.startswith('final'):
        prefix= 'final_train'
    
    train_flag = f'{prefix}_train'
    ds_train = WeatherDataSet(flag=train_flag,tabular_data=load_arr, wea_arr=wea_arr, timestamp=t, config=config, sce_ind=ind)
    
    #loader_train = DataLoader(ds_train, **train_loader_settings)
    # add the batch dim
    wea_input_shape = [1, *wea_arr.shape]
    m = TSWeatherNet(wea_arr_shape=wea_input_shape, config=config)
    m.setup_mean(scaler=ds_train.scaler, target_mean=ds_train.target_mean)
    trainer = get_trainer(config)
    tuner = Tuner(trainer)

    #m.to('mps')
    #ldm_loader_train = pl.LightningDataModule(loader_train)
    def train_dl():
        return DataLoader(ds_train, batch_size=m.batch_size)
    
    def test_dl():
        return DataLoader(ds_test, batch_size=m.batch_size)
    
    def val_dl():
        return DataLoader(ds_val, batch_size=m.batch_size)
    
    dm = TsWeaDataModule(batch_size=m.batch_size)
    dm.train_dataloader = train_dl
    dm.test_dataloader = test_dl
    dm.val_dataloader = val_dl
    sub_task = args['sub']
    if sub_task == 'find_batch_size':
        tuner.scale_batch_size(m, datamodule=dm)
        return
    if sub_task == 'find_lr':
        lr_finder = tuner.lr_find(m, datamodule=dm)
        print(lr_finder.results)
        # Plot with
        fig = lr_finder.plot(suggest=True)
        fig.show()

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()
        logger.info(f'suggested lr: {new_lr}')
        return

    # update hparams of the model
    #m.hparams.lr = new_lr
    # Fit model
    trainer.fit(m, datamodule=dm)

    test_res = trainer.test(m, datamodule=dm, verbose=False)
    logger.info(f'test results: {test_res}')



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
# python -m pytools.weather_task -cfg pytools/config/albany_test.toml --create task_1 
# python -m pytools.weather_task -cfg pytools/config/albany_test.toml task_2 -fh 2 --n-cores 1 -year 2020 -flag hf
# python -m pytools.weather_task -cfg pytools/config/albany_prod.toml task_2 --n-cores 1 -year 2018 -flag h
# python -m pytools.weather_task -cfg pytools/config/albany_prod.toml task_2 -fh 48 --n-cores 1 -year 2024 -flag f
    main(sys.argv[1:])
