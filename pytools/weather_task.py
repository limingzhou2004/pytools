import copy
from pathlib import Path
import pickle
import sys
import datetime as dt
from functools import partial
import os.path as osp
from typing import Tuple

import torch
import numpy as np
import pandas as pd

# from torch.utils.data.dataloader import DataLoader

# from pytools.modeling.dataset import WeatherDataSet
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.tuner import Tuner
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from tqdm import tqdm

from pytools.arg_class import ArgClass
from pytools.modeling.scaler import Scaler, load
from pytools.data_prep.herbie_wrapper import download_hist_fst_data
from pytools.modeling.dataset import WeatherDataSet, check_fix_missings, create_datasets, create_rolling_fst_data, get_hourly_fst_data, read_past_weather_data_from_config, read_weather_data_from_config
from pytools.modeling.rolling_forecast import RollingForecast
from pytools.modeling.ts_weather_net import TSWeatherNet, TsWeaDataModule
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

train_loader_settings = {'batch_size':30, 'shuffle':True, 'drop_last':True, 'pin_memory':True, 'num_workers':7}
test_loader_settings = {'batch_size':20, 'shuffle':False, 'drop_last':False, 'num_workers':7}
val_loader_settings = {'batch_size':20, 'shuffle':False, 'drop_last':False, 'num_workers':7}

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


def past_fst_weather_prepare(config_file, fst_hour=48, year=-1, month=-1):
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
    

def load_training_data(config:Config, yrs):
    if yrs=='-1':
        return read_weather_data_from_config(config, year=-1)
    years = yrs.split('-')
    y0 = int(years[0])
    y1 = int(years[-1])
    w_timestamp_list = []
    w_data_list = []
    inds = []
    for yr in range(y0, y1+1):
        load_data, w_paras, w_timestamp, w_data = read_weather_data_from_config(config, year=yr)
        w_timestamp_list.append(w_timestamp)
        w_data_list.append(w_data)
        inds.append(w_data.shape[0])
    return load_data, w_paras, np.concatenate(w_timestamp_list,axis=0), np.concatenate(w_data_list,axis=0)


def get_trainer(config:Config, use_val:bool=True):
    model_path = osp.join(config.site_parent_folder, 'model')
    setting = config.model_pdt.hyper_options
    early_stop_callback = EarlyStopping(
        monitor='val RMSE loss' if use_val else 'training RMSE loss',
        stopping_threshold=setting['stopping_threshold'],
        min_delta=setting['min_delta'],
        patience=setting['patience'],
        verbose=False,
        mode='min',
    )
    tb_logger = TensorBoardLogger(model_path, name='tensorboard_logger')
    csv_logger = CSVLogger(save_dir=model_path, name='csv_logger')
    trainer = pl.Trainer(
        default_root_dir=model_path,
        num_nodes=setting['num_nodes'],
       # callbacks=early_stop_callback,
        check_val_every_n_epoch=setting['check_val_every_n_epoch'],
        max_epochs=setting['max_epochs'],   
        logger=[tb_logger, csv_logger],
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
    load_data, w_paras, w_timestamp, w_data = load_training_data(config=config, yrs=args['years']) 
    p_list = [ a[1] for a in list(w_paras.item().items()) ]
    plist=[]
    for p in p_list:
        plist+=p
    p_adopted = [plist[i] for i in config.model_pdt.weather_para_to_adopt]    
    w_data=w_data[...,config.model_pdt.weather_para_to_adopt]
    logger.info(f'Use these weather parameters... {w_paras}')
    logger.info(f'Chosen {p_adopted}')
    load_arr, wea_arr, t = check_fix_missings(load_arr=load_data, w_timestamp=w_timestamp, w_arr=w_data)
    # import matplotlib.pyplot as plt
    # plt.scatter(load_arr[:,0],wea_arr[:,11,11,0])
    # #plt.show()
    # plt.savefig('test.png')
    # return

    wea_arr = wea_arr.astype(np.float32)
    load_arr = load_arr.astype(np.float32)
    num_worker = args['number_of_worker']
    ind = args['ind']
    ds_train, ds_val, ds_test = create_datasets(config, flag, tabular_data=load_arr, wea_arr=wea_arr,timestamp=t,sce_ind=ind)
 
    # add the batch dim
    wea_input_shape = [1, *wea_arr.shape]
    del wea_arr
    m = TSWeatherNet(wea_arr_shape=wea_input_shape, config=config)
    m.setup_mean(scaler=ds_train.scaler, target_std=ds_train.target_std, target_mean=ds_train.target_mean)
    if config.model_pdt.train_frac>=1:
        use_val = False
    else:
        use_val = True
    trainer = get_trainer(config, use_val=use_val)
    tuner = Tuner(trainer)   
    dm = TsWeaDataModule(batch_size=m.batch_size,num_worker=num_worker, ds_test=ds_test, ds_train=ds_train,ds_val=ds_val)
    if not ds_test:
        dm.test_dataloader = None
    if not ds_val:
        dm.val_dataloader = None
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
    trainer.fit(m, datamodule=dm,ckpt_path=trainer.ckpt_path)

    test_res = trainer.test(m, datamodule=dm, verbose=False)
    logger.info(f'test results: {test_res}')
    #$model_name='test.ckpt'
    model_name = args['model_name']+'.ckpt'
    ckpt_path = osp.join(config.site_parent_folder,'model', model_name)
    trainer.save_checkpoint(ckpt_path)

    #m2 =TSWeatherNet.load_from_checkpoint(ckpt_path)

    #m2.eval()

    #y = m2()


def task_4(**args):
    # load the model for predictions
    config = Config(args['config_file'])
    year = args['year']
    ckpt_path = osp.join(config.site_parent_folder, 'model', args['model_name']+'.ckpt')
    model:TSWeatherNet = TSWeatherNet.load_from_checkpoint(ckpt_path)
    # get weather data
    if args['t0'] != 'latest':
       load_data, wea_data =  read_past_weather_data_from_config(config=config, year=year)
    
    spot_t_list = wea_data[0]
    spot_t_list = [t.tz_localize('UTC') for t in spot_t_list]
    fst_t = wea_data[1][0][0]
    fst_t = [t.tz_localize('UTC') for t in fst_t]
    wea_arr = np.stack(wea_data[1][0][1], axis=0)

    load_timestamp_list = list(load_data[:,0])

    seq_length = model._seq_length
    #seq_length = config.model_pdt.seq_length
    #fst_horizon = args['rolling_fst_hzn']

    buffer = 10
    i = 0
    t_pointer = 0
    wea_arr_list = []
    tab_data_list = []
    w_timestamp = []
    rolling_fst_horizon = args['rolling_fst_hzn']
 
    while i<len(fst_t)-rolling_fst_horizon-buffer:
        #if spot_t_list[t_pointer] == fst_t[i]+pd.Timedelta(-1,'h'):
        cur_t:pd.Timestamp = spot_t_list[t_pointer]
        load_ind = load_timestamp_list.index(cur_t)
        load_ind_start = load_ind-seq_length - buffer
        load_ind_end = load_ind + rolling_fst_horizon + buffer
        ind_end = i
        for j in range(i, i+rolling_fst_horizon+buffer):
            if fst_t[j] > fst_t[j+1]:
                ind_end = j
                break 
        wea_start_ind = i - seq_length
        if wea_start_ind < 0:
            wea_start_ind = 0
        wea_arr_list.append(wea_arr[wea_start_ind:ind_end+1,:,:,config.model_pdt.weather_para_to_adopt])
        tab_data_list.append(load_data[0 if load_ind_start<0 else load_ind_start:load_ind_end,:])
        w_timestamp.append(fst_t[wea_start_ind:ind_end+1])
        if abs((fst_t[i] - cur_t)/pd.Timedelta(1,'h')) > 24:
            logger.warning(f'time difference between spot time {cur_t} and first fst time {fst_t[i]}exceeds 24 hours!')
        i = ind_end + 1
        t_pointer += 1

    # get scalers for target, wea array
    _fn_scaler = config.get_model_file_name(class_name='scaler')
 
    if Path(_fn_scaler).exists():
        scaler:Scaler = load(_fn_scaler)
    else:
        raise ValueError(f'sclaer file {_fn_scaler} does not exist!')
    
    res_spot_time = []
    res_fst_time = []
    res_actual_y = []
    res_fst_y = []

    for t, tdata, wt, wdata in tqdm(zip(spot_t_list, tab_data_list, w_timestamp, wea_arr_list)):
        logger.info(f'processing {t}...')
        # col 0 is timestamp, col 1 is the load/target
        df_load = pd.DataFrame(tdata).set_index(0)
        df, wea = create_rolling_fst_data(load_data=df_load,cur_t=t, w_timestamp=wt, wea_data=wdata, rolling_fst_horizon=rolling_fst_horizon,default_seq_length=seq_length)
        scaled_target = copy.deepcopy(scaler.scale_target(df.values[:,0]))
        #find the ind of hr=0
        ind_0 = list(df.index).index(t)
        scaled_wea = np.stack(scaler.scale_arr(wea), axis=0)

        for hr in range(1, 2): #rolling_fst_horizon+1):
            seq_wea_arr, seq_ext_arr, seq_target, wea_arr, ext_arr, target = \
            get_hourly_fst_data(target_arr=scaled_target, 
                                    ext_arr=df.values[:,1:], 
                                    wea_arr=scaled_wea, 
                                    hr=hr, seq_length=seq_length)
            with torch.no_grad():
                y = model(seq_wea_arr=seq_wea_arr,
                          seq_ext_arr=seq_ext_arr,
                          seq_target=seq_target,
                          wea_arr=wea_arr,
                          ext_arr=ext_arr)
            y = scaler.unscale_target(y)
            res_spot_time.append(t)
            res_fst_time.append(t+pd.Timedelta(hr, 'h'))
            res_actual_y.append(target.item())
            res_fst_y.append(y.item())
            scaled_target[ind_0+hr] = y

    res_df = pd.DataFrame(res_spot_time,columns=['spot_time'])
    res_df['fst_time'] = res_fst_time
    res_df['target'] = res_actual_y
    res_df['fst'] = res_fst_y

    res_df['mae'] = abs(res_df['fst'] - res_df['target'])
    mae = res_df['mae'].mean()
    mean_target = res_df['target'].mean()
    rmae = mae/mean_target
    logger.info(f'mae = {mae}, rmae={rmae}, mean_target={mean_target}')
    res_df.to_pickle(f'past-test-{year}.pkl')






    #df_result =  # timestamp, actual load, fst load



    #y = m2()


    return 


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
# -fh, forecast hours
# python -m pytools.weather_task -cfg pytools/config/albany_test.toml task_2 -fh 2 --n-cores 1 -year 2020 -flag hf
# python -m pytools.weather_task -cfg pytools/config/albany_prod.toml task_2 --n-cores 1 -year 2018 -flag h
# python -m pytools.weather_task -cfg pytools/config/albany_prod.toml task_2 -fh 48 --n-cores 1 -year 2024 -flag f

# python -m pytools.weather_task -cfg pytools/config/albany_test.toml task_3 --flag cv --ind 0 -sb [find_batch_size|find_lr|fit] -mn test0
    main(sys.argv[1:])
