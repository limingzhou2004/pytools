# import os
import datetime as dt

# from typing import List

import pytest
import pandas as pd
from dateutil import parser


from pytools.data_prep.data_prep_manager_builder import DataPrepManagerBuilder as dpmb

from pytools.data_prep import data_prep_manager as dm

# from pytools.pytools.data_prep import weather_data_prep as wp
from pytools.config import Config


class TestDataManager:
    def test_training_data_save(
        self,
        cur_toml_file,
        train_t0,
        train_t1,
        suffix_hrrr,
        suffix_nam,
        mock_train_load,
        mock_predict_load,
        mock_max_date,
    ):
        dm0, config = dpmb(
            config_file=cur_toml_file, train_t0=train_t0, train_t1=train_t1
        ).build_dm_from_config()
        dm.save(config=config, dmp=dm0["hrrr"], suffix=suffix_hrrr)
        dm1: dm.DataPrepManager = dm.load(config=config, suffix=suffix_hrrr)
        dm.save(config=config, dmp=dm0["nam"], suffix=suffix_nam)
        dm2 = dm.load(config=config, suffix=suffix_nam)
        tt0 = "2018-12-29"
        tt1 = "2019-01-02"
        tt0_nam = "2019-02-01"
        tt1_nam = "2019-02-03"
        ld = dm1.get_prediction_load(t0=tt0, t1=tt1)
        dm1.standardize_predictions(ld)
        ld2 = dm2.get_prediction_load(t0=tt0_nam, t1=tt1_nam)
        dm2.standardize_predictions(ld2)
        dm2.process_load_data()
        # dm.save(config, dm1, suffix="model_pred_hrrr.pickle")
        # dm.save(config, dm2, suffix="model_pred_nam.pickle")
        assert dm1.load_data.train_data.shape == (40, 8)

    @pytest.mark.skip("not used as the binary data are not available to test")
    @pytest.mark.parametrize(
        "file_name,datetime",
        [
            ("model_pred_hrrr.pickle", "2019-01-04 15:00"),
            ("model_pred_nam.pickle", "2019-02-02 02:00"),
        ],
    )
    @pytest.mark.skip("no grib data to test")
    def test_make_npy_data(self, cur_toml_file, file_name, datetime):
        config = Config(filename=cur_toml_file)
        d: dm = dm.load(config=config, suffix=file_name)
        d.build_weather(
            weather_folder=config.weather_folder,
            jar_address=config.jar_config,
            center=config.site["center"],
            rect=config.site["rect"],
        )
        d.make_npy_train()
        d.make_npy_predict(time_after=dt.datetime.strptime(datetime, "%Y-%m-%d %H:%M"))

    @pytest.fixture(
        params=[
            ("model_pred_hrrr.pickle", "2018-12-26"),
            ("model_pred_nam.pickle", "2018-12-26"),
        ],
        ids=["hrrr", "nam"],
    )
    def weather_para(self, request):
        return request.param

    @pytest.fixture(params=["2018-12-26", "2018-12-26"])
    def datetime_cut(self, request):
        return pd.to_datetime(request.param)

    def test_make_npy_data_with_filter(self, cur_toml_file, weather_para):
        config = Config(filename=cur_toml_file)
        file = weather_para[0]
        tt = weather_para[1]
        d: dm.DataPrepManager = dm.load(config=config, suffix=file)
        d.build_weather(
            weather_folder=config.weather_folder,
            jar_address=config.jar_config,
            center=config.site["center"],
            rect=config.site["rect"],
        )
        # build a filter for weather files
        # timezone shift is handled in the weather object

        def fn_train(x: str) -> bool:
            spot_time = d.weather.extract_datetime_from_grib_filename(
                x, nptime=True, get_fst_hour=False
            )
            return spot_time > parser.parse(tt)

        d.make_npy_train(filter_func=fn_train, parallel=True)
        d.make_npy_predict(
            out_folder=None, time_after=parser.parse(tt), filter_func=fn_train
        )
        dm.save(config, d, suffix=file)

    def test_reconcile(self, cur_toml_file, weather_para):
        config = Config(filename=cur_toml_file)
        file = weather_para[0]
        tt = weather_para[1]
        tt1 = "2018-12-27"
        d: dm.DataPrepManager = dm.load(config=config, suffix=file)
        h_weather = d.get_train_weather()
        # for hrrr, get all files in the folder between t0 and t1
        p_weather = d.get_predict_weather()
        join_load, join_wdata = d.reconcile(
            d.load_data.train_data, d.load_data.date_col, h_weather
        )
        join_load_pre, join_wdata_pre = d.reconcile(
            d.load_data.query_predict_data(t0=tt, t1=tt1),
            d.load_data.date_col,
            p_weather,
        )
        assert join_load.shape[0] == join_wdata.shape[0]
        assert join_load_pre.shape[0] == join_wdata_pre.shape[0]
