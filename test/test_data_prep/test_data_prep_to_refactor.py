import datetime as dt
import dill as pickle
import os

import numpy as np
import pandas as pd
import pytest

from pytools.data_prep import calendar_data_prep as CD
from pytools.data_prep import load_data_prep as ldp
from pytools.data_prep import query_str_fill
from pytools.data_prep import data_prep_manager as dpm
from pytools.data_prep import py_jar as Pj
from pytools.data_prep import weather_data_prep as wdp


@pytest.mark.skip("to drop")
class TestDataPrep:
    def test_query_str_fill(self):
        ret = query_str_fill(qstr="test{t0}test", t0="_filled_")
        assert ret == "test_filled_test"

    def test_calendar_data(self):
        cd = CD.CalendarData()
        cd.construct_calendar_data()
        # cd.load_to_db(schema="zhoul", table="calendar")
        # cd.load_daylightsaving_to_db(schema="zhoul", table="daylightsaving")

    def test_daylightsaving_data(self):
        cd = CD.CalendarData()
        assert cd.is_daylightsaving(np.datetime64("2018-10-01 13:00"))

    def test_hourofday(self):
        df = pd.DataFrame(
            pd.to_datetime(["2018-01-01 23:00", "2018-01-01 00:00"]),
            columns=["timestamp"],
        )
        hourofday = CD.CalendarData.get_hourofday(df["timestamp"])
        dayofweek = CD.CalendarData.get_dayofweek(df["timestamp"])
        assert dayofweek
        assert hourofday

    def test_load_data(self, yaml_file):
        mwh = ldp.LoadData.build_nyiso_load(
            yaml_file, "nyiso_hist_load", "CAPITL", t0="2019-01-01", t1="2019-01-03"
        )
        # last_time = mwh.get_last_load_time()
        fn = "../temp.pkl"
        with open(fn, "wb") as dill_file:
            pickle.dump(mwh, dill_file)
        # print(last_time)

    def test_data_prep_manager(self):
        ny_manager = dpm.DataPrepManager.build_nyiso_hist_load_prep(
            site_name="CAPITL",
            site_description="capital",
            site_folder="/users/limingzhou/zhoul/work/me/xaog_ops/modeling",
            t0="2018-01-01",
            t1="2018-11-30",
        )
        assert ny_manager.data_standard_load_lag.shape[1] == 168

    # def test_py_jar(self):
    #     data_folder = "/Users/limingzhou/zhoul/work/me"
    #     f_in = data_folder + "/testdata/hrrrsub_2018_10_06_00F0"
    #     f_out = data_folder + "/testdata/output2/test_hrrr"
    #     # paras_file = "/Users/limingzhou/zhoul/work/me/xaog_ops/modeling/sites/nyiso/nam_paras.txt"
    #     pj = Pj.PyJar(
    #         jar_address=self.jar_address,
    #         folder_in=f_in,
    #         folder_out=f_out,
    #         paras_file=self.paras_file,
    #         center='"(43,-73.0)"',
    #         rect='"(100.0,100.0)"',
    #     )
    #     # pj.process_a_grib(f_in=f_in, f_out=f_out)
    #     f_in = data_folder + "/testdata/nam.t00z.conusnest.hiresf00.tm00.grib2"
    #     f_out = data_folder + "/testdata/output2/test_nem"
    #     pj.process_a_grib(f_in=f_in, f_out=f_out)

    # def test_py_jar_process_folder(self):
    #     data_in_folder = "/Users/limingzhou/zhoul/work/me/testdata/naminput"
    #     data_out_folder = "/Users/limingzhou/zhoul/work/me/testdata/output3"
    #     pj = Pj.PyJar(
    #         jar_address=self.jar_address,
    #         folder_in=data_in_folder,
    #         folder_out=data_out_folder,
    #         paras_file=self.nam_paras_file,
    #         center='"(43,-73.0)"',
    #         rect='"(100.0,100.0)"',
    #     )
    #     pj.process_folders(out_prefix="nam_test_", out_suffix=".npy", parallel=True)

    def test_weather_data_prep(self):
        npy_folder = "/Users/limingzhou/zhoul/work/me/testdata/output5"
        w = wdp.WeatherDataPrep.build_hrrr(
            weather_folder=self.hrrr_data_in_folder, dest_npy_folder=npy_folder
        )
        t = dt.datetime(2018, 12, 26)
        w.make_npy_data(
            center='"(43,-73.0)"',
            rect='"(100.0,100.0)"',
            prefix="hrrr_weather_test_",
            last_time=t,
        )

    def test_weather_nam_data_prep(self):
        npy_folder = "/Users/limingzhou/zhoul/work/me/testdata/output6"
        w = wdp.WeatherDataPrep.build_nam(
            weather_folder=self.nam_data_in_folder, dest_npy_folder=npy_folder
        )
        t = dt.datetime(2018, 12, 24)
        w.make_npy_data(center='"(43,-73.0)"', rect='"(100.0,100.0)"', last_time=t)
        w2 = wdp.WeatherDataPrep.build_nam(
            weather_folder=[self.nam_data_in_folder], dest_npy_folder=npy_folder
        )
        # w2.make_npy_data(center="\"(43,-73.0)\"", rect="\"(100.0,100.0)\"", last_time=t)
        assert w2

    def test_load_npy(self):
        npy_folder = "/Users/limingzhou/zhoul/work/me/testdata/output5"
        fn = os.path.join(
            npy_folder, "hrrr_weather_test_hrrrsub_2018_12_26_00F0.grib2.npy"
        )
        w = wdp.WeatherDataPrep.build_hrrr(
            weather_folder=[self.hrrr_data_in_folder], dest_npy_folder=npy_folder
        )
        # res = w.load_a_npy(fn, 13)
        # train_data = w.get_weather_train()
        # predict_data = w.get_weather_predict()

        assert fn
        assert w
        assert 1 == 1
