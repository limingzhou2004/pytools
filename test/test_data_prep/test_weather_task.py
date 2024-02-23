import mlflow
import numpy as np
import pytest

from pytools.data_prep.weather_data_prep import GribType
from pytools.data_prep.load_data_prep import LoadData
from pytools.mocking_utils import mock_train_load, mock_predict_load, mock_max_date
from pytools.data_prep.weather_task import (
    hist_load,
    hist_weather_prepare,
    main,
    train_data_assemble,
    train_model,
)
import pytools.data_prep.weather_data_prep as wp
from pytools.utilities import get_absolute_path


class TestWeatherTask:
    config_file = get_absolute_path(__file__, "../../pytools/config/albany_test.toml")

    def test_hist_load(self, train_t0, train_t1, monkeypatch):
        monkeypatch.setattr(LoadData, "query_train_data", mock_train_load)
        res = hist_load(
            config_file=self.config_file,
            grib_type=GribType.hrrr,
            t0=train_t0,
            t1=train_t1,
        )
        assert res.load_data.train_data.shape == (2, 4)

    def test_hist_weather(self):
        dm = hist_load(config_file=self.config_file, grib_type=GribType.hrrr)
        assert dm.weather_type == GribType.hrrr
        hist_weather_prepare(
            config_file=self.config_file, grib_type=GribType.hrrr, t_after="2018-12-24"
        )

    @pytest.mark.parametrize(
        "gribtype, shape_cal, shape_weather",
        [
            (GribType.hrrr, (32, 6), (32, 34, 34, 13)),
        ],
    )
    def test_train_data_assemble(self, gribtype, shape_cal, shape_weather):
        weather_data, lag_load, calendar, target_load = train_data_assemble(
            config_file=self.config_file, grib_type=gribtype
        )
        assert calendar.shape == shape_cal
        assert weather_data.shape == shape_weather

    @pytest.mark.parametrize(
        "argument_str, shape",
        [
            (
                "-c ../../pytools/config/albany_test.toml task_1 -cr -t0 1/1/2020 -t1 1/5/2020 ",
                (97, 8),
            ),
        ],
    )
    def test_main_1(self, argument_str, shape, monkeypatch):
        monkeypatch.setattr(LoadData, "query_train_data", mock_train_load)
        dm = main(argument_str.split())
        assert dm.load_data.train_data.shape == shape

    @pytest.mark.parametrize(
        "argument_str, shape",
        [
            ("-c ../../pytools/config/albany_test.toml task_2 -ta 1/1/2020", ()),
        ],
    )
    def test_main_2(self, argument_str, shape):
        dm = main(argument_str.split())
        assert dm.weather.grib_type == wp.GribType.hrrr

    @pytest.mark.parametrize(
        "argument_str, shape",
        [
            ("-c ../../pytools/config/albany_test.toml task_3 ", ()),
        ],
    )
    def test_main_3(self, argument_str, shape):
        join_wdata, join_load_lag, join_load_cal, join_load_target = main(
            argument_str.split()
        )
        assert join_wdata.shape == (97, 34, 34, 12)
        assert join_load_lag.shape == (97, 168)
        assert join_load_cal.shape == (97, 6)
        assert join_load_target.shape == (97, 1)
        assert np.allclose(
            join_load_lag.values[1:, 0], join_load_target.values[0:-1, 0]
        )

    @pytest.mark.parametrize(
        "argument_str",
        [
            (
                "-c ../../pytools/config/albany_test.toml task_4 "
                '-to {"batch_size":10,"cat_fraction":[1,0,0],"epoch_num":3} -ah 1 -env albany3'
            ),
            (
                "-c ../../pytools/config/albany_test.toml task_4 "
                '-to {"batch_size":10,"cat_fraction":[0.8,0.1,0.1]} -ah 6 --tag cat=test,name=test -env albany3'
            ),
        ],
    )
    def test_main_4(self, argument_str):
        """
        Train models

        Args:
            argument_str:

        Returns:

        """
        trainer = main(argument_str.split())
        assert trainer.current_epoch > 0

    @pytest.mark.parametrize(
        "argument_str, row_count",
        [
            (
                "-c ../../pytools/config/albany_test.toml task_5 "
                "-mha 48 --rebuild-npy yes",
                0,
            ),
            (
                "-c ../../pytools/config/albany_test.toml task_5 "
                "-mha 28 -tc 2020-1-9T15:00 --rebuild-npy yes ",
                29,
            ),
        ],
    )
    def test_main_5(self, argument_str, row_count, monkeypatch):
        """
        Prepare weather prediction npy data.

        Args:
            argument_str:
            row_count:
            monkeypatch:

        Returns:

        """
        monkeypatch.setattr(LoadData, "query_max_load_time", mock_max_date)
        monkeypatch.setattr(LoadData, "query_predict_data", mock_predict_load)
        r = main(argument_str.split())
        assert r == row_count

    @pytest.mark.parametrize(
        "argument_str",
        [
            "-c ../../pytools/config/albany_test.toml task_6 -mha 28 -tc 2020-1-9T15:00",
        ],
    )
    def test_main_6(self, argument_str, monkeypatch):

        monkeypatch.setattr(LoadData, "query_max_load_time", mock_max_date)
        monkeypatch.setattr(LoadData, "query_predict_data", mock_predict_load)
        r = main(argument_str.split())

        assert 1
