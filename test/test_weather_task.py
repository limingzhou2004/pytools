#import mlflow
import numpy as np
import pytest

from pytools.config import Config
from pytools.data_prep.load_data_prep import LoadData
from pytools.mocking_utils import mock_train_load, mock_predict_load, mock_max_date
from pytools.weather_task import (
    hist_load,
   # hist_weather_prepare,
    hist_weather_prepare_from_report,
    load_training_data,
    main,
    past_fst_weather_prepare,
   # train_model,
)
import pytools.data_prep.weather_data_prep as wp
from pytools.utilities import get_absolute_path


class TestWeatherTask:
    config_file = get_absolute_path(__file__, '../pytools/config/albany_test.toml')

    def test_commandline_task1(self):
        cmd_str =f'-cfg {self.config_file} task_2 -flag h'
        main(cmd_str.split(' '))
        #cmd_str = f'-cfg {self.config_file} task_2 -fh 2 -flag f'
        #main(cmd_str.split(' '))

    def test_commandline_task3(self):
        cmd_str = f'-cfg {self.config_file} task_3 --flag cv -ind 0 -mn test0 -yr -1 -nworker 3'
        main(cmd_str.split(' '))
        assert 1==1

    def test_commandline_task3_prod(self):
        cfile = get_absolute_path(__file__,'../pytools/config/albany_prod.toml')
        cmd_str = f'-cfg {cfile} task_3 --flag cv -ind 0 -mn prod0 -yr 2018-2023'
        main(cmd_str.split(' '))
        assert 1==1

    def test_commandLine_task4(self):
        cmd_str = f'-cfg {self.config_file} task_4 --flag test -mn test0 -ind 0'
        main(cmd_str.split(' '))
        assert 1==1

    def test_task1_hist_load(self, ):
        #monkeypatch.setattr(LoadData, "query_train_data", mock_train_load)
        res = hist_load(config_file=self.config_file,create=True)
        assert res.load_data.train_data.shape[1] == 10
        assert res.load_data.train_data.shape[0] >= 2

    def test_hist_weather_from_inventory(self):
        dm = hist_weather_prepare_from_report(config_file=self.config_file, n_cores=4)
        assert dm.weather.weather_train_data.standardized_data.shape==(49, 21, 21, 16)

    def test_load_weather_data(self, ):
        c = Config(get_absolute_path(__file__,'../pytools/config/albany_prod.toml'))
        res = load_training_data(config=c, yrs='2020-2023')
        assert res[3].shape[0] > 30000

    def test_past_weather_fst(self):
        past_fst_weather_prepare(self.config_file, fst_hour=2, year=2020)

        assert 1==1


        

    @pytest.mark.parametrize(
        "shape_cal, shape_weather",
        [
            ((68, 7), (68, 35, 35, 16)),
        ],
    )
    def test_train_data_assemble(self, shape_cal, shape_weather):
        weather_data, lag_load, calendar, target_load = train_data_assemble(
            config_file=self.config_file
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
