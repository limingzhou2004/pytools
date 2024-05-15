import numpy as np
import pandas as pd

from pytools.data_prep.load_data_prep import build_from_toml
from pytools.data_prep import calendar_data_prep as CD

from pytools.data_prep import load_data_prep as ldp
from pytools.data_prep import query_str_fill

# from pytools.data_prep import data_prep_manager as dpm
# from pytools.data_prep import py_jar as Pj
# from pytools.data_prep import weather_data_prep as wdp


class TestLoadDataPrep:
    def test_build_from_toml(self, cur_toml_file, train_t0, train_t1):
        mwh = build_from_toml(config_file=cur_toml_file, t0=train_t0, t1=train_t1)
        assert mwh.train_data.shape == (40, 8)

    def test_query_str_fill(self):
        ret = query_str_fill(qstr="test{t0}test", t0="_filled_")
        assert ret == "test_filled_test"

    @DeprecationWarning
    def test_calendar_data(self):
        cd = CD.CalendarData()
        res = cd.construct_calendar_data()
        assert res.shape[1] == 5
        # cd.load_to_db(schema="zhoul", table="calendar")
        # cd.load_daylightsaving_to_db(schema="zhoul", table="daylightsaving")

    @DeprecationWarning
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
        assert dayofweek[1].iloc[0, 1] == 1
        assert np.isclose(hourofday[1].iloc[0, 1], 0.9659258)


def test_build_from_toml(config, train_t0, train_t1):
    mwh = ldp.build_from_toml(config_file=config, t0=train_t0, t1=train_t1)
    assert mwh.train_data.shape == (40, 8)
