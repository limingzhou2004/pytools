from pytools.data_prep.data_prep_manager_builder import (
    DataPrepManagerBuilder as Dpmb,
)
from pytools.data_prep.weather_data_prep import GribType


class TestDataPrepMangerBuilder:
    def test_build_load_data_from_config(self, cur_toml_file, train_t0, train_t1):
        b = Dpmb(
            config_file=cur_toml_file, train_t0=train_t0, train_t1=train_t1
        ).build_dm_from_config_weather(weather_type=GribType.hrrr)
        assert b
