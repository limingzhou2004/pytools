import numpy as np

from pytools.data_prep import weather_data_prep as wp
from pytools.data_prep import data_prep_manager as dm
from pytools.config import Config


class TestWeatherDataPrep:
    def test_get_datetime_from_grib_file_name(self):
        fn = "nam_12_2019_02_03_14F1.grib2"
        fn_generate = wp.get_datetime_from_grib_file_name(fn, hour_offset=-5)
        assert fn_generate == np.datetime64("2019-02-03T08:00:00.000000")
        fn = "hrrrsub_2019_06_23_17F0.grib2"
        fn_generate = wp.get_datetime_from_grib_file_name(fn, hour_offset=-5)
        assert fn_generate == np.datetime64("2019-06-23T12:00:00.000000")

    def test_make_npy_data(self, cur_toml_file, weather_type):
        config = Config(filename=cur_toml_file)
        d: dm.DataPrepManager = dm.load(
            config=config, suffix=f"model_pred_{weather_type}.pickle"
        )
        # set up d.weather for both hrrr and nam, depending on self.weather_type
        d.build_weather(
            weather_folder=config.weather_folder,
            jar_address=config.jar_config,
            center=config.site["center"],
            rect=config.site["rect"],
        )
        d.weather.make_npy_data()
