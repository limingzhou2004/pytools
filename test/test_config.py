from pathlib import Path

from pytools.config import Config
from pytools.config import DataType

class TestConfig:

    def test_get_sample_segments(self, config):
        train_borders, test_borders, val_borders = config.get_sample_segmentation_borders(\
            15999, 0)
        
        train_borders = list(train_borders)
        test_borders = list(test_borders)
        val_borders = list(val_borders)

        assert train_borders[0] == 0 
        assert train_borders[7999:8001] == [7999, 8000]
        assert train_borders[-1] == 14772
        assert test_borders[0] == 8782
        assert test_borders[-1] == 15371
        assert val_borders[0] == 9381
        assert val_borders[-1] == 15969



    def test_get(self, config):
        assert config.site["name"] == "Albany-NY"

    def test_get_config_file_path(self, config:Config):
        assert config.automate_path('/abc')=='/abc'
        q = Path(config.automate_path('data_prep/hrrr_paras_cfgrib.txt'))
        assert q.exists()
        fp = config.automate_path('site_paras.toml')
        assert fp == config.site_pdt.base_folder + '/site_paras.toml'

    def test_load(self, config):
        assert config.load["datetime_column"] == "timestamp"

    def test_sql(self, config):
        assert len(config.sql) == 3

    def test_env_parse(self, config):
        assert config.site["envtest"] == "test an env"

    def test_center_radias(self, config):
        assert config.site['center'] == [-73.0, 43.0]
        assert config.site['rect'] == [30.0, 30.0, 30.0, 30.0]
        assert config.weather_pdt.envelope == [1548, 1568, 774, 794]

    def test_get_fst_hours(self, config):
        assert config.model_pdt.forecast_horizon==[[1,24], [25,48]]

    def test_get_full_filename(self, config):
        fn = config.get_load_data_full_fn(DataType.LoadData ,'npz')
        assert fn.endswith('LoadData.npz')
        fn = config.get_load_data_full_fn(DataType.LoadData ,'npz', year=2020)
        assert '_2020' in fn
        fn = config.get_load_data_full_fn(DataType.LoadData ,'npz', year=2021, month=2)
        assert '_2021_2' in fn



