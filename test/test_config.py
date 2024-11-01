from pathlib import Path

from pytools.config import Config
from pytools.config import DataType

class TestConfig:
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
        fn = config.get_load_data_full_fn(DataType.load ,'npz')
        assert fn.endswith('load.npz')
