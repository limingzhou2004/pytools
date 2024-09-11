from pathlib import Path

from pytools.config import Config


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

    def test_weather_folder(self, config):
        assert len(config.weather_folder["hrrr_hist"]) >= 1

    def test_env_parse(self, config):
        assert config.site["envtest"] == "test an env"

    def test_center_radias(self, config):
        assert config.site['center'] == [43.0, -73.0]
        assert config.site['radius'] == [100.0, 100.0, 100.0, 100.0]

    def test_get_fst_hours(self, config):

        assert config.load_pdt.fst_hours==[1,6, 24]
