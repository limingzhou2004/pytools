class TestConfig:
    def test_get(self, config):
        assert config.site["name"] == "Albany-NY"

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
