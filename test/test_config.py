class TestConfig:
    def test_get(self, config):
        assert config.site["name"] == "Albany-NY"

    def test_load(self, config):
        assert config.load["datetime_column"] == "timestamp"

    def test_sql(self, config):
        assert len(config.sql) == 3

    def test_weather_folder(self, config):
        assert len(config.weather_folder["hrrr_hist"]) == 1

    def test_env_parse(self, config):
        assert config.site["envtest"] == "test an env"
