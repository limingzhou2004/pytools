import os
from typing import List

import pytest
from pytools.data_prep.load_data_prep import LoadData

from pytools.config import Config


@pytest.fixture(scope="session")
def train_t0():
    return "2018-12-25"


@pytest.fixture(scope="session")
def train_t1():
    return "2018-12-26 15:00"


@pytest.fixture(scope="session")
def cur_toml_file():
    return os.path.join(os.path.dirname(__file__), "../pytools/config/albany_test.toml")


@pytest.fixture(scope="session")
def config(cur_toml_file):
    os.environ["ENV_TEST"] = "test an env"
    return Config(cur_toml_file)


@pytest.fixture(scope="session")
def jar_address(config):
    return config.jar_config


@pytest.fixture(scope="session")
def site_folder(config):
    return config.site["site_folder"]


@pytest.fixture(scope="session")
def center(config):
    return config.site["center"]


@pytest.fixture(scope="session")
def rect(config):
    return config.site["rect"]


@pytest.fixture(scope="session")
def hrrr_paras_file(config):
    return config.site["hrrr_paras_file"]


@pytest.fixture(scope="session")
def nam_paras_file(config):
    return config.site["nam_paras_file"]


@pytest.fixture(scope="session")
def hrrr_hist(config) -> List[str]:
    return config.weather_folder["hrrr_hist"]


@pytest.fixture(scope="session")
def hrrr_predict(config):
    return config.weather_folder["hrrr_predict"]


@pytest.fixture(scope="session")
def nam_hist(config):
    return config.weather_folder["nam_hist"]


@pytest.fixture(scope="session")
def nam_predict(config):
    return config.weather_folder["nam_predict"]


@pytest.fixture(scope="session")
def hrrr_include():
    return ["hrrrsub_2018_12_25_08F0.grib2", "hrrrsub_2018_12_25_09F0.grib2"]


@pytest.fixture(scope="session")
def hrrr_exclude():
    return []


@pytest.fixture(scope="session")
def suffix_hrrr():
    return "albany_hrrr.pickle"


@pytest.fixture(scope="session")
def suffix_nam():
    return "albany_nam.pickle"


@pytest.fixture(scope="session", params=["hrrr", "nam"])
def weather_type(request):
    return request.param


#
# @pytest.fixture
# def mock_max_date(monkeypatch):
#     def mock_get(*args, **kargs):
#         return pd.DataFrame({"max_date": ["12/26/2018 15:00"]})
#
#     monkeypatch.setattr(LoadData, "query_max_load_time", mock_get)
