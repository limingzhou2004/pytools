import pytest

from pytools import utilities as u, get_logger


@pytest.mark.skip("use absolute path on a specific machine")
def test_get_absolute_path():
    assert (
        u.get_absolute_path(__file__, "test.pkl")
        == "/Users/limingzhou/zhoul/work/me/xaog_proj/src/python/pytools/test/test.pkl"
    )


def test_logger():
    logger = get_logger(__name__)
    logger.error("err recorded")
