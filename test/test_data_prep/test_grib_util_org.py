import pytest
import os

from pytools.data_prep.grib_util_org import load_files, make_stats


@pytest.mark.skip(reason='grib files')
def test_load_files():
    load_files(-1)
    assert 1==1

@pytest.mark.skip(reason='grib files')
def test_make_stats():
    make_stats(i=1)

    assert 1==1