import pytest
import os

from pytools.data_prep.grib_util_org import load_files, make_stats


def test_load_files():
    load_files(-1)
    assert 1==1

def test_make_stats():
    make_stats()

    assert 1==1