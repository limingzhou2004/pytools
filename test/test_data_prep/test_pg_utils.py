

from pytools.data_prep.pg_utils import clean_tmp_tables


def test_clean_tmp_tables():
    clean_tmp_tables(schema='iso')