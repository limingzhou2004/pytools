import os

import pytest

from pytools.data_prep import py_jar


class TestPyJar:
    def test_teardown_class(self, config):
        pass

    @pytest.mark.skip("grib2 files are too big to include into code base")
    def test_hrrr_npy(
        self,
        jar_address,
        site_folder,
        config,
        center,
        rect,
        hrrr_hist,  # replace to hrrr_hist
        hrrr_predict,
        hrrr_include,
        hrrr_paras_file,
        nam_paras_file,
        cur_toml_file,
    ):
        p = py_jar.PyJar(
            folder_in=hrrr_predict,
            folder_out=os.path.join(
                cur_toml_file, "resources/test_data/", "hrrr_predict"
            ),
            paras_file=nam_paras_file,
            center=center,
            rect=rect,
            jar_address=jar_address,
        )
        os.path.join(os.path.dirname(__file__), "../../", "ss")
        p.process_folders(out_prefix="test_", parallel=True)
        assert p

    @pytest.mark.skip("grib2 files are too big to include into code base")
    def test_nam_npy(
        self,
        jar_address,
        site_folder,
        config,
        center,
        rect,
        nam_hist,  # replace to hrrr_hist
        nam_predict,
        hrrr_include,
        hrrr_paras_file,
        nam_paras_file,
        cur_toml_file,
    ):
        p = py_jar.PyJar(
            folder_in=nam_predict,
            folder_out=os.path.join(
                cur_toml_file, "resources/test_data/", "nam_predict"
            ),
            paras_file=nam_paras_file,
            center=center,
            rect=rect,
            jar_address=jar_address,
        )
        os.path.join(os.path.dirname(__file__), "../../", "ss")
        p.process_folders(out_prefix="test_", parallel=True)
        assert p
