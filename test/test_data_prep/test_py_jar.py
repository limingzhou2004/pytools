import os

import pytest

from pytools.data_prep import py_jar


class TestPyJar:
    def test_teardown_class(self, config):
        pass

    #@pytest.mark.skip("grib2 files are too big to include into code base")
    def test_hrrr_npy(
        self,
        jar_address,
        site_folder,
        config,
        center,
        radius,
        hrrr_hist,  # replace to hrrr_hist
        hrrr_predict,
        hrrr_include,
        hrrr_paras_file,
        cur_toml_file,
    ):
        p = py_jar.PyJar(
            folder_in=hrrr_hist,
            folder_out=os.path.join("resources/test_data/"),
            paras_file=hrrr_paras_file,
            center=center,
            rect=radius,
            #jar_address=jar_address,

        )
        #os.path.join(os.path.dirname(__file__), "../../", "ss")
        p.process_folders(out_prefix="test_", parallel=False)
        assert p


   