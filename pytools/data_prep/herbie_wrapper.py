from herbie import Herbie

from pytools.data_prep.grib_utils import get_herbie_str_from_cfgrib_file



def download_obs_data_as_files(t0:str, t1:str, paras_file, save_dir):
    fst_hr = 0
    paras_str = get_herbie_str_from_cfgrib_file(paras_file=paras_file)

    h = Herbie(
        dt,
        model='hrrr',
        product='sfc',
        fxx=fst_hr, save_dir=save_dir)
    h.download(paras_str, verbose=True, overwrite=True)
    
       

def download_data(dt, fst_hr):
    
    return


def extract_data_from_file(fn):

    return

