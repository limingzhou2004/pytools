

from pytools.data_prep.herbie_wrapper import download_latest_data, download_obs_data_as_files


def test_download_obs_data_as_files(config):
    pfile = config.automate_path(config.weather_pdt.hrrr_paras_file)
    download_obs_data_as_files(t0='2020-12-01 3:00',t1='2020-12-01 05:00',paras_file=pfile,save_dir='~/tmp_data')

def test_download_latest_data_file(config):
    pfile = config.automate_path(config.weather_pdt.hrrr_paras_file)

    data = download_latest_data(paras_file=pfile, max_hrs=3,envelopes=[])
    assert len(data)==3 

    data = download_latest_data(paras_file=pfile, max_hrs=[3, 78])




def test_download_data(config):

    pass