

from pytools.data_prep.herbie_wrapper import download_hist_fst_data, download_latest_data, download_obs_data_as_files
from pytools.data_prep.herbie_wrapper import main

def test_download_obs_data_as_files(config):
    pfile = config.automate_path(config.weather_pdt.hrrr_paras_file)
    download_obs_data_as_files(t0='2020-12-01 3:00',t1='2020-12-01 05:00',paras_file=pfile,save_dir='~/tmp_data', threads=2)

    download_obs_data_as_files(t0='2020-12-01 3:00',t1='2020-12-01 05:00',paras_file=pfile,save_dir='~/tmp_data')

def test_download_latest_data_file(config):
    pfile = config.automate_path(config.weather_pdt.hrrr_paras_file)
    # 78 is beyond the forecast horizon, and we expect it to fail with timeout errors.
    t1, data1 = download_latest_data(paras_file=pfile, max_hrs=[3, 5, 78], envelopes=[[1548, 1568, 774, 794]])
    assert len(t1) == 2 
    assert len(data1) == 1
    assert data1[0].shape==(2, 21, 21, 16)

    t, data = download_latest_data(paras_file=pfile, max_hrs=3, envelopes=[[1548, 1568, 774, 794]])
    assert len(t) == 3 
    assert len(data) == 1
    assert data[0].shape==(3, 21, 21, 16)


def test_download_hist_fst_data(config):
    pfile = config.automate_path(config.weather_pdt.hrrr_paras_file)

    timestamp, data = download_hist_fst_data(
        t_start='2023-12-20 01:00',
        t_end='2023-12-20 02:00', 
        fst_hr=2,paras_file=pfile,
        envelopes=[[1548, 1568, 774, 794]])

    assert len(timestamp) == 1
    assert len(data[0][1]) == 2


def test_main():
    arg_str_obs = '*.py -obs -t0 2020-01-01 -t1 2020-01-01 02:00 -save-dir /Users/limingzhou/zhoul/tmp/hist'
    arg_str_fst = '*.py -fst -t0 2020-01-01 -t1 2020-01-01 02:00 -fst_hr 3 -save-dir /Users/limingzhou/tmp/fst -fn albany-test'
    main(arg_str_obs.split(' '))
    main(arg_str_fst.split(' '))

    assert 1==1
