from datetime import datetime
import pytz
from zipfile import ZipFile

import pandas as pd
from pyiso import client_factory

from pytools.utilities import get_files_from_a_folder


nyiso_cols = ['Time Zone', 'Name', 'Integrated Load', 'timestamp']
nyiso_index =  ['Time Zone', 'Name', 'timestamp']
nyiso_fst_cols=['timestamp_utc', "Capitl",
	'Centrl',
	'Dunwod',
	'Genese',
	"Hud Vl",
	'Longil',
	"Mhk Vl",
	"Millwd",
	"N.Y.C.",
	'North',
	'West',
	'NYISO',]
nyiso_fst_index = ['timestamp_utc']


def read_a_hist_zip_file(fn):
    zip_file = ZipFile(fn)
    dfs = [pd.read_csv(zip_file.open(text_file.filename))
        for text_file in zip_file.infolist()
        if text_file.filename.endswith('.csv')]
    return dfs


def read_a_hist_zip_folder(fd: str):
    """
    Read a folders of zipped nyiso load files

    Args:
        fd (str): absolute folder path
    """
    files = get_files_from_a_folder(fd)
    dfs = []
    for f in files:
        dfs.extend( read_a_hist_zip_file(f) )
    # rename to match the timestamp name from API calls
    return pd.concat(dfs).rename(columns={'Time Stamp':'timestamp'})[nyiso_cols]


def get_forecast_load(client, t0, cur_time):

    return

def get_hist_load(client, t0, t1):
    c = client_factory('NYISO')

    now = pytz.utc.localize(datetime.utcnow())
    today = now.astimezone(pytz.timezone(c.TZ_NAME)).date()
    content_list = c.fetch_csvs(today, 'pal')
    assert(len(content_list), 1)
    assert(content_list[0].split('\r\n')[0],
                        '"Time Stamp","Time Zone","Name","PTID","Load"')
    return 


def get_latest_load(client, ):

    return