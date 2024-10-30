from datetime import datetime, timezone
import os.path as osp
import sys
import pytz
from zipfile import ZipFile

import pandas as pd
from pyiso import client_factory


from pytools.data_prep.pg_utils import get_pg_conn, upsert_df
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


def read_a_fst_zip_file(fn):

    zip_file = ZipFile(fn)
    spot_date_str = osp.split(fn)[1]
    spot_date = pd.to_datetime(spot_date_str[0:4] + '-' + spot_date_str[4:6] + '-' + spot_date_str[6:8])
    dfs = [pd.read_csv(zip_file.open(text_file.filename))
        for text_file in zip_file.infolist()
        if text_file.filename.endswith('.csv')]
    if dfs:
        rdf = pd.concat(dfs) #Time Stamp is local time
        rdf['Time Stamp'] = pd.to_datetime(rdf['Time Stamp'])
        rdf['timestamp_spot'] = spot_date
        return  rdf


def read_a_fst_zip_folder(fd:str):
    files = get_files_from_a_folder(fd)
    dfs = []
    #c = client_factory('NYISO')
    for f in files:
        if f:
            dfs = dfs + [read_a_fst_zip_file(f)]
    # rename to match the timestamp name from API calls
    df_all = pd.concat(dfs).rename(columns={'Time Stamp':'timestamp'})
    df_all.drop_duplicates(subset=['timestamp','timestamp_spot'],inplace=True)

    return df_all.set_index('timestamp')


def read_a_hist_zip_file(fn):
    zip_file = ZipFile(fn)
    dfs = [pd.read_csv(zip_file.open(text_file.filename))
        for text_file in zip_file.infolist()
        if text_file.filename.endswith('.csv')]
    if dfs:
        rdf = pd.concat(dfs)
        tz = rdf['Time Zone'].apply(lambda x: '-0500' if x=='EST' else '-0400')
        rdf['Time Stamp'] = pd.to_datetime(rdf['Time Stamp'] +' '+tz, utc=True)
        return  rdf

def read_a_hist_zip_folder(fd: str):
    """
    Read a folders of zipped nyiso load files

    Args:
        fd (str): absolute folder path
    """
    files = get_files_from_a_folder(fd)
    dfs = []
    #c = client_factory('NYISO')
    for f in files:
        if f:
            dfs = dfs + [read_a_hist_zip_file(f)]
    # rename to match the timestamp name from API calls
    df_all = pd.concat(dfs).rename(columns={'Time Stamp':'timestamp'})[nyiso_cols]
    return df_all.set_index(nyiso_index)




def get_forecast_load(client, t0, cur_time):
    raise NotImplementedError

    return

def get_hist_load(client, t0, t1):
    c = client_factory('NYISO')

    #now = pytz.utc.localize(datetime.utcnow())
    now = datetime.now(timezone.utc)
    today = now.astimezone(pytz.timezone(c.TZ_NAME)).date()
    content_list = c.fetch_csvs(today, 'pal')
    assert len(content_list) == 1
    assert content_list[0].split('\r\n')[0]=='"Time Stamp","Time Zone","Name","PTID","Load"'
    return 


def get_latest_load(client, ):
    raise NotImplementedError
    return


def upload_load_data(args):

    pg_dict = {'pg_server':'localhost', 'pg_db':'daf','pg_user':'postgres', 'pg_pwd':'***'}
    folder = '/'
    schema = 'iso'
    dest_table = ''
    option = 'hist'
    for i, p in enumerate(args):
        if p=='-option':
            option = args[i+1]
        if p=='-db':
            pg_dict['pg_db']=args[i+1]
        if p=='-folder':
            folder = args[i+1]
        if p=='-schema':
            schema = args[i+1]
        if p=='-dest_table':
            dest_table = args[i+1]
        if p=='-server':
            pg_dict['pg_server']=args[i+1]
        if p=='-user':
            pg_dict['pg_user'] = args[i+1]
        if p=='-password':
            pg_dict['pg_pwd']=args[i+1]

    df = read_a_hist_zip_folder(folder) if option=='hist' else read_a_fst_zip_folder(folder)
    eng = get_pg_conn(para_airflow=pg_dict)
    upsert_df(df,table_name=dest_table, engine=eng, schema=schema) 


if __name__ == '__main__':
    # python -m pytools.data_prep.nyiso.download_nyiso_load -option hist -dest_table nyiso_hist_load -db limingzhou -folder "/Users/limingzhou/zhoul/work/energy/iso-load/test/hist" -dest_table nyiso_hist_load -password $pass

    # python -m pytools.data_prep.nyiso.download_nyiso_load -option fst  -db limingzhou -dest_table nyiso_fst_load -folder "/Users/limingzhou/zhoul/work/energy/iso-load/test/hist" -dest_table nyiso_fst_load -password $pass
    upload_load_data(sys.argv[1:])