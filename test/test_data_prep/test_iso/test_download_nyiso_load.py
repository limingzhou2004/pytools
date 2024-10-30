from datetime import datetime, timezone
import pandas as pd
import pytz


from pyiso import client_factory

from pytools.data_prep.nyiso.download_nyiso_load import read_a_hist_zip_folder, upload_load_data
from pytools.data_prep.nyiso.download_nyiso_load import nyiso_cols, nyiso_index, nyiso_fst_cols,nyiso_fst_index
from pytools.data_prep.pg_utils import get_pg_conn, upsert_df


zip_folder = "/Users/limingzhou/zhoul/work/energy/iso-load/nyiso-load"


def test_populate_nyiso_load_compare():
    df_zip = read_a_hist_zip_folder(zip_folder)
    c = client_factory('NYISO')
    data = c.get_load(latest=False, yesterday=False, start_at='10/23/2023 0:00', end_at="10/23/2023 19:00", integrated_1h=True, freq='hourly')
    df = pd.DataFrame(data)[nyiso_cols]

    assert set(df_zip.columns) == set(df.columns) 


def test_populate_api_call_data(config):
    c = client_factory('NYISO')
    eng = get_pg_conn()
    schema = config.load['schema']
    table = config.load['table']
    fst_table= config.load['table_iso_fst']

    data = c.get_load(latest=True,integrated_1h=True, freq='hourly')
    df = pd.DataFrame(data)[nyiso_cols] 
    df = df.set_index(nyiso_index)
    res = upsert_df(df,table_name=f'{table}', engine=eng, schema=schema)

    data = c.get_load(latest=True, forecast=True, freq='hourly')
    df2 = pd.DataFrame(data)[nyiso_fst_cols]
    df2['timestamp_spot'] = pd.Timestamp.now()
    df2.set_index(nyiso_fst_index, inplace=True)
    res = upsert_df(df2,table_name=f'{fst_table}', engine=eng, schema=schema)

    assert res


def test_populate_api_forecast(config):
    c = client_factory('NYISO')
    df = c.get_load(latest=True, yesterday=True, forecast=True, freq='hourly')
    assert df.shape[0] > 1
    

def test_read_a_hist_zip_folder(config):
    df = read_a_hist_zip_folder(zip_folder)
    eng = get_pg_conn()
    schema = config.load['schema']
    table = config.load['table']
    upsert_df(df=df, table_name=table,engine=eng,schema=schema)
    assert df.shape[1] == 1

def test_get_hist_load():
    c = client_factory('NYISO')

    now = pytz.utc.localize(datetime.utcnow())
    now = datetime.now(timezone.utc)
    today = now.astimezone(pytz.timezone(c.TZ_NAME)).date()
    content_list = c.fetch_csvs(today, 'isolf')
    assert(len(content_list)==1)
  

def test_get_nyiso_load():
    c = client_factory('NYISO')
    
    # get data
    # start=, end=, | latest
    data = c.get_load(latest=False, yesterday=False, start_at='10/23/2023 0:00', end_at="10/23/2023 19:00", integrated_1h=True, freq='hourly')
    df = pd.DataFrame(data)
    assert(df.shape[0]>0)

    #http://mis.nyiso.com/public/csv/palIntegrated/20230802palIntegrated.csv


def test_get_nyiso_fst_load():
    c = client_factory('NYISO')
    data = c.get_load(latest=True, forecast=True, freq='hourly')
    df = pd.DataFrame(data)
    assert df.shape[0] >1

    
def test_upload_load_data():
    args = '-option hist  -db limingzhou -folder /Users/limingzhou/zhoul/work/energy/iso-load/test/hist -dest_table nyiso_hist_load -password $password'

    # args = '-option hist  -db limingzhou -folder /Users/limingzhou/zhoul/work/energy/iso-load/nyiso-load-hist -dest_table nyiso_hist_load -password $password'
    #upload_load_data(args.split(' '))


    args = '-option fst  -db limingzhou -folder /Users/limingzhou/zhoul/work/energy/iso-load/test/fst -dest_table nyiso_fst_load -password $password'
    upload_load_data(args.split(' '))
    assert 1==1

 