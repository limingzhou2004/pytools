from datetime import datetime
import pandas as pd
import pytz


from pyiso import client_factory

from pytools.data_prep.nyiso.download_nyiso_load import read_a_hist_zip_folder
from pytools.data_prep.nyiso.download_nyiso_load import nyiso_cols
from pytools.data_prep.pg_utils import upsert_df


zip_folder = "/Users/limingzhou/zhoul/work/energy/iso-load/nyiso-load"


def test_populate_nyiso_load_compare():
    df_zip = read_a_hist_zip_folder(zip_folder)
    c = client_factory('NYISO')
    data = c.get_load(latest=False, yesterday=False, start_at='10/23/2023 0:00', end_at="10/23/2023 19:00", integrated_1h=True, freq='hourly')
    df = pd.DataFrame(data)[nyiso_cols]

    assert set(df_zip.columns) == set(df.columns) 


def test_populate_api_call_data():
    c = client_factory('NYISO')
    data = c.get_load(latest=True, yesterday=True, integrated_1h=True, freq='hourly')
    df = pd.DataFrame(data)[nyiso_cols]
    upsert_df(df,table_name=, engine=)




def test_read_a_hist_zip_folder():
    df = read_a_hist_zip_folder(zip_folder)
    assert df.shape[1] == 5

def test_get_hist_load():
    c = client_factory('NYISO')

    now = pytz.utc.localize(datetime.utcnow())
    today = now.astimezone(pytz.timezone(c.TZ_NAME)).date()
    content_list = c.fetch_csvs(today, 'isolf')
    assert(len(content_list)==1)
  #  assert(content_list[0].split('\n')[0] ==
    #                    '"Time Stamp","Time Zone","Name","PTID","Load"')
    

def test_get_nyiso_load():
    c = client_factory('NYISO')
    
    # get data
    # start=, end=, | latest
    data = c.get_load(latest=False, yesterday=False, start_at='10/23/2023 0:00', end_at="10/23/2023 19:00", integrated_1h=True, freq='hourly')
    df = pd.DataFrame(data)
    assert(df.shape[0]>0)

    #http://mis.nyiso.com/public/csv/palIntegrated/20230802palIntegrated.csv

    


 