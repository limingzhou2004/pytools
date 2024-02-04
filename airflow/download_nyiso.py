from datetime import timedelta
from textwrap import dedent 

from airflow import DAG
# from airflow.operators.bash import BashOperator
from airflow.operators.python import ExternalPythonOperator
# from airflow.operators.empty import EmptyOperator
from airflow.operators.latest_only import LatestOnlyOperator
#from airflow.decorators import dag, task
from airflow.models import Variable
import pendulum as pu

# won't work in the airflow env as pyiso not installed there
#from pyiso import client_factory

# from pytools.data_prep.nyiso.download_nyiso_load import read_a_hist_zip_folder
# from pytools.data_prep.pg_utils import get_pg_conn, upsert_df
# from pytools.data_prep.nyiso.download_nyiso_load import nyiso_cols, nyiso_index, nyiso_fst_cols,nyiso_fst_index


args={
    'owner' : 'liming',
    'time_out':timedelta(hours=48),
    'retries': 5,
    'retry_delay': timedelta(minutes=5),
    'start_date':pu.now(tz='UTC').add(days=-2)# 1 means yesterday
}

with DAG(
    "nyiso", start_date=args['start_date'],
    dagrun_timeout=args['time_out'],
    schedule="15 * * * *", catchup=True, tags=['nyiso','liming']
) as dag:
    # airflow variables set [-h] [-j] [-v] key VALUE    
    py_path = Variable.get('py_path',default_var=None)
    iso_schema = Variable.get('iso_schema',default_var='nyiso')
    nyiso_hist_load_table = Variable.get('nyiso_hist_load_table',default_var='nyiso_hist_load')
    nyiso_fst_load_table = Variable.get('nyiso_fst_load_table',default_var='nyiso_fst_load')

    critical_time = int(Variable.get('critical_time_mm', default_var=50))

    if not py_path:
        py_path = '/Users/limingzhou/miniforge3/envs/energy_x86/bin/python'


    def download_nyiso_load_data(schema, hist_table, fst_table):
       # from pytools.data_prep.grib_utils import download_hrrr_by_hour
        import pendulum as pu 
        from pyiso import client_factory
        import pandas as pd 

        from pytools.data_prep.pg_utils import get_pg_conn, upsert_df
        from pytools.data_prep.nyiso.download_nyiso_load import nyiso_cols, nyiso_index, nyiso_fst_cols,nyiso_fst_index

        c = client_factory('NYISO')
        eng = get_pg_conn()

        data = c.get_load(latest=True, yesterday=True, integrated_1h=True, freq='hourly')
        df = pd.DataFrame(data)[nyiso_cols] 
        df = df.set_index(nyiso_index)
        res = upsert_df(df,table_name=f'{hist_table}', engine=eng, schema=schema)

        data = c.get_load(latest=True, forecast=True, freq='hourly')
        df2 = pd.DataFrame(data)[nyiso_fst_cols]
        df2['timestamp_spot'] = pd.Timestamp.now()
        df2.set_index(nyiso_fst_index, inplace=True)
        res = upsert_df(df2,table_name=f'{fst_table}', engine=eng, schema=schema)


    t1 = ExternalPythonOperator(
        python=py_path, 
        op_kwargs={
            'schema':iso_schema,
            'hist_table': nyiso_hist_load_table,
            'fst_table':nyiso_fst_load_table,
          'execution_date_str': '{{ ts }}',      
          'external_trigger': '{{ dag_run.external_trigger }}',
          'critical_time': critical_time,
        },
        retries=args['retries'], 
        retry_delay=args['retry_delay'], 
        task_id='download-nyiso-hist-load', 
        python_callable=download_nyiso_load_data, 
        expect_airflow=True, 
        expect_pendulum=True,
        dag=dag,  
       )  

    