from datetime import timedelta
from textwrap import dedent 

from airflow import DAG
# from airflow.operators.bash import BashOperator
from airflow.operators.python import ExternalPythonOperator
# from airflow.operators.empty import EmptyOperator
from airflow.operators.latest_only import LatestOnlyOperator
from airflow.models import Variable
#from airflow.decorators import dag, task
from airflow.models import Variable
import pendulum as pu

from pytools.data_prep.grib_utils import download_hrrr_by_hour


args={
    'owner' : 'Anti',
    'time_out':timedelta(seconds=1800),
    'retries': 5,
    'retry_delay': timedelta(minutes=5),
    'start_date':pu.now(tz='UTC').add(hours=-3)# 1 means yesterday
}

with DAG(
    "hrrr_obs", start_date=pu.datetime(2023, 1, 1, tz="UTC"),
    dagrun_timeout=args['time_out'],
    schedule="10 * * * *", catchup=False
) as dag:
    # airflow variables set [-h] [-j] [-v] key VALUE    
    py_path = Variable.get('py_path',default_var=None)
    if not py_path:
        py_path = '/Users/limingzhou/miniforge3/envs/energy_x86/bin/python'
    #exe_date = Variable.get('execution_date')
    obs_dest_path = Variable.get('obs_dest_path', default_var=None)
    if not obs_dest_path:
        obs_dest_path = '.'

    t0 = LatestOnlyOperator(task_id='latest-start')  

    def download_data(execution_date, **context):
        kwarg = {
        'exe_date': execution_date, 
        'fst_hour':0, 
        'tgt_folder':obs_dest_path},
        download_hrrr_by_hour(**kwarg)

    t1 = ExternalPythonOperator(python=py_path, retries=args['retries'], retry_delay=args['retry_delay'], task_id='download-hrrr-obs',python_callable=download_hrrr_by_hour, expect_airflow=True, expect_pendulum=True)  

    t0.set_downstream(t1)
    #op0>>task0


if __name__ == "__main__":
    dag.test()