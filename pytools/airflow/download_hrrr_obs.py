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


args={
    'owner' : 'liming',
    'time_out':timedelta(hours=48),
    'retries': 5,
    'retry_delay': timedelta(minutes=5),
    'start_date':pu.now(tz='UTC').add(hours=-3)# 1 means yesterday
}

with DAG(
    "hrrr_obs", start_date=pu.datetime(2023, 1, 1, tz="UTC"),
    dagrun_timeout=args['time_out'],
    schedule="20 * * * *", catchup=False, tags=['hrrr','liming']
) as dag:
    # airflow variables set [-h] [-j] [-v] key VALUE    
    py_path = Variable.get('py_path',default_var=None)
    if not py_path:
        py_path = '/Users/limingzhou/miniforge3/envs/energy_x86/bin/python'
    obs_dest_path = Variable.get('obs_dest_path', default_var=None)
    if not obs_dest_path:
        obs_dest_path = '.'

    t0 = LatestOnlyOperator(task_id='latest-start', dag=dag)  

    def download_data(tgt_folder, fst_hour,  execution_date):
        from pytools.data_prep.grib_utils import download_hrrr_by_hour
        import pendulum as pu  
 
        kwarg = {
        'exe_date': execution_date.add(hours=-1), 
        'fst_hour':fst_hour, 
        'tgt_folder':tgt_folder,
        }
        print(kwarg)
        download_hrrr_by_hour(**kwarg)

    t1 = ExternalPythonOperator(
        python=py_path, 
        op_kwargs={
          'execution_date': '{{ data_interval_end }}', 'tgt_folder': obs_dest_path, 'fst_hour':0
        },
        retries=args['retries'], 
        retry_delay=args['retry_delay'], 
        task_id='download-hrrr-obs', 
        python_callable=download_data, 
        expect_airflow=True, 
        expect_pendulum=True,
        dag=dag,  
       )  

    t0.set_downstream(t1)


if __name__ == "__main__":
    dag.test()