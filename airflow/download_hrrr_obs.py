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
    schedule="51 * * * *", catchup=False, tags=['hrrr','liming']
) as dag:
    # airflow variables set [-h] [-j] [-v] key VALUE    
    py_path = Variable.get('py_path',default_var=None)
    critical_time = int(Variable.get('critical_time_mm', default_var=50))

    if not py_path:
        py_path = '/Users/limingzhou/miniforge3/envs/energy_x86/bin/python'
    obs_dest_path = Variable.get('obs_dest_path', default_var=None)
    if not obs_dest_path:
        obs_dest_path = '.'

    t0 = LatestOnlyOperator(task_id='latest-start', dag=dag)  

    def download_data(tgt_folder, fst_hour,  execution_date_str, external_trigger, critical_time):
        from pytools.data_prep.grib_utils import download_hrrr_by_hour
        import pendulum as pu  
        print(f'trigger... {external_trigger}')
        
        exe_date = pu.parse(execution_date_str) if external_trigger == 'True' else pu.parse(execution_date_str).add(hours=1)
        # the hrrr data are found to be generated at 50 min past the starting hour
        print(exe_date)
        if exe_date.minute < critical_time:
            exe_date = exe_date.add(hours=-1)
            print(exe_date)

        kwarg = {
        'exe_date': exe_date, 
        'fst_hour':fst_hour, 
        'tgt_folder':tgt_folder,
        }
        print(kwarg)
        download_hrrr_by_hour(**kwarg)

    t1 = ExternalPythonOperator(
        python=py_path, 
        op_kwargs={
          'execution_date_str': '{{ ts }}', 
          'tgt_folder': obs_dest_path, 
          'fst_hour':0,
          'external_trigger': '{{ dag_run.external_trigger }}',
          'critical_time': critical_time,
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