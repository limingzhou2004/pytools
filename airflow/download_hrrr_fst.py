from datetime import timedelta
import logging
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
    'retry_delay': timedelta(minutes=10),
    'start_date':pu.now(tz='UTC').add(hours=-3)
}

# for forecast of 48 hours, use 0,6,12,18
with DAG(
    "hrrr_fst", start_date=pu.now('UTC').add(days=-2),
    dagrun_timeout=args['time_out'],
    schedule="56 6 * * *", catchup=False, tags=['hrrr','liming'] 
) as dag:
    # airflow variables set [-h] [-j] [-v] key VALUE    
    py_path = Variable.get('py_path',default_var=None)
    if not py_path:
        py_path = '/Users/limingzhou/miniforge3/envs/energy_x86/bin/python'
    obs_dest_path = Variable.get('fst_dest_path', default_var=None)
    if not obs_dest_path:
        obs_dest_path = '.'

    t0 = LatestOnlyOperator(task_id='latest-start', dag=dag)  

    def download_data(tgt_folder, max_fst_hour,  execution_date_str, external_trigger, critical_time):
        from pytools.data_prep.grib_utils import download_hrrr_by_hour
        import pendulum as pu  
        import logging
         
        # round the hour to 0, 6, 12, 18
        exe_date = pu.parse(execution_date_str) if external_trigger == 'True'  else pu.parse(execution_date_str).add(hours=6*4) # need to match the schedule
        exe_date = exe_date.add(minutes=-critical_time)

        print(exe_date)
        hrs = exe_date.hour % 6
        print(f'hours into (hrs)={hrs}')
        exe_date = exe_date.add(hours=-hrs)

        for i in range(max_fst_hour): 
            kwarg = {
            'exe_date': exe_date,
            'fst_hour':i, 
            'tgt_folder':tgt_folder,
            }
            print(kwarg)
            
            #logging.info(f'{kwarg}')
            download_hrrr_by_hour(**kwarg)

    max_hours = Variable.get('max_fst_hours', default_var=3)
    critical_time = int(Variable.get('critical_time_mm', default_var=20))

    tu = ExternalPythonOperator(
    python=py_path, 
    op_kwargs={
    'execution_date_str': '{{ ts }}', 'tgt_folder': obs_dest_path, 'max_fst_hour':int(max_hours)+1,
    'external_trigger': '{{ dag_run.external_trigger }}',
    'critical_time': critical_time,
    },
    retries=args['retries'], 
    retry_delay=args['retry_delay'], 
    task_id=f'download-hrrr-fst-hour-{int(max_hours)}', 
    python_callable=download_data, 
    expect_airflow=True, 
    expect_pendulum=True,
    dag=dag,  
    ) 
    t0.set_downstream(tu)


if __name__ == "__main__":
    dag.test()