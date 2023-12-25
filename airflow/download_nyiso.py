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
    'start_date':pu.now(tz='UTC').add(days=-2)# 1 means yesterday
}

with DAG(
    "nyiso", start_date=args['start_date'],
    dagrun_timeout=args['time_out'],
    schedule="15 * * * *", catchup=True, tags=['nyiso','liming']
) as dag:
    # airflow variables set [-h] [-j] [-v] key VALUE    
    py_path = Variable.get('py_path',default_var=None)
    critical_time = int(Variable.get('critical_time_mm', default_var=50))

    if not py_path:
        py_path = '/Users/limingzhou/miniforge3/envs/energy_x86/bin/python'


    