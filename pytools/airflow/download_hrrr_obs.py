import os 

from airflow import DAG
# from airflow.operators.bash import BashOperator
from airflow.operators.python import ExternalPythonOperator
# from airflow.operators.empty import EmptyOperator
from airflow.operators.latest_only import LatestOnlyOperator
from airflow.models import Variable
from airflow.decorators import dag, task
from airflow.models import Variable
import pendulum

from pytools.data_prep.grib_utils import download_hrrr_by_hour

os.chdir('/home/lnx/test/')

with DAG(
    "my_dag_name", start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    schedule="@daily", catchup=False
) as dag:
    py_path = Variable.get('py_path')
    op0 = LatestOnlyOperator(task_id='latest start')  

    task0 = ExternalPythonOperator(python=py_path, python_callable=download_hrrr_by_hour, op_kwargs={}, expect_airflow=True, expect_pendulum=True)  


    
