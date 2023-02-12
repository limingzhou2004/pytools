import datetime 
import os
from pathlib import Path
import sys

from airflow import DAG
from airflow.models.taskinstance import TaskInstance
from airflow.models import Variable
from airflow.operators.python import ExternalPythonOperator
import pendulum as pu
import pytest
from pytools.data_prep.weather_data_prep import get_datetime_from_grib_file_name

from airflow import DAG
from airflow.utils.state import DagRunState, TaskInstanceState
from airflow.utils.types import DagRunType
from airflow.operators.python import get_current_context


DATA_INTERVAL_START = pu.datetime(2021, 9, 13, tz="UTC")
DATA_INTERVAL_END = DATA_INTERVAL_START + datetime.timedelta(days=1)

TEST_DAG_ID = "my_custom_operator_dag7"
TEST_TASK_ID = "my_custom_operator_task7"
executable = Path(sys.executable).resolve()


@pytest.fixture()
def dag():

    def proxy(exe_date, fst_hour, tgt_folder):
        #get_datetime_from_grib_file_name(exe_date=dag., tgt_folder=tgt_folder, fst_hour=fst_hour)
        return

    with DAG(
        dag_id=TEST_DAG_ID,
        schedule="@daily",
        start_date=pu.now(tz='UTC'),
        #execution_timeout= datetime.timedelta(seconds=300),
    ) as dag:
        t1 = ExternalPythonOperator(
            task_id=TEST_TASK_ID,
            python=executable,
         python_callable = proxy, #get_datetime_from_grib_file_name, 
         op_kwargs={'tgt_folder':'.','fst_hour':0} ,
         expect_pendulum=True, 
         op_args=[
             "{{dag.get_latest_execution_date()}}",
         ] ,)
    return dag


def test_download_hrrr_obs(dag):
    # DAG(dag_id="hrrr", start_date=dt.datetime.now()):
    os.environ['AIRFLOW_HOME'] = '/Users/limingzhou/airflow'
    dagrun = dag.create_dagrun(
        state=DagRunState.RUNNING,
        execution_date=DATA_INTERVAL_START,
        data_interval=(DATA_INTERVAL_START, DATA_INTERVAL_END),
        start_date=DATA_INTERVAL_END,
        run_type=DagRunType.MANUAL,
    )
    ti = dagrun.get_task_instance(task_id=TEST_TASK_ID)
    ti.task = dag.get_task(task_id=TEST_TASK_ID)
    ti.run(ignore_ti_state=True)
    assert ti.state == TaskInstanceState.SUCCESS
  
