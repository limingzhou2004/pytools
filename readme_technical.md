## Introduction
This project is about using hrrr weather data for short term electric load forecast.
The input data include hrrr 2D weather data, lagged historical load, and calendar data.


### Airflow deployment

pip install "apache-airflow[celery]==2.10.2" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.2/constraints-3.9.txt"
- create the env  `conda create -n energy_ops python=3.9`
- `conda activate energy_ops && pip install -r requirements.txt` 
- install pytools `git checkout v0.01 && pip install -e .`
- set up the variables in the [airflowo web console](http://127.0.0.1:8080/home), `py_path=/Users/limingzhou/miniforge3/envs/energy/bin/python`, and `obs_dest_path=/Users/limingzhou/energy/data/hrrr_obs`, `fst_dest_path=/Users/limingzhou/energy/data_fst`
- copy the dag file to airflow home `cp airflow/*.py ~/airflow/dags/`
- test, `airflow test dag_id task_id 2023-02-13`

- remove example dags by change the setting in airflow.cfg `load_examples = True`

- check data, `ls -lth ../energy/data/hrrr_obs` and `ls -lth ../energy/data/hrrr_fst`

- start, stop the scheduler, `airflow scheduler >~/airflow/airflow_scheduler.log 2>&1 < /dev/null &`,  `ps aux | grep scheduler`, `kill $(cat ~/airflow/airflow-scheduler.pid)`
- start the web server `airflow webserver --port 8080 --hostname 0.0.0.0 >~/airflow/airflow_web.log 2>&1 < /dev/null &`
- add the admin user `airflow users  create --role Admin --username admin --email admin --firstname admin --lastname admin --password admin`


### Fill missing hrrr data
The folders included are listed in pytools/data/hrrr_obs_folder.txt file, with stages ==0 for prod and <0 for test.
1. Create the pickle file to list each file for each hour, including missing hours, `python -m  pytools.data_prep.grib_util_org 1` The min and max timestamp are determined fromt he exising files. You may need to manually download the last hour data to expand the max timestamp.
2. Run the report of statistics, by not using additional args, `python -m  pytools.data_prep.grib_util_org summary 1` . __Remember__ to run step 1 to update the pickle file.
3. Run the Herbie downloading to fill missing hours,  `python -m pytools.data_prep.grib_util_org -fill "/Users/limingzhou/zhoul/work/energy/herbie" 1`
   
   After the missings are filled, rerun the statisitcal report, there should be no missings.

### Integration
#### Weather prep
The weather steps include:
1. Task 1, `python -m pytools.weather_task -cfg pytools/config/albany_test.toml --create task_1 `
2. To get the envelope based on the lon/lat, use grib2_utils notebook. Task 2, get historical weather `python -m pytools.weather_task -cfg pytools/config/albany_prod.toml task_2 --n-cores 1 -year 2018 -flag h`. To get past forecast weather, `python -m pytools.weather_task -cfg pytools/config/albany_prod.toml task_2 -fh 48 --n-cores 1 -year 2024 -flag f`
3. Task 3, sync load and weather data.
- CV training, `python -m pytools.weather_task -cfg pytools/config/albany_test.toml task_3 --flag cv --ind 0 -sb [find_batch_size|find_lr|fit] -mn test0 -yr 2018-2024`
  
