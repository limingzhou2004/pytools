## Introduction
This project is about using hrrr weather data for short term electric load forecast.
The input data include hrrr 2D weather data, lagged historical load, and calendar data.



----------
## Code Structure

### Data Prep

- Remove small files from grib folders.  `find folder -type f -name "*" -size -1600k -delete`
Use division to define a region.
For each site, a folder of training data, a folder of predict data, a folder of model


#### DataManager
Initiated from load, get train weather , get pred weather


#### Config
Use a Config object to configure the folders, file names

#### LoadDataPrep
set up database environmental varialbes in ~/.zprofile, pg_server, pg_user, pg_pwd
load hist data
- NYISO load data
- set

load pred data

#### Weather Data, hrrr and nam
* WeatherDataPrep <- WeatherData <- PyJar
generate npy files from hrrr
extract npy files for WeatherData

---------------------

## Usage

### weather data






------
#### Calendar data and load data

* category or table_name
* site
* t0 start time
* t1 end time

#### tokens 
timezone_utc = {
    "est": -5,
    "edt": -4,
    "cst": -6,
    "cdt": -5,
    "pst": -8,
    "pdt": -7,
    "mst": -7,
    "mdt": -6,
}
##### token used in sql query as column names
token_load = "load"
token_timestamp = "timestamp"
token_query_max_date: str = "query_max_date"
token_query_train: str = "query_train"
token_query_predict: str = "query_predict"
token_site_name = "site_name"
token_table_name = "table_name"
token_t0 = "t0"
token_t1 = "t1"
token_max_load_time = "max_load_time"

### Modeling
TODO
### 


-----
## Deployment

### Database
create the airflow_db database
airflow db migrate

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
2. Task 2, get historical weather `python -m pytools.weather_task -cfg pytools/config/albany_prod.toml task_2 --n-cores 1 -year 2018 -flag h`. To get past forecast weather, `python -m pytools.weather_task -cfg pytools/config/albany_prod.toml task_2 -fh 48 --n-cores 1 -year 2024 -flag f`
3. Task 3, sync load and weather data.
- CV training, `python -m pytools.weather_task -cfg pytools/config/albany_test.toml task_3 --flag cv --ind 0 `
- 
