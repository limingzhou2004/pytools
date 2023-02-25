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
load hist data
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
### Airflow deployment

- create the env  `conda create -n energy_x86 python=3.9`
- `conda activate energy_x86 && pip install -r requirements.txt && CONDA_SUBDIR=osx-64 conda install pynio `
- install pytools `git checkout v0.01 && pip install -e .`
- set up the variables in the [airflowo web console](http://192.168.1.9:8080/home), `py_path=/Users/limingzhou/miniconda/envs/energy_x86/bin/python`, and `obs_dest_path=/Users/limingzhou/energy/data/hrrr_obs`, `fst_dest_path=/Users/limingzhou/energy/data_fst`
- copy the dag file to airflow home `cp airflow/*.py ~/airflow/dags/`
- test, `airflow test dag_id task_id 2023-02-13`

- remove example dags by change the setting in airflow.cfg `load_examples = True`

- check data, `ls -lth ../energy/data/hrrr_obs` and `ls -lth ../energy/data/hrrr_fst`


- start, stop the scheduler, `airflow scheduler >~/airflow/airflow_scheduler.log 2>&1 < /dev/null &`,  `ps aux | grep scheduler`, `kill $(cat ~/airflow/airflow-scheduler.pid)`


### Fill missing hrrr data
1. Create the pickle file, `python -m  pytools.data_prep.grib_util_org 0`.
2. Run the report, `python -m  pytools.data_prep.grib_util_org`
3. Run the Utah downloading to fill missing hours, `python -m pytools.data_prep.grib_utils /Users/limingzhou/zhoul/work/energy/utah_2`