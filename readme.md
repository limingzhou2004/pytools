## Introduction
This project is about using hrrr weather data for short term electric load forecast.
The input data include hrrr 2D weather data, lagged historical load, and calendar data.



----------
## Code Structure

### Data Prep
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



## Installation
- create a conda yaml file, `conda env export | grep -v "^prefix: " >  energy_conda.yaml`
- create the env from the file `conda env create --name energy_x86 --file=energy_conda.yaml`
- copy the dag file to airflow home
- 