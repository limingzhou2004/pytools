#!/bin/bash

cd src/python/pytools
echo $(pwd)

config_file="pytools/config/albany_test.toml"
echo $config_file
fst_ahead=$1

python -m pytools.data_prep.weather_task -c $config_file task_6 -mha 28 -tc 2020-1-9T15:00