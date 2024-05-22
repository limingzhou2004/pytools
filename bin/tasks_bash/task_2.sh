#!/bin/bash

cd src/python/pytools
echo $(pwd)

config_file="pytools/config/albany_test.toml"
echo $config_file
python -m pytools.data_prep.weather_task -c $config_file task_2 -n 2

