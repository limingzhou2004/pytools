#!/bin/bash

cd src/python/pytools
echo $(pwd)

config_file="pytools/config/albany_test.toml"
echo $config_file
fst_ahead=$1
python -m pytools.data_prep.weather_task -c $config_file task_5 -mha 48 --rebuild-npy yes
#-tc 2020-1-9T15:00 -mha 48 --rebuild-npy yes
