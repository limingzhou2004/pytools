#!/bin/bash

cd src/python/pytools
echo $(pwd)

config_file="pytools/config/albany_test.toml"
echo $config_file
# historical 
python -m pytools.weather_task -cfg $config_file task_2 --n-cores 1 -year 2018 -flag h

# past forecast
#python -m pytools.weather_task -cfg pytools/config/albany_prod.toml task_2 -fh 48 --n-cores 1 -year 2024 -flag f
