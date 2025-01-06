#!/bin/bash

cd src/python/pytools
echo $(pwd)

config_file="pytools/config/albany_prod.toml"
echo $config_file
python -m pytools.weather_task -cfg $config_file --create task_1 





