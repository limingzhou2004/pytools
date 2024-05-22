#!/bin/bash

cd src/python/pytools
echo $(pwd)

config_file="../pytools/config/albany_test.toml"
echo $config_file
python -m pytools.weather_task -cfg $config_file -cr #task_1 -t0 1/1/2020 -t1 1/3/2020





