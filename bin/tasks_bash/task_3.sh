#!/bin/bash

cd src/python/pytools
echo $(pwd)

config_file="pytools/config/albany_prod.toml"
echo $config_file

# python -m pytools.weather_task -cfg pytools/config/albany_prod.toml task_3 --flag cv -ind 0 -sb find_batch_size -mn prod0 -yr 2018-2024

python -m pytools.weather_task -cfg pytools/config/albany_prod.toml task_3 --flag cv -ind 3 -sb fit -mn prod2-1h-v2005b -yr 2019-2023 --model-month -1 -nworker 7