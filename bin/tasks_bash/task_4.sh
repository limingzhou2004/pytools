#!/bin/bash

cd src/python/pytools
echo $(pwd)

config_file="pytools/config/albany_test.toml"
echo $config_file
fst_ahead=$1
python -m pytools.data_prep.weather_task -c $config_file task_4 -to {"batch_size":10,"cat_fraction":[1,0,0],"epoch_num":3} -ah 1 -env albany3

#-ah $fst_ahead -to "{\"batch_size\":100,\"learning_rate\":0.01}"

