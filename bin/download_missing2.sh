#!/bin/zsh
export source="//Users//limingzhou//zhoul//work//energy//grib2//hrrrdata,//Users//limingzhou//zhoul//work//energy//grib2//hrrr2"
export dest="/Users/limingzhou/zhoul/work/energy/grib2/utah"

export source="/Volumes/LaCie/weather/grib2_2018b"
export dest="/Users/limingzhou/zhoul/work/energy/utah"

python -m pytools.data_prep.grib_utils   $source $dest  "2021-09-27 05:00"  "2021-12-31 23:00"  


#"2022-12-20 16:00"  