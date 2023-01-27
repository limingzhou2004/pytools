#!/bin/zsh
export source="//Users//limingzhou//zhoul//work//energy//grib2//hrrrdata,//Users//limingzhou//zhoul//work//energy//grib2//hrrr2"
export dest="/Users/limingzhou/zhoul/work/energy/grib2/utah"

export source="/Volumes/LaCie/weather/grib2_2018b"
export dest="/Users/limingzhou/zhoul/work/energy/utah"

python -m pytools.data_prep.grib_utils   $source $dest  "2022-09-23 17:00"  "2022-09-30 23:00"  


#"2022-12-20 16:00"  