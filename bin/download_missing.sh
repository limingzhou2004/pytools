#!/bin/zsh
export source="//Users//limingzhou//zhoul//work//energy//grib2//hrrrdata,//Users//limingzhou//zhoul//work//energy//grib2//hrrr2"
export dest="/Users/limingzhou/zhoul/work/energy/grib2/utah"

export source="/Volumes/LaCie/weather/grib2_2018b"
export dest="/Users/limingzhou/zhoul/work/energy/utah"

python -m pytools.data_prep.grib_utils   $source $dest  "2022-10-01 10:00"  "2022-12-30 16:00"  


#"2022-12-20 16:00"  