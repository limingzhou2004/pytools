# pd.set_option('mode.chained_assignment', 'raise')

import numpy as np
import pendulum as pu


def get_datetime_from_utah_file_name(filename:str, get_fst_hour=False, numpy=True):
    t = pu.parse(filename[0:8])
    chour = int(filename[15:17])
    fhour = int(filename[26:28])
    t = t.add(hours=chour)
    if numpy:
        t = np.datetime64(t)

    if get_fst_hour:
        return t, fhour
    else:
        return t