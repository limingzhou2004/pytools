
from functools import partial
import glob
from itertools import chain
import os
from math import ceil, floor
import shutil
import sys
import time
from typing import List, Tuple, Union, Dict

from pytools.retry.api import retry
import pandas as pd
import pendulum as pu


# give a folder list all grib files, nc files greater than 1mb.

