
from enum import Enum
import math
import os
import os.path as osp
from pathlib import Path
from typing import Dict, List, Tuple, Union
from itertools import chain


import numpy as np
import pandas as pd
from pydantic import BaseModel, FilePath, field_validator
import toml
import envtoml

from pytools.utilities import get_absolute_path, get_file_path

US_state_abbvs=['AL',
'AK',
'AZ',
'AR',
'CA',
'CO',
'CT',
'DE',
'DC',
'FL',
'GA',
'HI',
'ID',
'IL',
'IN',
'IA',
'KS',
'KY',
'LA',
'ME',
'MD',
'MA',
'MI',
'MN',
'MS',
'MO',
'MT',
'NE',
'NV',
'NH',
'NJ',
'NM',
'NY',
'NC',
'ND',
'OH',
'OK',
'OR',
'PA',
'RI',
'SC',
'SD',
'TN',
'TX',
'UT',
'VT',
'VA',
'WA',
'WV',
'WI',
'WY',
]

US_timezones=['US/Alaska',
 'US/Aleutian',
 'US/Arizona',
 'US/Central',
 'US/East-Indiana',
 'US/Eastern',
 'US/Hawaii',
 'US/Indiana-Starke',
 'US/Michigan',
 'US/Mountain',
 'US/Pacific',
 'US/Samoa']

class DataType(Enum):
    LoadData =1 
    CalendarData = 2
    Hist_weatherData = 3
    Past_fst_weatherData = 4 
    Latest_fst_weatherData = 5


class Site(BaseModel):
    timezone:str
    state:str
    folder_name:str
    name:str
    base_folder: str
    center: Tuple[float, float]
    rect: Tuple[float, float, float, float]
    back_fst_window: List[str]
    t0: str
    t1: str
    hrrr_paras_file: FilePath
    sql_location: str 
    #site_folder: str 
    description: str 

    @field_validator('state',mode='after')
    @classmethod
    def state_must_be_abbrevs(cls, state):
        if state not in US_state_abbvs:
            raise ValueError(f'{state} is not in the two letter abbvs for states, in {US_state_abbvs}')
        return state
        
    @field_validator('timezone', mode='after')
    @classmethod
    def timezone_must_be_in_list(cls, timezone):
        if timezone not in US_timezones:
            raise ValueError(f'{timezone} not in the list {US_timezones}')
        return timezone


class Load(BaseModel):
    db_name:str
    db_schema:str
    table:str #for hist
    table_iso_fst: str
    table_our_fst: str
    unit:str
    datetime_column:str 
    daylightsaving_col:str 
    load_column:str 
    sql_template_file: str
    limit: Tuple[float,float]
    lag_hours:int
    utc_to_local_hours:int 
    #load_lag_start:int # to delete
    #fst_hours: List[int]


class Weather(BaseModel):
    envelope: List[int]
    hist_weather_pickle: str
    fullfile_col_name: str 
    filename_col_name: str 
    hrrr_paras_file: str
    type_col_name: str 
    #hrrr_hist: List[str]
    hrrr_predict: str


class Model(BaseModel):
    y_label: str
    scaler_type: str
    sample_data_seq_dim: int
    model_settings: Dict
    frac_yr1: float
    frac_split: list
    final_train_frac_yr1: float
    final_train_frac: list
    cv_settings: List[List]
    forecast_horizon: List[List]
    final_train_hist: List
    target_ind: int
    wea_ar_embedding_dim: int
    #wea_embedding_dim: int 
    ext_ar_embedding_dim: int
    seq_length: int 

    cov_net: Dict 
    ext_net: Dict
    filter_net: Dict
    mixed_net: Dict

    models: List


class Config:
    def __init__(self, filename: str = ""):
        self.filename = filename
        self.toml_dict = envtoml.load(filename)
        base_folder = self.toml_dict["site"].get("base_folder")
        home = str(Path.home())
        base_folder = base_folder.replace('~', home)
        self._base_folder = (
            base_folder if base_folder.endswith("/") else base_folder + "/"
        )
        # validate the settings via pydantic
        if not self.toml_dict["site"].get('hrrr_paras_file', None):
            self.toml_dict['site']['hrrr_paras_file']= get_absolute_path(__file__, 'data_prep/hrrr_paras_pynio.txt')
        else:
            self._add_base_folder(self.toml_dict["site"], "hrrr_paras_file")
        # use pydantic to validate the config
        self.site_pdt = Site(**self.toml_dict['site'])
        self.load_pdt = Load(**self.toml_dict['load'])
        self.weather_pdt = Weather(**self.toml_dict['weather'])
        self.model_pdt = Model(**self.toml_dict['model'])
        self.target_dim = 1 if isinstance(self.model_pdt.target_ind, int) else len(self.model_pdt.target_ind)
        self._filternet_input =  self.model_pdt.ext_net['output_channel'] + self.target_dim + self.model_pdt.cov_net['last']['channel']

        if self.model_pdt.seq_length < max(self.model_pdt.wea_ar_embedding_dim, self.model_pdt.ext_ar_embedding_dim):
            raise ValueError('weather or ext AR embedding exceeds seq length!') 

    def _add_base_folder(self, dict_to_update, key):
        dict_to_update[key] = os.path.join(self._base_folder, dict_to_update[key])

    def get(self, table: str) -> Dict:
        """
        Find a dict for a given table

        Args:
            table: table name in the toml file

        Returns: a dict

        """
        return self.toml_dict[table]
    
    def get_load_data_full_fn(self, data_type: DataType, extension:str, year=-1, month=-1):
        year_str = '' if year < 0 else f'_{year}'
        mon_str = '' if month < 0 else f'_{month}'
        return self.automate_path(f'data/{str(data_type.name)}{year_str}{mon_str}.{extension}')

    def get_model_file_name(
        self, class_name: str = "_data_manager_", prefix: str = "", suffix: str = "", extension='.pkl'
    ) -> str:
        """
        Return the full name of the model file name from the config file
        Args:
            class_name: type of object, data manager by default
            prefix:
            suffix:

        Returns: file name as a str

        """
        return os.path.join(
            self.site_parent_folder,
            prefix + self.site_pdt.folder_name + '_' + class_name + suffix + extension,
        )
    
    def get_logger_folder(self):
        return
    
    @property
    def filternet_input(self):
        return self._filternet_input

    @property
    def base_folder(self):
        return self._base_folder

    @base_folder.setter
    def base_folder(self, base_folder):
        self._base_folder = base_folder

    @property
    def site_parent_folder(self):
        return os.path.join(
            self._base_folder,
            #self.toml_dict["site"]["site_folder"],
            # self.toml_dict["category"]["name"],
            # self.toml_dict["site"]["folder_name"],
        )

    @property
    def category(self):
        return self.toml_dict["category"]
    
    def automate_path(self, fn:str)->str:
        """
        Automatically decide the absolute path, for the three situations.
        - absolute path
        - relative to data_prep folder
        - within the site folder

        Args:
            fn (str): file path in the config, relative or absolute

        Returns:
            str: Absolute path
        """
        if fn.startswith('/'): 
            return fn
        if fn.startswith('data_prep'):
            return get_file_path(fn=fn, this_file_path=__file__)
        
        return osp.join(self.site_parent_folder, fn)

    @property
    def load(self):
        return self.toml_dict["load"]

    @property
    def model(self):
        return self.toml_dict["model"]

    @property
    def site(self):
        return self.toml_dict['site']
    
    @property
    def weather(self):
        return self.toml_dict['weather']

    @property
    def sql(self):
        """
        From the sql template, get the query template for load data, including max, train, predict
        Returns:

        """
        sql_path = os.path.join(
            os.path.dirname(__file__),
            "config",
            self.toml_dict["load"]["sql_template_file"],
        )
        sql = toml.load(sql_path)
        return sql["sql"]

    def _join(self, base: str, target):
        if isinstance(target, str):
            return os.path.join(base, target)
        elif isinstance(target, List):
            return [os.path.join(base, t) for t in target]
        else:
            raise ValueError("type not supported, must be str or list")

    def report_predictions(
        self, df: pd.DataFrame, start_time: np.datetime64, hours_ahead: int
    ):
        df = pd.DataFrame(df)
        df["time_zone_code"] = self.load["utc_to_local_hours"]
        df["site_name"] = self.site["name"]
        out_dir = os.path.join(f"{self.site_parent_folder}", "report")
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        file_name = os.path.join(
            out_dir, f"{self.site['name']}-{str(start_time)}-{hours_ahead}.csv"
        )
        df.reset_index().to_csv(file_name, index=False)

    def get_sample_segmentation_borders(self, full_length, fst_scenario=0, first_yr_frac=0.5, fractions=[]):
        #, fractions=[0.5, (0.4, 0.3, 0.3)]):
        fraction_yr1 = first_yr_frac
        frac_train=fractions[0]
        frac_test=fractions[1]
        frac_val=fractions[2]
        # full ind 0: len(all samples) - pred_length 
        # first year + 40% 2nd year (train): 30% 2nd year(test): 30% 2nd year(validate)
        # fraction = [first percetage, (second train, test, validate)]
        pre_length = self.model_pdt.forecast_horizon[fst_scenario][-1]
        full_length -= pre_length
        train_borders = range(int(full_length*fraction_yr1))
        test_borders = range(1,0)
        val_borders = range(1,0)
        n_quarter = 4
        quarter = math.ceil(full_length * (1-fraction_yr1) / n_quarter)
        m = train_borders[-1]+1

        for i in range(n_quarter):
            train_borders = chain(train_borders, range(int(i*quarter+m), \
                                         int((i+frac_train)*quarter+m)))
            test_borders = chain(test_borders, range(int((i+frac_train)*quarter+m), \
                                       int((i+frac_train+frac_test)*quarter)+m))
            val_borders = chain(val_borders, range(int((i+frac_train+frac_test)*quarter)+m, \
                                      int((i+frac_train+frac_test+frac_val)*quarter)+m))

        def fun(x):
            return x<full_length
        return filter(fun, train_borders), filter(fun, test_borders), \
            filter(fun, val_borders)

    
    @property
    def center(self) -> Tuple[float, float]:
        return self.toml_dict['site']['center']
    
    @property
    def rect(self) -> Union[int,Tuple[int, int, int, int]]:
        return self.toml_dict['site']['rect']
    
    @property
    def t0(self) -> str:
        return self.site_pdt.t0 
    
    @property
    def t1(self) -> str:
        return self.site_pdt.t1


