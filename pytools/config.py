import os
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, FilePath
import toml
import envtoml


class Site(BaseModel):
    name:str
    base_folder: str
    center: Tuple[float, float]
    radius: Tuple[float, float, float, float]
    hrrr_paras_file: FilePath
    sql_location: str 
    site_folder: str 
    description: str 


class Load(BaseModel):
    schema:str
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
    load_lag_start:int



class Config:
    def __init__(self, filename: str = ""):
        self.filename = filename
        self.toml_dict = envtoml.load(filename)
        base_folder = self.toml_dict["site"].get("base_folder")
        self._base_folder = (
            base_folder if base_folder.endswith("/") else base_folder + "/"
        )
        # validate the settings via pydantic
        self._add_base_folder(self.toml_dict["site"], "hrrr_paras_file")
        # use pydantic to validate the config
        self.site_pdt = Site(**self.toml_dict['site'])
        self.load_pdt = Load(**self.toml_dict['load'])

        
        # jar and weather folder are processed separately as properties

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

    def get_model_file_name(
        self, class_name: str = "_data_manager_", prefix: str = "", suffix: str = ""
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
            self._base_folder,
            self.site_parent_folder,
            prefix + self.site["alias"] + class_name + suffix,
        )

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
            self.toml_dict["site"]["site_folder"],
            self.toml_dict["category"]["name"],
            self.toml_dict["site"]["alias"],
        )

    @property
    def category(self):
        return self.toml_dict["category"]

    @property
    def load(self):
        return self.toml_dict["load"]

    @property
    def jar_config(self):
        return self._join(self._base_folder, self.toml_dict["jar_config"]["address"])

    @property
    def model(self):
        return self.toml_dict["model"]

    @property
    def site(self):
        return self.toml_dict["site"]

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

    @property
    def weather_folder(self) -> Dict:
        path_dict = self.get("weather_folder")
        return {k: self._join(self._base_folder, path_dict[k]) for k in path_dict}
    
    @property
    def center(self) -> Tuple[float, float]:

        return self.toml_dict['site']['center']
    
    @property
    def radius(self) -> Union[int,Tuple[int, int, int, int]]:
        return self.toml_dict['site']['radius']
