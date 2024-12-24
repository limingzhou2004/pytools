from typing import Dict, Tuple
import uuid

from pytools.data_prep import data_prep_manager as dpm
from pytools.config import Config
from pytools.data_prep.data_prep_manager import GribType, DataPrepManager
from pytools.data_prep import load_data_prep as ldp


class DataPrepManagerBuilder:
    def __init__(self, config_file: str, t0: str=None, t1: str=None):
        self.uuid = uuid.uuid4()

        self.config = Config(config_file)
        if t0:
            self.t0 = t0
        else: 
            self.t0 = self.config.t0
        if t1:
            self.t1 = t1
        else: 
            self.t1 = self.config.t1
        self.load_data = None

    def build_load_data_from_config(self, config_file: Config = None) -> ldp.LoadData:
        if config_file is None:
            config_file = self.config

        return ldp.build_from_toml(config_file=config_file, t0=self.t0, t1=self.t1)

    def build_dm_from_config_weather(self, config: Config = None) -> dpm.DataPrepManager:
        """
        Build a data_manager for a given grib type

        Args:
            config: config object

        Returns: a dataManager, Config

        """
        if not config:
            config = self.config
        load_data = ldp.build_from_toml(config_file=config, t0=self.t0, t1=self.t1)

        dm = DataPrepManager(
            category=config.category["name"],
            site_name=config.site["name"],
            site_alias=config.site["folder_name"],
            site_description=config.site["description"],
            site_folder=config.site_parent_folder,
            t0=self.t0,
            t1=self.t1,
            load_data=load_data,
            load_limit=config.load["limit"],
            utc_to_local_hours=config.load["utc_to_local_hours"],
            load_name=config.load["load_column"],
            timestamp_name=config.load["datetime_column"],
        )
        dm.setup_grib_para_file(config.automate_path(config.weather_pdt.hrrr_paras_file))
        weather_predict_folder = config.automate_path(config.weather_pdt.hrrr_predict) 
        dm.setup_grib_predict_folder(weather_predict_folder)
        return dm

