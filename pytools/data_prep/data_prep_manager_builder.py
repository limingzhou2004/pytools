from typing import Dict, Tuple
import uuid

from pytools.data_prep import data_prep_manager as dpm
from pytools.config import Config
from pytools.data_prep.data_prep_manager import GribType, DataPrepManager
from pytools.data_prep import load_data_prep as ldp


class DataPrepManagerBuilder:
    def __init__(self, config_file: str, train_t0: str, train_t1: str):
        self.uuid = uuid.uuid4()
        self.t0 = train_t0
        self.t1 = train_t1
        self.config = Config(config_file)
        self.load_data = None

    def build_load_data_from_config(self, config_file: Config = None) -> ldp.LoadData:
        if config_file is None:
            config_file = self.config
        return ldp.build_from_toml(config_file=config_file, t0=self.t0, t1=self.t1)

    def build_dm_from_config_weather(
        self, weather_type: GribType, config: Config = None
    ) -> dpm.DataPrepManager:
        """
        Build a data_manager for a given grib type

        Args:
            config: config object
            weather_type: hrrr or nam

        Returns: a dataManager, Config

        """
        if config is None:
            config = self.config
        load_data = ldp.build_from_toml(config_file=config, t0=self.t0, t1=self.t1)

        dm = DataPrepManager(
            category=config.category["name"],
            site_name=config.site["name"],
            site_alias=config.site["alias"],
            site_description=config.site["description"],
            site_folder=config.site_parent_folder,
            t0=self.t0,
            t1=self.t1,
            load_data=load_data,
            load_limit=config.load["limit"],
            max_load_lag_start=config.load["load_lag_start"],
            utc_to_local_hours=config.load["utc_to_local_hours"],
            weather_type=weather_type,
            load_name=config.load["load_column"],
            timestamp_name=config.load["datetime_column"],
        )
        weather_para_file = (
            config.site["hrrr_paras_file"]
            if weather_type == GribType.hrrr
            else config.site["nam_paras_file"]
        )
        dm.setup_grib_para_file(weather_para_file)
        weather_predict_folder = (
            config.weather_folder["hrrr_predict"]
            if weather_type == GribType.hrrr
            else config.weather_folder["nam_predict"]
        )
        dm.setup_grib_predict_folder(weather_predict_folder)
        return dm

    def build_dm_from_config(self) -> Tuple[Dict[str, dpm.DataPrepManager], Config]:
        """
        Build to data_managers, one for hrrr, and one for nam

        Returns: a dict of hrrr:dm, nam:dm

        """
        dm2 = {
            "hrrr": self.build_dm_from_config_weather(
                weather_type=GribType.hrrr, config=self.config
            ),
            "nam": self.build_dm_from_config_weather(
                weather_type=GribType.nam, config=self.config
            ),
        }
        return dm2, self.config
