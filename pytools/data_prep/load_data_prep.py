from typing import List, Union

import numpy as np
import pandas as pd

# from pandas import DataFrame

from pytools.config import Config
from pytools.data_prep import (
    token_site_name,
    token_table_name,
    token_t0,
    token_t1,
    query_str_fill,
    token_max_load_time,
)
from pytools.data_prep import calendar_data_prep as Cp
from pytools.mysql_adapter0 import MySqlQuery as Mq
from pytools.pandas_pgsql import PandasSql as Ps


class LoadData:
    """
    Prepare load data
    """

    # nyiso_hist_load = "nyiso_hist_load"
    # t0_str = "'@t0'"
    # t1_str = "'@t1"

    def __init__(
        self,
        table_name: str,
        site_name: str,
        date_col: str,
        y_label: str,
        t0: str,
        t1: str,
        daylightsaving_col="daylightsaving",
        query_str_max_date: str = "",
        query_str_train: str = "",
        query_str_predict: str = "",
    ):
        """
        load data

        Args:
            table_name: table that holds the load data/ price data
            site_name: name of the site used in the sql table
            date_col: list of columns as datetime
            t0: start datetime str yyyy-mm-dd hh:mm
            t1: end datetime str yyyy-mm-dd hh:mm
            daylightsaving_col: [col_1, col_2]
            query_str_max_date: str = "",
            query_str_train: str ="",
            query_str_predict: str = "",
        """
        self.table_name = table_name
        self.site_name = site_name
        self.date_col = date_col
        self.daylightsaving_col = daylightsaving_col
        self.t0 = t0
        self.t1 = t1
        self.query_str_max_date = query_str_max_date
        self.query_str_train = query_str_train
        self.query_str_predict = query_str_predict
        self._train_data = self.query_train_data(t0=t0, t1=t1)
        self.y_label = y_label
        self.y_mean = self._train_data[self.y_label].mean()

    @property
    def train_data(self):
        return self._train_data

    def get_query_predict_str(self, t0: str, t1: str, t_max: str) -> str:
        match_args = {
            token_t1: t1,
            token_t0: t0,
            token_max_load_time: t_max,
            token_site_name: self.site_name,
            token_table_name: self.table_name,
        }
        return query_str_fill(self.query_str_predict, **match_args)

    def get_query_train_str(self, t0: str, t1: str) -> str:
        """
        Fill t0 and t1 for getting training data
        Args:
            t0: start time
            t1: end time

        Returns: filled query str

        """
        match_args = {
            token_t1: t1,
            token_t0: t0,
            token_site_name: self.site_name,
            token_table_name: self.table_name,
        }
        return query_str_fill(self.query_str_train, **match_args)

    def get_query_max_date_str(
        self,
    ):
        """
        Prepare the query string to get max date

        Returns: the query string

        """

        match_args = {
            token_site_name: self.site_name,
            token_table_name: self.table_name,
        }
        return query_str_fill(self.query_str_max_date, **match_args)

    def query_data(self, query: str) -> pd.DataFrame:
        """
        Query the data and add calendar columns

        Args:
            query: query string

        Returns: dataframe

        """
        data = self.sql_query(qstr=query, date_col=[self.date_col])
        data = self.add_hod(data, timestamp=self.date_col)
        data = self.add_dow(data, timestamp=self.date_col)
        return data

    def query_max_load_time(self, date_col=["max_date"]) -> np.datetime64:
        """
        Query the max load datetime

        Args:
            date_col: the column name to use when returning the result.

        Returns: latest datetime

        """
        max_qstr = self.get_query_max_date_str()
        return self.sql_query_scaler(qm_str=max_qstr, date_col=date_col)

    def query_predict_data(self, t0: str, t1: str, tc: str = None) -> pd.DataFrame:
        """
        Get the calendar and load data for predictions

        Args:
            t0: earliest load data time
            t1: forecast horizon end time
            tc: the latest known load time.

        Returns: dataframe for the calendar and load data

        """
        if not tc:
            tc = self.query_max_load_time()
        qstr = self.get_query_predict_str(t0=t0, t1=t1, t_max=str(tc))
        return self.query_data(query=qstr)

    def query_train_data(self, t0: str, t1: str) -> pd.DataFrame:
        qstr = self.get_query_train_str(t0=t0, t1=t1)
        return self.query_data(qstr)

    def add_hod(self, df: pd.DataFrame, timestamp="timestamp") -> pd.DataFrame:
        """
        add hour of day (hod)
        Args:
            df: DataFrame
            timestamp: timestamp column name

        Returns: dataframe with hour of day added

        """
        cols, dh = Cp.CalendarData().get_hourofday(df[timestamp])
        for c in cols:
            df[c] = dh[c]
        return df

    def add_dow(self, df, timestamp="timestamp"):
        cols, dw = Cp.CalendarData().get_dayofweek(df[timestamp])
        for c in cols:
            df[c] = dw[c]
        return df

    def sql_query_scaler(self, qm_str: str, date_col: List[str] = ["max_date"]):
        """
        query a scaler

        Args:
            qm_str: query string
            date_col: list of date columns

        Returns:
        """
        return Ps.read_sql_timeseries(
            Mq().get_sqlalchemy_engine(), qstr=qm_str, date_col=date_col
        ).values[0, 0]

    def sql_query(self, qstr: str, date_col: List[str] = []) -> pd.DataFrame:
        """
        sql query

        Args:
            qstr: query str
            date_col: which columns should be treated as datetime

        Returns:

        """
        if not isinstance(date_col, List):
            date_col = [date_col]
        return Ps.read_sql_timeseries(
            Mq().get_sqlalchemy_engine(), qstr=qstr, date_col=date_col
        )


def build_from_toml(config_file: Union[str, Config], t0: str, t1: str) -> LoadData:
    """
    Construct a LoadData from a toml file

    Args:
        config_file: toml file path, or a Config object
        t0: load start datetime, yyyy-mm-dd hh:mm
        t1: load end datetime, yyyy-mm-dd hh:mm

    Returns: a LoadData object

    """
    # if config_file is str, then it's the file name. Otherwise, it's a Config
    config = (
        Config(filename=config_file) if isinstance(config_file, str) else config_file
    )
    ld = LoadData(
        table_name=config.load["table"],
        site_name=config.site["sql_location"],
        date_col=config.load["datetime_column"],
        y_label=config.model["y_label"],
        query_str_max_date=config.sql["query_max_date"],
        query_str_train=config.sql["query_train"],
        query_str_predict=config.sql["query_predict"],
        t0=t0,
        t1=t1,
    )
    return ld
