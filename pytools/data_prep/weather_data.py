from enum import Enum
from typing import List, Tuple
from collections import OrderedDict


import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd


class Scaling(Enum):
    minmax = 0
    standard = 1


class WeatherData:
    def __init__(self, dict_data, scaling=Scaling.minmax, prediction=False, paras=None,
                 grid_x=None, grid_y=None):
        """
        WeatherData

        Args:
            dict_data (Dict): Use timestamp as the key
            scaling (Scaling): Scaler Defaults to Scaling.minmax.
            prediction (bool, optional): _description_. Defaults to False.
        """
        self.dict_data = OrderedDict(sorted(dict_data.items(), key=lambda t: t[0]))
        self.scaling = scaling
        self.timestamp = sorted(dict_data.keys())
        self._scaler = None
        self.data = None
        self.shape = None
        self.paras_array = np.array(list(paras.keys()))
        self.grid_x = grid_x
        self.grid_y = grid_y
        if not prediction:
            self.standardize()

    @property
    def scaler(self):
        return self._scaler

    @scaler.setter
    def scaler(self, sc):
        self._scaler = sc

    def get_timestamps(self):
        return self.timestamp

    def query_timestamp_sorted_data(
        self, query_timestamp: pd.DataFrame
    ) -> Tuple[np.array, np.array]:
        """
        Return an array of weather data sorted by query timestamp, and match both timestamps in weather and load

        Args:
            query_timestamp: timestamps to match in the weather data

        Returns: timestamp intersection, np.array

        """
        t_df = query_timestamp.reset_index().iloc[:, [1, 0]]
        name0, name1 = list(t_df)
        t_dict0 = t_df.to_dict(orient="records")
        t_dict = {d[name0]: d[name1] for d in t_dict0}
        zipped = [
            (t, self.dict_data[t])
            for t in self.get_timestamps()
            if t_dict.get(pd.Timestamp(t), -1) >= 0
        ]
        if not zipped:
            raise ValueError(
                "No overlapping data between load and weather. Are you going to do a forecast?"
            )
        t, y = zip(*zipped)

        return np.array(t), np.array(y)

    def standardize(self):
        """
        Define the scaler to transform weather data based on training weather, save for future use
        Returns:

        """
        if self._scaler is None:
            self._scaler = []
        data = np.array([d for d in self.dict_data.values()])
        self.shape = data.shape
        for p in range(data.shape[-1]):
            d1_arr = data[:, :, :, p]
            if len(self._scaler) <= p:
                sc = (
                    MinMaxScaler()
                    if self.scaling == Scaling.minmax
                    else StandardScaler()
                )
                scaled = sc.fit_transform(d1_arr.reshape(-1, 1))
                self._scaler.append(sc)
            else:
                sc = self._scaler[p]
                scaled = sc.transform(d1_arr.reshape(-1, 1))
            data[:, :, :, p] = scaled.reshape(d1_arr.shape)
        self.data = data

    def save_unscaled_npz(self, fn:str):
        """
        Save the original weather data and timestamp

        Args:
            fn (str): the npz file name
        """

        np.savez_compressed(fn, data=np.array([d for d in self.dict_data.values()]), timestamp=self.timestamp, paras=self.paras_array, 
        x_grid=self.grid_x, y_grid=self.grid_y )

    def transform(self, x_data: np.array = None, inverse: bool = False) -> np.array:
        """
        Standardize with saved scaler

        Args:
            x_data: raw np array data
            inverse: True for inverse transform

        Returns: standardized np array data, or inverse standardized np array

        """
        if x_data is None:
            x_data = self.data
        shape_original = x_data.shape
        if x_data is None:
            x_data = self.data
        if x_data.shape[1:] != self.shape[1:]:
            raise ValueError("Dimension of input data does not match historical data")
        for i in range(x_data.shape[-1]):
            if inverse:
                tmp = self._scaler[i].inverse_transform(
                    x_data[:, :, :, i].reshape(-1, 1)
                )
            else:
                tmp = self._scaler[i].transform(x_data[:, :, :, i].reshape(-1, 1))
            x_data[:, :, :, i] = tmp.reshape(shape_original[:-1])
        self.data = x_data
        return x_data


def build_predict_data(train_wd: WeatherData, dict_data_predict) -> WeatherData:
    """
    Prepare predict weather data, based on training weather data

    Args:
        train_wd: WeatherData with training data, for scaling etc.
        dict_data_predict: weather data dict, the key being the weather parameter, and value the 2D array spatial data

    Returns:

    """

    predict_wd: WeatherData = WeatherData(
        dict_data_predict, scaling=train_wd.scaling, prediction=True
    )
    predict_wd.scaler = train_wd.scaler
    predict_wd.standardize()
    return predict_wd
