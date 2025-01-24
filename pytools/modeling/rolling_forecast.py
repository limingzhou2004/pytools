from typing import Union, Dict
#import mlflow
import numpy as np
import pandas as pd
import torch
from torch import Tensor

from pytools.config import Config
from pytools.modeling.ts_weather_net import TsWeaDataModule


class RollingForecast:
    """
    Predict by steps
    """

    def __init__(
        self, config:Config, ):
        self.config = config

    def _check_data(self):
        raise NotImplementedError

    def _predict_step(self):

        return

    def _get_first_nan(self, df: pd.DataFrame, col_name: str):
        for row in df.itertuples():
            if np.isnan(getattr(row, col_name)):
                return row.Index

    def to_tensor32(self, x: np.ndarray):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.from_numpy(x).float().to(device)

    def predict(self, hrs: int):
        """
        Predict multiple steps, based on the existing model steps

        Args:
            hrs: hours ahead

        Returns: dataframe with predictions

        """
        assert self._model, "No models!"
        assert self._model[1], "No step 1 model"
        # remove step 1 model
        mdl_steps = sorted(self._model.keys())[1:]
        start_ind = self._get_first_nan(self._df_load, self._load_name)

        def prd_fun(ind, ld, cl, wt, start_ind):
            """
            Forecast and update the load df

            Args:
                ind: from which index to start the fst
                ld: load df
                cl: calendar df
                wt: weather array
                start_ind: the ind prediction period starts

            Returns: Update the self._df_load in situ

            """
            channel_num = 3
            new_channel_num = 1
            wt = np.moveaxis(wt, channel_num, new_channel_num)
            m: WeatherNet = self._model[ind]
            # add load data one more dimension, as it is 1d convolution
            # yp = m.forward(self.to_tensor32(wt), self.to_tensor32(ld), self.to_tensor32(cl))
            yp = m.model(
                self.to_tensor32(wt), self.to_tensor32(ld), self.to_tensor32(cl)
            )
            self._df_load.loc[
                start_ind + ind - 1, self._load_name
            ] = yp.data.detach().numpy()[0][0]

        max_ind = len(self._df_load)
        # Direct forecast big steps, 6, 12, 18, 24, 30, 36, 42...
        leap_steps = list(
            i - start_ind
            for i in range(start_ind, max_ind)
            if i - start_ind in mdl_steps
        )
        for s in leap_steps:
            load_embed, cal, wea = self._get_data_components(start_ind)
            prd_fun(s, load_embed, cal, wea, start_ind)
        # Rolling forecast at step 1
        for row in self._df_load.itertuples():
            if not np.isnan(getattr(row, self._load_name)):
                continue
            load_embed, cal, wea = self._get_data_components(row.Index)
            prd_fun(1, load_embed, cal, wea, row.Index)
            if hrs:
                if row.Index - start_ind >= hrs:
                    break
        return (
            self._df_load,
            self._df_load.loc[start_ind][self._timestamp_name],
            hrs if hrs else len(self._df_load) - start_ind,
        )

    def _get_data_components(self, cur_prd_ind):
        """
        Prepare the input data for the prediction of a given step； _df_load is updated by previous predictions

        Args:
            cur_prd_ind: current prediction index
            #prd_leap_dim: prediction steps (hours) ahead

        Returns: embedded load, calendar and weather data

        """
        df_copy = self._df_load.copy()
        load_embed = df_copy.loc[
            cur_prd_ind - self._load_embed_dim : cur_prd_ind - 1, self._load_name
        ].values.reshape((1, 1, -1))
        cal_data = (
            (
                df_copy.loc[
                    cur_prd_ind,
                    self._df_load.columns.difference(
                        [self._load_name, self._timestamp_name]
                    ),
                ]
            )
            .values.astype(np.float)
            .reshape((1, -1))
        )
        wea = np.expand_dims(
            self._wea.data[cur_prd_ind - self._load_embed_dim, :, :, :], 0
        )
        return load_embed, cal_data, wea
