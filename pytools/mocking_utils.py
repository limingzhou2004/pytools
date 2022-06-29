import datetime as dt
import numpy as np
import pandas as pd

from pytools.data_prep import calendar_data_prep as Cp


def mock_max_date(monkeypatch):
    return pd.DataFrame(
        {"max_date": [dt.datetime.strftime(dt.datetime.now(), "%m/%d/%Y %H:%M")]}
    )


# to use the mocking, monkeypatch.setattr(LoadData, "query_max_load_time", mock_function_name)


def mock_train_load(*args, **kargs):
    t0 = kargs.get("t0", "12/25/2018 8:00")
    t1 = kargs.get("t1", "12/26/2018 15:00")
    df = pd.date_range(t0, t1, freq="H").to_frame(name="timestamp", index=False)
    row_count = len(df)
    df["holiday"] = 0
    df["daylightsaving"] = 0
    df["load"] = np.random.random(row_count) * 100
    cols, dh = Cp.CalendarData().get_hourofday(df["timestamp"])
    for c in cols:
        df[c] = dh[c]
    cols, dw = Cp.CalendarData().get_dayofweek(df["timestamp"])
    for c in cols:
        df[c] = dw[c]
    return df


# to use the mocking,     monkeypatch.setattr(LoadData, "query_train_data", mock_function_name)


def mock_predict_load(*args, **kargs):
    t0 = kargs.get("t0", "12/25/2018 8:00")
    t1 = kargs.get("t1", "12/28/2018 15:00")
    tc = kargs.get("tc", "12/26/2018 15:00")
    df = pd.date_range(t0, t1, freq="H").to_frame(name="timestamp", index=False)
    row_count = len(df)
    df["holiday"] = 0
    df["daylightsaving"] = 0
    df["load"] = np.random.random(row_count) * 100
    cols, dh = Cp.CalendarData().get_hourofday(df["timestamp"])
    for c in cols:
        df[c] = dh[c]
    cols, dw = Cp.CalendarData().get_dayofweek(df["timestamp"])
    for c in cols:
        df[c] = dw[c]

    df.loc[df["timestamp"] >= tc, "load"] = np.nan
    return df

    # to use the mocking, monkeypatch.setattr(LoadData, "query_predict_data", mock_predict_load)
