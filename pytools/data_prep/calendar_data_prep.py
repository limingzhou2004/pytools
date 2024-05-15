import datetime as dt

import holidays
import numpy as np
import pandas as pd
import pytz

from pytools.pandas_pgsql import PandasSql as Ps
from pytools.pgsql_adapter import PGSqlQuery as Mq
from pytools import get_file_folder


class CalendarData:
    """
    Prepare calendar data. Provide convinences for load data prep.
    world wide: pip install holidays

    holidays using pandas.tseries.holiday, and observed holidays
    https://towardsdatascience.com/holiday-calendars-with-pandas-9c01f1ee5fee

    Global timezone and daylishgt saving
    https://ibexorigin.medium.com/giant-mess-dealing-with-timezones-and-daylight-saving-time-in-python-7222d37658cf

    daylightsavingtime: _.tm_isdst
    import time
    time.localtime()
    (2010, 5, 21, 21, 48, 51, 4, 141, 0)
     _.tm_isdst
    0

    """     

    @DeprecationWarning
    def construct_calendar_data(
        self,
        start_time=dt.datetime.strptime("2016-12-01", "%Y-%m-%d"),
        end_time=dt.datetime.strptime("2031-01-01", "%Y-%m-%d"),
    ):
        # hourly intervals
        # delta_time = (end_time - start_time).total_seconds() / 3600
        # timestamp = {"timestamp": [start_time + dt.timedelta(hours=t) for t in range(int(delta_time))]}
        timestamp = pd.date_range(start=start_time, end=end_time, freq="H")



        # match yyyy-MM-DD only for holidays
        cdate = pd.DataFrame(timestamp, columns=["timestamp"])
        cdate["date"] = pd.to_datetime(timestamp.date)
        cal_date = cdate.merge(
            self.holiday_calendar, how="left", left_on="date", right_on="date"
        )
        # cal_date.set_index("timestamp")
        cal_date.rename(columns={"date": "holiday_date"}, inplace=True)
        self.cal_date = cal_date
        return self.cal_date
    
    def _is_us_holiday(self, t, state='NY', year=2022):
        h =  holidays.US(subdiv=state, years=year)
        return t.date() in h
    
    def _is_daylightsaving(self, t, tz):
        # Checks if a given date is in daylight saving time.
        # Returns True if the date is in daylight saving time, False otherwise.
        #timezone = pytz.timezone(tz)
        t2 = t.tz_convert(tz)
        
        return t2.dst() != dt.timedelta(0)


    @DeprecationWarning
    def load_daylightsaving_to_db(self, schema, table):
        daylightsaving_data = pd.read_csv(
            self.daylightsaving_file, parse_dates=["start", "end"], index_col=0
        )
        self.load_to_db(schema=schema, table=table, df=daylightsaving_data)

    @DeprecationWarning
    def load_to_db(self, schema: str, table: str, df=None):
        """
        manually create a unique index on timestamp, and upload dataframe to database

        Args:
            schema:
            table:
            df:

        Returns: None

        """

        eng = Mq().get_sqlalchemy_engine()
        if df is None:
            df = self.cal_date
        Ps.df_to_sql(df, eng, schema=schema, tbl_name=table)

    @DeprecationWarning
    def is_daylightsaving(self, t: np.datetime64):
        year = pd.to_datetime(t).year
        tt = self.daylightsaving_data.loc[year]
        if tt[1] > t >= tt[0]:
            return True
        else:
            return False

    @DeprecationWarning
    def get_daylightsaving_data(self):
        return self.daylightsaving_data

    def get_hourofday(self, df: pd.Series):
        h = df.apply(lambda x: pd.Timestamp(x).hour).values
        h = np.squeeze(h)
        d = {
            "hourofday_sin": np.sin(h / 12 * np.pi),
            "hourofday_cos": np.cos(h / 12 * np.pi),
        }
        index = list(range(len(h)))
        dfr = pd.DataFrame(d, index=index)
        return list(dfr), dfr

    def get_dayofweek(self, df: pd.Series):
        h = df.apply(lambda x: pd.Timestamp(x).dayofweek).values
        h = np.squeeze(h)
        d = {
            "dayofweek_sin": np.sin(h / 7 * 2 * np.pi),
            "dayofweek_cos": np.cos(h / 7 * 2 * np.pi),
        }
        index = list(range(len(h)))
        dfr = pd.DataFrame(d, index=index)
        return list(dfr), dfr
    
    def get_holiday_dst(self, df:pd.Series, tz):
        h_hld = df.apply(self._is_us_holiday)
        h_dst = df.apply(self._is_daylightsaving, args=(tz,))
        index = list(range(len(h_hld)))
        d = {'holiday':h_hld, 'daylighsaving':h_dst}
        dfr = pd.DataFrame(d, index=index)
        return list(dfr), dfr

    
    


@DeprecationWarning
def main_load_daylightsaving_time(schema):
    cd = CalendarData()
    cd.load_daylightsaving_to_db(schema=schema, table="calendar")

@DeprecationWarning
def make_data():
    holiday_file = "../../resources/calendar/meta data - calendar.csv"
    dls_file = "../../resources/calendar/daylightsaving time.csv"
    holiday_df = pd.read_csv(holiday_file, parse_dates=["date"])
    holiday_df["date"] = holiday_df["date"].dt.date
    day_light_saving_df = pd.read_csv(dls_file, parse_dates=["start", "end"])
    df = pd.DataFrame(
        pd.date_range(start="1/1/2016", end="12/31/2030 23:00", freq="H"),
        columns=["timestamp"],
    )
    df["date"] = df["timestamp"].dt.date
    df["year"] = df.apply(lambda x: x["timestamp"].year, axis=1)
    df = df.merge(day_light_saving_df, on="year", how="left")
    df["daylightsaving"] = df.apply(
        lambda x: 1 if x["start"] < x["timestamp"] < x["end"] else 0, axis=1
    )
    df["holiday"] = df.apply(
        lambda x: 1 if x["date"] in set(holiday_df["date"]) else 0,
        axis=1,
    )
    del df["start"]
    del df["end"]
    del df["year"]
    df.to_pickle("calendar.pkl")

@DeprecationWarning
def upload_calendar_data(schema="nyiso", table="calendar"):
    df = pd.read_pickle("../calendar.pkl")
    cd = CalendarData()
    cd.load_to_db(schema, table, df)


if __name__ == "__main__":
    # main_load_daylightsaving_time(schema="nyiso")
    upload_calendar_data()
