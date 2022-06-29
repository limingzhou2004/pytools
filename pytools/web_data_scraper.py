import requests
import io
import datetime as dt
from datetime import timedelta
import pandas as pd
from zipfile import ZipFile
from . import pandas_pgsql
from . import pgsql_adapter
from dateutil.relativedelta import relativedelta


class WebDataScraper:
    """
    Get web load data from NYISO, and other ISOs
    NYISO load in mwh: http://mis.nyiso.com/public/csv/palIntegrated/YYYYMMDDpalIntegrated.csv
    nyiso load zip file  "http://mis.nyiso.com/public/csv/palIntegrated/20181101palIntegrated_csv.zip"

    """

    nyiso_base = (
        "http://mis.nyiso.com/public/csv/palIntegrated/{YYYYMMDD}palIntegrated.csv"
    )
    nyiso_zip_base = (
        "http://mis.nyiso.com/public/csv/palIntegrated/{YYYYMMDD}palIntegrated_csv.zip"
    )

    def __init__(self, base_url, parse_dates=None, schema="nyiso", **paras):
        self.base_url = base_url
        self.paras = paras
        self.parse_dates = parse_dates
        self.schema = schema
        self.url = base_url.format(**paras)

    def check_table_exist(self, tab_name, schema=None):
        schema = schema if schema else self.schema
        qstr = f"""SELECT *
                FROM pg_catalog.pg_tables
                WHERE schemaname='{schema}' and tablename='{tab_name}'
        """
        adapter = pgsql_adapter.PGSqlQuery()
        res = adapter.read_dict_query(qstr)
        if res:
            return True
        else:
            return False

    def read_db_last(self, schema="nyiso", table="", timestamp="timestamp"):
        qstr = f'select max("{timestamp}") max_date from {schema}."{table}"'
        adapter = pgsql_adapter.PGSqlQuery()
        res = adapter.read_dict_query(qstr)
        return res[0][0]

    def scrap_and_load(self, schema, tab_name, if_exists):
        """
        Create a new table, then use a staging table for updating

        :param schema: schema
        :param tab_name: table name
        :param if_exists:
        :return: col names as a list
        """
        adapter = pgsql_adapter.PGSqlQuery()
        engine = adapter.get_sqlalchemy_engine()
        df = self.scrap()
        new_dtype = pandas_pgsql.PandasSql.get_new_dtype(df)
        pandas_pgsql.PandasSql.df_to_sql(
            df, engine, schema, tab_name, new_dtype, if_exists=if_exists
        )
        return list(df)

    def upsert_new(self, schema, tab_name, unique_cols):
        # if the table does not exist, create it first
        adapter = pgsql_adapter.PGSqlQuery()
        if not self.check_table_exist(tab_name):
            self.scrap_and_load(schema, tab_name=tab_name, if_exists="replace")
            adapter.create_identity(table=tab_name, column="index")
            adapter.create_unique_constraint(table=tab_name, columns=unique_cols)
        cols = self.scrap_and_load(
            schema, tab_name=tab_name + "_staging", if_exists="replace"
        )
        col_str = ",".join(f'"{c}"' for c in cols)
        update_cols = list(set(cols) - set(unique_cols))
        update_str = ",".join(f'"{c}"=excluded."{c}"' for c in update_cols)
        qstr = f"""insert into {schema}.{tab_name} as t ({col_str}) 
        select {col_str} from {schema}.{tab_name}_staging
        on conflict on constraint unq_{tab_name}
        do update set {update_str}
        """
        print(qstr)
        adapter.non_read_query(qstr)

    @classmethod
    def build_nyiso_load_scraper(cls, hours_back=2, timestamp=dt.datetime.now()):
        return cls(
            cls.nyiso_base,
            ["Time Stamp"],
            YYYYMMDD=dt.datetime.strftime(
                (timestamp - timedelta(hours=hours_back)), "%Y%m%d"
            ),
        )

    @classmethod
    def build_nyiso_load_zip_scraper(cls, timestamp=dt.datetime.now(), months_back=0):
        return cls(
            cls.nyiso_zip_base,
            ["Time Stamp"],
            YYYYMMDD=(timestamp - relativedelta(months=months_back)).strftime("%Y%m01"),
        )

    def scrap(self, url=None, parse_dates=None):
        if not url:
            url = self.url
        if not parse_dates:
            parse_dates = self.parse_dates
        if url.endswith("zip"):
            req = requests.get(url)
            zip_file = ZipFile(io.BytesIO(req.content))
            dfs = [
                pd.read_csv(zip_file.open(text_file.filename), parse_dates=parse_dates)
                for text_file in zip_file.infolist()
                if text_file.filename.endswith(".csv")
            ]
            return pd.concat(dfs)
        else:
            return pd.read_csv(url, delim_whitespace=False, parse_dates=parse_dates)
