import pandas as pd
import sqlalchemy

from pytools import get_logger

logger = get_logger("pandas sql")


class PandasSql:
    """
    Install mysqldb on osx: brew install mysql-connector-c
    pip install configparser
    pip install

    Install mysqldb on ubuntu: sudo apt-get install mysql-server mysql-client

    """

    class sqlError(Exception):
        pass

    @staticmethod
    def clean_col_name(df):
        col = list(df)
        new_col = [c.replace("/", "_").replace(" ", "_") for c in col]
        df.columns = new_col
        return df

    @staticmethod
    def csv_to_df(infile, headers=[]):
        if len(headers) == 0:
            df = pd.read_csv(infile)
        else:
            df = pd.read_csv(infile, header=None)
            df.columns = headers
        for r in range(10):
            try:
                df.rename(
                    columns={"Unnamed: {0}".format(r): "Unnamed{0}".format(r)},
                    inplace=True,
                )
            except Exception as ex:
                logger.error(msg=str(ex))
        return df

    @staticmethod
    def dtype_mapping():
        return {
            "object": "TEXT",
            "int64": "INT",
            "float64": "FLOAT",
            "datetime64": "DATETIME",
            "bool": "TINYINT",
            "category": "varchar(20)",
            "timedelta[ns]": "TEXT",
        }

    @staticmethod
    def gen_tbl_cols_sql(df):
        dmap = PandasSql.dtype_mapping()
        sql = "pi_db_uid INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY"
        df1 = df.rename(columns={"": "nocolname"})
        hdrs = df1.dtypes.index
        hdrs_list = [("`" + hdr + "`", str(df1[hdr].dtype)) for hdr in hdrs]
        for i, hl in enumerate(hdrs_list):
            sql += " ,{0} {1}".format(hl[0], dmap[hl[1]])
        return sql

    @staticmethod
    def sql_dtype_map():
        return {
            "object": sqlalchemy.types.VARCHAR(50),
            "int64": sqlalchemy.types.INTEGER,
            "float64": sqlalchemy.types.FLOAT,
            "datetime64[ns]": sqlalchemy.types.TIMESTAMP,
            "bool": sqlalchemy.types.BOOLEAN,
            "category": sqlalchemy.types.VARCHAR(50),
            "timedelta[ns]": sqlalchemy.types.TEXT,
        }

    @staticmethod
    def get_new_dtype(df):
        dmap = PandasSql.sql_dtype_map()
        return {n: dmap[str(df[n].dtype)] for n in df.dtypes.index}

    @staticmethod
    def df_to_sql(df, engine, schema, tbl_name, new_dtype=None, if_exists="replace"):
        df.to_sql(tbl_name, engine, schema=schema, if_exists=if_exists, dtype=new_dtype)

    @staticmethod
    def read_sql_timeseries(engine, qstr):
        with engine.connect() as conn:
            return pd.read_sql(qstr, conn.connection)
