# https://www.postgresqltutorial.com/postgresql-python/connect/


import pandas as pd
import os

import psycopg2 as pg
from sqlalchemy import create_engine
import read_env
from typing import Dict


class PGSqlQuery:
    def __init__(
        self,
        server="localhost",
        port=5432,
        user="postgres",
        password=None,
        database="daf",
        schema="nyiso",
    ):
        read_env.read_env()
        env_paras = os.environ
        if password is None:
            password = env_paras.get("pgsql_password")
        self.server = env_paras.get("server", server)
        self.user = env_paras.get("user", user)
        self.port = env_paras.get("port", port)
        self.database = env_paras.get("database", database)
        self.password = password
        self.schema = env_paras.get("schema", schema)
        self.pgsql_str = f"postgresql+psycopg2://{self.user}:{self.password}@{self.server}:{self.port}/{self.database}"

    def get_sqlalchemy_engine(self):
        return create_engine(self.pgsql_str)

    def get_sql_connection(self) -> object:

        return pg.connect(
            f"host='{self.server}' port={self.port} dbname='{self.database}' user='{self.user}' password='{self.password}'"
        )

    def read_dict_query(self, qstr):
        ret = None
        db = self.get_sql_connection()
        try:
            cur = db.cursor()
            cur.execute(qstr)
            ret = []
            for r in cur:
                ret.append(r)
            cur.close()
        except Exception as ex:
            print(ex)
        finally:
            db.close()
        # return results in a dict
        # a column is a tuple, with the first element column name
        # {'columns':[('col1', 1, 1, 1),('col2', 2, 2, 2)],
        # 'eof':{'status_flag': 16385, 'warning_count':0}}
        return ret

    def non_read_query(self, qstr):
        """
        Run non-executive query

        Args:
            qstr:

        Returns: query results

        """
        res = None
        db = self.get_sql_connection()
        try:
            cur = db.cursor()
            res = cur.execute(qstr)
            db.commit()

        except Exception as ex:
            print(ex)
            raise ex
        finally:
            db.close()
        return res

    def non_read_multi_query(self, qstr):
        """
        Run multiple non-read query
        Args:
            qstr:

        Returns:

        """
        res = None
        db = self.get_sql_connection()
        cr = db.cursor()
        try:

            res = db.cmd_query_iter(qstr)
        except Exception as ex:
            print(ex)
        finally:
            cr.commit()
            cr.close()
        # for r in res:
        #     if "columns" in r:
        #         columns = r["columns"]
        #         rows = cr.get_row()
        return res

    def create_unique_constraint(self, table, columns):
        col_str = ",".join([f'"{c}"' for c in columns])
        qstr = f"alter table {self.schema}.{table} add constraint unq_{table} unique ({col_str}); "
        self.non_read_query(qstr)

    def create_identity(self, table, column):
        qstr = f'alter table {self.schema}.{table} modify column "{column}" int GENERATED ALWAYS AS IDENTITY'
        self.non_read_query(qstr)

    def sql2df(self, qstr: str, paras: Dict) -> pd.DataFrame:
        """
        Example:
        df = psql.read_sql(('select "Timestamp","Value" from "MyTable" '
                     'where "Timestamp" BETWEEN %(dstart)s AND %(dfinish)s'),
                   db,params={"dstart":datetime(2014,6,24,16,0),"dfinish":datetime(2014,6,24,17,0)},
                   index_col=['Timestamp'])

        Args:
            qstr:  To pass the values in the sql query, syntaxes possible: ?, :1, :name, %s, %(name)s
            paras: dict of parameters

        Returns: DataFrame from the qstr query

        """
        engine = self.get_sqlalchemy_engine()
        df = pd.read_sql(qstr, con=engine, params=paras)
        return df


