import sqlalchemy as sa
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
import uuid
import yaml
import os
import os.path as osp
# …


def get_pg_conn(port=5432, db='daf', schema='iso', para_airflow=None):
    if not para_airflow:
        with open(osp.join(osp.dirname(osp.abspath(__file__)),'../sql/sql_config.yaml'), 'r') as f:
            db_config = yaml.safe_load(f)
            server = os.getenv(db_config['server'])
            user = os.getenv(db_config['user'])
            pwd = os.getenv(db_config['password'])
    else:
        server=para_airflow['pg_server']
        user=para_airflow['pg_user']
        pwd=para_airflow['pg_pwd']


    engine = create_engine(f"postgresql+psycopg2://{user}:{pwd}@{server}:{port}/{db}", connect_args={'options': '-csearch_path={}'.format(schema)})

    return engine


def upsert_df(df: pd.DataFrame, table_name: str, engine: sqlalchemy.engine.Engine, schema):
    """Implements the equivalent of pd.DataFrame.to_sql(..., if_exists='update')
    (which does not exist). Creates or updates the db records based on the
    dataframe records.
    Conflicts to determine update are based on the dataframes index.
    This will set unique keys constraint on the table equal to the index names
    1. Create a temp table from the dataframe
    2. Insert/update from temp table into table_name
    Returns: True if successful
    """

    # If the table does not exist, we should just use to_sql to create it
    if not engine.execute(
        f"""SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE  table_schema = '{schema}'
            AND    table_name   = '{table_name}');
            """
    ).first()[0]:
        df.to_sql(table_name, engine)
        return True

    # If it already exists...
    temp_table_name = f"temp_{uuid.uuid4().hex[:6]}"
    df.to_sql(temp_table_name, engine, index=True)

    index = list(df.index.names)
    index_sql_txt = ", ".join([f'"{i}"' for i in index])
    columns = list(df.columns)
    headers = index + columns
    headers_sql_txt = ", ".join(
        [f'"{i}"' for i in headers]
    )  # index1, index2, ..., column 1, col2, ...

    # col1 = exluded.col1, col2=excluded.col2
    update_column_stmt = ", ".join([f'"{col}" = EXCLUDED."{col}"' for col in columns])

    # For the ON CONFLICT clause, postgres requires that the columns have unique constraint
    query_pk = f"""
    ALTER TABLE "{table_name}" DROP CONSTRAINT IF EXISTS unique_constraint_for_upsert_{table_name};
    ALTER TABLE "{table_name}" ADD CONSTRAINT unique_constraint_for_upsert_{table_name} UNIQUE ({index_sql_txt});
    """
    engine.execute(query_pk)

    # Compose and execute upsert query
    query_upsert = f"""
    INSERT INTO "{table_name}" ({headers_sql_txt}) 
    SELECT {headers_sql_txt} FROM "{temp_table_name}"
    ON CONFLICT ({index_sql_txt}) DO UPDATE 
    SET {update_column_stmt};
    """
    engine.execute(query_upsert)
    engine.execute(f"DROP TABLE {temp_table_name}")

    return True


def get_table_names_by_prefix(schema, prefix, engine):

    qstr=f"""SELECT table_name  FROM information_schema.tables 
        WHERE  table_schema = '{schema}'
        AND    table_name like '{prefix}%%'"""
    with engine.begin() as conn:
        dft=pd.read_sql_query(qstr, conn)
    return dft


def clean_tmp_tables(schema, conn, engine):
    tmps = get_table_names_by_prefix(schema, 'temp_')['table_name'].to_dict()

    with engine.begin() as conn:
        for id, t in enumerate(tmps):
            qstr=f"drop table if exists {schema}.{tmps[t]}"
            conn.exec_driver_sql(qstr)

    
