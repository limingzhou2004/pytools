__author__ = "limingzhou"

timezone_utc = {
    "est": -5,
    "edt": -4,
    "cst": -6,
    "cdt": -5,
    "pst": -8,
    "pdt": -7,
    "mst": -7,
    "mdt": -6,
}
# token used in sql query as column names
token_load = "load"
token_timestamp = "timestamp"
token_query_max_date: str = "query_max_date"
token_query_train: str = "query_train"
token_query_predict: str = "query_predict"
token_site_name = "site_name"
token_table_name = "table_name"
token_t0 = "t0"
token_t1 = "t1"
token_max_load_time = "max_load_time"
#
#


def query_str_fill(qstr: str, **kwargs):
    for k, v in kwargs.items():
        qstr = qstr.replace(f"{{{k}}}", v)
    return qstr
