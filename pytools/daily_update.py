import sys

from . import web_data_scraper as ws


class DailyUpdate:
    """
    Daily - Hourly update
    """

    def __init__(self, hour_back=2):
        self.ws = ws.WebDataScraper()

    @staticmethod
    def update_nyiso_hist_load(
        schema="nyiso",
        table="nyiso_hist_load",
        unique_cols=["Time Stamp", "Time Zone", "PTID", "Name"],
        hours_back=2,
    ):
        w = ws.WebDataScraper.build_nyiso_load_scraper(hours_back=hours_back)
        w.upsert_new(schema=schema, tab_name=table, unique_cols=unique_cols)


# The entrance is same for all types of data sources
def main(args):
    opt = None
    ag = iter(args)
    flag = True
    schema = "nyiso"
    table = None
    unique_cols = None
    hours_back = None
    while flag:
        a = next(ag, None)
        if a == "--schema":
            schema = next(ag)
        elif a == "--table":
            table = next(ag)
        elif a == "--unique_cols":
            unique_cols = [c.trim() for c in next(ag).split(",")]
        elif a == "--hours_back":
            hours_back = int(next(ag))
        elif a == "--opt":
            opt = next(ag)
        if a is None:
            flag = False

    if opt == "nyiso_hist_load":
        DailyUpdate.update_nyiso_hist_load(
            schema=schema, table=table, hours_back=hours_back
        )


if __name__ == "__main__":
    main(sys.argv)
