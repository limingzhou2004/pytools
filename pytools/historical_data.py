import sys
from . import web_data_scraper as ws


class HistoricalData:
    @staticmethod
    def process_historical_nyiso_load(
        months_back,
        schema,
        table_name,
        unique_cols=["Time Stamp", "Time Zone", "PTID", "Name"],
    ):
        w = ws.WebDataScraper.build_nyiso_load_zip_scraper(months_back=months_back)
        w.upsert_new(
            schema=schema,
            tab_name=table_name,
            unique_cols=unique_cols,
        )


def main(args):
    month_start = 1
    month_end = 24
    opt = "nyiso_load"
    for i in range(len(args)):
        if args[i] == "--month_start":
            month_start = int(args[i + 1])
        elif args[i] == "--month_end":
            month_end = int(args[i + 1])
        elif args[i] == "--opt":
            opt = args[i + 1]
    if opt == "nyiso_load":
        for m in range(month_start, month_end):
            print(f"months back...{m}")
            HistoricalData.process_historical_nyiso_load(m)


if __name__ == "__main__":
    main(sys.argv)
