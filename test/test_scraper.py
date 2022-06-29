from pytools import web_data_scraper as ws
import pytools.pgsql_adapter as sa
import pytest
import datetime as dt


class TestWebScrap:

    nyiso_base = (
        "http://mis.nyiso.com/public/csv/palIntegrated/{YYYYMMDD}palIntegrated.csv"
    )

    @pytest.fixture
    def test_nyiso(self):
        nyiso_base = (
            "http://mis.nyiso.com/public/csv/palIntegrated/{YYYYMMDD}palIntegrated.csv"
        )
        w = ws.WebDataScraper(
            nyiso_base, parse_dates=["Time Stamp"], YYYYMMDD="20210219"
        )
        data = w.scrap()
        data.head(5)
        return data

    def test_nyiso_zip(self):
        # "http://mis.nyiso.com/public/csv/palIntegrated/20181001palIntegrated_csv.zip"
        nyiso_base = "http://mis.nyiso.com/public/csv/palIntegrated/{YYYYMMDD}palIntegrated_csv.zip"
        w = ws.WebDataScraper(
            nyiso_base, parse_dates=["Time Stamp"], YYYYMMDD="20181001"
        )
        data = w.scrap()
        return data

    def test_nyiso_constructor(self):
        w = ws.WebDataScraper.build_nyiso_load_scraper(hours_back=3)
        data = w.scrap()
        data

    def test_write_db(self):
        data = ws.WebDataScraper.build_nyiso_load_scraper().scrap_and_load(
            schema="nyiso", tab_name="nyiso_hist_load", if_exists="replace"
        )
        assert data

    def test_create_unique_index(self):
        q = sa.PGSqlQuery()
        q.create_unique_constraint(
            "nyiso_hist_load", ["Time Stamp", "Time Zone", "PTID"]
        )

    # def test_identity(self):
    #     q = sa.PGSqlQuery()
    #     q.create_unique_index("nyiso_hist_load", "index")

    def test_read_last(
        self,
    ):
        w = ws.WebDataScraper.build_nyiso_load_scraper(hours_back=3)
        last_time = w.read_db_last(timestamp="Time Stamp", table="nyiso_hist_load")
        assert last_time

    def test_check_table_exists(self):
        w = ws.WebDataScraper.build_nyiso_load_scraper(hours_back=3)
        assert w.check_table_exist("nyiso_hist_load")
        assert not w.check_table_exist("nyiso_hist_load_staging")

    def test_upsert_new(self):
        w = ws.WebDataScraper.build_nyiso_load_scraper(
            hours_back=3, timestamp=dt.datetime.now() - dt.timedelta(days=1)
        )
        w.upsert_new(
            schema="nyiso",
            tab_name="nyiso_hist_load",
            unique_cols=["Time Stamp", "Time Zone", "PTID", "Name"],
        )
        assert 1

    def test_upsert_zip(self):
        w = ws.WebDataScraper.build_nyiso_load_zip_scraper(months_back=0)
        w.upsert_new(
            schema="nyiso",
            tab_name="nyiso_hist_load",
            unique_cols=["Time Stamp", "Time Zone", "PTID"],
        )


        ddd