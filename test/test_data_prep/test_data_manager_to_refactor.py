# import os

# import pytest
# from dateutil import parser
# from functools import partial

# import dill


# from pytools.data_prep import data_prep_manager as dm
# from pytools.data_prep.data_prep_manager import save
# import pytools.data_prep.get_datetime_from_grib_file_name
# from pytools.data_prep.load_data_prep import LoadData
# from pytools.data_prep.get_datetime_from_grib_file_name import get_datetime_from_grib_file_name
# from pytools import get_logger
# from pytools.data_prep import weather_data_prep as wp


# @pytest.mark.skip("to drop")
# class TestDataManager:
#     jar_address = (
#         '"/Users/limingzhou/zhoul/work/me/Scala-http/classes/artifacts/scalahttp_jar/*"'
#     )
#     site_folder = (
#         "/Users/limingzhou/zhoul/work/me/xaog_ops/modeling/sites/nyiso/nyiso_hist_load"
#     )

#     weather_folder = ["/Users/limingzhou/zhoul/work/me/testdata/input4"]
#     weather_pred_in = "/Users/limingzhou/zhoul/work/me/testdata/hrrr_fst_weather"

#     weather_folder_nam = ["/Users/limingzhou/zhoul/work/me/testdata/test_nem_data"]
#     weather_pred_in_nam = "/Users/limingzhou/zhoul/work/me/testdata/test_nem_data_pred"

#     table_name = "nyiso_hist_load"

#     def test_get_datetime_from_grib_file_name(self):
#         fn = "nam_12_2019_02_03_14F1.grib2"
#         pytools.data_prep.get_datetime_from_grib_file_name.get_datetime_from_grib_file_name(fn, hour_offset=-5)

#     def test_training_data_save(self):
#         table_name = "nyiso_hist_load"
#         site_folder = "/Users/limingzhou/zhoul/work/me/xaog_ops/modeling/sites/nyiso"
#         t0 = "2018-12-01"
#         t1 = "2019-03-04"
#         dm0 = dm.DataPrepManager.build_nyiso_load_prep(
#             "capitl",
#             site_description="nyiso hist load for capital (albany)",
#             site_parent_folder=site_folder,
#             t0=t0,
#             t1=t1,
#             category=table_name,
#             weather_type=wp.GribType.hrrr,
#         )
#         # fn = dm0.save(suffix="test")
#         # dm1 = dm.DataPrepManager.load(fn)
#         tt0 = "2018-12-29"
#         tt1 = "2019-01-02"
#         ld = dm0.get_prediction_load(t0=tt0, t1=tt1)
#         dm0.standardize_predictions(ld)
#         save(dm0, suffix="test")
#         assert 1 == 1

#     def test_training_data_save_nam(self):
#         table_name = "nyiso_hist_load"
#         site_folder = "/Users/limingzhou/zhoul/work/me/xaog_ops/modeling/sites/nyiso"
#         t0 = "2019-02-02"
#         t1 = "2019-02-04"
#         dm0 = dm.DataPrepManager.build_nyiso_load_prep(
#             "capitl",
#             site_description="nyiso hist load for capital (albany)",
#             site_parent_folder=site_folder,
#             t0=t0,
#             t1=t1,
#             category=table_name,
#             weather_type=wp.GribType.nam,
#         )
#         tt0 = "2019-02-02"
#         tt1 = "2019-02-04"
#         ld = dm0.get_prediction_load(t0=tt0, t1=tt1)
#         assert ld.shape[1] == 8
#         dm0.standardize_predictions(ld)
#         save(dm0, suffix="test_nam")

#     def test_predict_data(self):
#         d: dm.DataPrepManager = dm.load(
#             os.path.join(self.site_folder, "capitl", "capitl_data_manager_test")
#         )
#         d.build_weather(
#             weather_folder=self.weather_folder,
#             jar_address=self.jar_address,
#             center='"(43,-73.0)"',
#             rect='"(100,100.0)"',
#         )
#         d.make_npy_train()
#         d.make_npy_predict(
#             in_folder=self.weather_pred_in,
#             out_folder=None,
#             time_after=parser.parse("2018-12-31"),
#         )

#     def test_predict_data_nam(self):
#         d: dm.DataPrepManager = dm.load(
#             os.path.join(self.site_folder, "capitl", "capitl_data_manager_test_nam")
#         )

#         d.build_weather(
#             weather_folder=self.weather_folder_nam,
#             jar_address=self.jar_address,
#             center='"(43,-73.0)"',
#             rect='"(100,100.0)"',
#         )
#         d.make_npy_train()
#         d.make_npy_predict(
#             in_folder=self.weather_folder_nam,
#             out_folder=None,
#             time_after=parser.parse("2018-12-31"),
#         )
#         assert 1 == 1

#     def test_weather_data_pre_grib_filter(self):
#         fn = os.listdir(self.weather_pred_in)
#         dm.wp.grib_filter_func(fn)

#     def test_load_weather_data(self):
#         fn = os.path.join(self.site_folder, "capitl", "capitl_data_manager_test")
#         d: dm.DataPrepManager = dm.load(fn)
#         d.build_weather(
#             weather_folder=self.weather_folder,
#             jar_address=self.jar_address,
#             center='"(43,-73.0)"',
#             rect='"(100,100.0)"',
#         )
#         t0 = "2018-12-25"
#         t1 = "2019-01-09"
#         # d.make_npy_train()
#         # d.make_npy_predict(in_folder=self.weather_pred_in,
#         #                   out_folder=None,
#         #                   time_after=parser.parse("2018-12-31"))
#         h_weather = (
#             d.get_train_weather()
#         )  # for hrrr, get all files in the folder between t0 and t1
#         # for hrrr, get all files in the folder between t0 and t1
#         p_weather = d.get_predict_weather()
#         join_load, join_wdata = d.reconcile(
#             d.load_data.train_data, d.load_data.date_col, h_weather
#         )
#         join_load_pre, join_wdata_pre = d.reconcile(
#             d.load_data.query_predict_data(t0=t0, t1=t1),
#             d.load_data.date_col,
#             p_weather,
#         )
#         assert 1 == 1

#     def test_load_weather_data_nam(self):
#         d: dm.DataPrepManager = dm.load(
#             os.path.join(self.site_folder, "capitl", "capitl_data_manager_test_nam")
#         )
#         d.build_weather(
#             weather_folder=self.weather_folder_nam,
#             jar_address=self.jar_address,
#             center='"(43,-73.0)"',
#             rect='"(100,100.0)"',
#         )
#         t0 = "2019-02-02"
#         t1 = "2019-02-04"
#         hour_offset = -5
#         nam_hist_max_fst_hour = 5
#         filter_func_train = partial(
#             dm.wp.grib_filter_func,
#             func_timestamp=partial(
#                 get_datetime_from_grib_file_name,
#                 hour_offset=hour_offset,
#                 get_fst_hour=False,
#             ),
#             func_fst_hours=partial(
#                 get_datetime_from_grib_file_name,
#                 hour_offset=hour_offset,
#                 get_fst_hour=True,
#             ),
#             predict=False,
#             max_fst_hours=nam_hist_max_fst_hour,
#         )
#         filter_func_predict = partial(
#             dm.wp.grib_filter_func,
#             func_timestamp=partial(
#                 get_datetime_from_grib_file_name,
#                 hour_offset=hour_offset,
#                 get_fst_hour=False,
#             ),
#             func_fst_hours=partial(
#                 get_datetime_from_grib_file_name,
#                 hour_offset=hour_offset,
#                 get_fst_hour=True,
#             ),
#             predict=True,
#             max_fst_hours=nam_hist_max_fst_hour,
#         )
#         print("process train\n")
#         d.make_npy_train(filter_func=filter_func_train)
#         print("process predict\n")
#         d.make_npy_predict(
#             in_folder=self.weather_pred_in_nam,
#             out_folder=None,
#             time_after=parser.parse("2018-12-31"),
#             filter_func=filter_func_predict,
#         )

#         h_weather = d.get_train_weather()
#         p_weather = d.get_predict_weather()

#         join_load, join_wdata = d.reconcile(
#             d.load_data.train_data, d.load_data.date_col, h_weather
#         )
#         join_load_pre, join_wdata_pre = d.reconcile(
#             d.load_data.query_predict_data(t0=t0, t1=t1),
#             d.load_data.date_col,
#             p_weather,
#         )

#         assert 1 == 1

#     def test_save_weather_data_prep(self):
#         ld = LoadData(
#             table_name="", site_name="", query="", date_col=None, t0="", t1=""
#         )
#         fn = "../temp.pkl"
#         with open(fn, "wb") as dill_file:
#             dill.dump(ld, dill_file)

#     def test_logger(self):
#         logger = get_logger(__name__)
#         logger.error("err recorded")
