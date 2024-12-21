from argparse import ArgumentParser
import sys, os

from pytools.data_prep import weather_data_prep as wp


class ArgClass:
    """
    This class process the command line argument.
    The parent argument is option
    The get_taskX method returns the dict for each option
    task numbers are described in the operations-model.puml
    1 create dataManager with load data;
    2 extract grib2 files and create npy files for training;
    3 train the model;
    4 extract npy from grib2 data for weather prediction;
    5 predictions from a multi hour ahead model;
    6 roll out predictions from an hour-ahead model;
    7 ...
    """

    def __init__(self, args=None, fun_list=[]):
        self.tasks_dict = {
            1: self._add_task1,
            2: self._add_task2,
            3: self._add_task3,
            4: self._add_task4,
            5: self._add_task5,
            6: self._add_task6,
            7: self._add_task7,
        }
        parser = ArgumentParser(
            description="Generate data prep manager and prepare weather data",    
            add_help=False,        
        )
        self._args = args if args is not None else sys.argv
        parser.add_argument(
            "-cfg",
            "--config",
            dest="config_file",
            required=True,
            type=str,
            help="config file name",
        )
        parser.add_argument(
            "-sx",
            "--suffix",
            dest="suffix",
            required=False,
            type=str,
            help="suffix as an ID to add to the end of body of file names",
            default="v0",
        )
        parser.add_argument(
            "-cr",
            "--create",
            action="store_true",
            help="Create a new DataManager",
        )
        self.parser = parser
        self.sub_parsers = self.parser.add_subparsers()

        self._add_task1(fun_list[0])
        self._add_task2(fun_list[1])
        self._add_task3(fun_list[2])
        self._add_task4(fun_list[3])
        # self._add_task5()
        # self._add_task6()
        # self._add_task7()

    def construct_args_dict(self):
        """

        Returns: Extract the dict with all parameters to pass to the tasks.

        """
        args = self.parser.parse_args(self._args)
        a_dict = args.__dict__

        def clean_args(dct):
            return {k: dct[k] for k in dct if not k.startswith("__")}

        a_dict = clean_args(a_dict)
        fun = a_dict.pop('func')

        return fun, a_dict

    def _add_task1(self, fun):
        sub_parser = self.sub_parsers.add_parser("task_1")
        sub_parser.add_argument(
            "-t0",
            "--datetime0",
            dest="t0",
            required=False,
            type=str,
            help="start datetime",
        )
        sub_parser.add_argument(
            "-t1",
            "--datetime1",
            dest="t1",
            required=False,
            type=str,
            help="end datetime",
        )
        sub_parser.set_defaults(func=fun)

    def _add_task2(self, fun):
        sub_parser = self.sub_parsers.add_parser("task_2")
        sub_parser.add_argument(
            "-n",
            "--n-cores",
            dest="n_cores",
            required=False,
            default=1,
            type=int,
            help="number of cores to use, default 1",
        )
        sub_parser.add_argument(
            "-fh",
            "--fst-hour",
            dest="fst_hour",
            required=False,
            default=48,
            type=int,
            help="max weather forecast hours, default 48",
        )
        sub_parser.add_argument(
            "-flag",
            "--flag",
            dest="flag",
            required=False,
            default='hf',
            type=str,
            help='f-forecast, h-historical, use together as fh or separate as f or h',
        )
        sub_parser.add_argument(
            "-year",
            "--year",
            dest="year",
            required=False,
            default=-1,
            type=int,
            help='year, -1 for all years, 2020-',
        )
        
        sub_parser.add_argument(
            "-month",
            "--month",
            dest="month",
            required=False,
            default=-1,
            type=int,
            help='month, -1 for all months, 1-12',
        )
        sub_parser.set_defaults(func=fun)

    def _add_task3(self, fun):
        sub_parser = self.sub_parsers.add_parser("task_3")
        sub_parser.add_argument(
            "-flag",
            "--flag",
            dest="flag",
            required=False,
            default='cv_0',
            type=str,
            help="train type, e.g. cv_0|final_train_0",
        )
        sub_parser.add_argument(
            "-ind",
            "--sce-ind",
            dest="ind",
            required=False,
            default=0,
            type=int,
            help="cv or final_train ind, default 0",
        )
        sub_parser.add_argument(
            "-sb",
            "--sub-task",
            dest="sub",
            required=False,
            default='fit',
            type=str,
            help="subtask, find_batch_size|find_lr|fit, default fit",
        )
        sub_parser.add_argument(
            "-mn",
            "--model-name",
            dest="model_name",
            required=False,
            default='test',
            type=str,
            help="model name to save to checkpoint",
        )
        sub_parser.set_defaults(func=fun)

    def _add_task4(self, fun):
        sub_parser = self.sub_parsers.add_parser("task_4")
        sub_parser.add_argument(
            "-flag",
            "--flag",
            dest="flag",
            required=False,
            default='cv_0',
            type=str,
            help="test|predict",
        )
        sub_parser.add_argument(
            "-ind",
            "--sce-ind",
            dest="ind",
            required=False,
            default=0,
            type=int,
            help="ind for test with fst weather, default 0, -1 for real time weather forecast",
        )
        sub_parser.add_argument(
            "-mn",
            "--model-name",
            dest="model_name",
            required=False,
            default='test',
            type=str,
            help="model name to load from checkpoint",
        )
        sub_parser.set_defaults(func=fun)
        
    def _add_task5(self, fun):
        sub_parser = self.sub_parsers.add_parser("task_5")
        sub_parser.add_argument(
            "-mha",
            "--max-hours-ahead",
            dest="max_hours_ahead",
            required=False,
            default=20,
            type=int,
            help="hours ahead to forecast",
        )
        sub_parser.add_argument(
            "-tc",
            "--current-time",
            dest="current_time",
            required=False,
            default="",
            type=str,
            help="current time",
        )
        sub_parser.add_argument(
            "--rebuild-npy",
            default=True,
            type=bool,
            help="create npy files",
        )

        sub_parser.set_defaults(func=fun)

    def _add_task6(self, fun):
        sub_parser = self.sub_parsers.add_parser("task_6")
        sub_parser.add_argument(
            "-mha",
            "--max-hours-ahead",
            dest="max_hours_ahead",
            required=False,
            default=20,
            type=int,
            help="hours ahead to forecast",
        )
        sub_parser.add_argument(
            "-tc",
            "--current-time",
            dest="current_time",
            required=False,
            default="",
            type=str,
            help="current time",
        )
        sub_parser.add_argument(
            "-rt0",
            "--report-t0",
            dest="report_t0",
            required=False,
            default="",
            type=str,
            help="Starting report time",
        )
        sub_parser.add_argument(
            "-rt1",
            "--report-t1",
            dest="report_t1",
            required=False,
            default="",
            type=str,
            help="Ending report time",
        )

        sub_parser.set_defaults(func=fun)


    def _add_task7(self, fun):
        sub_parser = self.sub_parsers.add_parser("task_7")
        sub_parser.set_defaults(func=fun)

