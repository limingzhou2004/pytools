# import sys
# import numpy as np
# import pandas as pd
import os
import subprocess
from typing import Union, List
from subprocess import PIPE
from functools import partial


import dask.bag as bag

from pytools.data_prep.grib_utils import extract_a_file, extract_data_from_grib2, get_paras_from_pynio_file
import numpy as np

"""
--extract-npy --fIn ../testdata/hrrrsub_2018_10_06_00F0
--fOut ../testdata/output/hrrr2.npy --parasFile hrrr_paras_obsolete.txt --center "(43,-73)" --rect "(100,100)"
"""
 

class PyJar:
    """
    Wrapper for handling jar files, for getting grib2 data and processing grib2 data
    """

    def __init__(
        self,
        folder_in: List[str],
        folder_out: str,
        paras_file: Union[str, List[str]],
        center: str,  # = '"(43.0,-73.0)"',  # no space after the comma, or double quote
        rect: str,  # = '"(100.0,100.0)"',  # no space after the comma, or double quote
        #jar_address: str,  # ="/Users/limingzhou/zhoul/work/me/Scala-http/classes/artifacts/scalahttp_jar/*",
    ):
        """
        Constructor

        Args:
            folder_in: grib file folder as input
            folder_out: npy file folder as output
            paras_file: parameter definition
            center: center str "(x, y)"
            rect: rect str "(height, width)"
            jar_address:
        """
        #self.jar_address = jar_address
        if isinstance(folder_in, str):
            self.folder_in = [folder_in]
        else:
            self.folder_in: str = folder_in
        self.folder_out: str = folder_out
        self.paras_file = paras_file
        self.paras = get_paras_from_pynio_file(paras_file)
        self.center = center
        self.rect = rect
        os.makedirs(self.folder_out, exist_ok=True)

    def create_output_filename(self, fin, out_folder=None, prefix="", suffix=".npy"):
        if not out_folder:
            out_folder = self.folder_out
        return os.path.join(out_folder, prefix + os.path.basename(fin) + suffix)

    def process_a_file_list(self, file_list, prefix, suffix="", parallel=False):
        if parallel:
            out_func = partial(
                self.create_output_filename,
                out_folder=self.folder_out,
                prefix=prefix,
                suffix=suffix,
            )
            file_bag = bag.from_sequence(file_list, partition_size=1)
            file_bag.map(lambda x: self.process_a_grib(x, f_out=out_func(x))).compute()
        else:
            for f in file_list:
                self.process_a_grib(
                    f,
                    self.create_output_filename(
                        fin=f, out_folder=self.folder_out, prefix=prefix, suffix=suffix
                    ),
                )

    def process_folders(
        self,
        out_prefix: str,
        out_suffix: str = ".npy",
        parallel: bool = True,
        exclude: List = [],
        include_files: List = None,
    ) -> None:
        """
        Process grib2 folders.

        Args:
            out_prefix: npy files prefix
            out_suffix: npy files suffix
            parallel: process grib2 files in parallel or not
            exclude: exclude a list of grib2 files
            include_files: Only process the grib2 files in the include_files

        Returns: None

        """
        if not isinstance(self.folder_in, List):
            self.folder_in = [self.folder_in]

        for a_folder in self.folder_in:
            if include_files is None:
                files = [
                    os.path.join(a_folder, f)
                    for f in os.listdir(a_folder)
                    if not f.startswith(".")
                    if f not in exclude
                ]
            else:
                files = [
                    os.path.join(a_folder, f)
                    for f in os.listdir(a_folder)
                    if f in include_files and f not in exclude
                ]
            # print(files)
            self.process_a_file_list(
                files, prefix=out_prefix, suffix=out_suffix, parallel=parallel
            )

    def process_a_grib_jar(self, f_in, f_out):
        args = [
            "java",
            "-cp",
            self.jar_address,
            "com.xtreme.App",
            "--extract-npy",
            "--fIn",
            f_in,
            "--fOut",
            f_out,
            "--parasFile",
            self.paras_file,
            "--center",
            self.center,
            "--rect",
            self.rect,
        ]
        proc = subprocess.Popen(
            " ".join(args),
            # ["java", "-version"],
            stdout=PIPE,
            stderr=PIPE,
            shell=True,
        )
        if proc.wait() != 0:
            messages, error = proc.communicate()
            print(messages.decode("utf8"))
            print(error.decode("utf8"))


    def process_a_grib(self, f_in, f_out):
        # use py grib package to extract 
        data = extract_data_from_grib2(
                fn=f_in,
                paras=self.paras,
                lon=self.center[0],
                lat=self.center[1],
                radius=self.rect
                )
        # save the npy file
        np.save(file=f_out, arr=data)
        return data

    def set_folder_in(self, folder_in: Union[str, List[str]]):
        """
        Reset the folder for input grib files

        Args:
            folder_in:

        Returns: None

        """
        self.folder_in = folder_in

    def set_folder_out(self, folder_out):
        self.folder_out = folder_out
