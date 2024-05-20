
import pytest

from pytools.arg_class import ArgClass 


def test_parsetasks():

    arg_str = '-cfg config_file -cr task_1 -t0 1/1/2020 -t1 1/3/2020'
    ap = ArgClass(args=arg_str.split(' '))
    p_dict = ap.construct_args_dict()
    assert p_dict.get('suffix','na')=='v0'


    arg_str = '-cfg config_file task_1 -t0 1/1/2020 -t1 1/3/2020'
    ap = ArgClass(args=arg_str.split(' '))
    p_dict = ap.construct_args_dict()
    assert p_dict.get('cr')==False