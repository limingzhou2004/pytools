
import numpy as np

from pytools.modeling.scaler import Scaler,load


def test_sclaer(config):
    x = np.array([2,3,4])
    arr =[np.array([1,2,3]), np.random.rand(3,4,5), np.random.rand(4,6,7,2)]
    s = Scaler(x, arr, scaler_type='minmax')
    assert x.max() == 1
    assert x.min() ==0
    assert arr[0].max() ==1
    assert arr[2].max() ==1
    assert len(s._arr_scalers[-1])==2


    fn = config.get_model_file_name(class_name='scaler')
    s.save(fn)
    s = load(fn)


    assert s.unscale_target(np.array([4,6,8]) == np.array([1,2,3]))