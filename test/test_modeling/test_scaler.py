
import numpy as np

from pytools.modeling.scaler import Scaler,load


def test_sclaer(config):
    x = np.array([2,3,4])
    a = np.array([1,2,3])
    b= np.random.rand(3,4,5)+1
    c= np.random.rand(4,60,70,2)+1
    c[...,1] = c[...,1] * 3
    a0=a.copy()
    b0=b.copy()
    c0=c.copy()
    arr =[a, b, c]
    s = Scaler(x, arr, scaler_type='minmax')
    x1 = s.scale_target(x)
    arr=s.scale_arr(arr)
    assert x1.max() == 1
    assert x1.min() ==0
    assert np.isclose(arr[0].max() ,1)
    assert np.isclose(arr[2].max() ,1)
    assert len(s._arr_scalers)==3
    c2=arr[2]
    assert np.isclose(c2[...,0].min(),0,atol=1e-5)
    c21=c2[...,0]
    c01=c0[...,0]
    assert np.isclose((c21*(c01.max()-c01.min()) + c01.min()-c01).std(),0,atol=1e-8)
    assert (c2 * c0.max()-c0.min()+c0.min()).std()>0


    fn = config.get_model_file_name(class_name='scaler', suffix='_test')
    s.save(fn)
    s = load(fn)


    assert np.allclose(s.unscale_target(np.array([4,6,8])) , np.array([10,14,18]))