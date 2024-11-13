import pickle
from typing import List

from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Scaler:

    def __init__(self, target, data_list:List, scaler_type:str='standard'):
        self.scaler_type=scaler_type
        def _get_a_scaler():
            if self.scaler_type == 'standard':
                return StandardScaler() 
            elif self.scaler_type == 'minmax':
                return MinMaxScaler() 
            else:
                raise ValueError('The scaler type is standard|minmax!')
            
        self._arr_scalers = []
        self._target_scaler = _get_a_scaler()
        self._target_scaler.fit(target.reshape(-1,1))     

        for arr in data_list:
            self._arr_scalers.append(_get_a_scaler())
            last_dim = self._get_last_dim(arr.shape)
            self._arr_scalers[-1].fit(arr.reshape(-1, last_dim))

    def _get_last_dim(self, shape):
        last_dim = shape[-1]
        if len(shape) == 1:
            last_dim = 1
        return last_dim

    def scale_target(self,target):
        shape=target.shape
        return self._target_scaler.transform(target.reshape(-1,1)).reshape(shape)
    
    def sclae_arr(self, arr_list):
        ret = []
        for s, a in zip(self._arr_scalers, arr_list):
            last_dim = self._get_last_dim(a.shape)
            ret.append(s.transform(a.reshape(-1, last_dim)).reshape(a.shape))
        return ret
    
    def save(self, fn):
        with open(fn, 'wb') as fw:
            pickle.dump(self, fw)        
       
    def unscale_target(self, target):
        return self._target_scaler.inverse_transform(target.reshape(-1,1)).reshape(target.shape)
    
    def unscale_arr(self,arr_list):
        ret= []
        for i, arr in enumerate(arr_list):
            shape = arr.shape
            last_dim = shape[-1]
            if len(shape) == 1:
                last_dim = 1
            ret.append(self._arr_scalers[i].inverse_transform(arr.reshape(-1,last_dim)).reshape(shape))
        return ret
    

def load(fn):
    with open(fn, 'rb') as fr:
        return pickle.load(fr)