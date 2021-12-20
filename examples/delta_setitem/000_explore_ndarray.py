import numpy as np

class C(np.ndarray):
    a = 'pippo'

    def __setitem__(self, indx, val):
        print(f'{indx=} {val=}')
        prrr

a = np.array([1,2,3])

c= C(shape=a.shape, dtype=a.dtype, buffer=a.data, offset=0, strides=a.strides, order='C')
c.a

c1= C(shape=a.shape, dtype=a.dtype, buffer=a.data, offset=0, strides=a.strides, order='C')

c1.a
c.a='pluto'
c1.a
