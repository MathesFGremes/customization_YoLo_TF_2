from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import random

A = np.empty(60)
A[:] = np.NaN
A[0:0] = A[1:1]
print(A)

B = np.array([(2,3), (5,2), (3,10)])
print(B[-2])
print(B.sum())

for i in np.arange(3):
    print(i)