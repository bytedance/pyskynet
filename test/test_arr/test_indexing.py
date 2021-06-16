

import test_base
from test_base import TestObject

test_base.start()

import numpy as np
import random


arr1 = np.arange(1, 5*5*5*5 + 1).reshape(5,5,5,5)
o1 = TestObject(arr1)

a = np.array([[(i + j) %2==1 for i in range(5)] for j in range(5)])
o1[a,a]

o1[0]
o1[1,2]
o1[-1,2,-1]

o1[1:-3,-3:2]

o1[1,2]
o1[1,:]
o1[1,1:3]
o1[:,3,:]
o1[np.array([1,2])]
o1[np.array([1,2]), :, np.array([0,3])]
o1[:, np.array([-1,-2])]
o1[1, np.array([1,2])]
o1[np.array([True, True, False, False, True])]
o1[np.array([[random.random() > 5 for j in range(5)] for i in range(5)])]
o1[1,2,np.array([[random.random() > 5 for j in range(5)] for i in range(5)])]

arr2 = np.arange(1, 5*5 + 1).reshape(5,5)
o2 = TestObject(arr2)
o2[0] = np.array([1,1,1,1,1])
o2[-1] = np.array([2,2,2,2,2])
o2[:] = np.array([3,3,3,3,3])
o2[:,-1] = np.array([4,4,4,4,4])
o2[:,:] = np.array([[1,1,1,1,5]])
o2[-5,:] = np.array([[1,2,2,4,5]])
o2[1,np.array([1,2])] = np.array([0,0])
o2[np.array([1,2]),:] = np.array([[0,0,0,0,0],[1,2,3,4,5]])

arr3 = np.arange(1, 5*5*5*5*5 + 1).reshape(5,5,5,5,5)
o3 = TestObject(arr3)

a = np.array([1,2,3,4,1,1,1,2,3])
b = np.array([[i==1 or j==1 for i in range(5)] for j in range(5)])
o3[a,0,b]
