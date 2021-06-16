

import test_base
from test_base import TestObject

test_base.start()

import numpy as np

math_dtype_list = [
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.float32,
        np.float64,
        ]

bitwise_dtype_list = [
        np.bool,
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        ]

# test_math
math_obj_list = []
for dtype in math_dtype_list:
    if dtype != np.bool:
        arr = np.arange(-2, 3, dtype=dtype).reshape(5,1)
        arr[arr==0] = 127
    else:
        arr = (np.arange(-2, 3, dtype=np.int8).reshape(5,1) % 2).astype(np.bool)
    math_obj_list.append(TestObject(arr))

math_arr_list = [3213412321,-53, 321.0, -234.0]
for dtype in math_dtype_list:
    if dtype != np.bool:
        arr = np.arange(-2, 3, dtype=dtype).reshape(1,5)
        arr[arr==0] = 127
    else:
        arr = (np.arange(-2, 3, dtype=np.int8).reshape(1,5) % 2).astype(np.bool)
    math_arr_list.append(arr)

for o1 in math_obj_list:
    for o2 in math_arr_list:
        o1 + o2
        o1 - o2
        o1 * o2
        o1 / o2
        o1 // o2
