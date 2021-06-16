
import sys
sys.path.append("../../")

import pyskynet
import pyskynet.foreign as foreign

import numpy as np

pyskynet.start()

arr = pyskynet.test("""
        local ns = require "numsky"
        local arr = ns.zeros({3,3}, ns.float32)
        print(arr)
        for i=1,3 do
            for j=1,3 do
                arr[{i,j}] = i + j
            end
        end
        print(arr, arr.dtype, arr.dtype.name)
""")

print(arr)

pyskynet.join()
