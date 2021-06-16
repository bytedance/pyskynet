
import sys
sys.path.append("../../")

import pyskynet
import pyskynet.foreign as foreign

import numpy as np

pyskynet.start()

arr, = pyskynet.test("""
        local ns = require "numsky"
        local slice = ns.s(1,2,3)
        print(slice)
        for i=1, #slice do
            print(slice:get(i))
            slice:set(i, 2,2,2)
            print(slice:get(i))
        end
        return 0
""")


pyskynet.join()
