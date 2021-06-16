
import sys
sys.path.append("../../")

import pyskynet
import pyskynet.foreign as foreign

import numpy as np

pyskynet.start()

arr, = pyskynet.test("""
        local ns = require "numsky"
        local slice = ns.slice(1,2)
        print(slice)
        return 0
""")


pyskynet.join()
