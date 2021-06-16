
import sys
sys.path.append("../")

import pyskynet
import pyskynet.foreign as foreign

import numpy as np

pyskynet.start()


service = pyskynet.scriptservice("""
        local pyskynet = require "pyskynet"
        local foreign = require "pyskynet.foreign"
        local ns = require "numsky"
        foreign.dispatch("dosth", function(arr)
            return ns.array({1,2,3,4})
        end)
        pyskynet.start(function()
        end)
""")


arr = np.zeros(10000)
arr1, = foreign.call(service, "dosth", arr)

print(arr1, arr1.base)


pyskynet.join()
