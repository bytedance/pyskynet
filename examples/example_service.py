
import sys
sys.path.append("../")

import pyskynet
import pyskynet.foreign as foreign

import numpy as np

pyskynet.start()

service = pyskynet.newservice("example")

a,b = foreign.call(service, "func", "rewrwrwrwrewrew", 321)
print(a,b)


pyskynet.join()
