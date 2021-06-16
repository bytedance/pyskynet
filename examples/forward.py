

import sys
sys.path.append("../")

import pyskynet
import pyskynet.foreign as foreign

import numpy as np

pyskynet.start()

s1 = pyskynet.newservice("forward")


pyskynet.join()
