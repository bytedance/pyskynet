
import sys
sys.path.append("../../")

import pyskynet
import pyskynet.foreign as foreign

import numpy as np

pyskynet.start()

def test1():
    pyskynet.test("""
            local ns = require "numsky"
            local arr
            arr = ns.array({1,2,3,4})
            print(arr, arr.dtype, arr.shape)
            arr = ns.array({{1,2},{3,4}})
            print(arr, arr.dtype, arr.shape)
            arr = ns.array({{1,2},{3,4},{3,4}})
            print(arr, arr.dtype, arr.shape)
            arr = ns.array({{3, 4},{5,6},{1,3.0}}, ns.int32)
            print(arr, arr.dtype, arr.shape)
            arr = ns.array({{3.0, 4},{5,6},{1,3.0}})
            print(arr, arr.dtype, arr.shape)
            arr = ns.array({{3, 4.0},{5,6},{1,3}})
            print(arr, arr.dtype, arr.shape)
            arr = ns.array({{3, 4.5},{5,6},{1,3}})
            print(arr, arr.dtype, arr.shape)
            arr = ns.array({{4, 4.5},{5,6},{1,3}})
            print(arr, arr.dtype, arr.shape)
            arr = ns.array({{true, false}, {true, false}})
            print(arr, arr.dtype, arr.shape)
            arr = ns.array({{{}, {}}, {{}, {}}, {{}, {}}}, ns.int32)
            print(arr, arr.dtype, arr.shape)
            return 0
    """)

def test2():
    pyskynet.test("""
            local ns = require "numsky"
            local arr = ns.linspace(1,3,3, ns.int32)
            print(arr)
            return 0
    """)

test2()

pyskynet.join()
