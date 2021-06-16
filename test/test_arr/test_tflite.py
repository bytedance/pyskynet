
import sys
sys.path.append("../../")

import pyskynet
import pyskynet.foreign as foreign

import numpy as np

pyskynet.start()

arr, = pyskynet.test("""
        local ns = require "numsky"
        local tflite = require "tflite"
        local buf = (io.open("detect.tflite")):read("*a")
        local interpreter = tflite.interpreter(buf)
        interpreter:allocate_tensors()
        for i=1,#interpreter.input_tensors do
            local tensor = interpreter.input_tensors[i]
            print(i, tensor, tensor.ndim, tensor.shape, tensor.name, tensor.dtype)
            local arr = tensor:get()
            --arr[ns.s(1)(1)(1)(1)] = 0
            tensor:set(arr)
        end
        interpreter:invoke()
        for i=1,#interpreter.output_tensors do
            local tensor = interpreter.output_tensors[i]
            print(i, tensor, tensor.ndim, tensor.shape, tensor.name, tensor.dtype)
            local arr = tensor:get()
            print(arr)
        end
        print(interpreter)
        return 0
""")


pyskynet.join()
