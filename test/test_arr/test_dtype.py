
import sys
sys.path.append("../../")

import pyskynet
import pyskynet.foreign as foreign

import numpy as np

pyskynet.start()

arr, = pyskynet.test("""
        ns = require "numsky"
        local assert_expr = function(expr, g)
            if load("return "..expr, "temp", "t", setmetatable(g or {},{__index=_G}))() then
                print(expr, "ok")
            else
                print(expr, "test failed")
            end
        end
        assert_expr("ns.int16.char == 'h'")
        assert_expr("ns.int8(259) == 259-256")
        assert_expr("ns.bool(nil) == false")
        for k,v in pairs(getmetatable(ns.bool)) do
            if type(k) == "number" then
                assert_expr("string.char(k) == v.char", {v=v,k=k})
                assert_expr("ns[v.name] == v", {v=v})
            end
        end
        for k,v in pairs(getmetatable(ns.bool).fieldtable) do
            print(k,v)
        end
        return 0
""")


pyskynet.join()
