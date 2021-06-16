
import sys
sys.path.append("../")

import pyskynet

remote_addr = "127.0.0.1:8081"
pyskynet.start()

addr = pyskynet.scriptservice("""
local pyskynet = require "pyskynet"
local cluster = require "pyskynet.cluster"
local foreign = require "pyskynet.foreign"
local numsky = require "numsky"

foreign.dispatch("getbuf", function(a)
    return "jklrewklrewjklrw0"
end)

foreign.dispatch("getarr", function(a)
    local arr = numsky.zeros({3,4})
    for i=1,3 do
        for j=1,4 do
            arr[{i,j}] = i + j
        end
    end
    return arr
end)

pyskynet.start(function()
    cluster.open(8081)
end)
""")

pyskynet.scriptservice("""
local pyskynet = require "pyskynet"
local cluster = require "pyskynet.cluster"
local foreign = require "pyskynet.foreign"

local addr = tonumber(...)
pyskynet.start(function()
    local remote_addr = "127.0.0.1:8081"
    local a = cluster.call(remote_addr, addr, "getbuf")
    print(a)
    local b = cluster.call(remote_addr, addr, "getarr")
    print(b)
end)
""", str(addr))

import pyskynet.cluster

buf, = pyskynet.cluster.call(remote_addr, addr, "getbuf")
print(buf)

arr, = pyskynet.cluster.call(remote_addr, addr, "getarr")
print(arr)

#pyskynet.join()
