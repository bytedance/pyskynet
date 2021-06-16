
import sys
sys.path.append("../")

import pyskynet
import pyskynet.foreign as foreign

import numpy as np

pyskynet.start()

@foreign.dispatch("echo")
def echo(data):
    return "python pong"

lua_service = pyskynet.scriptservice("""
        local pyskynet = require "pyskynet"
        local foreign = require "pyskynet.foreign"
        foreign.dispatch("echo", function(a)
            return "lua pong"
        end)
        pyskynet.start(function()
        end)
""")

# call lua
lua_re = foreign.call(lua_service, "echo", "python ping")
print("call lua return:", lua_re)
# call python
py_re = foreign.call(".python", "echo", "python ping")
print("call python return:", py_re)


pyskynet.join()
