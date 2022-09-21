# PySkynet

PySkynet is a library for using [skynet](https://github.com/cloudwu/skynet) in python. Including a lua library *numsky* for dealing with numpy.ndarray object.

This repository has been archived for some reason. There's a personal fork [pyskynet](https://github.com/drinkwithwater/pyskynet).

### Install

Default install, find ssl by default. If ssl is not found, ltls.so will not be installed

~~~~sh
$ pip install pyskynet
~~~~

### Quick Start

Call lua from python

```python
import pyskynet
import pyskynet.foreign as foreign

pyskynet.start()

lua_service = pyskynet.scriptservice("""
		local pyskynet = require "pyskynet"
		local foreign = require "pyskynet.foreign"
		pyskynet.start(function()
			foreign.dispatch("echo", function(a)
				print("[lua]arg from python:", a)
				return "lua pong"
			end)
		end)
""")

lua_re = foreign.call(lua_service, "echo", "python ping")
print("[python]call lua return:", lua_re)

pyskynet.join()
```

Call python from lua

```python
import pyskynet
import pyskynet.foreign as foreign

pyskynet.start()

@foreign.dispatch("echo")
def echo(data):
	print("[python]arg from lua:", data)
	return "python pong"

lua_service = pyskynet.scriptservice("""
		local pyskynet = require "pyskynet"
		local foreign = require "pyskynet.foreign"
		pyskynet.start(function()
			local a = foreign.call(".python", "echo", "rewrew")
			print("[lua]return from python:", a)
		end)
""")

pyskynet.join()
```
