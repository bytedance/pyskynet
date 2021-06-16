
# pyskynet

## some design
* pyskynet
	* pyskynet is a python library based on [skynet](https://github.com/cloudwu/skynet).
	* pyskynet modify skynet's entry code, for using lua and skynet in python.
	* we include a lua library numsky for passing numpy.ndarray object between lua and python

* pyskynet.foreign
	* module pyskynet.foreign is using for communicating between service.
	* skynet's lua actors communicate by serializing and deserializing, we extend this for communicating with python.

* gevent
	* TODO
* object pass
	* TODO

## pyskynet api

* pyskynet.start() #
	* python : pyskynet.start(thread:int, path:List[str], cpath:List[str]) # launch pyskynet，thread is thread num，path cpath is just like package.path, package.cpath in lua
	* lua : pyskynet.start(func) -- lua service launch，func is callback after launch

* pyskynet.scriptservice(script, ...) # (string, string...) -> int # start a lua service and executing script, return service's address
* pyskynet.newservice(name, ...) # (string, string...) -> int # start lua service by file, return service's address
* pyskynet.uniqueservice(name, ...) # (string, string...) -> int # start or search a service by file, return service's address，only one in one process.
* pyskynet.self() # return address of current service

### pyskynet.foreign

* foreign.dispatch(cmd, func?) # (function|string|number|{string|number:function}, function?) # register callback for foreign message

```lua
-- register a function map to cmd "test"
foreign.dispatch("test", function(arg1, arg2, arg3)
	-- do something
	return 321
end)
-- register multi functions map to cmds
foreign.dispatch({
	func1=function()
	end,
	func2=function()
	end
})
-- register a callback for any cmd
foreign.dispatch(function(cmd, ...)
end)
```

```python
# besides lua's usage, we can write like decorator in python
@foreign.dispatch("test")
def func(arg1, arg2, arg3):
	# do something
	return 321

```

* foreign.call(address, cmd1, arg1, arg2, ...) # (int, string|number, function) -> tuple # call function map to cmd, block coroutine until return

* foreign.send(address, cmd, arg1, arg2, ...) # (int, string|number, function) # call function map to cmd, non-block


## other functions

* python and lua use different apis for other functions

### coroutine

* python use [gevent](http://www.gevent.org/)
* lua use skynet's coroutine，[skynet API](https://github.com/cloudwu/skynet/wiki/APIList)，some functions:
	* require module: local skynet = require "skynet"
	* skynet.sleep(time) -- wait time * 0.01s.
	* skynet.fork(func, ...) -- execute func in a coroutine.
	* skynet.time() -- return current time.
	* skynet.exit() -- exit current service

### network

* lua use skynet's network library [socket](https://github.com/cloudwu/skynet/wiki/Socket) and [http](https://github.com/cloudwu/skynet/wiki/Http)

### third-party library

* lua
	* [rapidjson](https://github.com/xpol/lua-rapidjson)
	* [protobuf](https://github.com/starwing/lua-protobuf)
