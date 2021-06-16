
# pyskynet

## 若干设计
* pyskynet
	* pyskynet基于开源框架[skynet](https://github.com/cloudwu/skynet).
	* skynet是一个以c为底层，上层使用lua编写业务逻辑的actor框架。
	* pyskynet替换了skynet的入口代码，从而使得我们能在python中调用lua与skynet的功能。
	* 该项目设计的初衷是考虑到python存在GIL问题，为了给python进程提供更高效的多线程能力，同时又不损失python作为解释型语言所具有的敏捷与灵活性，所以尝试引入lua。
	* 同时为了在lua层处理python的ndarray对象，实现了numsky这一lua库。

* pyskynet.foreign
	* pyskynet.foreign是用于跨actor交互的模块。
	* skynet本身使用序列化与反序列化的方式让lua actor之间进行交互，我们对其进行了扩展，使得这种方式也能用于与python通信。
	* 在skynet原有的消息类型之外增加了PTYPE_FOREIGN与PTYPE_FOREIGN_REMOTE的消息类型。
	* PTYPE_FOREIGN与PTYPE_FOREIGN_REMOTE是对skynet.PTYPE_LUA的拓展，增加了对numpy.ndarray的支持。
	* PTYPE_FOREIGN用于本进程内的service交互，对于numpy.ndarray类型会通过引用的方式避免拷贝
	* PTYPE_FOREIGN_REMOTE用于跨进程交互，TODO

* gevent
	* gevent是一个被广泛使用的、基于有栈协程greenlet的python并发库。
	* skynet的调度线程与python的主线程互相独立，经由一个名为.python的service互相通信。
	* python的主线程使用gevent内置的libuv模块作为事件循环。
	* 所有发送给.python的消息都会被放入一个专用的消息队列中，python主线程会使用libuv的事件循环响应该队列的消息，并对每条消息开一个协程进行处理。
	* gevent可以用于以非侵入的方式实现阻塞协程的代码。pyskynet基于此，支持阻塞式地调用skynet服务中的方法，阻塞期间可以在主线程的事件循环中同时处理其它协程的逻辑。

* 数据结构转换
	* python的数据结构传递到lua中时，python的string会被转成bytes
    * lua的数据结构传递到python中时，如果lua的table不含hash字段时，在python中会被转为list。如果lua的table有hash字段，在python中会被转为dict

## pyskynet的api

* pyskynet.start() #
	* python : pyskynet.start(thread:int, path:List[str], cpath:List[str]) # 启动pyskynet，thread为线程数，path和cpath会被添加到lua的查找路径中，类似package.path和package.cpath。
	* lua : pyskynet.start(func) -- lua服务启动，func为启动后的回调函数

* pyskynet.scriptservice(script, ...) # (string, string...) -> int # 启动一个lua服务执行script, 返回服务的地址
* pyskynet.newservice(name, ...) # (string, string...) -> int # 启动lua脚本name对应的服务, 返回服务的地址
* pyskynet.uniqueservice(name, ...) # (string, string...) -> int # 启动或查找一个lua脚本name对应的服务，返回它的地址，该服务唯一
* pyskynet.self() # 获取本服务的地址。

### pyskynet.foreign

* foreign.dispatch(cmd, func?) # (function|string|number|{string|number:function}, function?) # 注册foreign消息的回调函数

```lua
-- 注册一个cmd对应的单个函数
foreign.dispatch("test", function(arg1, arg2, arg3)
	-- do something
	return 321
end)
-- 同时注册多个cmd对应多个函数
foreign.dispatch({
	func1=function()
	end,
	func2=function()
	end
})
-- 直接注册一个函数处理包含cmd在内的所有参数
foreign.dispatch(function(cmd, ...)
end)
```

```python
除了lua的写法之外，python中还支持使用decorator的写法。
@foreign.dispatch("test")
def func(arg1, arg2, arg3):
	# do something
	return 321

```

* foreign.call(address, cmd1, arg1, arg2, ...) # (int, string|number, function) -> tuple # 调用cmd对应的回调函数，会阻塞以等待返回

* foreign.send(address, cmd, arg1, arg2, ...) # (int, string|number, function) # 调用cmd对应的回调函数，但不会阻塞，不会等待返回


## 其他功能

* 其余功能未统一python与lua的接口

### 协程

* python使用[gevent](http://www.gevent.org/)
* lua使用skynet封装协程，详见[skynet API](https://github.com/cloudwu/skynet/wiki/APIList)，以下是一些常用的：
	* 引入模块：local skynet = require "skynet"
	* skynet.sleep(time) 等待time * 0.01s 。
	* skynet.fork(func, ...) 使用协程执行函数 func 。
	* skynet.time() 返回当前时间
	* skynet.exit() 结束当前服务

### 网络

* lua使用skynet的网络库[socket](https://github.com/cloudwu/skynet/wiki/Socket)和[http](https://github.com/cloudwu/skynet/wiki/Http)

### 第三方库

* lua
	* [rapidjson](https://github.com/xpol/lua-rapidjson)
	* [protobuf](https://github.com/starwing/lua-protobuf)
