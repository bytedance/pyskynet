local skynet = require("skynet")
local foreign_seri = require("pyskynet.foreign_seri")
local remotepack = foreign_seri.remotepack
local remoteunpack = foreign_seri.remoteunpack

local clusterd
local cluster = {}
local sender = {}
local task_queue = {}

local function request_sender(q, node)
	local ok, c = pcall(skynet.call, clusterd, "lua", "sender", node)
	if not ok then
		skynet.error(c)
		c = nil
	end
	-- run tasks in queue
	local confirm = coroutine.running()
	q.confirm = confirm
	q.sender = c
	for _, task in ipairs(q) do
		if type(task) == "table" then
			if c then
				skynet.send(c, "lua", "push", task[1], remotepack(table.unpack(task, 2, task.n)))
			end
		else
			skynet.wakeup(task)
			skynet.wait(confirm)
		end
	end
	task_queue[node] = nil
	sender[node] = c
end

local function get_queue(t, node)
	local q = {}
	t[node] = q
	skynet.fork(request_sender, q, node)
	return q
end

setmetatable(task_queue, { __index = get_queue } )

local function get_sender(node)
	local s = sender[node]
	if not s then
		local q = task_queue[node]
		local task = coroutine.running()
		table.insert(q, task)
		skynet.wait(task)
		skynet.wakeup(q.confirm)
		return q.sender
	end
	return s
end

function cluster.call(node, address, ...)
	-- foreign.remotepack(...) will free by cluster.core.packrequest
	return remoteunpack(skynet.rawcall(get_sender(node), "lua", skynet.pack("req",  address, remotepack(...))))
end

function cluster.send(node, address, ...)
	-- push is the same with req, but no response
	local s = sender[node]
	if not s then
		table.insert(task_queue[node], table.pack(address, ...))
	else
		skynet.send(sender[node], "lua", "push", address, remotepack(...))
	end
end

function cluster.open(port)
	assert(type(port) == "number", "cluster.open's args must be number")
	skynet.call(clusterd, "lua", "listen", "0.0.0.0", port)
end

function cluster.reload(config)
	skynet.call(clusterd, "lua", "reload", config)
end

function cluster.proxy(node, name)
	return skynet.call(clusterd, "lua", "proxy", node, name)
end

function cluster.register(name, addr)
	assert(type(name) == "string")
	assert(addr == nil or type(addr) == "number")
	return skynet.call(clusterd, "lua", "register", name, addr)
end

function cluster.query(node, name)
	return skynet.call(get_sender(node), "lua", "req", 0, skynet.pack(name))
end

skynet.init(function()
	clusterd = skynet.uniqueservice("foreign_clusterd")
end)

return cluster
