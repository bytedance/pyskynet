
local foreign_seri = require("pyskynet.foreign_seri")
local numsky = require("numsky")

local skynet = require("skynet")

collectgarbage("setpause", 100)
collectgarbage("setstepmul", 200)

local foreign = {}

local PTYPE_FOREIGN_REMOTE = 254
local PTYPE_FOREIGN = 255

foreign.PTYPE_FOREIGN_REMOTE = PTYPE_FOREIGN_REMOTE
foreign.PTYPE_FOREIGN = PTYPE_FOREIGN

foreign.remotepack = assert(foreign_seri.remotepack)
foreign.remoteunpack = assert(foreign_seri.remoteunpack)
foreign.pack = assert(foreign_seri.pack)
foreign.unpack = assert(foreign_seri.unpack)

foreign.CMD = setmetatable({}, {
	__call=function(t, first, ...)
		return t[first](...)
	end
})

local function __foreign_dispatch(session, source, ...)
	if session ~= 0 then
		skynet.ret(foreign_seri.pack(foreign.CMD(...)))
	else
		foreign.CMD(...)
	end
end

local function __foreign_remote_dispatch(session, source, ...)
	if session ~= 0 then
		skynet.ret(foreign_seri.remotepack(foreign.CMD(...)))
     else
		foreign.CMD(...)
	end
end


do
	local REG = skynet.register_protocol

	REG {
		name = "foreign",
		id = PTYPE_FOREIGN,
		pack = foreign_seri.pack,
		unpack = foreign_seri.unpack,
		dispatch = __foreign_dispatch,
	}

	REG {
		name = "foreign_remote",
		id = PTYPE_FOREIGN_REMOTE,
		pack = foreign_seri.remotepack,
		unpack = foreign_seri.remoteunpack,
		dispatch = __foreign_remote_dispatch,
	}

end

function foreign.dispatch(cmd, func)
	if func then
		foreign.CMD[cmd] = func
	elseif type(cmd) == "table" then
		for k, v in pairs(cmd) do
			foreign.CMD[k] = v
		end
	elseif type(cmd) == "function" then
		foreign.CMD = cmd
	else
		error("dispatch failed for args")
	end
end

function foreign.call(addr, ...)
	return skynet.call(addr, PTYPE_FOREIGN, ...)
end

function foreign.send(addr, ...)
	skynet.send(addr, PTYPE_FOREIGN, cmd, ...)
end

return foreign
