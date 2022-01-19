
local foreign_seri = require("pyskynet.foreign_seri")
local numsky = require("numsky")

local skynet = require("skynet")
local csend = (require "skynet.core").send

collectgarbage("setpause", 100)
collectgarbage("setstepmul", 200)

local foreign = {}

local PTYPE_FOREIGN_REMOTE = 254
local PTYPE_FOREIGN = 255

foreign.PTYPE_FOREIGN_REMOTE = PTYPE_FOREIGN_REMOTE
foreign.PTYPE_FOREIGN = PTYPE_FOREIGN

foreign.remotepack = assert(foreign_seri.remotepack)
foreign.remoteunpack = assert(foreign_seri.remoteunpack)
foreign.refpack = assert(foreign_seri.refpack)
foreign.refunpack = assert(foreign_seri.refunpack)

foreign.CMD = setmetatable({}, {
	__call=function(t, first, ...)
		local f = t[first]
		if not f then
			error("cmd "..tostring(first).." not found")
		end
		return f(...)
	end
})

local function __foreign_dispatch(session, source, ...)
	if session ~= 0 then
        local msg_ptr, msg_size = foreign_seri.refpack(foreign.CMD(...))
		skynet.ret(msg_ptr, msg_size)
	else
		foreign.CMD(...)
	end
end

local function __foreign_remote_dispatch(session, source, ...)
	if session ~= 0 then
        local msg_ptr, msg_size = foreign_seri.remotepack(foreign.CMD(...))
		skynet.ret(msg_ptr, msg_size)
     else
		foreign.CMD(...)
	end
end


do
	local REG = skynet.register_protocol

	REG {
		name = "foreign",
		id = PTYPE_FOREIGN,
		pack = function()
            error("use foreign.someapi(xxx, ...) instead of skynet.someapi(xxx, 'foreign', ...) when packing foreign message")
        end,
		unpack = foreign_seri.refunpack,
		dispatch = __foreign_dispatch,
	}

	REG {
		name = "foreign_remote",
		id = PTYPE_FOREIGN_REMOTE,
		pack = function()
            error("foreign_remote pack is not recommand to used here")
        end,
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

local i = 1
while debug.getupvalue(skynet.rawcall, i) ~= "yield_call" do
    i=i+1
end
local _, yield_call = debug.getupvalue(skynet.rawcall, i)

function foreign.call(addr, ...)
    local msg_ptr, msg_size = foreign_seri.refpack(...)
	local session = csend(addr, PTYPE_FOREIGN , nil , msg_ptr, msg_size)
	if session == nil then
		error("call to invalid address " .. skynet.address(addr))
	end
	return foreign_seri.refunpack(yield_call(addr, session))
end

function foreign.send(addr, ...)
    local msg_ptr, msg_size = foreign_seri.refpack(...)
	return csend(addr, PTYPE_FOREIGN , 0, msg_ptr, msg_size)
end

return foreign
