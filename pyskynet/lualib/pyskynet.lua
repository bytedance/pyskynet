
local skynet = require("skynet")
local foreign = require("pyskynet.foreign")
local pyskynet_modify = require("pyskynet.modify")

local pyskynet = {}
pyskynet.foreign = foreign

pyskynet.start = skynet.start

----------------
-- proto api ---
----------------

pyskynet.PTYPE_TEXT = skynet.PTYPE_TEXT
pyskynet.PTYPE_CLIENT = skynet.PTYPE_CLIENT
pyskynet.PTYPE_SOCKET = skynet.PTYPE_SOCKET
pyskynet.PTYPE_LUA = skynet.PTYPE_LUA
pyskynet.PTYPE_FOREIGN_REMOTE = foreign.PTYPE_FOREIGN_REMOTE
pyskynet.PTYPE_FOREIGN = foreign.PTYPE_FOREIGN

pyskynet.rawcall = skynet.rawcall
pyskynet.rawsend = skynet.rawsend
pyskynet.ret = skynet.ret

------------------
-- service api ---
------------------

pyskynet.self = skynet.self

function pyskynet.getenv(k)
	local data = pyskynet_modify.getlenv(k)
	return (foreign.remoteunpack(data))
end

function pyskynet.setenv(k, v)
	if k ~= nil then
		assert(pyskynet.getenv(k) == nil, "Can't setenv exist key : " .. k)
	end
	local msg_ptr, msg_size = foreign.remotepack(v)
	local newkey = pyskynet_modify.setlenv(k, msg_ptr, msg_size)
	foreign.trash(msg_ptr)
	return newkey
end

function pyskynet.envs()
	local re = {}
	local function nextenv(t, k)
		return pyskynet_modify.nextenv(k)
	end
	for key in nextenv, nil, nil do
		re[key] = pyskynet.getenv(key)
	end
	return re
end

function pyskynet.newservice(...)
	for i=1, select("#", ...) do
		local arg = select(i, ...)
		assert(type(arg)=="string", "newservice's arg must be string")
	end
	return skynet.newservice(...)
end

function pyskynet.uniqueservice(...)
	for i=1, select("#", ...) do
		local arg = select(i, ...)
		assert(type(arg)=="string", "uniqueservice's arg must be string")
	end
	return skynet.uniqueservice(...)
end

function pyskynet.scriptservice(scriptaddr_or_loadargs, ...)
    local t1 = type(scriptaddr_or_loadargs)
    local scriptaddr
    if t1 == "string" and scriptaddr_or_loadargs:find("0x") == 1 then
        scriptaddr = scriptaddr_or_loadargs
    elseif t1 == "string" then
        scriptaddr = pyskynet.setenv(nil, {scriptaddr_or_loadargs})
    elseif t1 == "table" then
        scriptaddr = pyskynet.setenv(nil, scriptaddr_or_loadargs)
    end
	return pyskynet.newservice("script_service", scriptaddr, ...)
end

return pyskynet
