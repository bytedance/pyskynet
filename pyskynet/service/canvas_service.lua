local foreign_seri = require "pyskynet.foreign_seri"
local ns = require "numsky"
local skynet = require "skynet"
local foreign = require "pyskynet.foreign"
local pyskynet_modify = require "pyskynet.modify"

local info, file_name = ...
local msg_ptr, msg_size = pyskynet_modify.ptr_unwrap(info)
local script = foreign_seri.remoteunpack(msg_ptr, msg_size)
skynet.trash(msg_ptr, msg_size)

local canv = ns.canvas(script, file_name)

local CMD = {}

function CMD.reset(...)
	return canv:reset(...)
end

function CMD.render(...)
	return canv:render(...)
end

function CMD.exit()
	skynet.exit()
end

skynet.start(function()
	foreign.dispatch(CMD)
end)
