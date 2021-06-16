local foreign_seri = require "pyskynet.foreign_seri"
local pyskynet_modify = require "pyskynet.modify"
local skynet = require "skynet"

local kind = ...
local script, script_name
if kind == "script" then
	local info = select(2, ...)
	local msg_ptr, msg_size = pyskynet_modify.ptr_unwrap(info)
	script = foreign_seri.remoteunpack(msg_ptr, msg_size)
	script_name = string.format("script_service_%x", skynet.self())
	skynet.trash(msg_ptr, msg_size)
elseif kind == "file" then
	local filename = select(2, ...)
	script = io.open(filename,"r"):read("*a")
	script_name = filename
end

local func, err = load(script, script_name)
if func then
	func(select(3, ...))
else
	skynet.error("script service parsing failed:"..err)
end
