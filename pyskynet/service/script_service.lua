local foreign_seri = require "pyskynet.foreign_seri"
local pyskynet = require "pyskynet"
local skynet = require "skynet"

local script_addr = ...
local script, script_name = table.unpack(pyskynet.getenv(script_addr))
script_name = script_name or string.format("script_script_%s", script_addr)

local func, err = load(script, script_name)
if func then
	func(...)
else
	skynet.error("script service parsing failed:"..err)
end
