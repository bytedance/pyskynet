local foreign_seri = require "pyskynet.foreign_seri"
local ns = require "numsky"
local skynet = require "skynet"
local pyskynet = require "pyskynet"
local foreign = require "pyskynet.foreign"

local script_addr, file_name = ...
local script = pyskynet.getenv(script_addr)

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

pyskynet.start(function()
	foreign.dispatch(CMD)
end)
