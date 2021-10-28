local ns = require "numsky"
local skynet = require "skynet"
local pyskynet = require "pyskynet"
local foreign = require "pyskynet.foreign"

local CMD = {}
local canv = nil

function CMD.init(script, file_name)
	canv = ns.canvas(script, file_name)
end

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
