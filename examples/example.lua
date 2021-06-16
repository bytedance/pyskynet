
local pyskynet = require "pyskynet"
local foreign = require "pyskynet.foreign"
foreign.dispatch("func", function(a, b)
	return a,b
end)
pyskynet.start(function()
	-- call self
	local a,b = foreign.call(pyskynet.self(), "func", 321, 432)
	print(a,b)
end)
