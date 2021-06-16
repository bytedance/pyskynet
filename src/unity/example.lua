

local drlua = require 'drlua'
local ns = require "numsky"
local tflite = require "tflite"
local rapidjson = require "rapidjson"

local xorPreter = nil

-- (string, string) -> int
function drlua.newObj(name, arg)
	assert(name == "interpreter")
	xorPreter = tflite.interpreter(arg)
	xorPreter:allocate_tensors()
	return 0
end

-- (int, string, string) -> string
function drlua.objCall(id, name, arg)
	assert(name == "invoke")
	local input = rapidjson.decode(arg)
	xorPreter.input_tensors[1]:set(ns.array({input}, ns.float32))
	xorPreter:invoke()
	print(xorPreter.input_tensors[1]:get(), xorPreter.output_tensors[1]:get())
	return tostring(xorPreter.output_tensors[1]:get()[{1,1}])
end

-- (int)
function drlua.delObj(i)
end
