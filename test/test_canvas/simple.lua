
local ns = require "numsky"

local function diff(left, right)
	if type(left) == "table" then
		for k in pairs(left) do
			if diff(left[k], right[k]) then
				print(k, left[k], right[k])
				return true
			end
		end
	else
		local delta = (left - right)
		if type(delta) == "number" then
			return delta ~= 0
		else
			return ns.add:reduce(delta) ~= 0
		end
	end
	return false
end

local function test_non_args(xml_script, ...)
	local canv = ns.canvas(xml_script)
	local k = (debug.getuservalue(canv))
	canv:reset()
	local right = {...}
	if #right > 0 then
		local left = {canv:render()}
		if diff(left, right) then
			print("check false:")
			print(xml_script)
		else
			print("check ok")
		end
	else
		print("check ignore", canv:render())
	end
end

test_non_args([[
<arr ndim="1">
	<int8 for="k,v in pairs({4,3,2})"> k*v -- 电量不足
	</int8>
</arr>
]], ns.array({4,6,6}, ns.float32))


test_non_args([[
	<int8 for="k,v in pairs({4,3,2})"> k </int8>
]], 1,2,3)

test_non_args([[
<arr dtype="int32">
<var function="d(a,b)">
	return a+b, a-b
</var>
	<int8> d(1,2) </int8>
</arr>
]], ns.array({3, -1}, ns.int32))

test_non_args([[
<arr dtype="int32" for="k,v in pairs({1,2,3})">
	<int8 > k,v </int8>
</arr>
]], ns.array({1,1},ns.int32),ns.array({2,2},ns.int32),ns.array({3,3}, ns.int32))

test_non_args([[
<arr dtype="int32" for="k=1,3,1">
	<float32> k,k </float32>
</arr>
]], ns.array({1,1},ns.int32),ns.array({2,2},ns.int32),ns.array({3,3}, ns.int32))

test_non_args([[
<arr dtype="int32" shape="2,2" for="k,v in pairs({1,2,3})" if="k%2==1">
	<arr>
		<int8> k,v </int8>
	</arr>
	<arr>
		<int8> k,v </int8>
	</arr>
</arr>
]], ns.array({{1,1},{1,1}},ns.int32),ns.array({{3,3},{3,3}},ns.int32))



test_non_args([[
<arr>
		<int8 for="k,v in pairs({3,2,4})" sort="v"> k,v </int8>
		<proc> print("proc print") </proc>
</arr>
]], ns.array({2,2,1,3,3,4}, ns.int32))

test_non_args [[
<arr>
	<var local="yes">321
	</var>
	<block for="i=1,3">
		<block for="j=1,3">
			<int8> i,yes </int8>
		</block>
	</block>
</arr>
]]

test_non_args [[
<arr ndim="2">
	<block for="i=1,3">
		<arr>
			<block for="j=1,3">
				<int8> i,j </int8>
			</block>
		</arr>
		<arr ndim="1">
			<block for="j=1,3" len="2">
				<int8 len="2"> i,j </int8>
			</block>
		</arr>
	</block>
</arr>
]]

