
local ns = require "numsky"


local function test_non_args(xml_script, ...)
	local canv = ns.canvas(xml_script)
	canv:reset(...)
	print("return ", canv:render())
end

test_non_args([[
<?lua
local a1,a2 = ...
?>
<arr shape="a1,a2">
	<arr>
		<int8>1,2,3</int8>
	</arr>
	<arr>
		<int32 for="i=1,3">i</int32>
	</arr>
</arr>
]],2,3)

test_non_args([[
<arr dtype="int32" ndim="2" shape="1,2">
	<arr shape="2">
		<var function="d(a,b)">
			return a+b, a-b
		</var>
		<int8> d(1,2) </int8>
	</arr>
</arr>
]])

test_non_args([[
<arr dtype="int32" for="k,v in pairs({1,2,3})" len="2" shape="2">
	<int8 > k,v </int8>
</arr>
]])

test_non_args [[
<block>
	<arr ndim="1" shape="0">
		<float32>1,2,3</float32>
	</arr>
</block>
<arr shape="12,false,false">
	<block len="4" for="i=1,3">
		<block shape="5,2" for="i=1,3">
			<arr shape="5,2">
				<block shape="2" for="i=1,3">
					<arr>
						<int8> 1,2 </int8>
					</arr>
				</block>
				<arr>
					<int8> 4,3 </int8>
				</arr>
				<arr>
					<int8> 4,3 </int8>
				</arr>
			</arr>
		</block>
		<arr shape="5,2">
			<arr>
				<block len="2">
					<int8> 1,2 </int8>
				</block>
			</arr>
			<arr>
				<int8> 1,2 </int8>
			</arr>
			<arr>
				<int8> 1,2 </int8>
			</arr>
			<arr>
				<int8> 4,3 </int8>
			</arr>
			<arr>
				<int8> 1,2 </int8>
			</arr>
		</arr>
	</block>
</arr>
]]

test_non_args([[
<arr dtype="int32">
	<arr for="k,v in pairs({1,2,3})" len="4" shape="4">
		<int8 for="i=1,2" count="2" len="2"> k,v </int8>
	</arr>
</arr>
]])

