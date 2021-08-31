
local unqlite = require "unqlite"

local function test_store()
	local db1 = unqlite.open("test.db")

	db1:store("key1", "value1")

	db1:append("key1", "rewrw")

	db1:append("key2", "value2")

	db1:store("key3", "value3")

	db1:commit()

	db1:close()
end

local function test_fetch()
	local db2 = unqlite.open("test.db")

	local value = db2:fetch("key1")

	print(value)

	local value = db2:fetch("key2")

	print(value)

	local value = db2:fetch("key3")

	print(value)

	db2:delete("key1")

	db2:commit()

	db2:close()
end

local function test_cursor()
	local db3 = unqlite.open("test.db")

	--for k,v in db3:cursor() do
		--print(k,v)
	--end
	local sth = db3:cursor()
	for k,v in sth do
		print("cursor===", k,v)
	end
	--[[db3:cursor()
	cursor:first()
	while true do
		if not cursor:valid() then
			break
		else
			print("cursor====", cursor:key(), cursor:value())
			cursor:next()
		end
	end]]
end

test_store()
test_fetch()
test_cursor()
