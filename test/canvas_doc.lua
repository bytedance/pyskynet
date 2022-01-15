
local ns = require "numsky"

local tag2desc, attr_set = ns._canvas_enum_tag_attr()

local function filter_key(t, map)
	local l = {}
	for tag, desc in pairs(t) do
		if map(tag, desc) then
			l[#l + 1] = tag
		end
	end
	return l
end

local function check_tag_attr(format, tag, attr)
	local ok, msg = pcall(function()
		local sth = format:format(tag, attr, tag)
		ns.canvas(sth)
	end)
	return ok
end

local function dump_tag(format, tag_list)
	table.sort(tag_list)
	for i, tag in pairs(tag_list) do
		print(tag)
		--print("\tcontrol:")
		for attr,v in pairs(attr_set) do
			if attr:sub(1,2) == "x-" then
				-- print("\t", attr)
			end
		end
		print("\tstatic attr:")
		for attr,v in pairs(attr_set) do
			if attr:sub(1,2) ~= "x-"
				and string.byte(attr:sub(1,1)) >= string.byte("A")
				and string.byte(attr:sub(1,1)) <= string.byte("Z") then
				if check_tag_attr(format, tag, attr) then
					print("\t", attr)
				end
			end
		end
		print("\tdynamic attr:")
		for attr,v in pairs(attr_set) do
			if attr:sub(1,2) ~= "x-"
				and string.byte(attr:sub(1,1)) >= string.byte("a")
				and string.byte(attr:sub(1,1)) <= string.byte("z") then
				if check_tag_attr(format, tag, attr) then
					print("\t", attr)
				end
			end
		end
	end
end

-- dump control
dump_tag("<%s %s=''></%s>", filter_key(tag2desc, function(tag, desc)
	return desc == "control"
end))

-- dump lua
dump_tag("<%s %s=''></%s>", {"Camera", "Any", "Table", "Array"})

-- dump mesh
local mesh_format="<Camera><%s %s=''></%s></Camera>"
dump_tag(mesh_format, {"Mesh"})
dump_tag(mesh_format, filter_key(tag2desc, function(tag, desc)
	return tag ~= "Mesh" and desc == "mesh"
end))
