local utils = {}

local _class = {}

function utils.class(name, super)
	local class_type = {}
	class_type.ctor = false
	class_type.name = name
	class_type.super = super
	class_type.is = function(obj)
		local meta = getmetatable(obj)
		return meta and (meta.__index == _class[class_type])
	end
	class_type.new = function(...)
			local obj = {}
			setmetatable(obj, { __index = _class[class_type], __name = name })
			do
				local create
				create = function(c, ...)
					if c.super then
						create(c.super, ...)
					end
					if c.ctor then
						c.ctor(obj, ...)
					end
				end
				create(class_type, ...)
			end
			return obj
		end
	local vtbl = {}
	_class[class_type] = vtbl

	setmetatable(class_type, {
		__newindex=function(t, k, v)
			vtbl[k] = v
		end,
		__index=function(t, k)
			return vtbl[k]
		end
	})

	if super then
		setmetatable(vtbl, {__index=
			function(t, k)
				local ret = _class[super][k]
				vtbl[k] = ret
				return ret
			end
		})
	end

	return class_type
end

function utils.dumps(obj, indent)
    local getIndent, quoteStr, wrapKey, wrapVal, dumpObj
    getIndent = function(level)
        return indent and string.rep("\t", level) or ""
    end
    quoteStr = function(str)
        return '"' .. string.gsub(str, '"', '\\"') .. '"'
    end
    wrapKey = function(val)
        if type(val) == "number" then
            return "[" .. val .. "]"
        elseif type(val) == "string" then
            return "[" .. quoteStr(val) .. "]"
        else
            return "[" .. tostring(val) .. "]"
        end
    end
    wrapVal = function(val, level)
        if type(val) == "table" then
            return dumpObj(val, level)
        elseif type(val) == "number" then
            return val
        elseif type(val) == "string" then
            return quoteStr(val)
        else
            return tostring(val)
        end
    end
    dumpObj = function(obj, level)
        if type(obj) ~= "table" then
            return wrapVal(obj)
        end
        level = level + 1
        local tokens = {}
        tokens[#tokens + 1] = "{"
        for k, v in pairs(obj) do
            tokens[#tokens + 1] = getIndent(level) .. wrapKey(k) .. "=" .. wrapVal(v, level) .. ","
        end
        tokens[#tokens + 1] = getIndent(level - 1) .. "}"
        return table.concat(tokens, indent and "\n" or "")
    end
    return dumpObj(obj, 0)
end

return utils
