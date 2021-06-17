local skynet = require "skynet"
require "skynet.manager"

local mAddressToName = {[0]="0"}
local mNameToService = {}

local MSG_REPEAT = 10

local mPreLog = {0,""}
local mSameMsgCount = 1

local mFile 	= nil
local mFileName = nil
local mFilePath = nil
local function log2File(vFileName, vContent)
	if mFile == nil then
		mFile = io.open(vFileName, "a")
	elseif mFileName ~= vFileName then
		mFile:close()
		mFile = io.open(vFileName, "a")
	end
	mFile:write(vContent)
	mFile:flush()
	mFileName = vFileName
end

local function minify(str)
	local minstr = ""
	local service = ""
	while(true) do
		local first = str:find(" ")
		local oldStr = str
		if first then
			str = str:sub(first+1)
		end
		if (not first ) or #str == 0 then
			minstr = minstr.."-"..oldStr
			service = oldStr
			break
		else
			minstr = minstr..oldStr:sub(1,1)
		end
	end
	return minstr, service
end

local function update(address, msg)
	local name = ""
	local service = ""
	if msg:sub(1,6) == "LAUNCH" then
		name, service = minify(msg)
		mAddressToName[address] = name
		mNameToService[name] = service
	else
		name = mAddressToName[address]
		if not name then
			name = minify(msg)
			mAddressToName[address] = name
		end
	end
	return name
end

skynet.register_protocol {
	name = "text",
	id = skynet.PTYPE_TEXT,
	unpack = skynet.tostring,
	dispatch = function(_, address, msg)
		local name = update(address, msg)
		if mFilePath then
			local nFileName = mFilePath .. string.gsub(os.date("%x"), "/", "-") .. ".log"

			-- do not print same message for many times
			--[[if mPreLog[1] == address and mPreLog[2] == msg then
				mSameMsgCount = mSameMsgCount + 1
				if mSameMsgCount > MSG_REPEAT then
					return
				end
			else
				if mSameMsgCount > MSG_REPEAT then
					log2File(nFileName, "[WARNING] repeat "..mSameMsgCount.." times:")
					log2File(nFileName, mPreLog[2])
					log2File(nFileName, "\n")
				end
				mSameMsgCount = 1
				mPreLog={address, msg}
			end]]

			local nContent = string.format("[:%08x](%s.%02d)<%s>:%s\n", address, os.date("%X"), skynet.now()%100, name or "", msg)

			log2File(nFileName, nContent)
		else
			local nContent = string.format("[:%08x](%s.%02d)<%s>:%s", address, os.date("%X"), skynet.now()%100, name or "", msg)
			print(nContent)
		end
	end
}

skynet.register_protocol {
	name = "SYSTEM",
	id = skynet.PTYPE_SYSTEM,
	unpack = function(...) return ... end,
	dispatch = function()
		-- reopen signal
		print("SIGHUP")
	end
}

skynet.start(function()
	mFilePath = skynet.getenv "logpath"
	skynet.register ".logger"
end)
