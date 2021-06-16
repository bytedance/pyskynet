local pyskynet = require "pyskynet"
local skynet = require "skynet"
local socket = require "skynet.socket"

local function forward_fd(left_fd, right_fd)
	while true do
		local data = socket.read(left_fd)
		if not data then
			socket.close(right_fd)
			socket.close(left_fd)
			break
		end
		socket.write(right_fd, data)
	end
end

local dst_ip, dst_port = "10.227.5.113", 9001
pyskynet.start(function()
	local server_fd = socket.listen("127.0.0.1", 8001)
	print("Listen socket :", "127.0.0.1", 8001)
	socket.start(server_fd, function(left_fd, addr)
		print("connect from " .. addr .. " " .. left_fd)
		socket.start(left_fd)
		local right_fd = socket.open(dst_ip, dst_port)
        skynet.fork(forward_fd, left_fd, right_fd)
        skynet.fork(forward_fd, right_fd, left_fd)
	end)

end)
