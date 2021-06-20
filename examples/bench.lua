local skynet = require "skynet"
local socket = require "skynet.socket"
local pyskynet = require "pyskynet"
local foreign = require "pyskynet.foreign"

local scriptaddr, mode = ...

if mode == "client" then

skynet.start(function()
    skynet.dispatch("foreign", function (_,_,fd)
	   skynet.fork(function()
		   socket.start(fd)
		   while true do
			  local line = socket.readline(fd, "\r\n")
			  if not line then
				 break
			  end
			  if line == "PING" then
				 --local temp = foreign.fromstring("rewrwekljrwjklrjweklrjwerlw")
				 --local temp = "rewrwekljrwjklrjweklrjwerlw"
				 --skynet.call(".python", "foreign", sth)
				 socket.write(fd, "+PONG\r\n")
			  end
		   end
		   socket.close(fd)
	   end)
    end)
end)


elseif mode == "server" then
    skynet.start(function()
	    local agent = {}
	    for i= 1, 20 do
		    agent[i] = pyskynet.scriptservice(scriptaddr, "client")
	    end
	    local balance = 1
	    local id = socket.listen("0.0.0.0", 10114)
	    socket.start(id , function(id, addr)
		    skynet.error(string.format("%s connected, pass it to agent :%08x", addr, agent[balance]))
		    skynet.send(agent[balance], "foreign", id)
		    balance = balance + 1
		    if balance > #agent then
			    balance = 1
		    end
	    end)
    end)
end
