
LUA_INC = ../../skynet/3rd/lua

rapidxml.so: lua-rapidxml.cpp
	g++  -fPIC --shared -I$(LUA_INC) $^ -o $@
