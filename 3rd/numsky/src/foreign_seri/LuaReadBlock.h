
#pragma once

#include "foreign_seri/ReadBlock.h"

class LuaReadBlock : ReadBlock {
protected:
	SeriMode mMode;
	lua_State *L;
public:
	LuaReadBlock(SeriMode vMode, lua_State*vL) : ReadBlock(), mMode(vMode), L(vL) {}
	void unpack_table(int array_size);
	void unpack_one();
	void push_value(int type, int cookie);
	void push_string(int64_t len) {
		const char *p = get_string(len);
		lua_pushlstring(L,p,len);
	}
	int unpack();
};
