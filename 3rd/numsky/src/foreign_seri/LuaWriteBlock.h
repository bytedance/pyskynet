
#pragma once

#include "foreign_seri/WriteBlock.h"

class LuaWriteBlock : WriteBlock {
protected:
	lua_State *L;
public:
	LuaWriteBlock(SeriMode vMode, lua_State *vL): WriteBlock(vMode), L(vL) {}
	void pack() {
		int n = lua_gettop(L);
		for (int i=1;i<=n;i++) {
			pack_one(i, 0);
		}
	}
	int ret() {
		lua_pushlightuserdata(L, mBuffer);
		lua_pushinteger(L, mLen);
		mBuffer = NULL;
		return 2;
	}
	int wb_table_array(int index, int depth);
	void wb_table_hash(int index, int depth, int array_size);
	void wb_table_metapairs(int index, int depth);
	void wb_table(int index, int depth);
	void wb_ns_arr(struct numsky_ndarray *arr_obj);
	void pack_one(int index, int depth);
};
