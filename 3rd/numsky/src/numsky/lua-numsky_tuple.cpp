#include "numsky/lua-numsky.h"

static int lnumsky_tuple(lua_State *L) {
	int n = lua_gettop(L);
	lua_createtable(L, n, 0);
	luaL_setmetatable(L, NUMSKY_TUPLE_META);
	for (int i=1;i<=n;i++) {
		lua_pushvalue(L, i);
		lua_seti(L, n+1, i);
	}
	return 1;
}

static int lnumsky_tuple__tostring(lua_State *L) {
	int table_len = luaL_len(L, 1);
	std::string buf = "(";
	for(int table_i=1;table_i<=table_len;table_i++) {
		lua_geti(L, 1, table_i);
		buf += lua_tostring(L, -1);
		if(table_i!=table_len){
			buf += ',';
		}
		lua_settop(L, 1);
	}
	buf += ')';
	lua_pushstring(L, buf.c_str());
	return 1;
}

static int lnumsky_tuple__eq(lua_State *L) {
	int len1 = luaL_len(L, 1);
	int len2 = luaL_len(L, 2);
	if(len1 != len2) {
		lua_pushboolean(L, 0);
		return 1;
	} else {
		for(int i=1;i<=len1;i++) {
			lua_geti(L, 1, i);
			lua_geti(L, 2, i);
			if(!lua_compare(L, -1, -2, LUA_OPEQ)) {
				lua_pushboolean(L, 0);
				return 1;
			}
			lua_settop(L, 2);
		}
		lua_pushboolean(L, 1);
		return 1;
	}
}

void lnumsky_tuple_bind_lib(luabinding::Module_&m) {
	luaL_newmetatable(m.L, NUMSKY_TUPLE_META);
	lua_pushcfunction(m.L, lnumsky_tuple__tostring);
	lua_setfield(m.L, -2, "__tostring");
	lua_pushcfunction(m.L, lnumsky_tuple__eq);
	lua_setfield(m.L, -2, "__eq");

	m.setFunction("tuple", lnumsky_tuple);
}
