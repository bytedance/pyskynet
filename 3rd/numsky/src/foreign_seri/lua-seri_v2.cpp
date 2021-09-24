
#include "foreign_seri/WriteBlock.h"

static int lluapack(lua_State *L) {
	WriteBlock wb(MODE_LUA);
	return foreign_pack(L, MODE_LUA);
}

static int lluaunpack(lua_State *L) {
	return foreign_unpack(L, MODE_LUA);
}

static int lpack(lua_State *L) {
	return foreign_pack(L, MODE_FOREIGN);
}

static int lunpack(lua_State *L) {
	return foreign_unpack(L, MODE_FOREIGN);
}

static int lremotepack(lua_State *L) {
	return foreign_pack(L, MODE_FOREIGN_REMOTE);
}

static int lremoteunpack(lua_State *L) {
	return foreign_unpack(L, MODE_FOREIGN_REMOTE);
}

static const struct luaL_Reg l_methods[] = {
    { "luapack" , lluapack },
    { "luaunpack", lluaunpack },
    { "pack", lpack },
    { "unpack" , lunpack },
    { "remotepack", lremotepack },
    { "remoteunpack", lremoteunpack },
    { NULL,  NULL },
};

extern "C" {
	LUAMOD_API int
	luaopen_pyskynet_seri_v2(lua_State *L) {
		luaL_checkversion(L);

		luaL_newlib(L, l_methods);
		return 1;
	}
}
