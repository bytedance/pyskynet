
#include "foreign_seri/seri.h"
#include "foreign_seri/read_block.h"
#include "foreign_seri/write_block.h"

int luapack(lua_State *L) {
	return mode_pack(L, MODE_LUA);
}

int luaunpack(lua_State *L) {
	return mode_unpack(L, MODE_LUA);
}

int refpack(lua_State *L) {
	return mode_pack(L, MODE_FOREIGN_REF);
}

int refunpack(lua_State *L) {
	return mode_unpack(L, MODE_FOREIGN_REF);
}

int remotepack(lua_State *L) {
	return mode_pack(L, MODE_FOREIGN_REMOTE);
}

int remoteunpack(lua_State *L) {
	return mode_unpack(L, MODE_FOREIGN_REMOTE);
}

static const struct luaL_Reg l_methods[] = {
    { "luapack" , luapack },
    { "luaunpack", luaunpack },

    { "refpack" , refpack },
    { "refunpack", refunpack },

    { "remotepack", remotepack },
    { "remoteunpack", remoteunpack },
    { NULL,  NULL },
};

LUAMOD_API int
luaopen_pyskynet_seri_v2(lua_State *L) {
	luaL_checkversion(L);

	luaL_newlib(L, l_methods);
	return 1;
}
