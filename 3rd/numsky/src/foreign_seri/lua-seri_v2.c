
#include "foreign_seri/seri.h"
#include "foreign_seri/read_block.h"
#include "foreign_seri/write_block.h"

int refpack(lua_State *L) {
	return lua_pack(L, true);
}

int refunpack(lua_State *L) {
	return lua_unpack(L, true);
}

int remotepack(lua_State *L) {
	return lua_pack(L, false);
}

int remoteunpack(lua_State *L) {
	return lua_unpack(L, false);
}

static const struct luaL_Reg l_methods[] = {
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
