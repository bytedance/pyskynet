
#include "foreign_seri/LuaWriteBlock.h"
#include "foreign_seri/LuaReadBlock.h"

template <SeriMode mode> int lpack(lua_State *L) {
	LuaWriteBlock wb(mode, L);
	wb.pack();
	return wb.ret();
}

template <SeriMode mode> int lunpack(lua_State *L) {
	LuaReadBlock rb(mode, L);
	return rb.unpack();
}

static const struct luaL_Reg l_methods[] = {
    { "luapack" , lpack<MODE_LUA>},
    { "pack", lpack<MODE_FOREIGN> },
    { "remotepack", lpack<MODE_FOREIGN_REMOTE> },
    { "luaunpack" , lunpack<MODE_LUA>},
    { "unpack", lunpack<MODE_FOREIGN> },
    { "remoteunpack", lunpack<MODE_FOREIGN_REMOTE> },
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
