
#include "foreign_seri/seri.h"
#include "foreign_seri/read_block.h"
#include "foreign_seri/write_block.h"

static int refpack(lua_State *L) {
	return lmode_pack(MODE_FOREIGN_REF, L);
}

static int refunpack(lua_State *L) {
	return lmode_unpack(MODE_FOREIGN_REF, L);
}

static int remotepack(lua_State *L) {
	return lmode_pack(MODE_FOREIGN_REMOTE, L);
}

static int remoteunpack(lua_State *L) {
	return lmode_unpack(MODE_FOREIGN_REMOTE, L);
}

static int lpackhook(lua_State *L){
	if(lua_islightuserdata(L, 1)) {
		char * ptr = (char*)lua_touserdata(L,1);
		char ** hookptr = foreign_hook(ptr);
		if(hookptr == NULL) {
			lua_pushnil(L);
		} else {
			lua_pushlightuserdata(L, hookptr);
		}
		return 1;
	} else {
		return luaL_error(L, "packhook must take a lightuserdata");
	}
}

static int ltrash(lua_State *L) {
	if(lua_islightuserdata(L, 1)) {
		char * ptr = (char*)lua_touserdata(L,1);
        foreign_trash(ptr);
		return 0;
	} else {
		return luaL_error(L, "packhook must take a lightuserdata");
	}
}

static const struct luaL_Reg l_methods[] = {
    { "refpack" , refpack },
    { "refunpack", refunpack },

    { "remotepack", remotepack },
    { "remoteunpack", remoteunpack },

    { "packhook", lpackhook},
    { "trash", ltrash },

    { NULL,  NULL },
};

LUAMOD_API int luaopen_pyskynet_foreign_seri(lua_State *L) {
	luaL_checkversion(L);

	luaL_newlib(L, l_methods);
	return 1;
}
