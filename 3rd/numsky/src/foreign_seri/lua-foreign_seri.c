
#include "foreign_seri/seri.h"
#include "foreign_seri/read_block.h"
#include "foreign_seri/write_block.h"

static int luapack(lua_State *L) {
	return lmode_pack(MODE_LUA, L);
}

static int luaunpack(lua_State *L) {
	return lmode_unpack(MODE_LUA, L);
}

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

static int ltostring(lua_State *L) {
	int t = lua_type(L,1);
	switch (t) {
	case LUA_TSTRING: {
		lua_settop(L, 1);
		return 1;
	}
	case LUA_TLIGHTUSERDATA: {
		char * msg = (char*)lua_touserdata(L,1);
		int sz = luaL_checkinteger(L,2);
		lua_pushlstring(L,msg,sz);
		return 1;
	}
	default:
		return 0;
	}
}

static int ltrash(lua_State *L) {
	int t = lua_type(L,1);
	switch (t) {
	case LUA_TSTRING: {
		break;
	}
	case LUA_TLIGHTUSERDATA: {
		void * msg = lua_touserdata(L,1);
		luaL_checkinteger(L,2);
		skynet_free(msg);
		break;
	}
	default:
		luaL_error(L, "skynet.trash invalid param %s", lua_typename(L,t));
	}

	return 0;
}

static const struct luaL_Reg l_methods[] = {
    { "luapack" , luapack },
    { "luaunpack", luaunpack },

    { "pack" , refpack },
    { "unpack", refunpack },

    { "remotepack", remotepack },
    { "remoteunpack", remoteunpack },

    { "packhook", lpackhook},

    { "tostring", ltostring },
    { "trash", ltrash },

    { NULL,  NULL },
};

LUAMOD_API int
luaopen_pyskynet_foreign_seri(lua_State *L) {
	luaL_checkversion(L);

	luaL_newlib(L, l_methods);
	return 1;
}
