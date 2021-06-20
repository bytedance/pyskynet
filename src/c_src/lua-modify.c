#define LUA_LIB
#include "skynet.h"
#include "skynet_modify/skynet_py.h"

#include <lua.h>
#include <lauxlib.h>

#include <time.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

static int
lgetlenv(lua_State *L) {
    const char *key = luaL_checkstring(L, 1);
    size_t sz;
    const char *value = skynet_py_getlenv(key, &sz);
    if(value != NULL) {
        lua_pushlstring(L, value, sz);
    } else {
        lua_pushnil(L);
    }
    return 1;
}

static int
lsetlenv(lua_State *L) {
    const char *key;
	if(lua_isnil(L, 1)) {
		key = NULL;
	} else {
		key = luaL_checkstring(L, 1);
	}
	const char *newkey = NULL;
    size_t sz;
    const char *value;
    int t2 = lua_type(L,2);
    switch (t2) {
    case LUA_TSTRING: {
		value = lua_tolstring(L, 2, &sz);
		newkey = skynet_py_setlenv(key, value, sz);
		break;
	 }
    case LUA_TLIGHTUSERDATA: {
	    value = lua_touserdata(L, 2);
	    sz = luaL_checkinteger(L, 3);
	    newkey = skynet_py_setlenv(key, value, sz);
	    break;
	 }
    default:
	    luaL_error(L, "setlenv invalid param %s", lua_typename(L,t2));
    }
	if(key != NULL) {
		return 0;
	}
    char addr[32];
    sprintf(addr, "%p", newkey);
    lua_pushstring(L, addr);
	return 1;
}

static int
lnextenv(lua_State *L) {
    const char *key = NULL;
    if(lua_type(L,1) == LUA_TSTRING) {
	   key = lua_tostring(L, 1);
    }
    const char *nextkey = skynet_py_nextenv(key);
    if(nextkey == NULL) {
	   lua_pushnil(L);
    } else {
	   lua_pushstring(L, nextkey);
    }
    return 1;
}

static const struct luaL_Reg l_methods[] = {
    { "setlenv", lsetlenv},
    { "getlenv", lgetlenv},
    { "nextenv", lnextenv},
    { NULL,  NULL },
};

LUAMOD_API int luaopen_pyskynet_modify(lua_State *L) {
    luaL_checkversion(L);

    luaL_newlib(L, l_methods);

    return 1;
}

