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
lgetenv(lua_State *L) {
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
lsetenv(lua_State *L) {
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

static int
lptr_unwrap(lua_State *L) {
	const char * info = luaL_checkstring(L, 1);
	void * ptr;
	size_t sz;
    sscanf(info, "%p,%ld", &ptr, &sz);
	lua_pushlightuserdata(L, ptr);
	lua_pushinteger(L, sz);
    return 2;
}

static int
lptr_wrap(lua_State *L) {
	void * ptr;
	size_t sz;
	if (lua_type(L,1) == LUA_TLIGHTUSERDATA) {
		ptr = lua_touserdata(L,1);
		sz = luaL_checkinteger(L,2);
	} else {
		return luaL_error(L, "frominfo's first arg must be lightuserdata");
	}
    char addr[50];
    sprintf(addr, "%p,%ld", ptr, sz);
    lua_pushstring(L, addr);
    return 1;
}

static const struct luaL_Reg l_methods[] = {
    { "setenv", lsetenv},
    { "getenv", lgetenv},
    { "nextenv", lnextenv},
    { "ptr_wrap", lptr_wrap},
    { "ptr_unwrap", lptr_unwrap},
    { NULL,  NULL },
};

LUAMOD_API int luaopen_pyskynet_modify(lua_State *L) {
    luaL_checkversion(L);

    luaL_newlib(L, l_methods);

    return 1;
}

