#include "skynet.h"
#include "skynet_env.h"
#include "spinlock.h"
#include "skynet_py.h"

#include <lua.h>
#include <lauxlib.h>

#include <stdlib.h>
#include <assert.h>
#include <string.h>

struct skynet_env {
	struct spinlock lock;
	lua_State *L;
};

static struct skynet_env *E = NULL;

const char *skynet_py_getlenv(const char *key, size_t *sz) {
	SPIN_LOCK(E);
	lua_State *L = E->L;
	lua_getglobal(L, key);
	const char *value_data;
	if(lua_type(L, -1) == LUA_TSTRING) {
	    value_data = lua_tolstring(L, -1, sz);
	} else {
	    value_data = NULL;
	    *sz = 0;
	}
	lua_pop(L, 1);
	SPIN_UNLOCK(E);
	return value_data;
}

void *skynet_py_setlenv(const char *key, const char *value_str, size_t sz) {
	SPIN_LOCK(E)

	lua_State *L = E->L;
	void *newkey = NULL;
	if(key==NULL) {
		char addr[32];
		newkey = (void*)lua_pushlstring(L, value_str, sz);
		sprintf(addr, "%p", newkey);
		lua_setglobal(L, addr);
	} else {
		lua_getglobal(L, key);
		if(lua_isnil(L, -1) || skynet_py_address() == 0) {
			lua_pop(L, 1);
			lua_pushlstring(L, value_str, sz);
			lua_setglobal(L, key);
		} else {
			printf("can't set existed env after pyskynet start\n");
			lua_pop(L, 1);
		}
	}
	SPIN_UNLOCK(E)
	return newkey;
}

// nextkey
const char *skynet_py_nextenv(const char *key) {
    const char * next_key = NULL;

	SPIN_LOCK(E);
	lua_State *L = E->L;
	lua_pushglobaltable(L);
	if(key==NULL) {
	    lua_pushnil(L);
	} else {
	    lua_pushstring(L, key);
	}
	if(lua_next(L, -2) != 0){
	    next_key = luaL_checkstring(L, -2);
	    lua_pop(L, 3);
	}else {
	    lua_pop(L, 1);
	}
	SPIN_UNLOCK(E);

    return next_key;
}

void skynet_env_init() {
	E = skynet_malloc(sizeof(*E));
	SPIN_INIT(E)
	E->L = luaL_newstate();
}

#define MAX_COOKIE 32
#define TYPE_SHORT_STRING 4
#define TYPE_LONG_STRING 5
#define COMBINE_TYPE(t,v) ((t) | (v) << 3)

// for skynet_env.h
const char *skynet_getenv(const char *key) {
	size_t sz;
	const char *buffer = skynet_py_getlenv(key, &sz);
	if(buffer==NULL || sz <= 0) {
		return NULL;
	} else {
		uint8_t type = ((uint8_t*)buffer)[0];
		int cookie = type>>3;
		type = type & 0x7;
		if(type == TYPE_SHORT_STRING) {
			return buffer + 1;
		} else if(type == TYPE_LONG_STRING) {
			if(cookie == 2) {
				return buffer + 3;
			} else if(cookie == 4) {
				return buffer + 5;
			}
		} else {
			printf("pyskynet getenv but format error \n");
		}
		return NULL;
	}
}

static inline char* buffer_write(char* buffer, const void* data, size_t len) {
	memcpy(buffer, data, len);
	return buffer + len;
}

void skynet_setenv(const char *key, const char* value_str) {
	size_t len = strlen(value_str);
	char *buffer = skynet_malloc(len + 10);
	char *ptr = buffer;
	if (len < MAX_COOKIE) {
		uint8_t n = COMBINE_TYPE(TYPE_SHORT_STRING, len);
		ptr = buffer_write(ptr, &n, 1);
	} else {
		if (len < 0x10000) {
			uint8_t n = COMBINE_TYPE(TYPE_LONG_STRING, 2);
			ptr = buffer_write(ptr, &n, 1);
			uint16_t x = (uint16_t) len;
			ptr = buffer_write(ptr, &x, 2);
		} else {
			uint8_t n = COMBINE_TYPE(TYPE_LONG_STRING, 4);
			ptr = buffer_write(ptr, &n, 1);
			uint32_t x = (uint32_t) len;
			ptr = buffer_write(ptr, &x, 4);
		}
	}
	ptr = buffer_write(ptr, value_str, len);
	skynet_py_setlenv(key, buffer, ptr - buffer);
	skynet_free(buffer);
}
