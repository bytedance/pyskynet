#define LUA_LIB

#ifndef DRLUA_H
#define DRLUA_H

#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>

LUALIB_API int luaopen_tflite(lua_State *L);
LUALIB_API int luaopen_numsky(lua_State *L);
LUALIB_API int luaopen_numsky_tinygl(lua_State *L);
LUALIB_API int luaopen_rapidjson(lua_State *L);
LUALIB_API int luaopen_pb(lua_State *L);

#endif
