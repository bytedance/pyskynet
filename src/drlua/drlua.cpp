#define LUA_LIB
#if defined(_WIN32)
#define UNITY_API __declspec(dllexport)
#else
#define UNITY_API
#endif

extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
#include <stdio.h>
#include <string.h>


#ifdef BUILD_FOR_DRLUA
#include "drlua.h"
typedef void (*DrluaCallback)(lua_State*, const char*); // c# function, called by lua
#endif

}

#include <string>


namespace drlua {
	static int interface_newObj(lua_State *L) {
		return luaL_error(L, "drlua.newObj(strArg, byteArg)->objId not implement ");
	}

	static int interface_objCall(lua_State *L) {
		return luaL_error(L, "drlua.objCall(objId, funcName, byteArg)->byteRet not implement ");
	}

	static int interface_delObj(lua_State *L) {
		return luaL_error(L, "drlua.delObj(objId) not implement ");
	}

#ifdef BUILD_FOR_DRLUA
	// wrap csharp loader
	static int wrapper_loader(lua_State* L) {
		const char *path = lua_tostring(L, 1);
		lua_rawgetp(L, LUA_REGISTRYINDEX, reinterpret_cast<const void*>(wrapper_loader));
		auto csloader = reinterpret_cast<DrluaCallback>(lua_touserdata(L, -1));
		csloader(L, path);
		if(lua_type(L, -1) != LUA_TSTRING) {
			std::string s = "\n\tno result for customLoader('";
			s += path;
			s += "')";
			lua_pushstring(L, s.c_str());
			return 1;
		}
		size_t length;
		const char *buffer = luaL_checklstring(L, -1, &length);
		int err = luaL_loadbufferx(L, buffer, length, path, "t");
		if(err != LUA_OK) {
			return luaL_error(L, "%s", lua_tostring(L, -1));
		}
		return 1;
	}

	// wrap csharp panic
	static int wrapper_panic(lua_State* L) {
		const char* reason = lua_tostring(L, -1);
		lua_rawgetp(L, LUA_REGISTRYINDEX, reinterpret_cast<const void*>(wrapper_panic));
		auto cspanic = reinterpret_cast<DrluaCallback>(lua_touserdata(L, -1));
		cspanic(L, reason);
		return 0;
	}

	// wrap csharp print
	static int wrapper_print(lua_State*L) {
		int n = lua_gettop(L);
		std::string s;
		for (int i = 1; i <= n; i++)
		{
			size_t len;
			const char * temp = luaL_tolstring(L, i, &len);
			s += temp;
			if (i != n) {
				s += "\t";
			}
		}
		lua_rawgetp(L, LUA_REGISTRYINDEX, reinterpret_cast<const void*>(wrapper_print));
		auto csprint = reinterpret_cast<DrluaCallback>(lua_touserdata(L, -1));
		csprint(L, s.c_str());
		return 0;
	}

	// error handler
	static int wrapper_msgh(lua_State* L) {
		const char *msg = lua_tostring(L, 1);
		luaL_traceback(L, L, msg, 1);
		const char* reason = lua_tostring(L, -1);
		lua_rawgetp(L, LUA_REGISTRYINDEX, reinterpret_cast<const void*>(wrapper_panic));
		auto cspanic = reinterpret_cast<DrluaCallback>(lua_touserdata(L, -1));
		cspanic(L, reason);
		return 1;
	}
#endif
}

extern "C" {
	LUAMOD_API int luaopen_drlua(lua_State *L) {
		luaL_checkversion(L);

		struct luaL_Reg l_methods[] = {
			{ "newObj", drlua::interface_newObj},
			{ "objCall", drlua::interface_objCall},
			{ "delObj", drlua::interface_delObj},
			{ NULL,  NULL },
		};
		luaL_newlib(L, l_methods);
		lua_pushvalue(L, -1);
 		lua_rawsetp(L, LUA_REGISTRYINDEX, reinterpret_cast<const void*>(luaopen_drlua));

		return 1;
	}

#ifdef BUILD_FOR_DRLUA
	UNITY_API void drlua_pushbuffer(lua_State *L, const char * buffer, int len) {
		if(buffer == NULL) {
			lua_pushnil(L);
		} else {
			lua_pushlstring(L, buffer, len);
		}
	}

	UNITY_API int drlua_newObj(lua_State *L, const char * objName, const char * arg, int arg_len) {
 		lua_rawgetp(L, LUA_REGISTRYINDEX, reinterpret_cast<const void*>(luaopen_drlua));
		lua_getfield(L, -1, "newObj");
		lua_pushstring(L, objName);
		if(arg != NULL) {
			lua_pushlstring(L, arg, arg_len);
		} else {
			lua_pushnil(L);
		}
		lua_call(L, 2, 1);
		return luaL_checkinteger(L, -1);
	}

	UNITY_API const char* drlua_objCall(lua_State *L, int objId, const char * funcName, const char * arg, int arg_len, size_t * plen) {
 		lua_rawgetp(L, LUA_REGISTRYINDEX, reinterpret_cast<const void*>(luaopen_drlua));
		lua_getfield(L, -1, "objCall");
		lua_pushinteger(L, objId);
		lua_pushstring(L, funcName);
		if(arg != NULL) {
			lua_pushlstring(L, arg, arg_len);
		} else {
			lua_pushnil(L);
		}
		lua_call(L, 3, 1);
		return luaL_checklstring(L, -1, plen);
	}

	UNITY_API void drlua_delObj(lua_State *L, int objId) {
 		lua_rawgetp(L, LUA_REGISTRYINDEX, reinterpret_cast<const void*>(luaopen_drlua));
		lua_getfield(L, -1, "delObj");
		lua_pushinteger(L, objId);
		lua_call(L, 1, 0);
	}

	UNITY_API void drlua_popall(lua_State *L) {
		lua_settop(L, 0);
	}

	// called by csharp
	UNITY_API void drlua_boot(lua_State *L, const char* buffer, int length) {
		lua_pushcfunction(L, drlua::wrapper_msgh);
		int trace = lua_gettop(L);
		int err = luaL_loadbufferx(L, buffer, length, "boot script", "t");
		if(err != LUA_OK) {
			luaL_error(L, "%s", lua_tostring(L, -1));
		}
		err = lua_pcall(L, 0, 0, trace);
		if(err != LUA_OK) {
			luaL_error(L, "%s", lua_tostring(L, -1));
		}
		lua_settop(L, 0);
	}

	// called by csharp
	UNITY_API lua_State* drlua_new(DrluaCallback panic, DrluaCallback print, DrluaCallback loader){
		lua_State *L = luaL_newstate();
		// openlibs
		luaL_openlibs(L);
		// set panic
		lua_pushlightuserdata(L, (void*)panic);
		lua_rawsetp(L, LUA_REGISTRYINDEX, reinterpret_cast<const void*>(drlua::wrapper_panic));
		lua_atpanic(L, drlua::wrapper_panic);
		// set print
		lua_pushlightuserdata(L, (void*)print);
		lua_rawsetp(L, LUA_REGISTRYINDEX, reinterpret_cast<const void*>(drlua::wrapper_print));
		lua_pushcfunction(L, drlua::wrapper_print);
		lua_setglobal(L, "print");
		// set loader
		lua_pushlightuserdata(L, (void*)loader);
		lua_rawsetp(L, LUA_REGISTRYINDEX, reinterpret_cast<const void*>(drlua::wrapper_loader));
		lua_getglobal(L, "package");
		lua_getfield(L, -1, "searchers");
		int searchers = lua_gettop(L);
		for(int i = lua_rawlen(L, searchers); i >= 2; i--) {
			// just remove all loader, except preload
			lua_pushnil(L);
			lua_rawseti(L, searchers, i);
		}
		lua_pushcfunction(L, drlua::wrapper_loader);
		lua_rawseti(L, searchers, 2);
		// preload libs TODO
		struct luaL_Reg preload[] = {
			{"drlua", luaopen_drlua},
			{"numsky", luaopen_numsky},
			{"numsky.tinygl", luaopen_numsky_tinygl},
			{"tflite", luaopen_tflite},
			{"rapidjson", luaopen_rapidjson},
			{"pb", luaopen_pb},
			{ NULL,  NULL },
		};
		lua_getglobal(L, "package");
		lua_getfield(L, -1, "preload");
		luaL_setfuncs(L, preload, 0);
		lua_settop(L, 0);
		return L;
	}
#endif
}
