#include "foreign_seri/LuaReadBlock.h"

void LuaReadBlock::unpack_table(int array_size) {
	if (array_size == MAX_COOKIE-1) {
		uint8_t type = read<uint8_t>();
		int cookie = type >> 3;
		if ((type & 7) != TYPE_NUMBER || cookie == TYPE_NUMBER_REAL) {
			throw std::domain_error("invalid cookie");
		}
		array_size = get_integer(cookie);
	}
	luaL_checkstack(L,LUA_MINSTACK,NULL);
	lua_createtable(L,array_size,0);
	for (int i=1;i<=array_size;i++) {
		unpack_one();
		lua_rawseti(L,-2,i);
	}
	for (;;) {
		unpack_one();
		if (lua_isnil(L,-1)) {
			lua_pop(L,1);
			return;
		}
		unpack_one();
		lua_rawset(L,-3);
	}
}

void LuaReadBlock::unpack_one() {
	uint8_t type = read<uint8_t>();
	push_value(type & 0x7, type>>3);
}

void LuaReadBlock::push_value(int type, int cookie) {
	switch(type) {
	case TYPE_NIL:
		lua_pushnil(L);
		break;
	case TYPE_BOOLEAN:
		lua_pushboolean(L,cookie);
		break;
	case TYPE_NUMBER:
		if (cookie == TYPE_NUMBER_REAL) {
			lua_pushnumber(L, get_real());
		} else {
			lua_pushinteger(L, get_integer(cookie));
		}
		break;
	case TYPE_USERDATA:
		lua_pushlightuserdata(L, get_pointer());
		break;
	case TYPE_SHORT_STRING:
		push_string(cookie);
		break;
	case TYPE_LONG_STRING: {
		if (cookie == 2) {
			uint16_t n = read<uint16_t>();
			push_string(n);
		} else {
			if (cookie != 4) {
				throw std::domain_error("invalid stream");
			}
			uint32_t n = read<uint32_t>();
			push_string(n);
		}
		break;
	}
	case TYPE_TABLE: {
		unpack_table(cookie);
		break;
	}
	default: {
		throw std::domain_error("invalid type");
		break;
	}
	}
}

int LuaReadBlock::unpack() {
	const char * buffer;
	int64_t len;
	if (lua_type(L,1) == LUA_TSTRING) {
		size_t sz;
		buffer = lua_tolstring(L,1,&sz);
		len = (int)sz;
	} else {
		buffer = reinterpret_cast<char*>(lua_touserdata(L,1));
		len = luaL_checkinteger(L,2);
	}
	if (len == 0) {
		return 0;
	}
	if (buffer == NULL) {
		return luaL_error(L, "deserialize null pointer");
	}

	lua_settop(L,1);
	set_buffer(buffer, len);

	for (int i=0;;i++) {
		if (i%8==7) {
			luaL_checkstack(L,LUA_MINSTACK,NULL);
		}
		uint8_t type = read<uint8_t>();
		push_value(type & 0x7, type>>3);
	}

	// Need not free buffer

	return lua_gettop(L) - 1;
}
