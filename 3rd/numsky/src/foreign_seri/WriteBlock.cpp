#include "foreign_seri/WriteBlock.h"

void WriteBlock::wb_table_array(lua_State*L, int index, int depth) {
	int array_size = lua_rawlen(L, index);
	if (array_size >= MAX_COOKIE-1) {
		uint8_t n = COMBINE_TYPE(TYPE_TABLE, MAX_COOKIE-1);
		push(&n, 1);
		wb_integer(array_size);
	} else {
		uint8_t n = COMBINE_TYPE(TYPE_TABLE, array_size);
		push(&n, 1);
	}

	int i;
	for (i=1;i<=array_size;i++) {
		lua_rawgeti(L,index,i);
		pack_one(L, -1, depth);
		lua_pop(L, 1);
	}
	return array_size;
}

void WriteBlock::wb_table_hash(lua_State*L, int index, int depth, int array_size) {
	lua_pushnil(L);
	while (lua_next(L, index) != 0) {
		if (lua_type(L,-2) == LUA_TNUMBER) {
			if (lua_isinteger(L, -2)) {
				lua_Integer x = lua_tointeger(L,-2);
				if (x>0 && x<=array_size) {
					lua_pop(L,1);
					continue;
				}
			}
		}
		pack_one(L,-2,depth);
		pack_one(L,-1,depth);
		lua_pop(L, 1);
	}
	wb_nil();
}

void WriteBlock::wb_table(lua_State*L, int index, int depth) {
	luaL_checkstack(L, LUA_MINSTACK, NULL);
	if (index < 0) {
		index = lua_gettop(L) + index + 1;
	}
	if (luaL_getmetafield(L, index, "__pairs") != LUA_TNIL) {
		wb_table_metapairs(L, index, depth);
	} else {
		int array_size = wb_table_array(L, index, depth);
		wb_table_hash(L, index, depth, array_size);
	}
}

void WriteBlock::pack_one(lua_State *L, int index, int depth) {
	if (depth > MAX_DEPTH) {
		free_buffer();
		luaL_error(L, "serialize can't pack too depth table");
	}
	int type = lua_type(L,index);
	switch(type) {
	case LUA_TNIL:
		wb_nil(b);
		break;
	case LUA_TNUMBER: {
		if (lua_isinteger(L, index)) {
			lua_Integer x = lua_tointeger(L,index);
			wb_integer(x);
		} else {
			lua_Number n = lua_tonumber(L,index);
			wb_real(n);
		}
		break;
	}
	case LUA_TBOOLEAN:
		wb_boolean(lua_toboolean(L,index));
		break;
	case LUA_TSTRING: {
		size_t sz = 0;
		const char *str = lua_tolstring(L,index,&sz);
		wb_string(str, (int)sz);
		break;
	}
	case LUA_TLIGHTUSERDATA:
		wb_pointer(lua_touserdata(L,index));
		break;
	case LUA_TTABLE: {
		if (index < 0) {
			index = lua_gettop(L) + index + 1;
		}
		wb_table(L, index, depth+1);
		break;
	}
	case LUA_TUSERDATA: {
		struct numsky_ndarray* arr = *(struct numsky_ndarray**) (luaL_checkudata(L, index, NS_ARR_METANAME));
		if(arr->nd >= MAX_COOKIE) {
			free_buffer();
			luaL_error(L, "numsky.ndarray's nd must be <= 31");
		}
		if (mMode==MODE_FOREIGN) {
			if(arr->foreign_base == NULL) {
				free_buffer();
				luaL_error(L, "ns_arr.foreign_base can't be null");
				return ;
			}
			wb_ns_arr(arr);
		} else if (mMode == MODE_FOREIGN_REMOTE) {
			wb_ns_arr(arr);
		} else {
			free_buffer();
			luaL_error(L, "Unsupport type %s to serialize", lua_typename(L, type));
		}
		break;
	}
	default:
		free_buffer();
		luaL_error(L, "Unsupport type %s to serialize", lua_typename(L, type));
	}
}

void WriteBlock::pack_from(lua_State *L, int from) {
	int n = lua_gettop(L) - from;
	int i;
	for (i=1;i<=n;i++) {
		pack_one(L, from + i, 0);
	}
}

void WriteBlock::wb_ns_arr(lua_State *L, struct numsky_ndarray *arr_obj) {
	// 1. nd & type
	uint8_t n = COMBINE_TYPE(TYPE_FOREIGN_USERDATA, arr_obj->nd);
	push(&n, 1);
	struct numsky_dtype *dtype = arr_obj->dtype;
	// 2. typechar
	push(&(dtype->typechar), 1);
	// 3. dimension
	for(int i=0;i<arr_obj->nd;i++) {
		push_uint(arr_obj->dimensions[i]);
	}
	if (mMode == MODE_FOREIGN) {
		// 4. strides
		push(arr_obj->strides, sizeof(npy_intp)*arr_obj->nd);
		// 5. data
		skynet_foreign_incref(arr_obj->foreign_base);
		push(&(arr_obj->foreign_base), sizeof(arr_obj->foreign_base));
		push(&(arr_obj->dataptr), sizeof(arr_obj->dataptr));
	} else if(mMode == MODE_FOREIGN_REMOTE){
		// 4. data
		struct numsky_nditer * iter = numsky_nditer_create(arr_obj);
		for(int i=0;i<iter->ao->count;numsky_nditer_next(iter), i++) {
			push(iter->dataptr, dtype->elsize);
		}
		numsky_nditer_destroy(iter);
	}
}
