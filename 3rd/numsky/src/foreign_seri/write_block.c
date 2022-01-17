/*
	modify from https://github.com/cloudwu/lua-serialize
 */

#include "foreign_seri/write_block.h"

inline static void wb_push(struct write_block *wb, const void *data, int64_t sz) {
	int64_t newCapacity = wb->capacity;
	while(newCapacity < wb->len + sz) {
		newCapacity *= 2;
	}
	if(newCapacity != wb->capacity) {
		char *newBuffer = skynet_malloc(newCapacity);
		memcpy(newBuffer, wb->buffer, wb->len);
		skynet_free(wb->buffer);
		wb->buffer = newBuffer;
		wb->capacity = newCapacity;
	}
	memcpy(wb->buffer + wb->len, data, sz);
	wb->len += sz;
}

void wb_write(struct write_block *wb, const void *buf, int64_t sz) {
    wb_push(wb, buf, sz);
}

void wb_init(struct write_block *wb, int mode) {
	const int BLOCK_SIZE = 128;
	wb->buffer = skynet_malloc(BLOCK_SIZE);
	wb->nextbase = 0;
	wb->capacity = BLOCK_SIZE;
	wb->len = 0;
	wb->mode = mode;
	if(mode != MODE_LUA) {
		intptr_t empty = 0;
		wb_push(wb, &empty, sizeof(empty));
	}
}

void wb_free(struct write_block *wb) {
	if(wb->buffer){
		skynet_free(wb->buffer);
		wb->buffer = NULL;
	}
}

void wb_ref_base(struct write_block *wb, struct skynet_foreign* foreign_base, char *dataptr) {
	*((intptr_t*)(wb->buffer + wb->nextbase)) = wb->len;
	wb_push(wb, &(foreign_base), sizeof(foreign_base));
	wb_push(wb, &(dataptr), sizeof(dataptr));
	wb->nextbase = wb->len;
	intptr_t empty = 0;
	wb_push(wb, &empty, sizeof(empty));
}

void wb_nil(struct write_block *wb) {
	uint8_t n = TYPE_NIL;
	wb_push(wb, &n, 1);
}

void wb_boolean(struct write_block *wb, int boolean) {
	uint8_t n = COMBINE_TYPE(TYPE_BOOLEAN , boolean ? 1 : 0);
	wb_push(wb, &n, 1);
}

void wb_put_integer(struct write_block *wb, lua_Integer v) {
	int type = TYPE_NUMBER;
	if (v == 0) {
		uint8_t n = COMBINE_TYPE(type , TYPE_NUMBER_ZERO);
		wb_push(wb, &n, 1);
	} else if (v != (int32_t)v) {
		uint8_t n = COMBINE_TYPE(type , TYPE_NUMBER_QWORD);
		int64_t v64 = v;
		wb_push(wb, &n, 1);
		wb_push(wb, &v64, sizeof(v64));
	} else if (v < 0) {
		int32_t v32 = (int32_t)v;
		uint8_t n = COMBINE_TYPE(type , TYPE_NUMBER_DWORD);
		wb_push(wb, &n, 1);
		wb_push(wb, &v32, sizeof(v32));
	} else if (v<0x100) {
		uint8_t n = COMBINE_TYPE(type , TYPE_NUMBER_BYTE);
		wb_push(wb, &n, 1);
		uint8_t byte = (uint8_t)v;
		wb_push(wb, &byte, sizeof(byte));
	} else if (v<0x10000) {
		uint8_t n = COMBINE_TYPE(type , TYPE_NUMBER_WORD);
		wb_push(wb, &n, 1);
		uint16_t word = (uint16_t)v;
		wb_push(wb, &word, sizeof(word));
	} else {
		uint8_t n = COMBINE_TYPE(type , TYPE_NUMBER_DWORD);
		wb_push(wb, &n, 1);
		uint32_t v32 = (uint32_t)v;
		wb_push(wb, &v32, sizeof(v32));
	}
}

void wb_put_real(struct write_block *wb, double v) {
	uint8_t n = COMBINE_TYPE(TYPE_NUMBER , TYPE_NUMBER_REAL);
	wb_push(wb, &n, 1);
	wb_push(wb, &v, sizeof(v));
}

void wb_put_pointer(struct write_block *wb, void *v) {
	uint8_t n = TYPE_USERDATA;
	wb_push(wb, &n, 1);
	wb_push(wb, &v, sizeof(v));
}

void wb_put_string(struct write_block *wb, const char *str, int len) {
	if (len < MAX_COOKIE) {
		uint8_t n = COMBINE_TYPE(TYPE_SHORT_STRING, len);
		wb_push(wb, &n, 1);
		if (len > 0) {
			wb_push(wb, str, len);
		}
	} else {
		uint8_t n;
		if (len < 0x10000) {
			n = COMBINE_TYPE(TYPE_LONG_STRING, 2);
			wb_push(wb, &n, 1);
			uint16_t x = (uint16_t) len;
			wb_push(wb, &x, 2);
		} else {
			n = COMBINE_TYPE(TYPE_LONG_STRING, 4);
			wb_push(wb, &n, 1);
			uint32_t x = (uint32_t) len;
			wb_push(wb, &x, 4);
		}
		wb_push(wb, str, len);
	}
}

static void lwb_pack_one(lua_State *L, struct write_block *b, int index, int depth);

static int
lwb_table_array(lua_State *L, struct write_block * wb, int index, int depth) {
	int array_size = lua_rawlen(L,index);
	if (array_size >= MAX_COOKIE-1) {
		uint8_t n = COMBINE_TYPE(TYPE_TABLE, MAX_COOKIE-1);
		wb_push(wb, &n, 1);
		wb_put_integer(wb, array_size);
	} else {
		uint8_t n = COMBINE_TYPE(TYPE_TABLE, array_size);
		wb_push(wb, &n, 1);
	}

	int i;
	for (i=1;i<=array_size;i++) {
		lua_rawgeti(L,index,i);
		lwb_pack_one(L, wb, -1, depth);
		lua_pop(L,1);
	}

	return array_size;
}

static void
lwb_table_hash(lua_State *L, struct write_block * wb, int index, int depth, int array_size) {
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
		lwb_pack_one(L,wb,-2,depth);
		lwb_pack_one(L,wb,-1,depth);
		lua_pop(L, 1);
	}
	wb_nil(wb);
}

static void
lwb_table_metapairs(lua_State *L, struct write_block *wb, int index, int depth) {
	uint8_t n = COMBINE_TYPE(TYPE_TABLE, 0);
	wb_push(wb, &n, 1);
	lua_pushvalue(L, index);
	lua_call(L, 1, 3);
	for(;;) {
		lua_pushvalue(L, -2);
		lua_pushvalue(L, -2);
		lua_copy(L, -5, -3);
		lua_call(L, 2, 2);
		int type = lua_type(L, -2);
		if (type == LUA_TNIL) {
			lua_pop(L, 4);
			break;
		}
		lwb_pack_one(L, wb, -2, depth);
		lwb_pack_one(L, wb, -1, depth);
		lua_pop(L, 1);
	}
	wb_nil(wb);
}

static void
lwb_table(lua_State *L, struct write_block *wb, int index, int depth) {
	luaL_checkstack(L, LUA_MINSTACK, NULL);
	if (index < 0) {
		index = lua_gettop(L) + index + 1;
	}
	if (luaL_getmetafield(L, index, "__pairs") != LUA_TNIL) {
		lwb_table_metapairs(L, wb, index, depth);
	} else {
		int array_size = lwb_table_array(L, wb, index, depth);
		lwb_table_hash(L, wb, index, depth, array_size);
	}
}

void wb_uint(struct write_block* wb, npy_intp v) {
	static const int B = 128;
	uint8_t data = v | B;
	while (v >= B) {
		data = v | B;
		wb_push(wb, &data, 1);
		v >>= 7;
	}
	data = (uint8_t)v;
	wb_push(wb, &data, 1);
}

inline static void wb_ns_arr(struct write_block *wb, struct numsky_ndarray* arr_obj) {
	// 1. nd & type
	uint8_t n = COMBINE_TYPE(TYPE_FOREIGN_USERDATA, arr_obj->nd);
	wb_push(wb, &n, 1);
	struct numsky_dtype *dtype = arr_obj->dtype;
	// 2. typechar
	wb_push(wb, &(dtype->typechar), 1);
	// 3. dimension
	for(int i=0;i<arr_obj->nd;i++) {
		wb_uint(wb, arr_obj->dimensions[i]);
	}
	if (wb->mode == MODE_FOREIGN_REF) {
		// 4. strides
		wb_push(wb, arr_obj->strides, sizeof(npy_intp)*arr_obj->nd);
		// 5. data
		skynet_foreign_incref(arr_obj->foreign_base);
		wb_ref_base(wb, arr_obj->foreign_base, arr_obj->dataptr);
	} else if (wb->mode == MODE_FOREIGN_REMOTE) {
		// 4. data
		struct numsky_nditer * iter = numsky_nditer_create(arr_obj);
		for(int i=0;i<iter->ao->count;numsky_nditer_next(iter), i++) {
			wb_push(wb, iter->dataptr, dtype->elsize);
		}
		numsky_nditer_destroy(iter);
	} else {
		// error branch
	}
}

static void
lwb_pack_one(lua_State *L, struct write_block *wb, int index, int depth) {
	if (depth > MAX_DEPTH) {
		wb_free(wb);
		luaL_error(L, "serialize can't pack too depth table");
	}
	int type = lua_type(L,index);
	switch(type) {
	case LUA_TNIL:
		wb_nil(wb);
		break;
	case LUA_TNUMBER: {
		if (lua_isinteger(L, index)) {
			lua_Integer x = lua_tointeger(L,index);
			wb_put_integer(wb, x);
		} else {
			lua_Number n = lua_tonumber(L,index);
			wb_put_real(wb,n);
		}
		break;
	}
	case LUA_TBOOLEAN:
		wb_boolean(wb, lua_toboolean(L,index));
		break;
	case LUA_TSTRING: {
		size_t sz = 0;
		const char *str = lua_tolstring(L,index,&sz);
		wb_put_string(wb, str, (int)sz);
		break;
	}
	case LUA_TLIGHTUSERDATA:
		wb_put_pointer(wb, lua_touserdata(L,index));
		break;
	case LUA_TTABLE: {
		if (index < 0) {
			index = lua_gettop(L) + index + 1;
		}
		lwb_table(L, wb, index, depth+1);
		break;
	}
	case LUA_TUSERDATA: {
		struct numsky_ndarray* arr = *(struct numsky_ndarray**) (luaL_checkudata(L, index, NS_ARR_METANAME));
		if(arr->nd >= MAX_COOKIE) {
			wb_free(wb);
			luaL_error(L, "numsky.ndarray's nd must be <= 31");
		}
		if (wb->mode == MODE_FOREIGN_REF) {
			if(arr->foreign_base == NULL) {
				wb_free(wb);
				luaL_error(L, "foreign -base can't be null");
				return ;
			}
		} else if(wb->mode != MODE_FOREIGN_REMOTE) {
			wb_free(wb);
			luaL_error(L, "lua pack can't take numsky array");
		}
		wb_ns_arr(wb, arr);
		break;
	}
	default:
		wb_free(wb);
		luaL_error(L, "Unsupport type %s to serialize", lua_typename(L, type));
	}
}

char** mode_hook(int mode, char *buffer, intptr_t nextbase) {
	if((mode == MODE_FOREIGN_REF && nextbase != 0)) {
		char ** hookptr = skynet_malloc(sizeof(char*));
		*hookptr = buffer;
		return hookptr;
	} else {
		return NULL;
	}
}

int lmode_pack(int mode, lua_State *L) {
	struct write_block wb;
	wb_init(&wb, mode);
	int n = lua_gettop(L);
	for (int i=1;i<=n;i++) {
		lwb_pack_one(L, &wb, i, 0);
	}
	char ** hookptr = mode_hook(mode, wb.buffer, wb.nextbase);
	if(hookptr != NULL) {
		lua_pushlightuserdata(L, hookptr);
		lua_pushinteger(L, wb.len);
		lua_pushlightuserdata(L, wb.buffer);
		return 3;
	} else {
		lua_pushlightuserdata(L, wb.buffer);
		lua_pushinteger(L, wb.len);
		return 2;
	}
}
