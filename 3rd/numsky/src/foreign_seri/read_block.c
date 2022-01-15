/*
	modify from https://github.com/cloudwu/lua-serialize
 */

#include "foreign_seri/read_block.h"

void rball_init(struct read_block * rb, char * buffer, int size, bool refarr) {
	rb->buffer = buffer;
	rb->len = size;
	rb->ptr = 0;
	rb->refarr = refarr;
}

void *rb_read(struct read_block *rb, int sz) {
	if (rb->len < sz) {
		return NULL;
	}

	int ptr = rb->ptr;
	rb->ptr += sz;
	rb->len -= sz;
	return rb->buffer + ptr;
}

static inline void
invalid_stream_line(lua_State *L, struct read_block *rb, int line) {
	int len = rb->len;
	luaL_error(L, "Invalid serialize stream %d (line:%d)", len, line);
}

#define invalid_stream(L,rb) invalid_stream_line(L,rb,__LINE__)
inline static bool rb_uint(struct read_block* rb, npy_intp *value) {
	npy_intp result = 0;
	for (uint32_t shift = 0; shift <= 63; shift += 7) {
		uint8_t *p_byte = (uint8_t*)rb_read(rb, 1);
		if(p_byte==NULL) {
			return false;
		}
		npy_intp byte = p_byte[0];
		if (byte & 128) {
			// More bytes are present
			result |= ((byte & 127) << shift);
		} else {
			result |= (byte << shift);
			break;
		}
	}
	if(result < 0) {
		return false;
	}
	*value = result;
	return true;
}

static lua_Integer
get_integer(lua_State *L, struct read_block *rb, int cookie) {
	switch (cookie) {
	case TYPE_NUMBER_ZERO:
		return 0;
	case TYPE_NUMBER_BYTE: {
		uint8_t n;
		uint8_t * pn = (uint8_t *)rb_read(rb,sizeof(n));
		if (pn == NULL)
			invalid_stream(L,rb);
		n = *pn;
		return n;
	}
	case TYPE_NUMBER_WORD: {
		uint16_t n;
		uint16_t * pn = (uint16_t *)rb_read(rb,sizeof(n));
		if (pn == NULL)
			invalid_stream(L,rb);
		memcpy(&n, pn, sizeof(n));
		return n;
	}
	case TYPE_NUMBER_DWORD: {
		int32_t n;
		int32_t * pn = (int32_t *)rb_read(rb,sizeof(n));
		if (pn == NULL)
			invalid_stream(L,rb);
		memcpy(&n, pn, sizeof(n));
		return n;
	}
	case TYPE_NUMBER_QWORD: {
		int64_t n;
		int64_t * pn = (int64_t *)rb_read(rb,sizeof(n));
		if (pn == NULL)
			invalid_stream(L,rb);
		memcpy(&n, pn, sizeof(n));
		return n;
	}
	default:
		invalid_stream(L,rb);
		return 0;
	}
}

static double
get_real(lua_State *L, struct read_block *rb) {
	double n;
	double * pn = (double *)rb_read(rb,sizeof(n));
	if (pn == NULL)
		invalid_stream(L,rb);
	memcpy(&n, pn, sizeof(n));
	return n;
}

static void *
get_pointer(lua_State *L, struct read_block *rb) {
	void * userdata = 0;
	void ** v = (void **)rb_read(rb,sizeof(userdata));
	if (v == NULL) {
		invalid_stream(L,rb);
	}
	memcpy(&userdata, v, sizeof(userdata));
	return userdata;
}

static void
get_buffer(lua_State *L, struct read_block *rb, int len) {
	char * p = (char *)rb_read(rb,len);
	if (p == NULL) {
		invalid_stream(L,rb);
	}
	lua_pushlstring(L,p,len);
}

static void unpack_one(lua_State *L, struct read_block *rb);

static void
unpack_table(lua_State *L, struct read_block *rb, int array_size) {
	if (array_size == MAX_COOKIE-1) {
		uint8_t type;
		uint8_t *t = (uint8_t *)rb_read(rb, sizeof(type));
		if (t==NULL) {
			invalid_stream(L,rb);
		}
		type = *t;
		int cookie = type >> 3;
		if ((type & 7) != TYPE_NUMBER || cookie == TYPE_NUMBER_REAL) {
			invalid_stream(L,rb);
		}
		array_size = get_integer(L,rb,cookie);
	}
	luaL_checkstack(L,LUA_MINSTACK,NULL);
	lua_createtable(L,array_size,0);
	int i;
	for (i=1;i<=array_size;i++) {
		unpack_one(L,rb);
		lua_rawseti(L,-2,i);
	}
	for (;;) {
		unpack_one(L,rb);
		if (lua_isnil(L,-1)) {
			lua_pop(L,1);
			return;
		}
		unpack_one(L,rb);
		lua_rawset(L,-3);
	}
}

static struct numsky_ndarray*
unpack_ns_arr(struct read_block *rb, int nd) {
	// 1. get dtype
	char * p_typechar = (char *)rb_read(rb, 1);
	if(p_typechar == NULL){
		return NULL;
	}
	// 2. init from dimensions
	struct numsky_ndarray *arr = numsky_ndarray_precreate(nd, p_typechar[0]);
	for(int i=0;i<nd;i++){
		bool ok = rb_uint(rb, &arr->dimensions[i]);
		if(!ok) {
			numsky_ndarray_destroy(arr);
			return NULL;
		}
	}
	// 3. build
	struct skynet_foreign *foreign_base;
	char *dataptr;
	if(rb->refarr) {
		numsky_ndarray_autocount(arr);
		npy_intp *strides = (npy_intp*)rb_read(rb, sizeof(npy_intp)*nd);
		if(strides == NULL) {
			numsky_ndarray_destroy(arr);
			return NULL;
		}
		// 4. get strides,
		for(int i=0;i<nd;i++){
			arr->strides[i] = strides[i];
		}
		// 5. foreign_base, dataptr
		// get foreign_base
		void **v = (void **)rb_read(rb,sizeof(foreign_base));
		if (v == NULL) {
			numsky_ndarray_destroy(arr);
			return NULL;
		}
		memcpy(&foreign_base, v, sizeof(foreign_base));
		if(foreign_base == NULL) {
			numsky_ndarray_destroy(arr);
			printf("can't transfor numsky.ndarray with foreign_base == NULL\n");
			return NULL;
		}
		// get dataptr
		v = (void **)rb_read(rb,sizeof(dataptr));
		if (v == NULL) {
			numsky_ndarray_destroy(arr);
			return NULL;
		}
		memcpy(&dataptr, v, sizeof(dataptr));
	} else {
		numsky_ndarray_autostridecount(arr);
		// 4. alloc foreign_base
		size_t datasize = arr->count*arr->dtype->elsize;
		// 1) read & copy
		char * pdata = (char*)rb_read(rb, datasize);
		if(pdata==NULL){
			numsky_ndarray_destroy(arr);
			return NULL;
		}
		foreign_base = skynet_foreign_newbytes(datasize);
		dataptr = foreign_base->data;
		memcpy(dataptr, pdata, datasize);
	}
	numsky_ndarray_refdata(arr, foreign_base, dataptr);
	return arr;
}

static void
push_value(lua_State *L, struct read_block *rb, int type, int cookie) {
	switch(type) {
	case TYPE_NIL:
		lua_pushnil(L);
		break;
	case TYPE_BOOLEAN:
		lua_pushboolean(L,cookie);
		break;
	case TYPE_NUMBER:
		if (cookie == TYPE_NUMBER_REAL) {
			lua_pushnumber(L,get_real(L,rb));
		} else {
			lua_pushinteger(L, get_integer(L, rb, cookie));
		}
		break;
	case TYPE_USERDATA:
		lua_pushlightuserdata(L,get_pointer(L,rb));
		break;
	case TYPE_SHORT_STRING:
		get_buffer(L,rb,cookie);
		break;
	case TYPE_LONG_STRING: {
		if (cookie == 2) {
			uint16_t *plen = (uint16_t *)rb_read(rb, 2);
			if (plen == NULL) {
				invalid_stream(L,rb);
			}
			uint16_t n;
			memcpy(&n, plen, sizeof(n));
			get_buffer(L,rb,n);
		} else {
			if (cookie != 4) {
				invalid_stream(L,rb);
			}
			uint32_t *plen = (uint32_t *)rb_read(rb, 4);
			if (plen == NULL) {
				invalid_stream(L,rb);
			}
			uint32_t n;
			memcpy(&n, plen, sizeof(n));
			get_buffer(L,rb,n);
		}
		break;
	}
	case TYPE_TABLE: {
		unpack_table(L,rb,cookie);
		break;
	}
	case TYPE_FOREIGN_USERDATA: {
		struct numsky_ndarray *arr = unpack_ns_arr(rb, cookie);
		if(arr==NULL) {
			invalid_stream(L, rb);
		} else {
			*(struct numsky_ndarray**)(lua_newuserdata(L, sizeof(struct numsky_ndarray*))) = arr;
			luaL_getmetatable(L, NS_ARR_METANAME);
			if(lua_isnil(L, -1)) {
				luaL_error(L, "require 'numsky' before use foreign seri");
			}
			lua_setmetatable(L, -2);
		}
		break;
	}
	default: {
		invalid_stream(L,rb);
		break;
	}
	}
}

static void
unpack_one(lua_State *L, struct read_block *rb) {
	uint8_t type;
	uint8_t *t = (uint8_t*)rb_read(rb, sizeof(type));
	if (t==NULL) {
		invalid_stream(L, rb);
	}
	type = *t;
	push_value(L, rb, type & 0x7, type>>3);
}

int lua_unpack(lua_State *L, bool refarr) {
	if (lua_isnoneornil(L,1)) {
		return 0;
	}
	void * buffer;
	int len;
	if (lua_type(L,1) == LUA_TSTRING) {
		size_t sz;
		 buffer = (void *)lua_tolstring(L,1,&sz);
		len = (int)sz;
	} else {
		buffer = lua_touserdata(L,1);
		len = luaL_checkinteger(L,2);
	}
	if (len == 0) {
		return 0;
	}
	if (buffer == NULL) {
		return luaL_error(L, "deserialize null pointer");
	}

	lua_settop(L,1);
	struct read_block rb;
	rball_init(&rb, (char*)buffer, len, refarr);

	int i;
	for (i=0;;i++) {
		if (i%8==7) {
			luaL_checkstack(L,LUA_MINSTACK,NULL);
		}
		uint8_t type = 0;
		uint8_t *t = (uint8_t*)rb_read(&rb, sizeof(type));
		if (t==NULL)
			break;
		type = *t;
		push_value(L, &rb, type & 0x7, type>>3);
	}

	// Need not free buffer

	return lua_gettop(L) - 1;
}

