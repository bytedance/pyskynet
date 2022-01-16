/*
	modify from https://github.com/cloudwu/lua-serialize
 */

#include "foreign_seri/read_block.h"

void rb_init(struct read_block * rb, char * buffer, int64_t size, int mode) {
	rb->buffer = buffer;
	rb->len = size;
	rb->ptr = 0;
	rb->mode = mode;
	if(mode != MODE_LUA) {
		rb->nextbase = *((intptr_t*)rb_read(rb, sizeof(intptr_t)));
	}
}

void *rb_read(struct read_block *rb, int64_t sz) {
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

bool rb_get_integer(struct read_block *rb, int cookie, lua_Integer *pout) {
	switch (cookie) {
	case TYPE_NUMBER_ZERO:
		*pout = 0;
		return true;
	case TYPE_NUMBER_BYTE: {
		uint8_t *pn = (uint8_t *)rb_read(rb,sizeof(uint8_t));
		if (pn == NULL)
			return false;
		*pout = *pn;
		return true;
	}
	case TYPE_NUMBER_WORD: {
		uint16_t *pn = (uint16_t *)rb_read(rb,sizeof(uint16_t));
		if (pn == NULL)
			return false;
		*pout = *pn;
		return true;
	}
	case TYPE_NUMBER_DWORD: {
		int32_t *pn = (int32_t *)rb_read(rb,sizeof(int32_t));
		if (pn == NULL)
			return false;
		*pout = *pn;
		return true;
	}
	case TYPE_NUMBER_QWORD: {
		int64_t *pn = (int64_t *)rb_read(rb,sizeof(int64_t));
		if (pn == NULL)
			return false;
		*pout = *pn;
		return true;
	}
	default:
		return false;
	}
}

bool rb_get_real(struct read_block *rb, double * pout) {
	double * pn = (double *)rb_read(rb,sizeof(double));
	if (pn == NULL)
		return false;
	*pout = *pn;
	return true;
}

bool rb_get_pointer(struct read_block *rb, void ** pout) {
	void ** v = (void **)rb_read(rb,sizeof(void*));
	if (v == NULL) {
		return false;
	}
	*pout = *v;
	return true;
}

char* rb_get_string(struct read_block *rb, uint32_t ahead, size_t *psize) {
	int type = ahead & 0x7;
	int cookie = ahead >> 3;
	if(type == TYPE_SHORT_STRING){
		*psize = cookie;
		return (char*) rb_read(rb, cookie);
	} else if(cookie == 2) {
		uint16_t *plen = (uint16_t *)rb_read(rb, 2);
		if (plen == NULL) {
			return NULL;
		}
		*psize = *plen;
		return (char*) rb_read(rb, *plen);
	} else if (cookie == 4){
		uint32_t *plen = (uint32_t *)rb_read(rb, 4);
		if (plen == NULL) {
			return NULL;
		}
		*psize = *plen;
		return (char*) rb_read(rb, *plen);
	} else {
		return NULL;
	}
}

static void lrb_unpack_one(lua_State *L, struct read_block *rb);

static void lrb_unpack_table(lua_State *L, struct read_block *rb, lua_Integer array_size) {
	if (array_size == MAX_COOKIE-1) {
		uint8_t *t = (uint8_t *)rb_read(rb, sizeof(uint8_t));
		if (t==NULL) {
			invalid_stream(L,rb);
		}
		uint8_t type = *t;
		int cookie = type >> 3;
		if ((type & 7) != TYPE_NUMBER || cookie == TYPE_NUMBER_REAL) {
			invalid_stream(L,rb);
		}
		if(!rb_get_integer(rb, cookie, &array_size)) {
			invalid_stream(L,rb);
		}
	}
	luaL_checkstack(L,LUA_MINSTACK,NULL);
	lua_createtable(L,array_size,0);
	for (int i=1;i<=array_size;i++) {
		lrb_unpack_one(L,rb);
		lua_rawseti(L,-2,i);
	}
	for (;;) {
		lrb_unpack_one(L,rb);
		if (lua_isnil(L,-1)) {
			lua_pop(L,1);
			return;
		}
		lrb_unpack_one(L,rb);
		lua_rawset(L,-3);
	}
}

struct numsky_ndarray* unpack_ns_arr(struct read_block *rb, int nd) {
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
	if(rb->mode == MODE_FOREIGN_REF) {
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
		if(rb->nextbase != rb->ptr) {
			numsky_ndarray_destroy(arr);
			return NULL;
		}
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
		rb->nextbase = *((intptr_t*)rb_read(rb, sizeof(intptr_t)));
	} else if (rb->mode == MODE_FOREIGN_REMOTE){
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
	} else {
		numsky_ndarray_destroy(arr);
		return NULL;
	}
	numsky_ndarray_refdata(arr, foreign_base, dataptr);
	return arr;
}

static void
push_value(lua_State *L, struct read_block *rb, uint8_t ahead) {
	int type = ahead & 0x7;
	int cookie = ahead >> 3;
	switch(type) {
	case TYPE_NIL:
		lua_pushnil(L);
		break;
	case TYPE_BOOLEAN:
		lua_pushboolean(L,cookie);
		break;
	case TYPE_NUMBER:
		if (cookie == TYPE_NUMBER_REAL) {
			double value;
			if(rb_get_real(rb, &value)) {
				lua_pushnumber(L, value);
			} else {
				invalid_stream(L,rb);
			}
		} else {
			lua_Integer value;
			if(rb_get_integer(rb, cookie, &value)) {
				lua_pushinteger(L, value);
			} else {
				invalid_stream(L,rb);
			}
		}
		break;
	case TYPE_USERDATA: {
		void *value;
		if(rb_get_pointer(rb, &value)) {
			lua_pushlightuserdata(L, value);
		} else {
			invalid_stream(L,rb);
		}
		break;
	}
	case TYPE_SHORT_STRING:
	case TYPE_LONG_STRING: {
		size_t sz;
		char *p = rb_get_string(rb,ahead,&sz);
		if(p!=NULL) {
			lua_pushlstring(L,p,sz);
		} else {
			invalid_stream(L,rb);
		}
		break;
	}
	case TYPE_TABLE: {
		lrb_unpack_table(L,rb,cookie);
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
lrb_unpack_one(lua_State *L, struct read_block *rb) {
	uint8_t *t = (uint8_t*)rb_read(rb, 1);
	if (t==NULL) {
		invalid_stream(L, rb);
	}
	uint8_t ahead = *t;
	push_value(L, rb, ahead);
}

int mode_unpack(lua_State *L, int mode) {
	void * buffer;
	int64_t len;
	int type1 = lua_type(L, 1);
	if (type1 == LUA_TSTRING) {
		size_t sz;
		buffer = (void *)lua_tolstring(L,1,&sz);
		len = sz;
	} else if(type1 == LUA_TLIGHTUSERDATA) {
		buffer = lua_touserdata(L,1);
		len = luaL_checkinteger(L,2);
	} else {
		return luaL_error(L, "deserialize must take a string or lightuserdata & integer");
	}
	if (len == 0) {
		return 0;
	}
	if (buffer == NULL) {
		return luaL_error(L, "deserialize null pointer");
	}

	lua_settop(L,1);
	struct read_block rb;
	rb_init(&rb, (char*)buffer, len, mode);

	for (int i=0;;i++) {
		if (i%8==7) {
			luaL_checkstack(L,LUA_MINSTACK,NULL);
		}
		uint8_t *t = (uint8_t*)rb_read(&rb, 1);
		if (t==NULL)
			break;
		uint8_t ahead = *t;
		push_value(L, &rb, ahead);
	}

	// Need not free buffer

	return lua_gettop(L) - 1;
}

