
#include "lua-seri.c"

#include "skynet_foreign/skynet_foreign.h"
#include "skynet_foreign/numsky.h"

#define TYPE_FOREIGN_USERDATA 7

#define MODE_LUA 0
#define MODE_FOREIGN 1
#define MODE_FOREIGN_REMOTE 2

struct foreign_write_block {
    struct write_block wb;
    int mode;
};

struct foreign_read_block {
    struct read_block rb;
    int mode;
};

inline static struct write_block* wb_cast(struct foreign_write_block * wb) {
    return (struct write_block*) wb;
}

inline static struct read_block* rb_cast(struct foreign_read_block* rb) {
    return (struct read_block*) rb;
}

inline static void foreign_wb_init(struct foreign_write_block *wb , struct block *b, int mode) {
    wb_init(wb_cast(wb), b);
    wb->mode = mode;
}

inline static void foreign_rball_init(struct foreign_read_block * rb, char * buffer, int size, int mode) {
    rball_init(rb_cast(rb), buffer, size);
    rb->mode = mode;
}

inline static bool foreign_rb_uint(struct foreign_read_block* rb, npy_intp *value) {
	npy_intp result = 0;
	for (uint32_t shift = 0; shift <= 63; shift += 7) {
		uint8_t *p_byte = rb_read(rb_cast(rb), 1);
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

inline static void foreign_wb_uint(struct foreign_write_block* wb, npy_intp v) {
	static const int B = 128;
	uint8_t data = v | B;
	while (v >= B) {
		data = v | B;
		wb_push(wb_cast(wb), &data, 1);
		v >>= 7;
	}
	data = (uint8_t)v;
	wb_push(wb_cast(wb), &data, 1);
}



/*************
 * pack apis *
 *************/

/*used by foreign serialize, pass as a function pointer*/
static void foreign_wb_write(struct foreign_write_block *b, const void *buf, int sz) {
    wb_push(wb_cast(b), buf, sz);
}
static void *foreign_rb_read(struct foreign_read_block *rb, int sz) {
    return rb_read(rb_cast(rb), sz);
}

static inline void wb_foreign(struct foreign_write_block *wb, struct numsky_ndarray* arr_obj) {
	// 1. nd & type
	uint8_t n = COMBINE_TYPE(TYPE_FOREIGN_USERDATA, arr_obj->nd);
	foreign_wb_write(wb, &n, 1);
	struct numsky_dtype *dtype = arr_obj->dtype;
	// 2. typechar
	foreign_wb_write(wb, &(dtype->typechar), 1);
	// 3. dimension
	for(int i=0;i<arr_obj->nd;i++) {
		foreign_wb_uint(wb, arr_obj->dimensions[i]);
	}
	if (wb->mode == MODE_FOREIGN) {
		// 4. strides
		foreign_wb_write(wb, arr_obj->strides, sizeof(npy_intp)*arr_obj->nd);
		// 5. data
		skynet_foreign_incref(arr_obj->foreign_base);
		foreign_wb_write(wb, &(arr_obj->foreign_base), sizeof(arr_obj->foreign_base));
		foreign_wb_write(wb, &(arr_obj->dataptr), sizeof(arr_obj->dataptr));
	} else if(wb->mode == MODE_FOREIGN_REMOTE){
		// 4. data
		struct numsky_nditer * iter = numsky_nditer_create(arr_obj);
		for(int i=0;i<iter->ao->count;numsky_nditer_next(iter), i++) {
			foreign_wb_write(wb, iter->dataptr, dtype->elsize);
		}
		numsky_nditer_destroy(iter);
	}
}



/* override pack_one */
static void foreign_pack_one(lua_State *L, struct foreign_write_block *b, int index, int depth);

/* override wb_table_array */
static int foreign_wb_table_array(lua_State *L, struct foreign_write_block * wb, int index, int depth) {
	int array_size = lua_rawlen(L,index);
	if (array_size >= MAX_COOKIE-1) {
		uint8_t n = COMBINE_TYPE(TYPE_TABLE, MAX_COOKIE-1);
		foreign_wb_write(wb, &n, 1);
		wb_integer(wb_cast(wb), array_size);
	} else {
		uint8_t n = COMBINE_TYPE(TYPE_TABLE, array_size);
		foreign_wb_write(wb, &n, 1);
	}

	int i;
	for (i=1;i<=array_size;i++) {
		lua_rawgeti(L,index,i);
		foreign_pack_one(L, wb, -1, depth);
		lua_pop(L,1);
	}

	return array_size;
}

/* override wb_table_hash */
static void foreign_wb_table_hash(lua_State *L, struct foreign_write_block * wb, int index, int depth, int array_size) {
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
		foreign_pack_one(L,wb,-2,depth);
		foreign_pack_one(L,wb,-1,depth);
		lua_pop(L, 1);
	}
	wb_nil(wb_cast(wb));
}

/* override wb_table */
static void foreign_wb_table(lua_State *L, struct foreign_write_block *wb, int index, int depth) {
	luaL_checkstack(L, LUA_MINSTACK, NULL);
	if (index < 0) {
		index = lua_gettop(L) + index + 1;
	}
    int array_size = foreign_wb_table_array(L, wb, index, depth);
    foreign_wb_table_hash(L, wb, index, depth, array_size);
}

/* override pack_one */
static void foreign_pack_one(lua_State *L, struct foreign_write_block *wb, int index, int depth) {
	if (depth > MAX_DEPTH) {
		wb_free(wb_cast(wb));
		luaL_error(L, "serialize can't pack too depth table");
        return ;
	}
	int type = lua_type(L,index);
    switch(type) {
        case LUA_TUSERDATA: {
            struct numsky_ndarray* arr = *(struct numsky_ndarray**) (luaL_checkudata(L, index, NS_ARR_METANAME));
			if(arr->nd >= MAX_COOKIE) {
				luaL_error(L, "numsky.ndarray's nd must be <= 31");
			}
			if (wb->mode==MODE_FOREIGN) {
				if(arr->foreign_base == NULL) {
					luaL_error(L, "foreign -base can't be null");
					return ;
				}
				wb_foreign(wb, arr);
			} else if (wb->mode == MODE_FOREIGN_REMOTE) {
				wb_foreign(wb, arr);
			} else {
				luaL_error(L, "[ERROR]wb_foreign exception");
			}
            break;
        }
        case LUA_TTABLE: {
            if (index < 0) {
                index = lua_gettop(L) + index + 1;
            }
            foreign_wb_table(L, wb, index, depth+1);
            break;
        }
        default: {
            pack_one(L, wb_cast(wb), index, depth);
        }
    }
}

/* override pack_from */
static void foreign_pack_from(lua_State *L, struct foreign_write_block *b, int from) {
	int n = lua_gettop(L) - from;
	int i;
	for (i=1;i<=n;i++) {
		foreign_pack_one(L, b , from + i, 0);
	}
}

int foreign_pack(lua_State *L, int mode) {
	struct block temp;
	temp.next = NULL;
	struct foreign_write_block wb;
	foreign_wb_init(&wb, &temp, mode);
	foreign_pack_from(L,&wb,0);
	assert(wb.wb.head == &temp);
	seri(L, &temp, wb.wb.len);

	wb_free(wb_cast(&wb));

    return 2;
}


/***************
 * unpack apis *
 ***************/

static struct numsky_ndarray*
unpack_ns_arr(struct foreign_read_block *rb, int nd) {
	// 1. get dtype
	char * p_typechar = (char *)foreign_rb_read(rb, 1);
	if(p_typechar == NULL){
		return NULL;
	}
	// 2. init from dimensions
	struct numsky_ndarray *arr = numsky_ndarray_precreate(nd, p_typechar[0]);
	for(int i=0;i<nd;i++){
		bool ok = foreign_rb_uint(rb, &arr->dimensions[i]);
		if(!ok) {
			numsky_ndarray_destroy(arr);
			return NULL;
		}
	}
	// 3. build
	struct skynet_foreign *foreign_base;
	char *dataptr;
	if(rb->mode==MODE_FOREIGN) {
		numsky_ndarray_autocount(arr);
		npy_intp *strides = foreign_rb_read(rb, sizeof(npy_intp)*nd);
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
		void **v = (void **)foreign_rb_read(rb,sizeof(foreign_base));
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
		v = (void **)foreign_rb_read(rb,sizeof(dataptr));
		if (v == NULL) {
			numsky_ndarray_destroy(arr);
			return NULL;
		}
		memcpy(&dataptr, v, sizeof(dataptr));
	} else if(rb->mode==MODE_FOREIGN_REMOTE) {
		numsky_ndarray_autostridecount(arr);
		// 4. alloc foreign_base
		size_t datasize = arr->count*arr->dtype->elsize;
		// 1) read & copy
		char * pdata = foreign_rb_read(rb, datasize);
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

/* override unpack_one */
static void foreign_unpack_one(lua_State *L, struct foreign_read_block *rb);

/* override unpack_table */
static void foreign_unpack_table(lua_State *L, struct foreign_read_block *rb, int array_size) {
	if (array_size == MAX_COOKIE-1) {
		uint8_t type;
		uint8_t *t = foreign_rb_read(rb, sizeof(type));
		if (t==NULL) {
			invalid_stream(L,rb_cast(rb));
		}
		type = *t;
		int cookie = type >> 3;
		if ((type & 7) != TYPE_NUMBER || cookie == TYPE_NUMBER_REAL) {
			invalid_stream(L,rb_cast(rb));
		}
		array_size = get_integer(L,rb_cast(rb),cookie);
	}
	luaL_checkstack(L,LUA_MINSTACK,NULL);
	lua_createtable(L,array_size,0);
	int i;
	for (i=1;i<=array_size;i++) {
		foreign_unpack_one(L,rb);
		lua_rawseti(L,-2,i);
	}
	for (;;) {
		foreign_unpack_one(L,rb);
		if (lua_isnil(L,-1)) {
			lua_pop(L,1);
			return;
		}
		foreign_unpack_one(L,rb);
		lua_rawset(L,-3);
	}
}

/* override push_value */
static void foreign_push_value(lua_State *L, struct foreign_read_block *rb, int type, int cookie) {
	switch(type) {
        case TYPE_FOREIGN_USERDATA: {
            struct numsky_ndarray *arr = unpack_ns_arr(rb, cookie);
			if(arr==NULL) {
				invalid_stream(L, rb_cast(rb));
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
        case TYPE_TABLE: {
            foreign_unpack_table(L,rb,cookie);
            break;
        }
        default: {
            // other data type don't need recursive, just push_value
            push_value(L, rb_cast(rb), type, cookie);
            break;
         }
    }
}

static void foreign_unpack_one(lua_State *L, struct foreign_read_block *rb) {
	uint8_t type;
	uint8_t *t = foreign_rb_read(rb, sizeof(type));
	if (t==NULL) {
		invalid_stream(L, rb_cast(rb));
	}
	type = *t;
	foreign_push_value(L, rb, type & 0x7, type>>3);
}

int foreign_unpack(lua_State *L, int mode){
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
	struct foreign_read_block rb;
	foreign_rball_init(&rb, buffer, len, mode);

	int i;
	for (i=0;;i++) {
		if (i%8==7) {
			luaL_checkstack(L,LUA_MINSTACK,NULL);
		}
		uint8_t type = 0;
		uint8_t *t = foreign_rb_read(&rb, sizeof(type));
		if (t==NULL)
			break;
		type = *t;
		foreign_push_value(L, &rb, type & 0x7, type>>3);
	}

	// Need not free buffer

	return lua_gettop(L) - 1;
}

static int lluapack(lua_State *L) {
	return foreign_pack(L, MODE_LUA);
}

static int lluaunpack(lua_State *L) {
	return foreign_unpack(L, MODE_LUA);
}

static int lpack(lua_State *L) {
	return foreign_pack(L, MODE_FOREIGN);
}

static int lunpack(lua_State *L) {
	return foreign_unpack(L, MODE_FOREIGN);
}

static int lremotepack(lua_State *L) {
	return foreign_pack(L, MODE_FOREIGN_REMOTE);
}

static int lremoteunpack(lua_State *L) {
	return foreign_unpack(L, MODE_FOREIGN_REMOTE);
}

static const struct luaL_Reg l_methods[] = {
    { "luapack" , lluapack },
    { "luaunpack", lluaunpack },
    { "pack", lpack },
    { "unpack" , lunpack },
    { "remotepack", lremotepack },
    { "remoteunpack", lremoteunpack },
    { NULL,  NULL },
};

LUAMOD_API int
luaopen_pyskynet_foreign_seri(lua_State *L) {
	luaL_checkversion(L);

	luaL_newlib(L, l_methods);
    return 1;
}
