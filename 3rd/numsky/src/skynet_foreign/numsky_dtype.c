#include "skynet_foreign/hook_skynet_py.h"
#include "skynet_foreign/numsky.h"
#include <stdio.h>
#include <stdbool.h>


#define MAKE_DATAPTR_FUNC(T, lua_push, lua_check) \
static void dataptr_push_##T(lua_State*L, char* dataptr) { \
	lua_push(L, ((T*)(dataptr))[0]); \
} \
static void dataptr_check_##T(lua_State*L, char* dataptr, int stacki) { \
	((T*)(dataptr))[0] = lua_check(L, stacki); \
} \

MAKE_DATAPTR_FUNC(bool, lua_pushboolean, lua_toboolean);
MAKE_DATAPTR_FUNC(int8_t, lua_pushinteger, luaL_checkinteger);
MAKE_DATAPTR_FUNC(uint8_t, lua_pushinteger, luaL_checkinteger);
MAKE_DATAPTR_FUNC(int16_t, lua_pushinteger, luaL_checkinteger);
MAKE_DATAPTR_FUNC(uint16_t, lua_pushinteger, luaL_checkinteger);
MAKE_DATAPTR_FUNC(int32_t, lua_pushinteger, luaL_checkinteger);
MAKE_DATAPTR_FUNC(uint32_t, lua_pushinteger, luaL_checkinteger);
MAKE_DATAPTR_FUNC(int64_t, lua_pushinteger, luaL_checkinteger);
MAKE_DATAPTR_FUNC(float, lua_pushnumber, luaL_checknumber);
MAKE_DATAPTR_FUNC(double, lua_pushnumber, luaL_checknumber);

static struct numsky_dtype numsky_dtype_bool = {
    .type_num = NPY_BOOL,
    .typechar = '?',
	.kind = 'b',
    .elsize = 1,
    .name="bool",
	.dataptr_push=dataptr_push_bool,
	.dataptr_check=dataptr_check_bool,
};

static struct numsky_dtype numsky_dtype_int8 = {
    .type_num = NPY_INT8,
    .typechar = 'b',
	.kind = 'i',
    .elsize = 1,
    .name="int8",
	.dataptr_push=dataptr_push_int8_t,
	.dataptr_check=dataptr_check_int8_t,
};

static struct numsky_dtype numsky_dtype_uint8 = {
    .type_num = NPY_UINT8,
    .typechar = 'B',
	.kind = 'u',
    .elsize = 1,
    .name="uint8",
	.dataptr_push=dataptr_push_uint8_t,
	.dataptr_check=dataptr_check_uint8_t,
};

static struct numsky_dtype numsky_dtype_int16 = {
    .type_num = NPY_INT16,
    .typechar = 'h',
	.kind = 'i',
    .elsize = 2,
    .name="int16",
	.dataptr_push=dataptr_push_int16_t,
	.dataptr_check=dataptr_check_int16_t,
};

static struct numsky_dtype numsky_dtype_uint16 = {
    .type_num = NPY_UINT16,
    .typechar = 'H',
	.kind = 'u',
    .elsize = 2,
    .name="uint16",
	.dataptr_push=dataptr_push_uint16_t,
	.dataptr_check=dataptr_check_uint16_t,
};

static struct numsky_dtype numsky_dtype_int32 = {
    .type_num = NPY_INT32,
    .typechar = 'i',
	.kind = 'i',
    .elsize = 4,
    .name="int32",
	.dataptr_push=dataptr_push_int32_t,
	.dataptr_check=dataptr_check_int32_t,
};

static struct numsky_dtype numsky_dtype_uint32 = {
    .type_num = NPY_UINT32,
    .typechar = 'I',
	.kind = 'u',
    .elsize = 4,
    .name="uint32",
	.dataptr_push=dataptr_push_uint32_t,
	.dataptr_check=dataptr_check_uint32_t,
};

static struct numsky_dtype numsky_dtype_int64 = {
    .type_num = NPY_INT64,
    .typechar = 'l',
	.kind = 'i',
    .elsize = 8,
    .name="int64",
	.dataptr_push=dataptr_push_int64_t,
	.dataptr_check=dataptr_check_int64_t,
};

/*static struct numsky_dtype numsky_dtype_uint64 = {
    .type_num = NPY_UINT64,
    .typechar = 'L',
	.kind = 'u',
    .elsize = 8,
    .name="uint64",
};*/

static struct numsky_dtype numsky_dtype_float32 = {
    .type_num = NPY_FLOAT32,
    .typechar = 'f',
	.kind = 'f',
    .elsize = 4,
    .name="float32",
	.dataptr_push=dataptr_push_float,
	.dataptr_check=dataptr_check_float,
};

static struct numsky_dtype numsky_dtype_float64 = {
    .type_num = NPY_FLOAT64,
    .typechar = 'd',
	.kind = 'f',
    .elsize = 8,
    .name="float64",
	.dataptr_push=dataptr_push_double,
	.dataptr_check=dataptr_check_double,
};

char NS_DTYPE_CHARS[10] = {'?','b','B','h','H','i','I','l','f','d'};

struct numsky_dtype *numsky_get_dtype_by_char(char typechar) {
    switch(typechar){
    case '?': return &numsky_dtype_bool;
    case 'b': return &numsky_dtype_int8;
    case 'B': return &numsky_dtype_uint8;
    case 'h': return &numsky_dtype_int16;
    case 'H': return &numsky_dtype_uint16;
    case 'i': return &numsky_dtype_int32;
    case 'I': return &numsky_dtype_uint32;
    case 'l': return &numsky_dtype_int64;
    //case 'L': return &numsky_dtype_uint64;
    case 'f': return &numsky_dtype_float32;
    case 'd': return &numsky_dtype_float64;
    default: {
                 printf("ERROR!!!!!, get_dtype_by_char unexcept branch\n");
                 return NULL;
             }
    }
}
