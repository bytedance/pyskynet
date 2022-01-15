
#pragma once

extern "C" {
#include "lua.h"
#include "lauxlib.h"
}
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include <memory>
#include <string>

extern "C" {
#include "skynet_foreign/numsky.h"
}

#define NUMSKY_TUPLE_META "numsky.tuple"
#include "lua-binding.h"

struct numsky_ufunc {
	int nin;
	int nout;
	void *check_type; // return type
	void *check_oper; // return function for oper
	void *check_init; // return function for reduce init
	const char *name;
	int ufunc_num;
};

#define lnumsky_template_fp(L, typechar, tfunc) ([] (lua_State* _L, char tc) -> decltype(tfunc<bool>)* { \
decltype(tfunc<bool>) * name = NULL;\
switch(tc) { \
	case '?': { name = tfunc<bool>; break; } \
	case 'b': { name = tfunc<int8_t>; break; } \
	case 'B': { name = tfunc<uint8_t>; break; } \
	case 'h': { name = tfunc<int16_t>; break; } \
	case 'H': { name = tfunc<uint16_t>; break; } \
	case 'i': { name = tfunc<int32_t>; break; } \
	case 'I': { name = tfunc<uint32_t>; break; } \
	case 'l': { name = tfunc<int64_t>; break; } \
	case 'L': { luaL_error(_L, "uint64 not support"); break; } \
	case 'f': { name = tfunc<float>; break; } \
	case 'd': { name = tfunc<double>; break; } \
	default: { luaL_error(_L, "ERROR!!!!!, dtype_pushdata unexcept branch\n"); break; } \
} return name;})(L, typechar)

namespace luabinding {

template <> struct ClassTypeVariable<numsky_slice> {
   using ispointer = std::false_type;
};

} // namespace luabinding

namespace numsky {

// alloc in lua stack or in memory, only alloc nd, no shape, no stride, no count, no data
template <bool InLua> inline std::unique_ptr<numsky_ndarray, void(*)(numsky_ndarray*)> ndarray_new_preinit(lua_State* L, int nd, char typechar) {
	numsky_ndarray *arr = numsky_ndarray_precreate(nd, typechar);
	if(InLua) {
		luabinding::ClassUtil<numsky_ndarray>::newwrap(L, arr);
		std::unique_ptr<numsky_ndarray, void (*)(numsky_ndarray*)> ptr(arr, [](numsky_ndarray*){});
		return ptr;
	} else {
		std::unique_ptr<numsky_ndarray, void (*)(numsky_ndarray*)> ptr(arr, numsky_ndarray_destroy);
		return ptr;
	}
}
template <bool InLua> inline std::unique_ptr<numsky_ndarray, void(*)(numsky_ndarray*)> ndarray_new_alloc(lua_State* L, int nd, char typechar, const std::function<npy_intp(int i)> & func) {
	auto arr_ptr = ndarray_new_preinit<InLua>(L, nd, typechar);
	auto arr = arr_ptr.get();
	for(int i=0;i<nd;i++) {
		arr->dimensions[i] = func(i);
	}
	numsky_ndarray_autostridecountalloc(arr);
	return arr_ptr;
}
// cast a inmemory arr to inlua arr
inline void ndarray_mem2lua(lua_State* L, std::unique_ptr<numsky_ndarray, void(*)(numsky_ndarray*)> & arr_ptr){
	luabinding::ClassUtil<numsky_ndarray>::newwrap(L, arr_ptr.release());
}
template <typename T> inline T & dataptr_cast(char *dataptr) {
   T* ptr = reinterpret_cast<T*>(dataptr);
   return ptr[0];
}
template <typename T> inline void dataptr_copy(char *dst, char* src) {
	dataptr_cast<T>(dst) = numsky::dataptr_cast<T>(src);
}
static inline std::string shape_str(numsky_ndarray* arr) {
	std::string buf = "(";
	for(int i=0;i<arr->nd;i++){
		buf += std::to_string(arr->dimensions[i]);
		buf += ',';
	}
	buf += ')';
	return buf;
}

template <typename T> static inline void new_tuple(lua_State*L, int size, T* data) {
	lua_createtable(L, size, 0);
	luaL_setmetatable(L, NUMSKY_TUPLE_META);
	for (int i=1;i<=size;i++) {
		lua_pushinteger(L, data[i-1]);
		lua_seti(L, -2, i);
	}
}

template <typename T> struct generic;
template <> struct generic<bool> {
	static const char typechar = '?';
	static inline bool check(lua_State*L, int stacki) {
		return lua_toboolean(L, stacki);
	}
	static inline void push(lua_State*L, bool data){
		lua_pushboolean(L, data);
	}
};
template <> struct generic<int8_t> {
	static const char typechar = 'b';
	static inline int8_t check(lua_State*L, int stacki) {
		return luaL_checkinteger(L, stacki);
	}
	static inline void push(lua_State*L, int8_t data){
		lua_pushinteger(L, data);
	}
};
template <> struct generic<uint8_t> {
	static const char typechar = 'B';
	static inline uint8_t check(lua_State*L, int stacki) {
		return luaL_checkinteger(L, stacki);
	}
	static inline void push(lua_State*L, uint8_t data){
		lua_pushinteger(L, data);
	}
};
template <> struct generic<int16_t> {
	static const char typechar = 'h';
	static inline int16_t check(lua_State*L, int stacki) {
		return luaL_checkinteger(L, stacki);
	}
	static inline void push(lua_State*L, int16_t data){
		lua_pushinteger(L, data);
	}
};
template <> struct generic<uint16_t> {
	static const char typechar = 'H';
	static inline uint16_t check(lua_State*L, int stacki) {
		return luaL_checkinteger(L, stacki);
	}
	static inline void push(lua_State*L, uint16_t data){
		lua_pushinteger(L, data);
	}
};
template <> struct generic<int32_t> {
	static const char typechar = 'i';
	static inline int32_t check(lua_State*L, int stacki) {
		return luaL_checkinteger(L, stacki);
	}
	static inline void push(lua_State*L, int32_t data){
		lua_pushinteger(L, data);
	}
};
template <> struct generic<uint32_t> {
	static const char typechar = 'I';
	static inline uint32_t check(lua_State*L, int stacki) {
		return luaL_checkinteger(L, stacki);
	}
	static inline void push(lua_State*L, uint32_t data){
		lua_pushinteger(L, data);
	}
};
template <> struct generic<int64_t> {
	static const char typechar = 'l';
	static inline int64_t check(lua_State*L, int stacki) {
		return luaL_checkinteger(L, stacki);
	}
	static inline void push(lua_State*L, int64_t data){
		lua_pushinteger(L, data);
	}
};
template <> struct generic<float> {
	static const char typechar = 'f';
	static inline float check(lua_State*L, int stacki) {
		return luaL_checknumber(L, stacki);
	}
	static inline void push(lua_State*L, float data){
		lua_pushnumber(L, data);
	}
};
template <> struct generic<double> {
	static const char typechar = 'd';
	static inline double check(lua_State*L, int stacki) {
		return luaL_checknumber(L, stacki);
	}
	static inline void push(lua_State*L, double data){
		lua_pushnumber(L, data);
	}
};

template <typename T> inline void dataptr_check(lua_State*L, char *dataptr, int stacki) {
	numsky::dataptr_cast<T>(dataptr) = numsky::generic<T>::check(L, stacki);
}

template <typename T> inline void dataptr_push(lua_State*L, char *dataptr) {
	numsky::generic<T>::push(L, numsky::dataptr_cast<T>(dataptr));
}

template <typename T> inline double dataptr_to_float64(char *dataptr) {
	return numsky::dataptr_cast<T>(dataptr);
}

template <typename T> inline void dataptr_from_float64(char *dataptr, double value) {
	numsky::dataptr_cast<T>(dataptr) = value;
}

template <typename T> inline int64_t dataptr_to_int64(char *dataptr) {
	return numsky::dataptr_cast<T>(dataptr);
}

template <typename T> inline void dataptr_from_int64(char *dataptr, int64_t value) {
	numsky::dataptr_cast<T>(dataptr) = value;
}

} // namespace numsky

void lnumsky_tuple_bind_lib(luabinding::Module_& m);
