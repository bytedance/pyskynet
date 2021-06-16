#include <type_traits>
#include <string>
#include <cmath>

#include "numsky/lua-numsky.h"
#include "numsky/lua-numsky_module.h"
#include "numsky/ndarray/lua-numsky_ndarray.h"
#include "numsky/ufunc/lua-numsky_ufunc.h"

static void ufunc_call_21(lua_State*L, numsky_ufunc *ufunc, numsky_ndarray* left_arr, numsky_ndarray* right_arr) {
	auto check_oper = reinterpret_cast<numsky::generic_ufunc<2,1>::T_check_oper>(ufunc->check_oper);
	auto check_type = reinterpret_cast<numsky::generic_ufunc<2,1>::T_check_type>(ufunc->check_type);
	auto fp_oper = check_oper(L, left_arr->dtype->typechar, right_arr->dtype->typechar);
	char retypechar = check_type(L, left_arr->dtype->typechar, right_arr->dtype->typechar);
	/* step 1. build & alloc */
	std::vector<npy_intp> dimensions = numsky::ndarray_broadcasting_dimensions(L, left_arr, right_arr);
	auto new_arr = numsky::ndarray_new_alloc<true>(L, dimensions.size(), retypechar, [&](int i)->npy_intp {
			return dimensions[i];
	});
	/* step 2. broadcasting iter */
	char *new_dataptr = new_arr->dataptr;
	int itemsize = new_arr->dtype->elsize;
	numsky::ndarray_broadcasting_foreach(new_arr.get(), left_arr, right_arr, [&](numsky_nditer *left_iter, numsky_nditer *right_iter) {
		fp_oper(L, new_dataptr, left_iter->dataptr, right_iter->dataptr);
		new_dataptr += itemsize;
	});
}

template <typename T> static void ufunc_call_21(lua_State*L, numsky_ufunc *ufunc, numsky_ndarray* left_arr, T right_value) {
	auto check_oper = reinterpret_cast<numsky::generic_ufunc<2,1>::T_check_oper>(ufunc->check_oper);
	auto check_type = reinterpret_cast<numsky::generic_ufunc<2,1>::T_check_type>(ufunc->check_type);
	auto fp_oper = check_oper(L, left_arr->dtype->typechar, numsky::generic<T>::typechar);
	char retypechar = check_type(L, left_arr->dtype->typechar, numsky::generic<T>::typechar);
	/* step 1. build & alloc */
	auto new_arr_ptr = numsky::ndarray_new_alloc<true>(L, left_arr->nd, retypechar, [&](int i)->npy_intp {
			return left_arr->dimensions[i];
	});
	/* step 2. loop & assign */
	char *p_new = new_arr_ptr->dataptr;
	int itemsize = new_arr_ptr->dtype->elsize;
	numsky::ndarray_foreach(left_arr, [&](numsky_nditer* iter) {
		fp_oper(L, p_new, iter->dataptr, reinterpret_cast<char*>(&right_value));
		p_new += itemsize;
	});
}

template <typename T> static void ufunc_call_21(lua_State*L, numsky_ufunc *ufunc, T left_value, numsky_ndarray* right_arr) {
	auto check_oper = reinterpret_cast<numsky::generic_ufunc<2,1>::T_check_oper>(ufunc->check_oper);
	auto check_type = reinterpret_cast<numsky::generic_ufunc<2,1>::T_check_type>(ufunc->check_type);
	auto fp_oper = check_oper(L, numsky::generic<T>::typechar, right_arr->dtype->typechar);
	char retypechar = check_type(L, numsky::generic<T>::typechar, right_arr->dtype->typechar);
	/* step 1. build & alloc */
	auto new_arr_ptr = numsky::ndarray_new_alloc<true>(L, right_arr->nd, retypechar, [&](int i)->npy_intp {
			return right_arr->dimensions[i];
	});
	/* step 2. create iter */
	char *p_new = new_arr_ptr->dataptr;
	int itemsize = new_arr_ptr->dtype->elsize;
	numsky::ndarray_foreach(right_arr, [&](numsky_nditer* iter) {
		fp_oper(L, p_new, reinterpret_cast<char*>(&left_value), iter->dataptr);
		p_new += itemsize;
	});
}

int numsky::ufunc__call_21(lua_State *L, numsky_ufunc* ufunc, int lefti, int righti) {
	auto left_arr = luabinding::ClassUtil<numsky_ndarray>::test(L, lefti);
	auto right_arr = luabinding::ClassUtil<numsky_ndarray>::test(L, righti);
	if(left_arr!=NULL && right_arr!=NULL) {
		ufunc_call_21(L, ufunc, left_arr, right_arr);
		return 1;
	} else {
		if(left_arr != NULL) {
			int right_type = lua_type(L, righti);
			if(right_type==LUA_TNUMBER) {
				if(lua_isinteger(L, righti)) {
					ufunc_call_21(L, ufunc, left_arr, static_cast<int64_t>(lua_tointeger(L, righti)));
				} else {
					ufunc_call_21(L, ufunc, left_arr, static_cast<double>(lua_tonumber(L, righti)));
				}
				return 1;
			} else if(right_type == LUA_TBOOLEAN) {
				ufunc_call_21(L, ufunc, left_arr, lua_toboolean(L, righti) == 1);
				return 1;
			} else {
				return luaL_error(L, "numsky.ndarray can't operate with type=%s", lua_typename(L, right_type));
			}
		} else if (right_arr != NULL) {
			int left_type = lua_type(L, lefti);
			if(left_type==LUA_TNUMBER) {
				if(lua_isinteger(L, lefti)) {
					ufunc_call_21(L, ufunc, static_cast<int64_t>(lua_tointeger(L, lefti)), right_arr);
				} else {
					ufunc_call_21(L, ufunc, static_cast<double>(lua_tonumber(L, lefti)), right_arr);
				}
				return 1;
			} else if(left_type == LUA_TBOOLEAN) {
				ufunc_call_21(L, ufunc, lua_toboolean(L, lefti) == 1, right_arr);
				return 1;
			} else {
				return luaL_error(L, "numsky.ndarray can't operate with type=%s", lua_typename(L, left_type));
			}
		} else {
			auto check_oper = reinterpret_cast<numsky::generic_ufunc<2,1>::T_check_oper>(ufunc->check_oper);
			auto check_type = reinterpret_cast<numsky::generic_ufunc<2,1>::T_check_type>(ufunc->check_type);
			auto check_data = [] (lua_State* L, int stacki, char *outptr) -> char {
				char nType = lua_type(L, stacki);
				if(nType==LUA_TNUMBER) {
					if(lua_isinteger(L, stacki)) {
						numsky::dataptr_cast<int64_t>(outptr) = lua_tointeger(L, stacki);
						return numsky::generic<int64_t>::typechar;
					} else {
						numsky::dataptr_cast<double>(outptr) = lua_tonumber(L, stacki);
						return numsky::generic<double>::typechar;
					}
				} else if(nType == LUA_TBOOLEAN) {
					numsky::dataptr_cast<bool>(outptr) = (lua_toboolean(L, stacki) == 1);
					return numsky::generic<bool>::typechar;
				} else {
					luaL_error(L, "numsky.ndarray can't operate with type=%s", lua_typename(L, nType));
					return '\0';
				}
			};
			char left_data[16];
			char right_data[16];
			char left_typechar = check_data(L, lefti, left_data);
			char right_typechar = check_data(L, righti, right_data);
			auto fp_oper = check_oper(L, left_typechar, right_typechar);
			char retypechar = check_type(L, left_typechar, right_typechar);
			char result_data[16];
			fp_oper(L, result_data, left_data, right_data);
			lnumsky_template_fp(L, retypechar, numsky::dataptr_push)(L, result_data);
			return 1;
		}
	}
	return 1;
}

template <typename T> static void ufunc_call_11(lua_State*L, numsky_ufunc *ufunc, T arg_value) {
	auto check_oper = reinterpret_cast<numsky::generic_ufunc<1,1>::T_check_oper>(ufunc->check_oper);
	auto check_type = reinterpret_cast<numsky::generic_ufunc<1,1>::T_check_type>(ufunc->check_type);
	auto fp_oper = check_oper(L, numsky::generic<T>::typechar);
	char retypechar = check_type(L, numsky::generic<T>::typechar);
	char re[16];
	fp_oper(L, re, reinterpret_cast<char*>(&arg_value));
	lnumsky_template_fp(L, retypechar, numsky::dataptr_push)(L, re);
}

int numsky::ufunc__call_11(lua_State *L, numsky_ufunc* ufunc, int argi) {
	auto arr = luabinding::ClassUtil<numsky_ndarray>::test(L, argi);
	if(arr!=NULL) {
		auto check_oper = reinterpret_cast<numsky::generic_ufunc<1,1>::T_check_oper>(ufunc->check_oper);
		auto check_type = reinterpret_cast<numsky::generic_ufunc<1,1>::T_check_type>(ufunc->check_type);
		auto fp_oper = check_oper(L, arr->dtype->typechar);
		char retypechar = check_type(L, arr->dtype->typechar);
		/* step 1. build & alloc */
		auto new_arr_ptr = numsky::ndarray_new_alloc<true>(L, arr->nd, retypechar, [&](int i)->npy_intp {
			return arr->dimensions[i];
		});
		/* step 2. iter & assign */
		char *p_new = new_arr_ptr->dataptr;
		int itemsize = new_arr_ptr->dtype->elsize;
		numsky::ndarray_foreach(arr, [&](numsky_nditer* iter){
			fp_oper(L, p_new, iter->dataptr);
			p_new += itemsize;
		});
		return 1;
	} else {
		int arg_type = lua_type(L, argi);
		if(arg_type ==LUA_TNUMBER) {
			if(lua_isinteger(L, argi)) {
				ufunc_call_11(L, ufunc, static_cast<int64_t>(lua_tointeger(L, argi)));
			} else {
				ufunc_call_11(L, ufunc, static_cast<double>(lua_tonumber(L, argi)));
			}
			return 1;
		} else if(arg_type == LUA_TBOOLEAN) {
			ufunc_call_11(L, ufunc, lua_toboolean(L, argi)==1);
			return 1;
		} else {
			return luaL_error(L, "numsky.ndarray can't operate with type=%s", lua_typename(L, arg_type));
		}
	}
}
