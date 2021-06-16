#include <type_traits>
#include <string>
#include <cmath>

#include "numsky/lua-numsky.h"
#include "numsky/lua-numsky_module.h"
#include "numsky/ndarray/lua-numsky_ndarray.h"
#include "numsky/ufunc/lua-numsky_ufunc.h"

static int numsky_ufunc_reduce(lua_State*L, const numsky_ufunc *ufunc, numsky_ndarray *arr, int axis_stacki) {
	if(ufunc->nin!=2 || ufunc->nout!=1) {
		luaL_error(L, "only binary oper ufunc can reduce");
	}
	auto check_type = reinterpret_cast<numsky::generic_ufunc<2,1>::T_check_type>(ufunc->check_type);
	auto check_oper = reinterpret_cast<numsky::generic_ufunc<2,1>::T_check_oper>(ufunc->check_oper);
	auto check_init = reinterpret_cast<numsky::generic_ufunc<2,1>::T_check_init>(ufunc->check_init);
	char retypechar = check_type(L, arr->dtype->typechar, arr->dtype->typechar);
	if(retypechar != arr->dtype->typechar) {
		luaL_error(L, "[fatal error]%s.reduce(%s) retype not self", ufunc->name, arr->dtype->name);
	}
	auto fp_oper = check_oper(L, arr->dtype->typechar, arr->dtype->typechar);
	auto fp_init = check_init(L, retypechar);
	std::vector<int> data_axis;
	std::vector<int> reduce_axis;
	data_axis.resize(arr->nd);
	for(int i=0;i<arr->nd;i++) {
		data_axis[i] = i;
	}
	int axis_type = lua_type(L, axis_stacki);
	if(axis_type == LUA_TNIL || axis_type == LUA_TNONE) {
		data_axis.clear();
		reduce_axis.resize(arr->nd);
		for(int i=0;i<arr->nd;i++) {
			reduce_axis[i] = i;
		}
	} else {
		auto check_reduce_axis = [&](int stacki){
			int axis = luaL_checkinteger(L, stacki);
			if(axis <= 0) {
				axis = arr->nd + axis;
			} else {
				axis = axis - 1;
			}
			luaUtils::lassert(axis >= 0 && axis < arr->nd, L, "axis outof range");
			for(auto iter=data_axis.begin();iter!=data_axis.end();iter++) {
				if(*iter == axis) {
					data_axis.erase(iter);
					reduce_axis.push_back(axis);
					return ;
				}
			}
			luaL_error(L, "axis error when reduce");
		};
		if(axis_type == LUA_TTABLE) {
			int len = luaL_len(L, axis_stacki);
			for(int i=1;i<=len;i++) {
				lua_geti(L, axis_stacki, i);
				check_reduce_axis(-1);
			}
		} else if(axis_type == LUA_TNUMBER) {
			check_reduce_axis(axis_stacki);
		} else {
			luaL_error(L, "axis type error");
		}
	}
	if(arr->count <= 0) {
		luaL_error(L, "reduce array has no elements");
	}
	if(data_axis.size() > 0) {
		auto ret_arr = numsky::ndarray_new_alloc<true>(L, data_axis.size(), retypechar, [&](int i)-> npy_intp {
			return arr->dimensions[data_axis[i]];
		});
		char* ret_dataptr = ret_arr->dataptr;
		int itemsize = arr->dtype->elsize;
		numsky::ndarray_axis_foreach(arr, data_axis, [&](numsky_nditer *data_iter){
			fp_init(L, ret_dataptr, data_iter->dataptr);
			numsky::ndarray_axis_foreach(arr, reduce_axis, [&](numsky_nditer *reduce_iter){
				fp_oper(L, ret_dataptr, ret_dataptr, data_iter->dataptr + (reduce_iter->dataptr - arr->dataptr));
			});
			ret_dataptr += itemsize;
		});
		return 1;
	} else {
		char ret_dataptr[16];
		fp_init(L, ret_dataptr, arr->dataptr);
		numsky::ndarray_foreach(arr, [&](numsky_nditer *reduce_iter){
			fp_oper(L, ret_dataptr, ret_dataptr, reduce_iter->dataptr);
		});
		lnumsky_template_fp(L, arr->dtype->typechar, numsky::dataptr_push)(L, ret_dataptr);
		return 1;
	}
}

int numsky::ufunc_reduce(lua_State *L) {
	const numsky_ufunc* ufunc = luabinding::ClassUtil<numsky_ufunc>::check(L, 1);
	numsky_ndarray *arr = luabinding::ClassUtil<numsky_ndarray>::check(L, 2);
	return numsky_ufunc_reduce(L, ufunc, arr, 3);
}

int numsky::methods_sum(lua_State *L) {
	const numsky_ufunc* ufunc = &numsky::ufunc_instance<numsky::UFUNC_add>::ufunc;
	numsky_ndarray *arr = luabinding::ClassUtil<numsky_ndarray>::check(L, 1);
	return numsky_ufunc_reduce(L, ufunc, arr, 2);
}

int numsky::methods_prod(lua_State *L) {
	const numsky_ufunc* ufunc = &numsky::ufunc_instance<numsky::UFUNC_mul>::ufunc;
	numsky_ndarray *arr = luabinding::ClassUtil<numsky_ndarray>::check(L, 1);
	return numsky_ufunc_reduce(L, ufunc, arr, 2);
}

int numsky::methods_any(lua_State *L) {
	const numsky_ufunc* ufunc = &numsky::ufunc_instance<numsky::UFUNC_bor>::ufunc;
	numsky_ndarray *arr = luabinding::ClassUtil<numsky_ndarray>::check(L, 1);
	return numsky_ufunc_reduce(L, ufunc, arr, 2);
}

int numsky::methods_all(lua_State *L) {
	const numsky_ufunc* ufunc = &numsky::ufunc_instance<numsky::UFUNC_band>::ufunc;
	numsky_ndarray *arr = luabinding::ClassUtil<numsky_ndarray>::check(L, 1);
	return numsky_ufunc_reduce(L, ufunc, arr, 2);
}

int numsky::methods_max(lua_State *L) {
	const numsky_ufunc* ufunc = &numsky::ufunc_instance<numsky::UFUNC_fmax>::ufunc;
	numsky_ndarray *arr = luabinding::ClassUtil<numsky_ndarray>::check(L, 1);
	return numsky_ufunc_reduce(L, ufunc, arr, 2);
}

int numsky::methods_min(lua_State *L) {
	const numsky_ufunc* ufunc = &numsky::ufunc_instance<numsky::UFUNC_fmin>::ufunc;
	numsky_ndarray *arr = luabinding::ClassUtil<numsky_ndarray>::check(L, 1);
	return numsky_ufunc_reduce(L, ufunc, arr, 2);
}
