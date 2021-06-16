
#include <cmath>
#include <vector>

#include "numsky/lua-numsky.h"
#include "numsky/lua-numsky_module.h"
#include "numsky/ndarray/lua-numsky_ndarray.h"


/****************
 * numsky.array *
 ****************/

template <typename TLeft, typename TRight> static char* array_fill(numsky::ThrowableContext *ctx, numsky_ndarray *arr, char* dataptr, int depth, numsky_ndarray *sub_arr) {
	if(sub_arr->nd + depth != arr->nd) {
		ctx->throw_func("dim not match when constructor array");
		return NULL;
	}
	for(int i=0;i<sub_arr->nd;i++) {
		if(sub_arr->dimensions[i] != arr->dimensions[depth + i]) {
			ctx->throw_func("dim not match when constructor array");
			return NULL;
		}
	}
	numsky::ndarray_foreach(sub_arr, [&](numsky_nditer *iter){
		numsky::dataptr_cast<TLeft>(dataptr) = numsky::dataptr_cast<TRight>(iter->dataptr);
		dataptr += sizeof(TLeft);
	});
	return dataptr;
}
/* raise error if wrong type */
template <typename T> static char* table_fill(numsky::ThrowableContext *ctx, numsky_ndarray *arr, char* dataptr, int depth) {
	lua_State *L = ctx->L;
	int toptype = lua_type(L, -1);
	if(depth == arr->nd) {
		if(arr->dtype->typechar == '?' && toptype != LUA_TBOOLEAN) {
			ctx->throw_func("array(arg1,) error, arg1's content value type expect boolean");
			return NULL;
		}
		if(arr->dtype->typechar != '?' && toptype != LUA_TNUMBER) {
			ctx->throw_func("array(arg1,) error, arg1's content value type expect number");
			return NULL;
		}
		numsky::dataptr_cast<T>(dataptr) = numsky::generic<T>::check(L, -1);
		dataptr += sizeof(T);
	} else {
		if(toptype == LUA_TTABLE) {
			int dim = luaL_len(L, -1);
			if(dim != arr->dimensions[depth]) {
				ctx->throw_func("array(arg1,) error, content size not match");
				return NULL;
			}
			for(int i=0;i<dim;i++) {
				lua_geti(L, -1, i + 1);
				dataptr = table_fill<T>(ctx, arr, dataptr, depth + 1);
				lua_pop(L, 1);
			}
		} else if(toptype == LUA_TUSERDATA) {
			numsky_ndarray *sub_arr = luabinding::ClassUtil<numsky_ndarray>::test(L, -1);
			if(sub_arr==NULL) {
				ctx->throw_func("numsky.array constructor's content get unexcepted userdata");
			}
			dataptr = lnumsky_template_fp2t1(L, T, sub_arr->dtype->typechar, array_fill)(ctx, arr, dataptr, depth, sub_arr);
		} else {
			ctx->throw_func("array(arg1,) error, arg1's content must be table or numsky.ndarray");
			return NULL;
		}
	}
	return dataptr;
}


template <bool InLua> std::unique_ptr<numsky_ndarray, void(*)(numsky_ndarray*)>
numsky::table_to_array(numsky::ThrowableContext *ctx, int table_idx, char typechar) {
	lua_State*L = ctx->L;
	// step 1. get dims
	int ntop = lua_gettop(L);
	lua_pushvalue(L, table_idx);
	std::vector<int> dims;
	npy_intp count = 1;
	dims.reserve(10);
	int toptype = lua_type(L, table_idx);
	while(toptype == LUA_TTABLE) {
		int dim = luaL_len(L, -1);
		count *= dim;
		dims.push_back(dim);
		if(dim == 0) {
			break;
		} else {
			lua_geti(L, -1, 1);
			toptype = lua_type(L, -1);
		}
	}
	if(toptype == LUA_TUSERDATA) {
		numsky_ndarray *sub_arr = luabinding::ClassUtil<numsky_ndarray>::test(L, -1);
		if(sub_arr==NULL) {
			ctx->throw_func("numsky.array constructor's content get unexcepted userdata");
		}
		for(int i=0;i<sub_arr->nd;i++) {
			int dim = sub_arr->dimensions[i];
			count *= dim;
			dims.push_back(dim);
			if(dim==0) {
				break;
			}
		}
		if(typechar == '\0') {
			typechar = sub_arr->dtype->typechar;
		}
	} else if(toptype == LUA_TNUMBER || toptype == LUA_TBOOLEAN) {
		if(typechar == '\0') {
			if(count == 0) {
				typechar = numsky::generic<int64_t>::typechar;
			} else if(toptype == LUA_TNUMBER) {
				if(lua_isinteger(L, -1)) {
					typechar = numsky::generic<int64_t>::typechar;
				} else {
					typechar = numsky::generic<double>::typechar;
				}
			} else if(toptype == LUA_TBOOLEAN) {
				typechar = numsky::generic<bool>::typechar;
			}
		}
	} else {
		ctx->throw_func("cast table to array failed, arg1's inner content must be table or numsky.ndarray");
	}
	lua_settop(L, ntop);
	auto arr_ptr = numsky::ndarray_new_alloc<InLua>(L, dims.size(), typechar, [&](int i) -> npy_intp {
		return dims[i];
	});
	numsky_ndarray* arr = arr_ptr.get();
	lua_pushvalue(L, table_idx);
	auto fp_table_fill = lnumsky_template_fp(L, typechar, table_fill);
	fp_table_fill(ctx, arr, arr->dataptr, 0);
	if(InLua) {
		lua_settop(L, ntop + 1);
	} else {
		lua_settop(L, ntop);
	}
	return arr_ptr;
}

template std::unique_ptr<numsky_ndarray, void(*)(numsky_ndarray*)>
numsky::table_to_array<true>(numsky::ThrowableContext *ctx, int table_idx, char typechar);
template std::unique_ptr<numsky_ndarray, void(*)(numsky_ndarray*)>
numsky::table_to_array<false>(numsky::ThrowableContext *ctx, int table_idx, char typechar);

/*********************
 * routines creation *
 *********************/

static numsky_ndarray* lnumsky_empty_prebuild(lua_State *L) {
    char typechar = 'f';
    if(!lua_isnone(L, 2)) {
	   struct numsky_dtype *dtype = luabinding::ClassUtil<numsky_dtype>::check(L, 2);
	   typechar = dtype->typechar;
    }
	numsky_ndarray* arr = NULL;
	int shape_type = lua_type(L, 1);
	if(shape_type==LUA_TTABLE) {
		int nd = lua_rawlen(L, 1);
		arr = numsky::ndarray_new_alloc<true>(L, nd, typechar, [&](int i) -> npy_intp {
			lua_rawgeti(L, 1, i+1);
			int dim = luaL_checkinteger(L, -1);
			lua_pop(L, 1);
			return dim;
		}).get();
	} else if (shape_type == LUA_TNUMBER && lua_isinteger(L, 1)) {
		arr = numsky::ndarray_new_alloc<true>(L, 1, typechar, [&](int i) -> npy_intp {
			return lua_tointeger(L, 1);
		}).get();
	} else {
		luaL_error(L, "ctor args error");
	}
	return arr;
}

int numsky::ctor_empty(lua_State *L) {
	lnumsky_empty_prebuild(L);
    return 1;
}

int numsky::ctor_zeros(lua_State *L) {
	auto arr = lnumsky_empty_prebuild(L);
    memset(arr->dataptr, 0, arr->dtype->elsize * arr->count);
    return 1;
}

template <typename T> static void fill_one(void *dataptr, size_t count) {
	T* data = reinterpret_cast<T*>(dataptr);
	std::fill_n(data, count, (T)(1));
}

int numsky::ctor_ones(lua_State *L) {
	auto arr = lnumsky_empty_prebuild(L);
	lnumsky_template_fp(L, arr->dtype->typechar, fill_one)(arr->dataptr, arr->count);
    return 1;
}

/********************
 * numerical ranges *
 ********************/

template <typename T> static int fill_arange(lua_State *L, bool checkstep){
	T start = numsky::generic<T>::check(L, 1);
	T stop = numsky::generic<T>::check(L, 2);
	T step;
	if(checkstep) {
		step = numsky::generic<T>::check(L, 3);
		luaUtils::lassert(step != 0, L, "step can't be zero");
	} else {
		step = 1;
	}
	int dim0 = ((stop-start)/step) + 1;
	auto arr_ptr = numsky::ndarray_new_alloc<true>(L, 1, numsky::generic<T>::typechar, [&](int i) -> npy_intp {
			return dim0;
	});
	T *data = reinterpret_cast<T*>(arr_ptr->dataptr);
	for(int i=0;i<dim0;i++){
		data[i] = start + i * step;
	}
	return 1;
}

int numsky::ctor_arange(lua_State *L) {
	int top = lua_gettop(L);
	luaUtils::lassert(top>=2, L, "numsky.arange(start, stop, [num,] [dtype,]) got wrong args");
	auto dtype = luabinding::ClassUtil<numsky_dtype>::test(L, -1);
	if (dtype == NULL) {
		luaUtils::lassert(top == 2 || top == 3, L, "numsky.arange(start, stop, [step,] [dtype,]) got wrong args");
		if(lua_isinteger(L, 1) && lua_isinteger(L, 2)) {
			if(lua_isinteger(L, 3) || lua_isnone(L, 3)) {
				return fill_arange<int64_t>(L, top == 3);
			}
		}
		return fill_arange<double>(L, top == 3);
	} else {
		luaUtils::lassert(dtype->typechar != '?', L, "numsky.arange can't use bool as dtype");
		luaUtils::lassert(top==3 || top == 4, L, "numsky.arange(start, stop, [step,] [dtype,]) got wrong args");
		return lnumsky_template_fp(L, dtype->typechar, fill_arange)(L, top == 4);
	}
}

template <typename T> static int fill_linspace(lua_State *L, double start, double stop, int num, bool endpoint){
	double step = 0.0;
	luaUtils::lassert(num >= 0, L, "numsky.linspace's num must >= 0");
	if(num > 1) {
		if(endpoint) {
			step = (stop - start)/(num - 1);
		} else {
			step = (stop - start)/num;
		}
	}
	auto arr_ptr = numsky::ndarray_new_alloc<true>(L, 1, numsky::generic<T>::typechar, [&](int)-> npy_intp {
			return num;
	});
	T *data = reinterpret_cast<T*>(arr_ptr->dataptr);
	for(int i=0;i<num;i++){
		data[i] = start + i * step;
	}
	if(num > 0 && endpoint) {
		data[num-1] = stop;
	}
	return 1;
}

int numsky::ctor_linspace(lua_State *L) {
	luaUtils::lassert(lua_gettop(L)>=2, L, "numsky.linspace(start, stop, num, endpoint=true, dtype=int64) got wrong args");
	auto dtype = luabinding::ClassUtil<numsky_dtype>::test(L, -1);
	if (dtype != NULL) {
		luaUtils::lassert(dtype->typechar != '?', L, "numsky.linspace can't use bool as dtype");
		lua_pop(L, 1);
	} else {
		dtype = numsky_get_dtype_by_char('d');
	}
	bool endpoint = true;
	if(lua_type(L, -1) == LUA_TBOOLEAN) {
		endpoint = lua_toboolean(L, -1);
		lua_pop(L, 1);
	}
	luaUtils::lassert(lua_gettop(L)==3, L, "numsky.linspace(start, stop, num, endpoint=true, dtype=int64) got wrong args");
	return lnumsky_template_fp(L, dtype->typechar, fill_linspace)(L, luaL_checknumber(L, 1), luaL_checknumber(L, 2), luaL_checkinteger(L, 3), endpoint);
}

/* not used frequence
template <typename T> static int fill_logspace(lua_State *L, double start, double stop, int num, double base){
	double step = 0.0;
	luaUtils::lassert(num >= 0, L, "numsky.logspace's num must >= 0");
	if(num > 1) {
		step = (stop - start)/(num - 1);
	}
	auto arr = numsky::ndarray_new_alloc<true>(L, 1, numsky::generic<T>::typechar, [&](int)->npy_intp{
			return num;
	}).get();
	T *data = reinterpret_cast<T*>(arr->dataptr);
	for(int i=0;i<num-1;i++){
		data[i] = pow(base, start + i * step);
	}
	if(num > 0) {
		data[num-1] = pow(base, stop);
	}
	return 1;
}

int numsky::ctor_logspace(lua_State *L) {
	int top = lua_gettop(L);
	luaUtils::lassert(top>=2, L, "numsky.logspace(start, stop, [num,] [dtype,] [base,]) got wrong args");
	auto dtype = luabinding::ClassUtil<numsky_dtype>::test(L, 3);
	if(dtype != NULL) {
		if(top == 3){
			return lnumsky_template_fp(L, dtype->typechar, fill_logspace)(L, luaL_checknumber(L, 1), luaL_checknumber(L, 2), 50, 10);
		} else if(top == 4) {
			return lnumsky_template_fp(L, dtype->typechar, fill_logspace)(L, luaL_checknumber(L, 1), luaL_checknumber(L, 2), 50, luaL_checknumber(L, 4));
		} else {
			return luaL_error(L, "numsky.logspace(start, stop, [num,] [dtype,] [base,]) got wrong args");
		}
	}
	dtype = luabinding::ClassUtil<numsky_dtype>::test(L, 4);
	if(dtype != NULL) {
		if(top == 4){
			return lnumsky_template_fp(L, dtype->typechar, fill_logspace)(L, luaL_checknumber(L, 1), luaL_checknumber(L, 2), luaL_checkinteger(L, 3), 10);
		} else if(top == 5) {
			return lnumsky_template_fp(L, dtype->typechar, fill_logspace)(L, luaL_checknumber(L, 1), luaL_checknumber(L, 2), luaL_checkinteger(L, 3), luaL_checknumber(L, 5));
		} else {
			return luaL_error(L, "numsky.logspace(start, stop, [num,] [dtype,] [base,]) got wrong args");
		}
	}
	if(top == 2) {
		return fill_logspace<double>(L, luaL_checknumber(L, 1), luaL_checknumber(L, 2), 50, 10);
	} else if(top == 3) {
		return fill_logspace<double>(L, luaL_checknumber(L, 1), luaL_checknumber(L, 2), luaL_checkinteger(L, 3), 10);
	} else if(top == 4) {
		return fill_logspace<double>(L, luaL_checknumber(L, 1), luaL_checknumber(L, 2), luaL_checkinteger(L, 3), luaL_checknumber(L ,4));
	} else {
		return luaL_error(L, "numsky.logspace(start, stop, [num,] [dtype,] [base,]) got wrong args");
	}
}
*/
