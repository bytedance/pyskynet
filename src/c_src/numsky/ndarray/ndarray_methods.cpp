#include "numsky/lua-numsky.h"
#include "numsky/lua-numsky_module.h"

#include "numsky/ndarray/lua-numsky_ndarray.h"

/*******************
 * ndarray methods *
 *******************/

/*************************************
 * flatten & reshape & copy & astype *
 *************************************/

template <typename Told, typename Tnew> static void _ndarray_copy(numsky_ndarray* old_arr, numsky_ndarray* new_arr) {
	char *new_dataptr = new_arr->dataptr;
	numsky::ndarray_foreach(old_arr, [&](numsky_nditer* iter) {
		numsky::dataptr_cast<Tnew>(new_dataptr) = numsky::dataptr_cast<Told>(iter->dataptr);
		new_dataptr += sizeof(Tnew);
	});
}

int numsky::ndarray_methods_flatten(lua_State *L) {
	auto old_arr = luabinding::ClassUtil<numsky_ndarray>::check(L, 1);
	auto new_arr_ptr = numsky::ndarray_new_alloc<true>(L, 1, old_arr->dtype->typechar, [&](int)-> npy_intp {
			return old_arr->count;
	});
	lnumsky_template_fp2(L, old_arr->dtype->typechar, new_arr_ptr->dtype->typechar, _ndarray_copy)(old_arr, new_arr_ptr.get());
	return 1;
}

int numsky::ndarray_methods_reshape(lua_State *L) {
	numsky_ndarray* old_arr = luabinding::ClassUtil<numsky_ndarray>::check(L, 1);
	numsky_ndarray* new_arr = NULL;
	int shape_type = lua_type(L, 2);
	if(shape_type==LUA_TTABLE) {
		int nd = lua_rawlen(L, 2);
		auto new_arr_ptr = numsky::ndarray_new_alloc<true>(L, nd, old_arr->dtype->typechar, [&](int i) -> npy_intp {
			lua_rawgeti(L, 2, i+1);
			int dim = luaL_checkinteger(L, -1);
			lua_pop(L,1);
			return dim;
		});
		new_arr = new_arr_ptr.get();
	} else if (shape_type == LUA_TNUMBER && lua_isinteger(L, 2)) {
		int nd = lua_gettop(L) - 1;
		auto new_arr_ptr = numsky::ndarray_new_alloc<true>(L, nd, old_arr->dtype->typechar, [&](int i) -> npy_intp {
			return luaL_checkinteger(L, i + 2);
		});
		new_arr = new_arr_ptr.get();
	} else {
		return luaL_error(L, "reshape args error");
	}
	if(old_arr->count != new_arr->count) {
		return luaL_error(L, "reshape %s -> %s failed",
				numsky::shape_str(old_arr).c_str(),
				numsky::shape_str(new_arr).c_str());
	}
	lnumsky_template_fp2(L, old_arr->dtype->typechar, new_arr->dtype->typechar, _ndarray_copy)(old_arr, new_arr);
	return 1;
}

int numsky::ndarray_methods_astype(lua_State *L) {
	auto old_arr = luabinding::ClassUtil<numsky_ndarray>::check(L, 1);
	auto new_dtype = luabinding::ClassUtil<numsky_dtype>::check(L, 2);
	auto new_arr_ptr = numsky::ndarray_new_alloc<true>(L, old_arr->nd, new_dtype->typechar, [&](int i)-> npy_intp {
			return old_arr->dimensions[i];
	});
	auto new_arr = new_arr_ptr.get();
	lnumsky_template_fp2(L, old_arr->dtype->typechar, new_arr->dtype->typechar, _ndarray_copy)(old_arr, new_arr);
	return 1;
}

int numsky::ndarray_methods_copy(lua_State *L) {
	auto old_arr = luabinding::ClassUtil<numsky_ndarray>::check(L, 1);
	auto new_arr_ptr = numsky::ndarray_new_alloc<true>(L, old_arr->nd, old_arr->dtype->typechar, [&](int i)-> npy_intp {
			return old_arr->dimensions[i];
	});
	lnumsky_template_fp2(L, old_arr->dtype->typechar, new_arr_ptr->dtype->typechar, _ndarray_copy)(old_arr, new_arr_ptr.get());
	return 1;
}

int numsky::ndarray_methods_roll(lua_State *L) {
	auto old_arr = luabinding::ClassUtil<numsky_ndarray>::check(L, 1);
	npy_intp shift = luaL_checkinteger(L, 2);
	int axis = 0;
	int top = lua_gettop(L);
	if(top == 3) {
		int axis_args = luaL_checkinteger(L, 3);
		if(axis_args < 0) {
			axis = old_arr->nd + axis_args;
		} else {
			axis = axis_args - 1;
		}
	} else if(top != 2) {
		luaL_error(L, "roll take arguments (arr, shift[,axis])");
	}
	if(axis < 0 || axis >= old_arr->nd) {
		luaL_error(L, "axis out of range");
	}
	auto new_arr_ptr = numsky::ndarray_new_alloc<true>(L, old_arr->nd, old_arr->dtype->typechar, [&](int i)-> npy_intp {
			return old_arr->dimensions[i];
	});
	auto item_copy_func = lnumsky_template_fp(L, old_arr->dtype->typechar, numsky::dataptr_copy);
    ndarray_foreach(new_arr_ptr.get(), [&](numsky_nditer*iter) {
		char* old_dataptr = old_arr->dataptr;
		for(int i=0;i<old_arr->nd;i++) {
			if(i != axis) {
				old_dataptr += iter->coordinates[i] * old_arr->strides[i];
			} else {
				npy_intp dimi = old_arr->dimensions[i];
				npy_intp offset = (iter->coordinates[i] - shift) % dimi;
				if(offset < 0) {
					offset += dimi;
				}
				old_dataptr += offset * old_arr->strides[i];
			}
		}
		item_copy_func(iter->dataptr, old_dataptr);
	});
	return 1;
}

/********
 * TODO *
 ********/

/********
 * fill *
 ********/

/*********************
 * tobytes, tostring *
 *********************/

/***********
 * tolist *
 ***********/

/***********
 * squeeze *
 ***********/
